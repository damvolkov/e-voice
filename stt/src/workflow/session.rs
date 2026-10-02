use std::collections::VecDeque;
use std::collections::vec_deque::Drain;
use std::time::Duration;

use crate::config::pipeline::{GateClose, Overload, PipelineConfig};
use crate::schema::audio::Audio;
use crate::schema::emotion::Emotion;
use crate::schema::error::NodeError;
use crate::schema::event::{Event, FinalEvent, PartialEvent, SpeechEvent, SpeechState, WakeEvent};
use crate::schema::lang::Lang;
use crate::schema::mode::Mode;
use crate::schema::segment::{Segment, SegmentId, SegmentSpan};
use crate::workflow::gate::{Gate, GatePolicy, GateRoute};
use crate::workflow::join::{Branch, Join, Slot};

/// The required ASR node: how it consumes speech and how long it may take once speech ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrPlan {
    pub mode: Mode,
    pub deadline: Duration,
}

/// The optional SER node: its deadline and the shortest segment, in samples, worth classifying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerPlan {
    pub deadline: Duration,
    pub min: u64,
}

/// Everything one session decides by; built per stream from configuration and backend shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionPlan {
    pub lang: Lang,
    pub asr: AsrPlan,
    pub ser: Option<SerPlan>,
    pub gate: GatePolicy,
    pub pending: usize,
    pub overload: Overload,
    pub stall: Duration,
}

impl SessionPlan {
    /// `mode` comes from the ASR backend; `gated` is whether a wake-word backend exists.
    #[must_use]
    pub fn new(config: &PipelineConfig, lang: Lang, mode: Mode, gated: bool, classify: bool) -> Self {
        let idle = config.gate.idle;
        let gate = match (gated, config.gate.close) {
            (false, _) => GatePolicy::Disabled,
            (true, GateClose::Utterance) => GatePolicy::Utterance { idle },
            (true, GateClose::Window) => GatePolicy::Window { idle },
            (true, GateClose::Session) => GatePolicy::Session,
        };
        let ser = classify.then(|| SerPlan {
            deadline: config.ser.deadline,
            min: Audio::length(config.ser.min),
        });
        Self {
            lang,
            asr: AsrPlan {
                mode,
                deadline: config.asr.deadline,
            },
            ser,
            gate,
            pending: config.pending,
            overload: config.overload,
            stall: config.stall,
        }
    }
}

/// Facts fed into a session: node outputs and the passage of time.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionInput {
    Wake(WakeEvent),
    Start {
        at: u64,
    },
    End {
        span: SegmentSpan,
        audio: Audio,
    },
    Partial {
        segment: SegmentId,
        text: String,
    },
    Transcript {
        segment: SegmentId,
        result: Result<String, NodeError>,
    },
    Emotion {
        segment: SegmentId,
        result: Result<Emotion, NodeError>,
    },
    Tick,
    Drain,
    Cancel,
}

/// Work the runner must perform on the session's behalf.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionCommand {
    Route(GateRoute),
    Begin(SegmentId),
    Finish(SegmentId),
    Transcribe(Segment),
    Classify(Segment),
    Abort(SegmentId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionOutput {
    Command(SessionCommand),
    Event(Event),
}

/// One stream's pipeline as a sans-IO state machine: inputs in, commands and events out.
///
/// Guarantees: one `Final` per segment, in segment order; at most `pending` live segments;
/// a `Closed` event, once, after `Drain` or `Cancel` when nothing is pending.
#[derive(Debug)]
pub struct Session {
    plan: SessionPlan,
    gate: Gate,
    join: Join,
    active: Option<SegmentId>,
    draining: bool,
    closed: bool,
    out: VecDeque<SessionOutput>,
}

impl Session {
    // ##### PRIVATE #####

    fn common_commands(&mut self, commands: impl IntoIterator<Item = SessionCommand>) {
        self.out.extend(commands.into_iter().map(SessionOutput::Command));
    }

    fn common_events(&mut self, events: impl IntoIterator<Item = Event>) {
        self.out.extend(events.into_iter().map(SessionOutput::Event));
    }

    fn common_route(&mut self, route: Option<GateRoute>) {
        self.common_commands(route.map(SessionCommand::Route));
    }

    fn step_wake(&mut self, now: Duration, wake: WakeEvent) {
        let Some(route) = self.gate.enabled().then(|| self.gate.wake(now)) else {
            return;
        };
        self.common_events([Event::Wake(wake)]);
        self.common_route(route);
    }

    fn step_start(&mut self, now: Duration, at: u64) {
        self.gate.touch(now);
        self.active = match (self.active, self.draining) {
            (None, false) => Some(self.step_open(now, at)),
            (active @ Some(_), _) | (active @ None, true) => active,
        };
    }

    fn step_open(&mut self, now: Duration, at: u64) -> SegmentId {
        let full = self.join.live() >= self.plan.pending.max(1);
        let evicted = match (full, self.plan.overload) {
            (true, Overload::Evict) => self.join.oldest(),
            (true, Overload::Reject) | (false, _) => None,
        };
        let aborts = evicted.map(|id| self.step_evict(id));
        let dropped = full && self.plan.overload == Overload::Reject;
        let id = self.join.open(at, now, dropped);
        let begin = match (self.plan.asr.mode, dropped) {
            (Mode::Streaming, false) => Some(SessionCommand::Begin(id)),
            (Mode::Batch, _) | (Mode::Streaming, true) => None,
        };
        self.common_events((!dropped).then_some(Event::Speech(SpeechEvent {
            segment: id,
            state: SpeechState::Started,
            at,
        })));
        self.common_commands(aborts.into_iter().chain(begin));
        id
    }

    fn step_evict(&mut self, id: SegmentId) -> SessionCommand {
        if let Some(slot) = self.join.slot(id) {
            slot.fail(&NodeError::Overload);
        }
        self.active = self.active.filter(|active| *active != id);
        SessionCommand::Abort(id)
    }

    fn step_end(&mut self, now: Duration, span: SegmentSpan, audio: Audio) {
        let route = self.gate.end(now);
        self.common_route(route);
        let id = match (self.active.take(), self.draining) {
            (Some(id), _) => Some(id),
            (None, false) => Some(self.step_open(now, span.start)),
            (None, true) => None,
        };
        if let Some(id) = id {
            self.step_close(now, &Segment { id, span, audio });
        }
    }

    fn step_close(&mut self, now: Duration, segment: &Segment) {
        let plan = self.plan;
        let Some(slot) = self.join.slot(segment.id) else {
            return;
        };
        let open = std::mem::replace(&mut slot.open, false);
        slot.span = segment.span;
        let stopped = (open && !slot.dropped).then_some(Event::Speech(SpeechEvent {
            segment: segment.id,
            state: SpeechState::Stopped,
            at: segment.span.end,
        }));
        let commands = match (open, slot.dropped) {
            (true, false) => Self::step_close_branches(&plan, now, slot, segment),
            (true, true) => {
                slot.fail(&NodeError::Overload);
                Vec::new()
            }
            (false, _) => Vec::new(),
        };
        self.common_events(stopped);
        self.common_commands(commands);
    }

    fn step_close_branches(
        plan: &SessionPlan,
        now: Duration,
        slot: &mut Slot,
        segment: &Segment,
    ) -> Vec<SessionCommand> {
        let asr = match (&slot.asr, plan.asr.mode) {
            (Branch::Waiting, Mode::Streaming) => Some(SessionCommand::Finish(segment.id)),
            (Branch::Waiting, Mode::Batch) => Some(SessionCommand::Transcribe(segment.clone())),
            (Branch::Running { .. } | Branch::Ready(_), _) => None,
        };
        if asr.is_some() {
            slot.asr = Branch::Running {
                deadline: now.saturating_add(plan.asr.deadline),
            };
        }
        let ser = match (&slot.ser, plan.ser) {
            (Branch::Waiting, Some(ser)) if segment.span.len() >= ser.min => {
                slot.ser = Branch::Running {
                    deadline: now.saturating_add(ser.deadline),
                };
                Some(SessionCommand::Classify(segment.clone()))
            }
            (Branch::Waiting, _) => {
                slot.ser.settle(Ok(Emotion::default()));
                None
            }
            (Branch::Running { .. } | Branch::Ready(_), _) => None,
        };
        asr.into_iter().chain(ser).collect()
    }

    fn step_partial(&mut self, segment: SegmentId, text: String) {
        let live = self.join.slot(segment).is_some_and(|slot| !slot.asr.ready());
        self.common_events(live.then_some(Event::Partial(PartialEvent { segment, text })));
    }

    fn step_cancel(&mut self) {
        self.draining = true;
        self.active = None;
        let aborts: Vec<SessionCommand> = self
            .join
            .slots()
            .filter(|(_, slot)| !slot.done())
            .filter_map(|(id, slot)| {
                slot.fail(&NodeError::Cancelled);
                (!slot.dropped).then_some(SessionCommand::Abort(id))
            })
            .collect();
        self.common_commands(aborts);
    }

    fn step_apply(&mut self, now: Duration, input: SessionInput) {
        match input {
            SessionInput::Wake(wake) => self.step_wake(now, wake),
            SessionInput::Start { at } => self.step_start(now, at),
            SessionInput::End { span, audio } => self.step_end(now, span, audio),
            SessionInput::Partial { segment, text } => self.step_partial(segment, text),
            SessionInput::Transcript { segment, result } => {
                if let Some(slot) = self.join.slot(segment) {
                    slot.asr.accept(result);
                }
            }
            SessionInput::Emotion { segment, result } => {
                if let Some(slot) = self.join.slot(segment) {
                    slot.ser.accept(result);
                }
            }
            SessionInput::Tick => {
                let route = self.gate.tick(now, self.active.is_some());
                self.common_route(route);
            }
            SessionInput::Drain => self.draining = true,
            SessionInput::Cancel => self.step_cancel(),
        }
    }

    fn step_expire(&mut self, now: Duration) {
        let stall = self.plan.stall;
        let expired: Vec<(SegmentId, bool)> = self
            .join
            .slots()
            .filter_map(|(id, slot)| {
                let stalled = slot.open && now.saturating_sub(slot.opened) >= stall;
                stalled.then(|| slot.fail(&NodeError::Stalled));
                let asr = slot.asr.expire(now);
                let ser = slot.ser.expire(now);
                (stalled || asr || ser).then_some((id, slot.dropped))
            })
            .collect();
        self.active = self.active.filter(|active| expired.iter().all(|(id, _)| id != active));
        self.common_commands(
            expired
                .into_iter()
                .filter(|(_, dropped)| !dropped)
                .map(|(id, _)| SessionCommand::Abort(id)),
        );
    }

    fn step_release(&mut self) {
        let lang = self.plan.lang;
        let finals: Vec<Event> = std::iter::from_fn(|| self.join.release())
            .map(|(segment, slot)| {
                let (text, error) = match slot.asr {
                    Branch::Ready(Ok(text)) => (text, None),
                    Branch::Ready(Err(error)) => (String::new(), Some(error)),
                    Branch::Waiting | Branch::Running { .. } => (String::new(), Some(NodeError::Timeout)),
                };
                let emotion = match slot.ser {
                    Branch::Ready(Ok(emotion)) => emotion,
                    Branch::Ready(Err(_)) | Branch::Waiting | Branch::Running { .. } => Emotion::default(),
                };
                Event::Final(FinalEvent {
                    segment,
                    span: slot.span,
                    lang,
                    text,
                    emotion,
                    error,
                })
            })
            .collect();
        self.common_events(finals);
    }

    fn step_seal(&mut self) {
        self.closed = self.draining && self.active.is_none() && self.join.is_empty();
        self.common_events(self.closed.then_some(Event::Closed));
    }

    // ##########################################################

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(plan: SessionPlan) -> Self {
        let gate = Gate::new(plan.gate);
        let out = VecDeque::from([SessionOutput::Command(SessionCommand::Route(gate.route()))]);
        Self {
            plan,
            gate,
            join: Join::default(),
            active: None,
            draining: false,
            closed: false,
            out,
        }
    }

    /// Feeds one input observed at monotonic time `now`; results accumulate until [`Session::poll`].
    pub fn step(&mut self, now: Duration, input: SessionInput) {
        if self.closed {
            return;
        }
        self.step_apply(now, input);
        self.step_expire(now);
        self.step_release();
        self.step_seal();
    }

    pub fn poll(&mut self) -> Drain<'_, SessionOutput> {
        self.out.drain(..)
    }

    #[must_use]
    pub fn pending(&self) -> usize {
        self.join.live()
    }

    /// Ended segments still waiting on nodes; a lossless intake stops reading audio at the limit.
    #[must_use]
    pub fn backlog(&self) -> usize {
        self.join.backlog()
    }

    #[must_use]
    pub fn limit(&self) -> usize {
        self.plan.pending.max(1)
    }

    #[must_use]
    pub const fn closed(&self) -> bool {
        self.closed
    }
}
