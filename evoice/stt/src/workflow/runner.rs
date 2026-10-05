use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Duration;

use e_voice_core::audio::AudioGain;
use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::{AbortHandle, block_in_place, spawn_blocking};
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::config::asr::AsrEngine;
use crate::config::pipeline::PipelineConfig;
use crate::schema::audio::{Audio, RATE};
use crate::schema::emotion::Emotion;
use crate::schema::event::Event;
use crate::schema::segment::{Segment, SegmentId, SegmentSpan};
use crate::schema::transcript::Transcript;
use crate::workflow::asr::base::AsrSession;
use crate::workflow::asr::registry::AsrBackend;
use crate::workflow::denoise::base::DenoiseSession;
use crate::workflow::gate::GateRoute;
use crate::workflow::nodes::Nodes;
use crate::workflow::session::{Session, SessionCommand, SessionInput, SessionOutput, SessionPlan};
use crate::workflow::vad::base::{VadEvent, VadSession};
use crate::workflow::ww::base::WwSession;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunnerError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("{0} panicked on the frame path; stream closed")]
    Panicked(&'static str),
    #[error("the run ended before closing its session")]
    Incomplete,
}

/// How audio enters a run. `Live` follows the configured gate and sheds load by the overload policy;
/// `File` skips the gate and stops reading audio while the backlog is full, so no segment is lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerIntake {
    Live,
    File,
}

/// Drives one stream: audio in, events out. The frame path (wake word → VAD → streaming ASR) runs
/// in order on this task; segment work runs on the blocking pool under a semaphore shared by all
/// streams. Every decision is the [`Session`]'s; the runner only executes its commands.
#[derive(Debug, Clone)]
pub struct Runner {
    nodes: Arc<Nodes>,
    config: Arc<PipelineConfig>,
    jobs: Arc<Semaphore>,
}

struct Stream {
    asr: AsrBackend,
    denoise: Option<Box<dyn DenoiseSession>>,
    gain: AudioGain,
    session: Session,
    route: GateRoute,
    ww: Option<Box<dyn WwSession>>,
    vad: Option<(Box<dyn VadSession>, u64)>,
    decoding: Option<(SegmentId, Box<dyn AsrSession>, u64)>,
    start: u64,
    ring: VecDeque<f32>,
    pos: u64,
    keep: usize,
    work: HashMap<SegmentId, Vec<AbortHandle>>,
    results: mpsc::UnboundedSender<SessionInput>,
    events: mpsc::Sender<Event>,
    clock: Instant,
}

impl Runner {
    // ##### PRIVATE #####

    fn common_guard<T>(node: &'static str, work: impl FnOnce() -> T) -> Result<T, RunnerError> {
        block_in_place(|| catch_unwind(AssertUnwindSafe(work))).map_err(|_| RunnerError::Panicked(node))
    }

    fn run_spawn<T: Send + 'static>(
        &self,
        stream: &mut Stream,
        id: SegmentId,
        work: impl FnOnce() -> Result<T, NodeError> + Send + 'static,
        wrap: fn(SegmentId, Result<T, NodeError>) -> SessionInput,
    ) {
        let (jobs, results) = (Arc::clone(&self.jobs), stream.results.clone());
        let task = tokio::spawn(async move {
            let _permit = jobs.acquire_owned().await;
            let result = spawn_blocking(work)
                .await
                .unwrap_or_else(|_| Err(NodeError::Backend("backend panicked".to_owned())));
            results.send(wrap(id, result)).ok();
        });
        stream.work.entry(id).or_default().push(task.abort_handle());
    }

    fn run_route(&self, stream: &mut Stream, route: GateRoute) -> Result<(), RunnerError> {
        stream.route = route;
        stream.vad = match route {
            GateRoute::Vad => Some((self.nodes.vad.open()?, stream.pos)),
            GateRoute::Wake => None,
        };
        Ok(())
    }

    fn run_begin(stream: &mut Stream, id: SegmentId, lang: Lang) -> Result<(), RunnerError> {
        let AsrBackend::Streaming(asr) = &stream.asr else {
            return Ok(());
        };
        stream.decoding = Some((id, asr.open(lang)?, stream.start));
        Ok(())
    }

    fn run_command(&self, stream: &mut Stream, command: SessionCommand, lang: Lang) -> Result<(), RunnerError> {
        match command {
            SessionCommand::Route(route) => self.run_route(stream, route)?,
            SessionCommand::Begin(id) => Self::run_begin(stream, id, lang)?,
            SessionCommand::Finish(id) => match stream.decoding.take() {
                Some((active, session, _)) if active == id => {
                    self.run_spawn(
                        stream,
                        id,
                        move || session.finish(),
                        |segment, result| SessionInput::Transcript {
                            segment,
                            result,
                            lang: None,
                        },
                    );
                }
                other => stream.decoding = other,
            },
            SessionCommand::Transcribe(Segment { id, audio, .. }) => {
                if let AsrBackend::Batch(asr) = &stream.asr {
                    let (asr, lid) = (Arc::clone(asr), self.nodes.lid.clone());
                    let min = usize::try_from(Audio::length(self.config.lid.min)).unwrap_or(usize::MAX);
                    self.run_spawn(
                        stream,
                        id,
                        move || {
                            let heard = lid.filter(|_| audio.len() >= min).and_then(|lid| {
                                lid.identify(&audio)
                                    .inspect_err(|error| tracing::warn!(%error, "lid.failed"))
                                    .ok()
                                    .flatten()
                            });
                            asr.transcribe(heard.unwrap_or(lang), &audio).map(|text| (text, heard))
                        },
                        |segment, result| match result {
                            Ok((text, lang)) => SessionInput::Transcript {
                                segment,
                                result: Ok(text),
                                lang,
                            },
                            Err(error) => SessionInput::Transcript {
                                segment,
                                result: Err(error),
                                lang: None,
                            },
                        },
                    );
                }
            }
            SessionCommand::Classify(Segment { id, audio, .. }) => {
                if let Some(ser) = &self.nodes.ser {
                    let ser = Arc::clone(ser);
                    self.run_spawn(
                        stream,
                        id,
                        move || ser.classify(&audio),
                        |segment, result: Result<Emotion, _>| SessionInput::Emotion { segment, result },
                    );
                }
            }
            SessionCommand::Abort(id) => {
                stream
                    .work
                    .remove(&id)
                    .into_iter()
                    .flatten()
                    .for_each(|task| task.abort());
                stream.decoding = stream.decoding.take().filter(|(active, ..)| *active != id);
            }
        }
        Ok(())
    }

    async fn run_step(&self, stream: &mut Stream, input: SessionInput, lang: Lang) -> Result<(), RunnerError> {
        stream.session.step(stream.clock.elapsed(), input);
        let outputs: Vec<SessionOutput> = stream.session.poll().collect();
        for output in outputs {
            match output {
                SessionOutput::Command(command) => self.run_command(stream, command, lang)?,
                SessionOutput::Event(event) => {
                    if let Event::Final(done) = &event {
                        stream.work.remove(&done.segment);
                    }
                    if stream.events.send(event).await.is_err() {
                        stream.session.step(stream.clock.elapsed(), SessionInput::Cancel);
                    }
                }
            }
        }
        Ok(())
    }

    async fn run_vad(
        &self,
        stream: &mut Stream,
        events: Vec<VadEvent>,
        offset: u64,
        lang: Lang,
    ) -> Result<(), RunnerError> {
        for event in events {
            let input = match event {
                VadEvent::Start { at } => {
                    stream.start = at.saturating_add(offset);
                    SessionInput::Start { at: stream.start }
                }
                VadEvent::End { span, audio } => SessionInput::End {
                    span: SegmentSpan {
                        start: span.start.saturating_add(offset),
                        end: span.end.saturating_add(offset),
                    },
                    audio,
                },
            };
            self.run_step(stream, input, lang).await?;
        }
        Ok(())
    }

    async fn run_frame(&self, stream: &mut Stream, chunk: &[f32], lang: Lang) -> Result<(), RunnerError> {
        let clean = match stream.denoise.as_mut() {
            Some(denoise) => Self::common_guard("denoise", || denoise.push(chunk))?,
            None => chunk.to_vec(),
        };
        self.run_clean(stream, clean, lang).await
    }

    async fn run_clean(&self, stream: &mut Stream, mut gained: Vec<f32>, lang: Lang) -> Result<(), RunnerError> {
        if stream.gain.enabled() {
            stream.gain.apply(&mut gained);
        }
        let chunk = gained.as_slice();
        stream.ring.extend(chunk);
        stream.pos = stream.pos.saturating_add(chunk.len() as u64);
        stream.ring.drain(..stream.ring.len().saturating_sub(stream.keep));
        let scored = match (stream.route, stream.ww.as_mut()) {
            (GateRoute::Wake, Some(ww)) => Self::common_guard("ww", || ww.push(chunk))?,
            (GateRoute::Wake, None) | (GateRoute::Vad, _) => Ok(None),
        };
        let wake = scored.unwrap_or_else(|error| {
            tracing::warn!(%error, "ww.failed");
            None
        });
        if let Some(wake) = wake {
            self.run_step(stream, SessionInput::Wake(wake), lang).await?;
            return Ok(());
        }
        let detected = match stream.vad.as_mut() {
            Some((vad, offset)) => Some((Self::common_guard("vad", || vad.push(chunk))?, *offset)),
            None => None,
        };
        if let Some((events, offset)) = detected {
            self.run_vad(stream, events, offset, lang).await?;
        }
        self.run_feed(stream, lang).await
    }

    async fn run_feed(&self, stream: &mut Stream, lang: Lang) -> Result<(), RunnerError> {
        let Some((id, asr, fed)) = stream.decoding.as_mut() else {
            return Ok(());
        };
        let first = stream.pos.saturating_sub(stream.ring.len() as u64);
        let skip = usize::try_from(fed.saturating_sub(first)).unwrap_or(usize::MAX);
        let audio: Vec<f32> = stream.ring.iter().skip(skip).copied().collect();
        *fed = stream.pos;
        let segment = *id;
        let partial = Self::common_guard("asr", || asr.push(&audio))?;
        match partial {
            Some(text) => {
                self.run_step(stream, SessionInput::Partial { segment, text }, lang)
                    .await
            }
            None => Ok(()),
        }
    }

    async fn run_flush(&self, stream: &mut Stream, lang: Lang) -> Result<(), RunnerError> {
        let rest = match stream.denoise.as_mut() {
            Some(denoise) => Some(Self::common_guard("denoise", || denoise.flush())?),
            None => None,
        };
        if let Some(rest) = rest.filter(|rest| !rest.is_empty()) {
            self.run_clean(stream, rest, lang).await?;
        }
        let flushed = match stream.vad.as_mut() {
            Some((vad, offset)) => Some((Self::common_guard("vad", || vad.flush())?, *offset)),
            None => None,
        };
        if let Some((events, offset)) = flushed {
            self.run_vad(stream, events, offset, lang).await?;
        }
        self.run_step(stream, SessionInput::Drain, lang).await
    }

    // ##########################################################

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(nodes: Arc<Nodes>, config: PipelineConfig) -> Self {
        let jobs = Arc::new(Semaphore::new(config.jobs.max(1)));
        Self {
            nodes,
            config: Arc::new(config),
            jobs,
        }
    }

    /// This runner with uploads transcribed by `engine`; `None` when that engine is not loaded.
    #[must_use]
    pub fn engine(&self, engine: AsrEngine) -> Option<Self> {
        if self.config.asr.file() == Some(engine) {
            return Some(self.clone());
        }
        let (_, asr) = self.nodes.extra.iter().find(|(loaded, _)| *loaded == engine)?;
        let nodes = Nodes {
            offline: Some(asr.clone()),
            ..(*self.nodes).clone()
        };
        Some(Self {
            nodes: Arc::new(nodes),
            config: Arc::clone(&self.config),
            jobs: Arc::clone(&self.jobs),
        })
    }

    /// Runs until the session closes: input end drains every pending segment, `cancel` (or the event
    /// consumer going away) finalizes them as cancelled. Always ends with [`Event::Closed`] unless a
    /// frame-path backend panics.
    ///
    /// # Errors
    /// A backend could not open a session, or panicked on the frame path.
    pub async fn run(
        &self,
        lang: Lang,
        intake: RunnerIntake,
        mut audio: mpsc::Receiver<Audio>,
        events: mpsc::Sender<Event>,
        cancel: CancellationToken,
    ) -> Result<(), RunnerError> {
        let gated = intake == RunnerIntake::Live && self.nodes.ww.is_some();
        let asr = match (intake, &self.nodes.offline) {
            (RunnerIntake::File, Some(offline)) => offline.clone(),
            (RunnerIntake::File, None) | (RunnerIntake::Live, _) => self.nodes.asr.clone(),
        };
        let plan = SessionPlan::new(&self.config, lang, asr.mode(), gated, self.nodes.ser.is_some());
        let (results, mut done) = mpsc::unbounded_channel();
        let ww = self
            .nodes
            .ww
            .as_ref()
            .filter(|_| gated)
            .map(|ww| ww.open())
            .transpose()?;
        let keep = usize::try_from(Audio::length(self.config.preroll)).unwrap_or(usize::MAX);
        let gain = self.config.gain;
        let denoise = self.nodes.denoise.as_ref().map(|denoise| denoise.open()).transpose()?;
        let mut stream = Stream {
            asr,
            denoise,
            gain: AudioGain::new(RATE, gain.peak, gain.max, gain.noise, gain.release),
            session: Session::new(plan),
            route: GateRoute::Wake,
            ww,
            vad: None,
            decoding: None,
            start: 0,
            ring: VecDeque::with_capacity(keep),
            pos: 0,
            keep,
            work: HashMap::new(),
            results,
            events,
            clock: Instant::now(),
        };
        self.run_step(&mut stream, SessionInput::Tick, lang).await?;
        let mut ticker = tokio::time::interval(self.config.tick.max(Duration::from_millis(1)));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let (mut open, mut cancelled) = (true, false);
        while !stream.session.closed() {
            tokio::select! {
                biased;
                () = cancel.cancelled(), if !cancelled => {
                    cancelled = true;
                    self.run_step(&mut stream, SessionInput::Cancel, lang).await?;
                }
                Some(input) = done.recv() => self.run_step(&mut stream, input, lang).await?,
                chunk = audio.recv(), if open && !(intake == RunnerIntake::File && stream.session.backlog() >= stream.session.limit()) => match chunk {
                    Some(chunk) => self.run_frame(&mut stream, &chunk, lang).await?,
                    None => {
                        open = false;
                        self.run_flush(&mut stream, lang).await?;
                    }
                },
                _ = ticker.tick() => self.run_step(&mut stream, SessionInput::Tick, lang).await?,
            }
        }
        stream.work.into_values().flatten().for_each(|task| task.abort());
        Ok(())
    }

    /// Runs a whole decoded file through a lossless [`RunnerIntake::File`] run on its own task.
    /// Events end with [`Event::Closed`]; a channel that ends without it means the run failed.
    /// Dropping the receiver cancels the run.
    #[must_use]
    pub fn file(&self, lang: Lang, samples: Vec<f32>) -> mpsc::Receiver<Event> {
        let (audio_tx, audio_rx) = mpsc::channel(4);
        let (events_tx, events_rx) = mpsc::channel(64);
        let runner = self.clone();
        let chunk = usize::try_from(Audio::length(Duration::from_secs(1))).unwrap_or(usize::MAX);
        tokio::spawn(async move {
            let feed = async move {
                for piece in samples.chunks(chunk) {
                    if audio_tx.send(Audio::from(piece.to_vec())).await.is_err() {
                        return;
                    }
                }
            };
            let run = runner.run(lang, RunnerIntake::File, audio_rx, events_tx, CancellationToken::new());
            let ((), outcome) = tokio::join!(feed, run);
            if let Err(error) = outcome {
                tracing::error!(%error, "file.failed");
            }
        });
        events_rx
    }

    /// Collects [`Runner::file`] into one [`Transcript`].
    ///
    /// # Errors
    /// The run ended without closing its session.
    pub async fn transcribe(&self, lang: Lang, samples: Vec<f32>) -> Result<Transcript, RunnerError> {
        let total = samples.len() as u64;
        let mut events = self.file(lang, samples);
        let mut segments = Vec::new();
        while let Some(event) = events.recv().await {
            match event {
                Event::Final(done) => segments.push(done),
                Event::Closed => {
                    return Ok(Transcript {
                        lang,
                        samples: total,
                        segments,
                    });
                }
                Event::Wake(_) | Event::Speech(_) | Event::Partial(_) => {}
            }
        }
        Err(RunnerError::Incomplete)
    }
}
