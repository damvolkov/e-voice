use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use e_voice_stt::config::pipeline::Overload;
use e_voice_stt::schema::audio::Audio;
use e_voice_stt::schema::emotion::{Emotion, EmotionLabel};
use e_voice_stt::schema::error::NodeError;
use e_voice_stt::schema::event::{Event, FinalEvent, WakeEvent};
use e_voice_stt::schema::lang::Lang;
use e_voice_stt::schema::mode::Mode;
use e_voice_stt::schema::segment::{SegmentId, SegmentSpan};
use e_voice_stt::workflow::gate::{GatePolicy, GateRoute};
use e_voice_stt::workflow::session::{
    AsrPlan, SerPlan, Session, SessionCommand, SessionInput, SessionOutput, SessionPlan,
};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Wake,
    Start,
    End(u64),
    Partial(u64),
    Transcript(u64, bool),
    Emotion(u64, bool),
    Tick(u64),
    Drain,
}

#[derive(Debug, Clone, Copy)]
enum Finale {
    Drain,
    Cancel,
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn plan() -> impl Strategy<Value = SessionPlan> {
    let mode = prop_oneof![Just(Mode::Streaming), Just(Mode::Batch)];
    let ser = proptest::option::of((100u64..3_000, 0u64..16_000).prop_map(|(d, min)| SerPlan { deadline: ms(d), min }));
    let gate = prop_oneof![
        Just(GatePolicy::Disabled),
        (500u64..10_000).prop_map(|i| GatePolicy::Utterance { idle: ms(i) }),
        (500u64..10_000).prop_map(|i| GatePolicy::Window { idle: ms(i) }),
        Just(GatePolicy::Session),
    ];
    let overload = prop_oneof![Just(Overload::Reject), Just(Overload::Evict)];
    (mode, 100u64..3_000, ser, gate, 0usize..4, overload, 1_000u64..20_000).prop_map(
        |(mode, asr, ser, gate, pending, overload, stall)| SessionPlan {
            lang: Lang::Es,
            asr: AsrPlan {
                mode,
                deadline: ms(asr),
            },
            ser,
            gate,
            pending,
            overload,
            stall: ms(stall),
        },
    )
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        1 => Just(Op::Wake),
        3 => Just(Op::Start),
        3 => (1u64..64_000).prop_map(Op::End),
        2 => (0u64..12).prop_map(Op::Partial),
        3 => (0u64..12, any::<bool>()).prop_map(|(id, ok)| Op::Transcript(id, ok)),
        3 => (0u64..12, any::<bool>()).prop_map(|(id, ok)| Op::Emotion(id, ok)),
        3 => (0u64..4_000).prop_map(Op::Tick),
        1 => Just(Op::Drain),
    ]
}

fn input(op: &Op, cursor: &mut u64) -> SessionInput {
    match op {
        Op::Wake => SessionInput::Wake(WakeEvent {
            keyword: "evoice".into(),
            score: 0.9,
        }),
        Op::Start => SessionInput::Start { at: *cursor },
        Op::End(len) => {
            let span = SegmentSpan {
                start: *cursor,
                end: *cursor + len,
            };
            *cursor = span.end;
            SessionInput::End {
                span,
                audio: Audio::from(vec![0.0; 8]),
            }
        }
        Op::Partial(id) => SessionInput::Partial {
            segment: SegmentId(*id),
            text: "hola".into(),
        },
        Op::Transcript(id, ok) => SessionInput::Transcript {
            segment: SegmentId(*id),
            result: ok
                .then(|| "hola mundo".to_owned())
                .ok_or_else(|| NodeError::Backend("boom".into())),
        },
        Op::Emotion(id, ok) => SessionInput::Emotion {
            segment: SegmentId(*id),
            result: ok
                .then(|| Emotion {
                    label: EmotionLabel::Happy,
                    scores: BTreeMap::new(),
                    model: Some("fake".into()),
                })
                .ok_or_else(|| NodeError::Backend("boom".into())),
        },
        Op::Tick(_) => SessionInput::Tick,
        Op::Drain => SessionInput::Drain,
    }
}

fn run(plan: SessionPlan, ops: &[Op], finale: Finale) -> (Vec<SessionOutput>, usize) {
    let mut session = Session::new(plan);
    let (mut now, mut cursor, mut peak) = (Duration::ZERO, 0u64, 0usize);
    let mut outputs: Vec<SessionOutput> = session.poll().collect();
    for op in ops {
        now += match op {
            Op::Tick(delta) => ms(*delta),
            _ => Duration::ZERO,
        };
        session.step(now, input(op, &mut cursor));
        peak = peak.max(session.pending());
        outputs.extend(session.poll());
    }
    let closing = match finale {
        Finale::Drain => SessionInput::Drain,
        Finale::Cancel => SessionInput::Cancel,
    };
    session.step(now, closing);
    session.step(now + Duration::from_secs(86_400), SessionInput::Tick);
    outputs.extend(session.poll());
    assert!(session.closed());
    (outputs, peak)
}

fn finals(outputs: &[SessionOutput]) -> Vec<&FinalEvent> {
    outputs
        .iter()
        .filter_map(|output| match output {
            SessionOutput::Event(Event::Final(event)) => Some(event),
            _ => None,
        })
        .collect()
}

fn work(command: &SessionCommand) -> Option<SegmentId> {
    match command {
        SessionCommand::Begin(id) | SessionCommand::Finish(id) => Some(*id),
        SessionCommand::Transcribe(segment) | SessionCommand::Classify(segment) => Some(segment.id),
        SessionCommand::Route(_) | SessionCommand::Abort(_) => None,
    }
}

fn finale() -> impl Strategy<Value = Finale> {
    prop_oneof![Just(Finale::Drain), Just(Finale::Cancel)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn test_final_once_per_segment_in_order(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let (outputs, _) = run(plan, &ops, end);
        let ids: Vec<u64> = finals(&outputs).iter().map(|event| event.segment.0).collect();
        prop_assert_eq!(ids, (0..finals(&outputs).len() as u64).collect::<Vec<_>>());
    }

    #[test]
    fn test_closed_once_and_last(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let (outputs, _) = run(plan, &ops, end);
        let closed = outputs.iter().filter(|output| matches!(output, SessionOutput::Event(Event::Closed))).count();
        prop_assert_eq!(closed, 1);
        prop_assert!(matches!(outputs.last(), Some(SessionOutput::Event(Event::Closed))));
    }

    #[test]
    fn test_pending_never_exceeds_limit(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let (_, peak) = run(plan, &ops, end);
        prop_assert!(peak <= plan.pending.max(1));
    }

    #[test]
    fn test_no_work_after_final(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let (outputs, _) = run(plan, &ops, end);
        let mut released = HashSet::new();
        for output in &outputs {
            match output {
                SessionOutput::Event(Event::Final(event)) => { released.insert(event.segment); }
                SessionOutput::Command(command) => {
                    prop_assert!(work(command).is_none_or(|id| !released.contains(&id)));
                }
                SessionOutput::Event(_) => {}
            }
        }
    }

    #[test]
    fn test_mode_dictates_asr_commands(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let (outputs, _) = run(plan, &ops, end);
        let mut begun = HashSet::new();
        for output in &outputs {
            match (plan.asr.mode, output) {
                (Mode::Streaming, SessionOutput::Command(SessionCommand::Begin(id))) => { begun.insert(*id); }
                (Mode::Streaming, SessionOutput::Command(SessionCommand::Finish(id))) => prop_assert!(begun.contains(id)),
                (Mode::Streaming, SessionOutput::Command(SessionCommand::Transcribe(_)))
                | (Mode::Batch, SessionOutput::Command(SessionCommand::Begin(_) | SessionCommand::Finish(_))) => {
                    prop_assert!(false, "asr command contradicts mode");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_without_ser_emotion_is_unknown(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let plan = SessionPlan { ser: None, ..plan };
        let (outputs, _) = run(plan, &ops, end);
        prop_assert!(finals(&outputs).iter().all(|event| event.emotion.label == EmotionLabel::Unknown));
        prop_assert!(!outputs.iter().any(|output| matches!(output, SessionOutput::Command(SessionCommand::Classify(_)))));
    }

    #[test]
    fn test_disabled_gate_routes_only_to_vad(plan in plan(), ops in prop::collection::vec(op(), 0..80), end in finale()) {
        let plan = SessionPlan { gate: GatePolicy::Disabled, ..plan };
        let (outputs, _) = run(plan, &ops, end);
        let routes: Vec<&GateRoute> = outputs.iter().filter_map(|output| match output {
            SessionOutput::Command(SessionCommand::Route(route)) => Some(route),
            _ => None,
        }).collect();
        prop_assert_eq!(routes, vec![&GateRoute::Vad]);
        prop_assert!(!outputs.iter().any(|output| matches!(output, SessionOutput::Event(Event::Wake(_)))));
    }
}

fn batch_plan() -> SessionPlan {
    SessionPlan {
        lang: Lang::En,
        asr: AsrPlan {
            mode: Mode::Batch,
            deadline: ms(1_000),
        },
        ser: Some(SerPlan {
            deadline: ms(500),
            min: 1_000,
        }),
        gate: GatePolicy::Disabled,
        pending: 2,
        overload: Overload::Reject,
        stall: ms(30_000),
    }
}

#[test]
fn test_finals_wait_for_earlier_segments() {
    let mut session = Session::new(batch_plan());
    let span = |start| SegmentSpan {
        start,
        end: start + 2_000,
    };
    session.step(
        ms(0),
        SessionInput::End {
            span: span(0),
            audio: Audio::default(),
        },
    );
    session.step(
        ms(10),
        SessionInput::End {
            span: span(4_000),
            audio: Audio::default(),
        },
    );
    session.step(
        ms(20),
        SessionInput::Transcript {
            segment: SegmentId(1),
            result: Ok("second".into()),
        },
    );
    session.step(
        ms(20),
        SessionInput::Emotion {
            segment: SegmentId(1),
            result: Ok(Emotion::default()),
        },
    );
    assert!(finals(&session.poll().collect::<Vec<_>>()).is_empty());
    session.step(
        ms(30),
        SessionInput::Transcript {
            segment: SegmentId(0),
            result: Ok("first".into()),
        },
    );
    session.step(
        ms(30),
        SessionInput::Emotion {
            segment: SegmentId(0),
            result: Ok(Emotion::default()),
        },
    );
    let outputs: Vec<_> = session.poll().collect();
    let texts: Vec<&str> = finals(&outputs).iter().map(|event| event.text.as_str()).collect();
    assert_eq!(texts, ["first", "second"]);
}

#[test]
fn test_late_ser_falls_back_to_unknown_without_error() {
    let mut session = Session::new(batch_plan());
    session.step(
        ms(0),
        SessionInput::End {
            span: SegmentSpan { start: 0, end: 2_000 },
            audio: Audio::default(),
        },
    );
    session.step(
        ms(100),
        SessionInput::Transcript {
            segment: SegmentId(0),
            result: Ok("hola".into()),
        },
    );
    session.step(ms(600), SessionInput::Tick);
    let outputs: Vec<_> = session.poll().collect();
    let event = finals(&outputs)[0].clone();
    assert_eq!(
        (event.text.as_str(), event.emotion.label, event.error),
        ("hola", EmotionLabel::Unknown, None)
    );
    assert!(outputs.contains(&SessionOutput::Command(SessionCommand::Abort(SegmentId(0)))));
}

#[test]
fn test_late_asr_fails_with_timeout() {
    let mut session = Session::new(batch_plan());
    session.step(
        ms(0),
        SessionInput::End {
            span: SegmentSpan { start: 0, end: 500 },
            audio: Audio::default(),
        },
    );
    session.step(ms(1_000), SessionInput::Tick);
    let outputs: Vec<_> = session.poll().collect();
    assert_eq!(finals(&outputs)[0].error, Some(NodeError::Timeout));
}

#[test]
fn test_reject_marks_new_segment_overloaded() {
    let plan = SessionPlan {
        pending: 1,
        ..batch_plan()
    };
    let mut session = Session::new(plan);
    session.step(
        ms(0),
        SessionInput::End {
            span: SegmentSpan { start: 0, end: 2_000 },
            audio: Audio::default(),
        },
    );
    session.step(
        ms(10),
        SessionInput::End {
            span: SegmentSpan {
                start: 3_000,
                end: 5_000,
            },
            audio: Audio::default(),
        },
    );
    session.step(ms(5_000), SessionInput::Tick);
    let outputs: Vec<_> = session.poll().collect();
    let errors: Vec<Option<NodeError>> = finals(&outputs).iter().map(|event| event.error.clone()).collect();
    assert_eq!(errors, [Some(NodeError::Timeout), Some(NodeError::Overload)]);
}

#[test]
fn test_final_event_serializes_with_type_tag() {
    let event = Event::Final(FinalEvent {
        segment: SegmentId(3),
        span: SegmentSpan { start: 0, end: 16_000 },
        lang: Lang::Es,
        text: "hola".into(),
        emotion: Emotion::default(),
        error: None,
    });
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "final");
    assert_eq!(json["segment"], 3);
    assert_eq!(json["lang"], "es");
    assert_eq!(json["emotion"]["label"], "unknown");
    assert!(json["error"].is_null());
}
