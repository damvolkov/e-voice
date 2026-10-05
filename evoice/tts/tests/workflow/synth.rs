use std::time::Duration;

use e_voice_core::schema::lang::Lang;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::schema::error::TrainError;
use e_voice_tts::workflow::synth::base::Synth;

use crate::conformance::conform;
use crate::fake::{FakeSynth, FakeWhole};

const STEP: Duration = Duration::from_millis(2);

#[test]
fn test_conform_accepts_frame_streaming() {
    assert_eq!(conform(&FakeSynth { delay: STEP }, Lang::Es, None), Ok(()));
}

#[test]
fn test_conform_rejects_sentence_at_once() {
    let verdict = conform(&FakeWhole { delay: STEP }, Lang::Es, None);
    assert!(
        verdict.as_ref().is_err_and(|reason| reason.contains("not streaming")),
        "{verdict:?}"
    );
}

#[test]
fn test_train_defaults_to_unsupported() {
    let clips = [Audio::from(vec![0.5; 10])];
    assert_eq!(
        FakeWhole { delay: STEP }.train(&clips, None),
        Err(TrainError::Unsupported)
    );
    assert!(FakeSynth { delay: STEP }.train(&clips, None).is_ok());
}

#[test]
fn test_speak_stops_when_dropped() {
    let synth = FakeSynth {
        delay: Duration::from_millis(20),
    };
    let mut session = synth.open(Lang::Es, None).unwrap();
    let start = std::time::Instant::now();
    let taken = session.speak(crate::conformance::LONG).take(2).count();
    assert_eq!(taken, 2);
    assert!(start.elapsed() < Duration::from_millis(200));
}
