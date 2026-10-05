use std::sync::Arc;

use e_voice_stt::config::ww::{WwBackend, WwConfig};
use e_voice_stt::schema::event::WakeEvent;
use e_voice_stt::workflow::ww::base::Ww;
use e_voice_stt::workflow::ww::registry::WwRegistry;

use crate::fixture;

fn ww() -> Arc<dyn Ww> {
    WwRegistry::build(
        &WwConfig {
            backend: WwBackend::Oww,
            keyword: "hey_jarvis".to_owned(),
            ..WwConfig::default()
        },
        &fixture::store(),
    )
    .unwrap()
    .unwrap()
}

fn detections(ww: &Arc<dyn Ww>, audio: &[f32], chunk: usize) -> Vec<WakeEvent> {
    let mut session = ww.open().unwrap();
    audio
        .chunks(chunk)
        .filter_map(|chunk| session.push(chunk).unwrap())
        .collect()
}

fn wake() -> Vec<f32> {
    [
        fixture::silence(2.0),
        fixture::speak("Hey Jarvis."),
        fixture::silence(1.0),
    ]
    .concat()
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_detects_keyword() {
    let found = detections(&ww(), &wake(), 1_280);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].keyword, "hey_jarvis");
    assert!(found[0].score >= 0.5 && found[0].score <= 1.0, "{found:?}");
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_ignores_other_speech_and_silence() {
    let ww = ww();
    let speech = [
        fixture::silence(2.0),
        fixture::wav("parakeet-v3-int8", "es.wav"),
        fixture::wav("parakeet-v3-int8", "en.wav"),
        fixture::speak("Hey, are you there? What time is it?"),
    ]
    .concat();
    assert!(detections(&ww, &speech, 1_280).is_empty());
    assert!(detections(&ww, &fixture::silence(5.0), 1_280).is_empty());
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_detection_does_not_depend_on_chunking() {
    let ww = ww();
    let audio = wake();
    let reference = detections(&ww, &audio, 1_280);
    for chunk in [7, 160, 4_000, 16_000] {
        assert_eq!(detections(&ww, &audio, chunk), reference, "chunk {chunk}");
    }
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_cooldown_merges_close_repeats_only() {
    let ww = ww();
    let phrase = fixture::speak("Hey Jarvis.");
    let close = [
        fixture::silence(2.0),
        phrase.clone(),
        fixture::silence(0.3),
        phrase.clone(),
        fixture::silence(1.0),
    ]
    .concat();
    let apart = [
        fixture::silence(2.0),
        phrase.clone(),
        fixture::silence(3.0),
        phrase,
        fixture::silence(1.0),
    ]
    .concat();
    assert_eq!(detections(&ww, &close, 1_280).len(), 1);
    assert_eq!(detections(&ww, &apart, 1_280).len(), 2);
}
