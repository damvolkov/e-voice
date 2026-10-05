use std::sync::Arc;

use e_voice_stt::config::ww::{WwBackend, WwConfig};
use e_voice_stt::ops::voice::Voice;
use e_voice_stt::schema::event::WakeEvent;
use e_voice_stt::workflow::ww::base::Ww;
use e_voice_stt::workflow::ww::kws::KwsWw;
use e_voice_stt::workflow::ww::registry::WwRegistry;

use crate::fixture;

fn config(keyword: &str) -> WwConfig {
    WwConfig {
        backend: WwBackend::Kws,
        keyword: keyword.to_owned(),
        ..WwConfig::default()
    }
}

fn ww(keyword: &str) -> Arc<dyn Ww> {
    WwRegistry::build(&config(keyword), &fixture::store()).unwrap().unwrap()
}

fn detections(ww: &Arc<dyn Ww>, audio: &[f32], chunk: usize) -> Vec<WakeEvent> {
    let mut session = ww.open().unwrap();
    audio
        .chunks(chunk)
        .filter_map(|chunk| session.push(chunk).unwrap())
        .collect()
}

fn said(phrase: &str) -> Vec<f32> {
    [fixture::silence(1.0), fixture::speak(phrase), fixture::silence(1.0)].concat()
}

fn utterances(phrase: &str) -> Vec<Vec<f32>> {
    let store = fixture::store();
    let voices: Vec<Voice> = Voice::dirs(|id| store.dir(id).ok())
        .iter()
        .map(|dir| Voice::open(dir).unwrap())
        .collect();
    voices
        .iter()
        .flat_map(|voice| [0.9, 1.0, 1.1].map(|speed| voice.speak(phrase, speed).unwrap()))
        .map(|speech| [fixture::silence(1.0), speech, fixture::silence(1.0)].concat())
        .collect()
}

fn reliable(ww: &Arc<dyn Ww>, phrase: &str) -> Vec<f32> {
    utterances(phrase)
        .into_iter()
        .find(|audio| detections(ww, audio, 1_600).len() == 1)
        .unwrap()
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_encodes_keyword_into_model_pieces() {
    let dir = fixture::store().dir("kws-gigaspeech").unwrap();
    assert_eq!(
        KwsWw::encode(&dir, &config("hey eager")).unwrap(),
        "▁HE Y ▁E AGE R :3 #0.1 @hey_eager"
    );
    assert_eq!(
        KwsWw::encode(&dir, &config("Hey Siri")).unwrap(),
        "▁HE Y ▁S I RI :3 #0.1 @hey_siri"
    );
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_detects_default_phrase() {
    let ww = ww(&WwConfig::default().keyword);
    let found: Vec<Vec<WakeEvent>> = utterances("Hey eager.")
        .iter()
        .map(|audio| detections(&ww, audio, 1_600))
        .collect();
    let hits = found.iter().filter(|events| events.len() == 1).count();
    assert!(hits >= 3, "{hits}/9 detected: {found:?}");
    assert!(found.iter().all(|events| events.len() <= 1));
    assert!(found.iter().flatten().all(|event| event.keyword == "hey_eager"));
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_detects_any_phrase_without_training() {
    let found = detections(&ww("hey siri"), &said("Hey Siri."), 1_600);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].keyword, "hey_siri");
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_ignores_other_speech_and_silence() {
    let ww = ww("hey eager");
    let speech = [
        fixture::silence(1.0),
        fixture::wav("parakeet-v3-int8", "en.wav"),
        fixture::wav("parakeet-v3-int8", "es.wav"),
        fixture::speak("The eagle landed near the river before dinner."),
    ]
    .concat();
    assert!(detections(&ww, &speech, 1_600).is_empty());
    assert!(detections(&ww, &fixture::silence(5.0), 1_600).is_empty());
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_detection_does_not_depend_on_chunking() {
    let ww = ww("hey eager");
    let audio = reliable(&ww, "Hey eager.");
    let reference = detections(&ww, &audio, 1_600);
    for chunk in [160, 4_000, 16_000] {
        assert_eq!(detections(&ww, &audio, chunk).len(), reference.len(), "chunk {chunk}");
    }
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_cooldown_merges_close_repeats_only() {
    let ww = ww("hey eager");
    let heard = reliable(&ww, "Hey eager.");
    let phrase = heard[16_000..heard.len() - 16_000].to_vec();
    let close = [
        fixture::silence(1.0),
        phrase.clone(),
        fixture::silence(0.2),
        phrase.clone(),
        fixture::silence(1.0),
    ]
    .concat();
    let apart = [
        fixture::silence(1.0),
        phrase.clone(),
        fixture::silence(3.0),
        phrase,
        fixture::silence(1.0),
    ]
    .concat();
    assert_eq!(detections(&ww, &close, 1_600).len(), 1);
    assert_eq!(detections(&ww, &apart, 1_600).len(), 2);
}
