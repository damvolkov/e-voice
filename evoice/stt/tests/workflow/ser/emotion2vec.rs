use std::sync::Arc;

use e_voice_core::schema::lang::Lang;
use e_voice_stt::config::asr::AsrConfig;
use e_voice_stt::config::ser::{SerBackend, SerConfig};
use e_voice_stt::schema::emotion::EmotionLabel;
use e_voice_stt::workflow::asr::registry::{AsrBackend, AsrRegistry};
use e_voice_stt::workflow::ser::base::Ser;
use e_voice_stt::workflow::ser::registry::SerRegistry;

use crate::fixture;

fn ser() -> Arc<dyn Ser> {
    SerRegistry::build(&SerConfig::default(), &fixture::store())
        .unwrap()
        .unwrap()
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_classify_yields_distribution_over_closed_labels() {
    let emotion = ser().classify(&fixture::wav("parakeet-v3-int8", "es.wav")).unwrap();
    assert_eq!(emotion.scores.len(), 9);
    assert!((emotion.scores.values().sum::<f32>() - 1.0).abs() < 1e-4, "{emotion:?}");
    let best = emotion
        .scores
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(label, _)| *label);
    assert_eq!(Some(emotion.label), best);
    assert_eq!(emotion.model.as_deref(), Some("emotion2vec-plus-base"));
    println!("es.wav → {:?} {:?}", emotion.label, emotion.scores);
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_classify_is_deterministic_under_concurrency() {
    let ser = ser();
    let audio = fixture::wav("parakeet-v3-int8", "en.wav");
    let reference = ser.classify(&audio).unwrap();
    let results: Vec<_> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4).map(|_| scope.spawn(|| ser.classify(&audio).unwrap())).collect();
        workers.into_iter().map(|worker| worker.join().unwrap()).collect()
    });
    assert!(results.iter().all(|emotion| *emotion == reference));
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_classify_rejects_too_short_segment() {
    assert!(ser().classify(&fixture::silence(0.05)).is_err());
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_shares_onnxruntime_with_sherpa() {
    let ser = ser();
    let AsrBackend::Streaming(asr) = AsrRegistry::build(&AsrConfig::default(), &fixture::store()).unwrap() else {
        panic!("nemotron must stream");
    };
    let audio = fixture::wav("parakeet-v3-int8", "es.wav");
    let mut session = asr.open(Lang::Es).unwrap();
    audio.chunks(1600).for_each(|chunk| drop(session.push(chunk)));
    let emotion = ser.classify(&audio).unwrap();
    assert!(session.finish().unwrap().contains("por tu país"));
    assert_ne!(emotion.scores.get(&EmotionLabel::Neutral), None);
}

#[test]
#[ignore = "requires the local export: make export"]
fn test_large_runs_its_1024_dim_head_on_the_same_pipeline() {
    let config = SerConfig {
        backend: SerBackend::Emotion2vecLarge,
        ..SerConfig::default()
    };
    let large = SerRegistry::build(&config, &fixture::store()).unwrap().unwrap();
    let emotion = large.classify(&fixture::wav("parakeet-v3-int8", "en.wav")).unwrap();
    assert_eq!(emotion.scores.len(), 9);
    assert!((emotion.scores.values().sum::<f32>() - 1.0).abs() < 1e-4, "{emotion:?}");
    assert_eq!(emotion.model.as_deref(), Some("emotion2vec-plus-large"));
    assert_ne!(emotion.label, EmotionLabel::Unknown);
}
