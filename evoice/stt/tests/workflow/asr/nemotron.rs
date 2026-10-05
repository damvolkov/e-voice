use std::sync::Arc;

use e_voice_core::schema::lang::Lang;
use e_voice_stt::config::asr::AsrConfig;
use e_voice_stt::workflow::asr::base::StreamingAsr;
use e_voice_stt::workflow::asr::registry::{AsrBackend, AsrRegistry};

use crate::fixture;

fn asr() -> Arc<dyn StreamingAsr> {
    match AsrRegistry::build(&AsrConfig::default(), &fixture::store()).unwrap() {
        AsrBackend::Streaming(asr) => asr,
        AsrBackend::Batch(_) => panic!("nemotron must stream"),
    }
}

fn decode(asr: &Arc<dyn StreamingAsr>, lang: Lang, name: &str) -> (Vec<String>, String) {
    let mut session = asr.open(lang).unwrap();
    let audio = fixture::wav("parakeet-v3-int8", name);
    let partials = audio.chunks(1600).filter_map(|chunk| session.push(chunk)).collect();
    (partials, session.finish().unwrap())
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_transcribes_spanish_and_english() {
    let asr = asr();
    let (_, es) = decode(&asr, Lang::Es, "es.wav");
    let (_, en) = decode(&asr, Lang::En, "en.wav");
    assert!(es.starts_with("No preguntes qué puede hacer tu país por ti"), "{es}");
    assert!(es.contains("por tu país"), "{es}");
    assert!(en.starts_with("Ask not what your country can do for you"), "{en}");
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_partials_grow_toward_final() {
    let (partials, text) = decode(&asr(), Lang::Es, "es.wav");
    assert!(partials.len() >= 2, "{partials:?}");
    assert!(
        partials
            .iter()
            .all(|partial| !partial.is_empty() && !partial.contains("  "))
    );
    assert!(
        partials.last().is_some_and(|last| last.len() <= text.len()),
        "{partials:?} → {text}"
    );
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_interleaved_sessions_match_sequential() {
    let asr = asr();
    let (_, es) = decode(&asr, Lang::Es, "es.wav");
    let (_, en) = decode(&asr, Lang::En, "en.wav");
    let (spanish, english) = (
        fixture::wav("parakeet-v3-int8", "es.wav"),
        fixture::wav("parakeet-v3-int8", "en.wav"),
    );
    let (mut left, mut right) = (asr.open(Lang::Es).unwrap(), asr.open(Lang::En).unwrap());
    let longest = spanish.len().max(english.len());
    for start in (0..longest).step_by(1600) {
        left.push(
            spanish
                .get(start..(start + 1600).min(spanish.len()))
                .unwrap_or_default(),
        );
        right.push(
            english
                .get(start..(start + 1600).min(english.len()))
                .unwrap_or_default(),
        );
    }
    assert_eq!((left.finish().unwrap(), right.finish().unwrap()), (es, en));
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_concurrent_sessions_share_one_backend() {
    let asr = asr();
    let (_, reference) = decode(&asr, Lang::Es, "es.wav");
    let texts: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| decode(&asr, Lang::Es, "es.wav").1))
            .collect();
        workers.into_iter().map(|worker| worker.join().unwrap()).collect()
    });
    assert!(texts.iter().all(|text| *text == reference), "{texts:?}");
}
