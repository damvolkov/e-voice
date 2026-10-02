use std::sync::Arc;

use e_voice_stt::config::asr::{AsrBackend as AsrChoice, AsrConfig};
use e_voice_stt::schema::lang::Lang;
use e_voice_stt::workflow::asr::base::{BatchAsr, StreamingAsr};
use e_voice_stt::workflow::asr::registry::{AsrBackend, AsrRegistry};

use crate::fixture;

fn build(backend: AsrChoice) -> AsrBackend {
    let config = AsrConfig {
        backend,
        ..AsrConfig::default()
    };
    AsrRegistry::build(&config, &fixture::store()).unwrap()
}

fn batch(backend: AsrChoice) -> Arc<dyn BatchAsr> {
    match build(backend) {
        AsrBackend::Batch(asr) => asr,
        AsrBackend::Streaming(_) => panic!("{backend:?} must be batch"),
    }
}

fn streaming(backend: AsrChoice) -> Arc<dyn StreamingAsr> {
    match build(backend) {
        AsrBackend::Streaming(asr) => asr,
        AsrBackend::Batch(_) => panic!("{backend:?} must stream"),
    }
}

fn heard(asr: &Arc<dyn BatchAsr>, lang: Lang) -> String {
    let name = format!("{}.wav", lang.code());
    asr.transcribe(lang, &fixture::wav("parakeet-v3-int8", &name))
        .unwrap()
        .to_lowercase()
}

fn assert_kennedy(es: &str, en: &str) {
    assert!(es.contains("no preguntes") && es.contains("por tu país"), "{es}");
    assert!(en.contains("ask not what your country can do for you"), "{en}");
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_canary_transcribes_spanish_and_english_and_silence() {
    let asr = batch(AsrChoice::Canary);
    assert_kennedy(&heard(&asr, Lang::Es), &heard(&asr, Lang::En));
    assert_eq!(asr.transcribe(Lang::Es, &fixture::silence(2.0)).unwrap(), "");
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_cohere_transcribes_spanish_and_english() {
    let asr = batch(AsrChoice::Cohere);
    assert_kennedy(&heard(&asr, Lang::Es), &heard(&asr, Lang::En));
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_whisper_transcribes_spanish_and_english() {
    let asr = batch(AsrChoice::Whisper);
    assert_kennedy(&heard(&asr, Lang::Es), &heard(&asr, Lang::En));
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_kroko_streams_each_language_on_its_model() {
    let asr = streaming(AsrChoice::Kroko);
    let decode = |lang: Lang| {
        let mut session = asr.open(lang).unwrap();
        let audio = fixture::wav("parakeet-v3-int8", &format!("{}.wav", lang.code()));
        let partials: Vec<String> = audio.chunks(1600).filter_map(|chunk| session.push(chunk)).collect();
        (partials, session.finish().unwrap().to_lowercase())
    };
    let ((partials, es), (_, en)) = (decode(Lang::Es), decode(Lang::En));
    assert!(partials.len() >= 2, "{partials:?}");
    assert_kennedy(&es, &en);
}
