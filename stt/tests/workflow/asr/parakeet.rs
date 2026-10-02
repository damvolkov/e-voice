use std::sync::Arc;

use e_voice_stt::config::asr::{AsrBackend as AsrChoice, AsrConfig};
use e_voice_stt::schema::lang::Lang;
use e_voice_stt::workflow::asr::base::BatchAsr;
use e_voice_stt::workflow::asr::registry::{AsrBackend, AsrRegistry};

use crate::fixture;

fn asr() -> Arc<dyn BatchAsr> {
    match AsrRegistry::build(
        &AsrConfig {
            backend: AsrChoice::Parakeet,
            ..AsrConfig::default()
        },
        &fixture::store(),
    )
    .unwrap()
    {
        AsrBackend::Batch(asr) => asr,
        AsrBackend::Streaming(_) => panic!("parakeet must be batch"),
    }
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_transcribes_spanish_and_english() {
    let asr = asr();
    let es = asr
        .transcribe(Lang::Es, &fixture::wav("parakeet-v3-int8", "es.wav"))
        .unwrap();
    let en = asr
        .transcribe(Lang::En, &fixture::wav("parakeet-v3-int8", "en.wav"))
        .unwrap();
    assert_eq!(
        es,
        "No preguntes qué puede hacer tu país por ti, pregunta qué puedes hacer tú por tu país."
    );
    assert_eq!(
        en,
        "Ask not what your country can do for you, ask what you can do for your country."
    );
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_silence_transcribes_to_empty() {
    assert_eq!(asr().transcribe(Lang::Es, &fixture::silence(2.0)).unwrap(), "");
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_concurrent_calls_share_one_backend() {
    let asr = asr();
    let audio = fixture::wav("parakeet-v3-int8", "es.wav");
    let reference = asr.transcribe(Lang::Es, &audio).unwrap();
    let texts: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| asr.transcribe(Lang::Es, &audio).unwrap()))
            .collect();
        workers.into_iter().map(|worker| worker.join().unwrap()).collect()
    });
    assert!(texts.iter().all(|text| *text == reference), "{texts:?}");
}
