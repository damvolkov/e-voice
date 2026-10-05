use std::path::PathBuf;

use e_voice_core::schema::lang::Lang;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::workflow::synth::base::Synth;
use e_voice_tts::workflow::synth::neutts::{NeuttsOptions, NeuttsSynth, Phonemizer};

use crate::conformance::{LONG, conform};

/// Stand-in graphs with the `NeuTTS` IO (`eval/export/neutts_fixture.py`): code 7 until 1000 tokens, then END.
fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/resources/neutts")
}

fn synth() -> NeuttsSynth {
    let dir = fixtures();
    let options = NeuttsOptions {
        threads: 1,
        workers: 1,
        temperature: 1.0,
        espeak: PathBuf::from("/bin/echo"),
    };
    let models = [
        (Lang::Es, "neutts-es".to_owned(), dir.as_path()),
        (Lang::En, "neutts-en".to_owned(), dir.as_path()),
    ];
    NeuttsSynth::new(&models, &dir, options).unwrap()
}

fn clip() -> Audio {
    Audio::from((0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect::<Vec<f32>>())
}

#[test]
fn test_neutts_needs_a_voice_with_its_transcript() {
    let synth = synth();
    assert!(synth.open(Lang::Es, None).is_err());
    let mute = synth.train(&[clip()], None).unwrap();
    assert!(synth.open(Lang::Es, Some(&mute)).is_err());
}

#[test]
fn test_neutts_streams_windows_and_conforms() {
    let synth = synth();
    let voice = synth.train(&[clip()], Some("Hola, esto es la referencia.")).unwrap();
    assert_eq!(conform(&synth, Lang::Es, Some(&voice)), Ok(()));
    let mut session = synth.open(Lang::En, Some(&voice)).unwrap();
    let chunks: Vec<Audio> = session.speak(LONG).map(Result::unwrap).collect();
    assert!(chunks.len() > 2);
    assert!(
        chunks
            .iter()
            .flat_map(|chunk| chunk.iter())
            .all(|sample| (sample - 0.1).abs() < 1e-4)
    );
}

#[test]
fn test_phonemizer_keeps_punctuation_between_espeak_segments() {
    let out = Phonemizer::run(&PathBuf::from("/bin/echo"), "es", "Hola, mundo.").unwrap();
    assert_eq!(out, "-q --ipa -v es -- Hola, -q --ipa -v es -- mundo.");
}
