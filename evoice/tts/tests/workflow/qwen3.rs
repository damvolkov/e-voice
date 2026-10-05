use std::path::PathBuf;

use e_voice_core::schema::lang::Lang;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::workflow::synth::base::Synth;
use e_voice_tts::workflow::synth::qwen3::{Qwen3Options, Qwen3Synth};

use crate::conformance::{LONG, conform};

const OPTIONS: Qwen3Options = Qwen3Options {
    threads: 4,
    workers: 1,
    temperature: 0.9,
    context: 50,
    first: 4,
    chunk: 16,
};

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/tts")
}

fn clip() -> Audio {
    let file = std::fs::File::open(data().join("ops/exports/qwen3/golden.npz")).unwrap();
    let mut npz = npyz::npz::NpzArchive::new(std::io::BufReader::new(file)).unwrap();
    Audio::from(npz.by_name("audio").unwrap().unwrap().into_vec::<f32>().unwrap())
}

#[test]
#[ignore = "requires the installed model: e-voice-tts pull qwen3-tts"]
fn test_qwen3_streams_frames_and_conforms_in_both_cloning_modes() {
    let synth = Qwen3Synth::new(&data().join("models/qwen3-tts"), OPTIONS).unwrap();
    assert!(synth.open(Lang::Es, None).is_err());
    let speaker = synth.train(&[clip()], None).unwrap();
    assert_eq!(conform(&synth, Lang::Es, Some(&speaker)), Ok(()));
    let context = synth
        .train(&[clip()], Some("Esto es lo que se dice en la referencia."))
        .unwrap();
    let mut session = synth.open(Lang::En, Some(&context)).unwrap();
    let samples: usize = session.speak(LONG).map(|chunk| chunk.unwrap().len()).sum();
    assert!(samples > 24_000, "{samples}");
}
