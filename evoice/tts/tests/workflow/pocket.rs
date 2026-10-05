use std::path::PathBuf;
use std::time::Instant;

use e_voice_core::schema::lang::Lang;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::workflow::synth::base::Synth;
use e_voice_tts::workflow::synth::pocket::{PocketOptions, PocketSynth};

use crate::conformance::conform;

const GREEDY: PocketOptions = PocketOptions {
    threads: 4,
    workers: 1,
    temperature: 0.0,
    steps: 1,
    quantized: false,
};

fn bundle(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/tts/ops/exports")
        .join(name)
}

fn golden(name: &str, key: &str) -> Vec<f32> {
    let file = std::fs::File::open(bundle(name).join("golden/golden.npz")).unwrap();
    let mut npz = npyz::npz::NpzArchive::new(std::io::BufReader::new(file)).unwrap();
    npz.by_name(key).unwrap().unwrap().into_vec::<f32>().unwrap()
}

#[test]
#[ignore = "requires the exported bundle: make pocket"]
fn test_pocket_reproduces_the_python_reference_loop() {
    let dir = bundle("pocket-spanish");
    let synth = PocketSynth::new(&[(Lang::Es, "pocket-spanish".to_owned(), dir.as_path())], GREEDY).unwrap();
    let voice = synth
        .train(&[Audio::from(golden("pocket-spanish", "audio"))], None)
        .unwrap();
    let text = std::fs::read_to_string(dir.join("golden/golden.txt")).unwrap();
    let mut session = synth.open(Lang::Es, Some(&voice)).unwrap();
    let pcm: Vec<f32> = session.speak(&text).flat_map(|chunk| chunk.unwrap().to_vec()).collect();
    let expected = golden("pocket-spanish", "pcm");
    assert_eq!(pcm.len(), expected.len());
    let worst = pcm
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst < 1e-3, "max abs diff {worst}");
}

#[test]
#[ignore = "requires the exported bundle: make pocket"]
fn test_pocket_streams_frames_and_conforms() {
    let dir = bundle("pocket-spanish");
    let options = PocketOptions {
        temperature: 0.3,
        quantized: true,
        ..GREEDY
    };
    let synth = PocketSynth::new(&[(Lang::Es, "pocket-spanish".to_owned(), dir.as_path())], options).unwrap();
    let voice = synth
        .train(&[Audio::from(golden("pocket-spanish", "audio"))], None)
        .unwrap();
    let start = Instant::now();
    assert_eq!(conform(&synth, Lang::Es, Some(&voice)), Ok(()));
    assert!(start.elapsed().as_secs() < 60);
}

#[test]
#[ignore = "requires the exported bundle: make pocket"]
fn test_pocket_refuses_to_speak_without_a_voice() {
    let dir = bundle("pocket-spanish");
    let synth = PocketSynth::new(&[(Lang::Es, "pocket-spanish".to_owned(), dir.as_path())], GREEDY).unwrap();
    assert!(synth.open(Lang::Es, None).is_err());
    assert!(synth.open(Lang::En, None).is_err());
}
