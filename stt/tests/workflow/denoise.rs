use e_voice_stt::config::denoise::{DenoiseBackend, DenoiseConfig};
use e_voice_stt::workflow::denoise::registry::DenoiseRegistry;

use crate::fixture;

fn energy(audio: &[f32]) -> f32 {
    audio.iter().map(|sample| sample * sample).sum::<f32>() / audio.len().max(1) as f32
}

#[test]
fn test_off_builds_nothing() {
    assert!(
        DenoiseRegistry::build(&DenoiseConfig::default(), &fixture::store())
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_gtcrn_keeps_length_and_attenuates_noise_more_than_speech() {
    let config = DenoiseConfig {
        backend: DenoiseBackend::Gtcrn,
        ..DenoiseConfig::default()
    };
    let denoise = DenoiseRegistry::build(&config, &fixture::store()).unwrap().unwrap();
    let run = |audio: &[f32]| {
        let mut session = denoise.open().unwrap();
        let mut out: Vec<f32> = audio.chunks(1_600).flat_map(|chunk| session.push(chunk)).collect();
        out.extend(session.flush());
        out
    };
    let speech = fixture::wav("parakeet-v3-int8", "es.wav");
    let noise: Vec<f32> = (0u32..32_000)
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 16) as f32 / 65_536.0 - 0.5) * 0.2)
        .collect();
    let (clean, hiss) = (run(&speech), run(&noise));
    assert!(
        clean.len().abs_diff(speech.len()) <= 1_600,
        "{} vs {}",
        clean.len(),
        speech.len()
    );
    let kept = energy(&clean) / energy(&speech);
    let left = energy(&hiss) / energy(&noise);
    assert!(left < kept / 4.0, "noise kept {left:.3}, speech kept {kept:.3}");
}
