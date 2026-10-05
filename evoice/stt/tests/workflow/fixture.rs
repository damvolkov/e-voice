use std::path::PathBuf;

use e_voice_core::config::ops::{ModelsVerify, OpsConfig};
use e_voice_core::models::ModelStore;
use e_voice_stt::schema::audio::RATE;
use sherpa_onnx::{
    GenerationConfig, LinearResampler, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig,
    Wave,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn data() -> PathBuf {
    root().join("../../data/stt")
}

pub fn store() -> ModelStore {
    let config = OpsConfig {
        data: data(),
        manifest: root().join("models.toml"),
        verify: ModelsVerify::Stamp,
    };
    ModelStore::open(&config).unwrap()
}

pub fn wav(model: &str, name: &str) -> Vec<f32> {
    let path = data().join("models").join(model).join("test_wavs").join(name);
    let wave = Wave::read(path.to_str().unwrap()).unwrap();
    let resampler = LinearResampler::create(wave.sample_rate(), RATE as i32).unwrap();
    resampler.resample(wave.samples(), true)
}

pub fn silence(seconds: f32) -> Vec<f32> {
    vec![0.0; (seconds * RATE as f32) as usize]
}

pub fn dialogue() -> Vec<f32> {
    [
        silence(1.0),
        wav("parakeet-v3-int8", "es.wav"),
        silence(2.0),
        wav("parakeet-v3-int8", "en.wav"),
        silence(1.5),
    ]
    .concat()
}

pub fn speak(text: &str) -> Vec<f32> {
    let dir = data().join("ops/models/piper-en-amy-low");
    let path = |name: &str| Some(dir.join(name).display().to_string());
    let config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            vits: OfflineTtsVitsModelConfig {
                model: path("en_US-amy-low.onnx"),
                tokens: path("tokens.txt"),
                data_dir: path("espeak-ng-data"),
                noise_scale: 1e-6,
                noise_scale_w: 1e-6,
                ..OfflineTtsVitsModelConfig::default()
            },
            num_threads: 2,
            ..OfflineTtsModelConfig::default()
        },
        ..OfflineTtsConfig::default()
    };
    let tts = OfflineTts::create(&config).unwrap();
    let audio = tts
        .generate_with_config(text, &GenerationConfig::default(), None::<fn(&[f32], f32) -> bool>)
        .unwrap();
    let resampler = LinearResampler::create(audio.sample_rate(), RATE as i32).unwrap();
    resampler.resample(audio.samples(), true)
}
