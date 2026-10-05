use e_voice_core::schema::lang::Lang;
use serde::{Deserialize, Serialize};

/// Synthesis engine. Only real streaming backends exist: audio leaves while a sentence is generated.
/// `pocket`: Kyutai Pocket TTS 100M, fastest. `qwen3`: Qwen3-TTS 12Hz 0.6B, higher fidelity cloning,
/// about real time on one stream. `neutts`: NeuTTS Nano (licence: free under $5M annual revenue),
/// needs espeak-ng and voices learned with their transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SynthBackend {
    #[default]
    Pocket,
    Qwen3,
    Neutts,
}

/// Model size: `base` (6 layers) or `large` (24 layers: higher quality, ~4× the compute; Spanish only,
/// English stays on base).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SynthSize {
    #[default]
    Base,
    Large,
}

/// The backend, its size, and per-stream compute: `workers` concurrent streams per language, each
/// with `threads` onnxruntime threads. `temperature` scales sampling noise (unset: the backend's
/// upstream default — Pocket 0.3, Qwen3 0.9); `steps` (Pocket's flow solver) and `quantized` (Pocket's
/// int8 graphs) apply to Pocket only. `size` applies to Pocket; Qwen3 has one multilingual model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SynthConfig {
    pub backend: SynthBackend,
    pub size: SynthSize,
    pub threads: u16,
    pub workers: usize,
    pub temperature: Option<f32>,
    pub steps: u16,
    pub quantized: bool,
    /// espeak-ng binary (NeuTTS phonemizes text with it).
    pub espeak: std::path::PathBuf,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            backend: SynthBackend::Pocket,
            size: SynthSize::Base,
            threads: 2,
            workers: 4,
            temperature: None,
            steps: 1,
            quantized: true,
            espeak: "espeak-ng".into(),
        }
    }
}

impl SynthConfig {
    /// Manifest id of the codec encoder NeuTTS turns reference clips into codes with.
    pub const ENCODER: &'static str = "neucodec-encoder";

    /// Manifest id of the model serving `lang`.
    #[must_use]
    pub const fn model(&self, lang: Lang) -> &'static str {
        match (self.backend, self.size, lang) {
            (SynthBackend::Pocket, SynthSize::Base, Lang::Es) => "pocket-es",
            (SynthBackend::Pocket, SynthSize::Large, Lang::Es) => "pocket-es-24l",
            (SynthBackend::Pocket, _, Lang::En) => "pocket-en",
            (SynthBackend::Qwen3, _, _) => "qwen3-tts",
            (SynthBackend::Neutts, _, Lang::Es) => "neutts-es",
            (SynthBackend::Neutts, _, Lang::En) => "neutts-en",
        }
    }

    /// Manifest ids of every model the configured backend loads.
    #[must_use]
    pub fn models(&self) -> Vec<String> {
        let mut models: Vec<String> = Lang::ALL.iter().map(|lang| self.model(*lang).to_owned()).collect();
        models.dedup();
        if self.backend == SynthBackend::Neutts {
            models.push(Self::ENCODER.to_owned());
        }
        models
    }
}
