use std::fmt::{self, Debug};
use std::path::Path;

use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig};

use crate::workflow::asr::base::BatchAsr;
use crate::workflow::asr::offline::Offline;
use crate::workflow::parts::Parts;

/// Whisper large-v3 turbo; it identifies the language itself, so `lang` is not applied. Inputs stay
/// under its 30 s window because the VAD caps segments.
pub struct WhisperAsr {
    recognizer: OfflineRecognizer,
}

impl Debug for WhisperAsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WhisperAsr").finish_non_exhaustive()
    }
}

impl WhisperAsr {
    /// # Errors
    /// Model files missing from `dir`, or the runtime rejecting them.
    pub fn new(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let mut sherpa = OfflineRecognizerConfig::default();
        sherpa.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(Parts::onnx(dir, "encoder")?),
            decoder: Some(Parts::onnx(dir, "decoder")?),
            language: None,
            task: Some("transcribe".to_owned()),
            tail_paddings: -1,
            enable_token_timestamps: false,
            enable_segment_timestamps: false,
        };
        sherpa.model_config.tokens = Some(Parts::tokens(dir)?);
        Ok(Self {
            recognizer: Offline::create(sherpa, threads, "whisper")?,
        })
    }
}

impl BatchAsr for WhisperAsr {
    fn transcribe(&self, _lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        Offline::decode(&self.recognizer, |_| {}, audio, "whisper")
    }
}
