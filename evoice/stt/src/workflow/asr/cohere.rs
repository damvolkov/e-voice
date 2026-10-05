use std::fmt::{self, Debug};
use std::path::Path;

use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use sherpa_onnx::{OfflineCohereTranscribeModelConfig, OfflineRecognizer, OfflineRecognizerConfig};

use crate::workflow::asr::base::BatchAsr;
use crate::workflow::asr::offline::Offline;
use crate::workflow::parts::Parts;

/// Cohere Transcribe (14 languages), punctuated; the language is a per-stream option. It does not
/// detect the language and transcribes background noise, so it relies on the VAD in front.
pub struct CohereAsr {
    recognizer: OfflineRecognizer,
}

impl Debug for CohereAsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CohereAsr").finish_non_exhaustive()
    }
}

impl CohereAsr {
    /// # Errors
    /// Model files missing from `dir`, or the runtime rejecting them.
    pub fn new(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let mut sherpa = OfflineRecognizerConfig::default();
        sherpa.model_config.cohere_transcribe = OfflineCohereTranscribeModelConfig {
            encoder: Some(Parts::onnx(dir, "encoder")?),
            decoder: Some(Parts::onnx(dir, "decoder")?),
            language: Some(Lang::default().code().to_owned()),
            use_punct: true,
            use_itn: true,
        };
        sherpa.model_config.tokens = Some(Parts::tokens(dir)?);
        Ok(Self {
            recognizer: Offline::create(sherpa, threads, "cohere")?,
        })
    }
}

impl BatchAsr for CohereAsr {
    fn transcribe(&self, lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        Offline::decode(
            &self.recognizer,
            |stream| stream.set_option("language", lang.code()),
            audio,
            "cohere",
        )
    }
}
