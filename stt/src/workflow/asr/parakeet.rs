use std::fmt::{self, Debug};
use std::path::Path;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

use crate::schema::error::{BackendError, NodeError};
use crate::schema::lang::Lang;
use crate::workflow::asr::base::BatchAsr;
use crate::workflow::asr::offline::Offline;
use crate::workflow::asr::transducer::Transducer;

/// Parakeet TDT v3 offline transducer; it identifies the language itself, so `lang` is not applied.
pub struct ParakeetAsr {
    recognizer: OfflineRecognizer,
}

impl Debug for ParakeetAsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParakeetAsr").finish_non_exhaustive()
    }
}

impl ParakeetAsr {
    /// # Errors
    /// Model files missing from `dir`, or the runtime rejecting them.
    pub fn new(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let files = Transducer::locate(dir)?;
        let mut sherpa = OfflineRecognizerConfig::default();
        sherpa.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(files.encoder),
            decoder: Some(files.decoder),
            joiner: Some(files.joiner),
        };
        sherpa.model_config.tokens = Some(files.tokens);
        Ok(Self {
            recognizer: Offline::create(sherpa, threads, "parakeet")?,
        })
    }
}

impl BatchAsr for ParakeetAsr {
    fn transcribe(&self, _lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        Offline::decode(&self.recognizer, |_| {}, audio, "parakeet")
    }
}
