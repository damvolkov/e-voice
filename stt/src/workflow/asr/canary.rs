use std::fmt::{self, Debug};
use std::path::Path;

use sherpa_onnx::{OfflineCanaryModelConfig, OfflineRecognizer, OfflineRecognizerConfig};

use crate::schema::error::{BackendError, NodeError};
use crate::schema::lang::Lang;
use crate::workflow::asr::base::BatchAsr;
use crate::workflow::asr::offline::Offline;
use crate::workflow::parts::Parts;

/// NeMo Canary 180M flash (en, es, de, fr), punctuated. Its source language is fixed per recognizer,
/// so one recognizer is built per supported language.
pub struct CanaryAsr {
    es: OfflineRecognizer,
    en: OfflineRecognizer,
}

impl Debug for CanaryAsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CanaryAsr").finish_non_exhaustive()
    }
}

impl CanaryAsr {
    // ##### PRIVATE #####

    fn new_for(dir: &Path, threads: u16, lang: Lang) -> Result<OfflineRecognizer, BackendError> {
        let mut sherpa = OfflineRecognizerConfig::default();
        sherpa.model_config.canary = OfflineCanaryModelConfig {
            encoder: Some(Parts::onnx(dir, "encoder")?),
            decoder: Some(Parts::onnx(dir, "decoder")?),
            src_lang: Some(lang.code().to_owned()),
            tgt_lang: Some(lang.code().to_owned()),
            use_pnc: true,
        };
        sherpa.model_config.tokens = Some(Parts::tokens(dir)?);
        Offline::create(sherpa, threads, "canary")
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// Model files missing from `dir`, or the runtime rejecting them.
    pub fn new(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        Ok(Self {
            es: Self::new_for(dir, threads, Lang::Es)?,
            en: Self::new_for(dir, threads, Lang::En)?,
        })
    }
}

impl BatchAsr for CanaryAsr {
    fn transcribe(&self, lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        let recognizer = match lang {
            Lang::Es => &self.es,
            Lang::En => &self.en,
        };
        Offline::decode(recognizer, |_| {}, audio, "canary")
    }
}
