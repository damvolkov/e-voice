use std::fmt::{self, Debug};
use std::path::Path;
use std::sync::Arc;

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::BackendError;
use e_voice_core::schema::lang::Lang;
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineTransducerModelConfig};

use crate::config::asr::AsrConfig;
use crate::schema::audio::Audio;
use crate::workflow::asr::base::{AsrSession, StreamingAsr};
use crate::workflow::asr::online::OnlineSession;
use crate::workflow::asr::transducer::Transducer;

/// Kroko streaming zipformers, one model per language (CC-BY-SA community release).
pub struct KrokoAsr {
    es: Arc<OnlineRecognizer>,
    en: Arc<OnlineRecognizer>,
    lead: usize,
    tail: usize,
}

impl Debug for KrokoAsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KrokoAsr")
            .field("lead", &self.lead)
            .field("tail", &self.tail)
            .finish_non_exhaustive()
    }
}

impl KrokoAsr {
    // ##### PRIVATE #####

    fn new_for(dir: &Path, threads: u16) -> Result<Arc<OnlineRecognizer>, BackendError> {
        let files = Transducer::locate(dir)?;
        let mut sherpa = OnlineRecognizerConfig::default();
        sherpa.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(files.encoder),
            decoder: Some(files.decoder),
            joiner: Some(files.joiner),
        };
        sherpa.model_config.tokens = Some(files.tokens);
        sherpa.model_config.num_threads = i32::from(threads);
        sherpa.model_config.provider = Some(Runtime::provider());
        sherpa.decoding_method = Some("greedy_search".to_owned());
        sherpa.enable_endpoint = false;
        OnlineRecognizer::create(&sherpa)
            .map(Arc::new)
            .ok_or(BackendError::Load("kroko"))
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// Model files missing from either directory, or the runtime rejecting them.
    pub fn new(es: &Path, en: &Path, config: &AsrConfig) -> Result<Self, BackendError> {
        let samples = |duration| usize::try_from(Audio::length(duration)).unwrap_or(usize::MAX);
        Ok(Self {
            es: Self::new_for(es, config.threads)?,
            en: Self::new_for(en, config.threads)?,
            lead: samples(config.lead()),
            tail: samples(config.tail()),
        })
    }
}

impl StreamingAsr for KrokoAsr {
    fn open(&self, lang: Lang) -> Result<Box<dyn AsrSession>, BackendError> {
        let recognizer = match lang {
            Lang::Es => &self.es,
            Lang::En => &self.en,
        };
        Ok(Box::new(OnlineSession::open(recognizer, |_| {}, self.lead, self.tail)))
    }
}
