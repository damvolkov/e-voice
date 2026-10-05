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

/// Nemotron 3.5 cache-aware streaming transducer; the language is a per-stream prompt.
pub struct NemotronAsr {
    recognizer: Arc<OnlineRecognizer>,
    lead: usize,
    tail: usize,
}

impl Debug for NemotronAsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NemotronAsr")
            .field("lead", &self.lead)
            .field("tail", &self.tail)
            .finish_non_exhaustive()
    }
}

impl NemotronAsr {
    /// # Errors
    /// Model files missing from `dir`, or the runtime rejecting them.
    pub fn new(dir: &Path, config: &AsrConfig) -> Result<Self, BackendError> {
        let files = Transducer::locate(dir)?;
        let mut sherpa = OnlineRecognizerConfig::default();
        sherpa.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(files.encoder),
            decoder: Some(files.decoder),
            joiner: Some(files.joiner),
        };
        sherpa.model_config.tokens = Some(files.tokens);
        sherpa.model_config.num_threads = i32::from(config.threads);
        sherpa.model_config.provider = Some(Runtime::provider());
        sherpa.decoding_method = Some("greedy_search".to_owned());
        sherpa.enable_endpoint = false;
        let recognizer = OnlineRecognizer::create(&sherpa).ok_or(BackendError::Load("nemotron"))?;
        let samples = |duration| usize::try_from(Audio::length(duration)).unwrap_or(usize::MAX);
        Ok(Self {
            recognizer: Arc::new(recognizer),
            lead: samples(config.lead()),
            tail: samples(config.tail()),
        })
    }
}

impl StreamingAsr for NemotronAsr {
    fn open(&self, lang: Lang) -> Result<Box<dyn AsrSession>, BackendError> {
        Ok(Box::new(OnlineSession::open(
            &self.recognizer,
            |stream| stream.set_option("language", lang.locale()),
            self.lead,
            self.tail,
        )))
    }
}
