use std::fmt::{self, Debug};
use std::path::Path;
use std::sync::Arc;

use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, OnlineTransducerModelConfig};

use crate::config::asr::AsrConfig;
use crate::core::runtime::Runtime;
use crate::schema::audio::{Audio, RATE};
use crate::schema::error::{BackendError, NodeError};
use crate::schema::lang::Lang;
use crate::workflow::asr::base::{AsrSession, StreamingAsr};
use crate::workflow::asr::transducer::Transducer;

const SAMPLE_RATE: i32 = RATE.cast_signed();

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
        let stream = self.recognizer.create_stream();
        stream.set_option("language", lang.locale());
        stream.accept_waveform(SAMPLE_RATE, &vec![0.0; self.lead]);
        Ok(Box::new(NemotronAsrSession {
            recognizer: Arc::clone(&self.recognizer),
            stream,
            tail: self.tail,
            last: String::new(),
        }))
    }
}

pub struct NemotronAsrSession {
    recognizer: Arc<OnlineRecognizer>,
    stream: OnlineStream,
    tail: usize,
    last: String,
}

impl Debug for NemotronAsrSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NemotronAsrSession")
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl NemotronAsrSession {
    fn common_decode(&self) -> Option<String> {
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
        self.recognizer
            .get_result(&self.stream)
            .map(|result| result.text.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

impl AsrSession for NemotronAsrSession {
    fn push(&mut self, audio: &[f32]) -> Option<String> {
        self.stream.accept_waveform(SAMPLE_RATE, audio);
        let text = self.common_decode().filter(|text| *text != self.last)?;
        self.last.clone_from(&text);
        Some(text)
    }

    fn finish(self: Box<Self>) -> Result<String, NodeError> {
        self.stream.accept_waveform(SAMPLE_RATE, &vec![0.0; self.tail]);
        self.stream.input_finished();
        self.common_decode()
            .ok_or_else(|| NodeError::Backend("nemotron produced no result".to_owned()))
    }
}
