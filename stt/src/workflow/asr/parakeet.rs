use std::fmt::{self, Debug};
use std::path::Path;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

use crate::core::runtime::Runtime;
use crate::schema::audio::RATE;
use crate::schema::error::{BackendError, NodeError};
use crate::schema::lang::Lang;
use crate::workflow::asr::base::BatchAsr;
use crate::workflow::asr::transducer::Transducer;

const SAMPLE_RATE: i32 = RATE.cast_signed();

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
        sherpa.model_config.num_threads = i32::from(threads);
        sherpa.model_config.provider = Some(Runtime::provider());
        sherpa.decoding_method = Some("greedy_search".to_owned());
        let recognizer = OfflineRecognizer::create(&sherpa).ok_or(BackendError::Load("parakeet"))?;
        Ok(Self { recognizer })
    }
}

impl BatchAsr for ParakeetAsr {
    fn transcribe(&self, _lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE, audio);
        self.recognizer.decode(&stream);
        stream
            .get_result()
            .map(|result| result.text.split_whitespace().collect::<Vec<_>>().join(" "))
            .ok_or_else(|| NodeError::Backend("parakeet produced no result".to_owned()))
    }
}
