use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::{BackendError, NodeError};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineStream};

use crate::schema::audio::RATE;

const SAMPLE_RATE: i32 = RATE.cast_signed();

/// Shared mechanics of the sherpa offline engines: recognizer creation and one-shot decoding.
#[derive(Debug)]
pub struct Offline;

impl Offline {
    /// Fills threads, provider and greedy decoding, then creates the recognizer.
    ///
    /// # Errors
    /// The runtime rejects the model files.
    pub fn create(
        mut config: OfflineRecognizerConfig,
        threads: u16,
        name: &'static str,
    ) -> Result<OfflineRecognizer, BackendError> {
        config.model_config.num_threads = i32::from(threads);
        config.model_config.provider = Some(Runtime::provider());
        config.decoding_method = Some("greedy_search".to_owned());
        OfflineRecognizer::create(&config).ok_or(BackendError::Load(name))
    }

    /// Decodes `audio` on a fresh stream prepared by `prepare`; whitespace is normalized.
    ///
    /// # Errors
    /// The runtime produced no result.
    pub fn decode(
        recognizer: &OfflineRecognizer,
        prepare: impl FnOnce(&OfflineStream),
        audio: &[f32],
        name: &'static str,
    ) -> Result<String, NodeError> {
        let stream = recognizer.create_stream();
        prepare(&stream);
        stream.accept_waveform(SAMPLE_RATE, audio);
        recognizer.decode(&stream);
        stream
            .get_result()
            .map(|result| result.text.split_whitespace().collect::<Vec<_>>().join(" "))
            .ok_or_else(|| NodeError::Backend(format!("{name} produced no result")))
    }
}
