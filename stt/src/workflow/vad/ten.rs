use std::path::Path;

use sherpa_onnx::{TenVadModelConfig, VadModelConfig};

use crate::config::vad::VadConfig;
use crate::core::runtime::Runtime;
use crate::schema::audio::RATE;
use crate::schema::error::BackendError;
use crate::workflow::vad::detector::DetectorVad;

const FILE: &str = "ten-vad.onnx";
const WINDOW: usize = 256;

/// TEN VAD on sherpa-onnx: 16 ms windows.
#[derive(Debug)]
pub struct TenVad;

impl TenVad {
    /// # Errors
    /// The model file is missing from `dir`.
    pub fn build(dir: &Path, config: &VadConfig) -> Result<DetectorVad, BackendError> {
        let model = dir.join(FILE);
        let ten_vad = TenVadModelConfig {
            model: Some(model.display().to_string()),
            threshold: config.threshold,
            min_silence_duration: config.min_silence.as_secs_f32(),
            min_speech_duration: config.min_speech.as_secs_f32(),
            window_size: i32::try_from(WINDOW).unwrap_or(i32::MAX),
            max_speech_duration: config.max_speech.as_secs_f32(),
        };
        let sherpa = VadModelConfig {
            ten_vad,
            sample_rate: i32::try_from(RATE).unwrap_or(i32::MAX),
            num_threads: i32::from(config.threads),
            provider: Some(Runtime::provider()),
            ..VadModelConfig::default()
        };
        DetectorVad::new(&model, sherpa, WINDOW, config)
    }
}
