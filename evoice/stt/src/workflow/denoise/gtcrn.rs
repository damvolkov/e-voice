use std::fmt::{self, Debug};
use std::path::{Path, PathBuf};

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::BackendError;
use sherpa_onnx::{
    OfflineSpeechDenoiserGtcrnModelConfig, OfflineSpeechDenoiserModelConfig, OnlineSpeechDenoiser,
    OnlineSpeechDenoiserConfig,
};

use crate::schema::audio::RATE;
use crate::workflow::denoise::base::{Denoise, DenoiseSession};

const SAMPLE_RATE: i32 = RATE.cast_signed();

/// GTCRN (48k parameters) streaming speech enhancement; each session owns its recurrent state.
#[derive(Debug, Clone)]
pub struct GtcrnDenoise {
    config: OnlineSpeechDenoiserConfig,
}

impl GtcrnDenoise {
    /// # Errors
    /// No `gtcrn*.onnx` in `dir`.
    pub fn new(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let model: PathBuf = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|extension| extension == "onnx"))
            .ok_or_else(|| BackendError::Missing(dir.join("gtcrn_simple.onnx")))?;
        let config = OnlineSpeechDenoiserConfig {
            model: OfflineSpeechDenoiserModelConfig {
                gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                    model: Some(model.display().to_string()),
                },
                num_threads: i32::from(threads),
                provider: Some(Runtime::provider()),
                ..OfflineSpeechDenoiserModelConfig::default()
            },
        };
        let gtcrn = Self { config };
        gtcrn.open()?;
        Ok(gtcrn)
    }
}

impl Denoise for GtcrnDenoise {
    fn open(&self) -> Result<Box<dyn DenoiseSession>, BackendError> {
        let denoiser = OnlineSpeechDenoiser::create(&self.config).ok_or(BackendError::Load("gtcrn"))?;
        Ok(Box::new(GtcrnSession { denoiser }))
    }
}

pub struct GtcrnSession {
    denoiser: OnlineSpeechDenoiser,
}

impl Debug for GtcrnSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GtcrnSession").finish_non_exhaustive()
    }
}

impl DenoiseSession for GtcrnSession {
    fn push(&mut self, audio: &[f32]) -> Vec<f32> {
        self.denoiser.run(audio, SAMPLE_RATE).samples
    }

    fn flush(&mut self) -> Vec<f32> {
        self.denoiser.flush().samples
    }
}
