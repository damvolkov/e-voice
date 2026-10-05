use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VadBackend {
    #[default]
    Silero,
    Ten,
}

/// Speech segmentation: durations bound what counts as speech and silence; `pad` is extra audio kept
/// on each side of a segment for the nodes (its span stays exact).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VadConfig {
    pub backend: VadBackend,
    pub threshold: f32,
    #[serde(with = "humantime_serde")]
    pub min_silence: Duration,
    #[serde(with = "humantime_serde")]
    pub min_speech: Duration,
    #[serde(with = "humantime_serde")]
    pub max_speech: Duration,
    #[serde(with = "humantime_serde")]
    pub pad: Duration,
    pub threads: u16,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            backend: VadBackend::Silero,
            threshold: 0.5,
            min_silence: Duration::from_millis(500),
            min_speech: Duration::from_millis(250),
            max_speech: Duration::from_secs(20),
            pad: Duration::from_millis(300),
            threads: 1,
        }
    }
}

impl VadConfig {
    #[must_use]
    pub const fn model(&self) -> &'static str {
        match self.backend {
            VadBackend::Silero => "silero-vad",
            VadBackend::Ten => "ten-vad",
        }
    }
}
