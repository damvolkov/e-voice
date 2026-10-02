use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Live ASR: Nemotron 3.5 streams partials while speech lasts; Parakeet v3 transcribes each segment
/// once it ends (more accurate, no partials).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrBackend {
    #[default]
    Nemotron,
    Parakeet,
}

/// Nemotron encoder chunk: shorter refreshes partials sooner, costs more CPU and accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NemotronChunk {
    #[serde(rename = "160ms")]
    Ms160,
    #[serde(rename = "560ms")]
    Ms560,
    #[default]
    #[serde(rename = "1120ms")]
    Ms1120,
}

/// ASR for uploaded files: Parakeet, or `live` to reuse the live backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OfflineBackend {
    #[default]
    Parakeet,
    Live,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OfflineConfig {
    pub backend: OfflineBackend,
    pub threads: u16,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            backend: OfflineBackend::Parakeet,
            threads: 4,
        }
    }
}

/// `chunk`, `lead` (silence fed before speech for encoder context) and `tail` (silence appended to
/// flush the last chunk) apply to Nemotron only. `deadline` counts from the end of speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrConfig {
    pub backend: AsrBackend,
    pub chunk: Option<NemotronChunk>,
    #[serde(with = "humantime_serde")]
    pub lead: Option<Duration>,
    #[serde(with = "humantime_serde")]
    pub tail: Option<Duration>,
    pub threads: u16,
    #[serde(with = "humantime_serde")]
    pub deadline: Duration,
    pub offline: OfflineConfig,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            backend: AsrBackend::Nemotron,
            chunk: None,
            lead: None,
            tail: None,
            threads: 4,
            deadline: Duration::from_secs(15),
            offline: OfflineConfig::default(),
        }
    }
}

impl AsrConfig {
    pub const PARAKEET: &'static str = "parakeet-v3-int8";

    #[must_use]
    pub fn chunk(&self) -> NemotronChunk {
        self.chunk.unwrap_or_default()
    }

    #[must_use]
    pub fn lead(&self) -> Duration {
        self.lead.unwrap_or(Duration::from_millis(300))
    }

    #[must_use]
    pub fn tail(&self) -> Duration {
        self.tail.unwrap_or(Duration::from_millis(1200))
    }

    /// Manifest id of the live backend's model.
    #[must_use]
    pub fn model(&self) -> &'static str {
        match (self.backend, self.chunk()) {
            (AsrBackend::Parakeet, _) => Self::PARAKEET,
            (AsrBackend::Nemotron, NemotronChunk::Ms160) => "nemotron-3.5-160ms-int8",
            (AsrBackend::Nemotron, NemotronChunk::Ms560) => "nemotron-3.5-560ms-int8",
            (AsrBackend::Nemotron, NemotronChunk::Ms1120) => "nemotron-3.5-1120ms-int8",
        }
    }

    /// Manifest id of the separate file backend, when one is configured.
    #[must_use]
    pub const fn offline_model(&self) -> Option<&'static str> {
        match self.offline.backend {
            OfflineBackend::Parakeet => Some(Self::PARAKEET),
            OfflineBackend::Live => None,
        }
    }
}
