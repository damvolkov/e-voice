use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LidBackend {
    #[default]
    Off,
    Whisper,
}

/// Spoken language identification per segment, before batch decoding: the detected language picks
/// the decoding language (Canary, Cohere) and is reported on the final. Streaming backends fix their
/// language at speech onset and keep the requested one. A detection outside es/en, or a segment
/// shorter than `min`, keeps the requested language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidConfig {
    pub backend: LidBackend,
    pub threads: u16,
    #[serde(with = "humantime_serde")]
    pub min: Duration,
}

impl Default for LidConfig {
    fn default() -> Self {
        Self {
            backend: LidBackend::Off,
            threads: 2,
            min: Duration::from_secs(1),
        }
    }
}

impl LidConfig {
    #[must_use]
    pub const fn model(&self) -> Option<&'static str> {
        match self.backend {
            LidBackend::Off => None,
            LidBackend::Whisper => Some("whisper-tiny"),
        }
    }
}
