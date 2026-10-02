use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DenoiseBackend {
    #[default]
    Off,
    Gtcrn,
}

/// Streaming speech enhancement on every frame, before gain and VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DenoiseConfig {
    pub backend: DenoiseBackend,
    pub threads: u16,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            backend: DenoiseBackend::Off,
            threads: 1,
        }
    }
}

impl DenoiseConfig {
    #[must_use]
    pub const fn model(&self) -> Option<&'static str> {
        match self.backend {
            DenoiseBackend::Off => None,
            DenoiseBackend::Gtcrn => Some("gtcrn"),
        }
    }
}
