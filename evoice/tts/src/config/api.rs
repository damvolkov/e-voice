use serde::{Deserialize, Serialize};

/// The TTS API: its port and the voice-clip upload cap in MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiConfig {
    pub port: u16,
    pub upload: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self { port: 5600, upload: 25 }
    }
}
