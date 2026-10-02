use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SerBackend {
    Off,
    #[default]
    Emotion2vec,
}

/// Speech emotion recognition; `off` makes every segment's emotion `unknown`. `deadline` counts from
/// the end of speech; segments shorter than `min` are not classified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SerConfig {
    pub backend: SerBackend,
    pub threads: u16,
    #[serde(with = "humantime_serde")]
    pub deadline: Duration,
    #[serde(with = "humantime_serde")]
    pub min: Duration,
}

impl Default for SerConfig {
    fn default() -> Self {
        Self {
            backend: SerBackend::Emotion2vec,
            threads: 4,
            deadline: Duration::from_secs(5),
            min: Duration::from_millis(500),
        }
    }
}

impl SerConfig {
    #[must_use]
    pub const fn model(&self) -> Option<&'static str> {
        match self.backend {
            SerBackend::Off => None,
            SerBackend::Emotion2vec => Some("emotion2vec-plus-base"),
        }
    }
}
