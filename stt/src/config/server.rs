use std::net::{IpAddr, Ipv4Addr};

use serde::{Deserialize, Serialize};

use crate::config::log::LogConfig;

/// How emotion reaches clients: `field` adds `emotion` objects to JSON responses and events, `tag`
/// also prefixes each segment's text (`[angry] …`) for clients that only read `text`, `off` drops it.
/// Requests can narrow it per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmotionMode {
    #[default]
    Field,
    Tag,
    Off,
}

/// Bind address, the transcription upload cap in MiB, emotion exposure and logging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: IpAddr,
    pub port: u16,
    pub upload: usize,
    pub emotion: EmotionMode,
    pub log: LogConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 5500,
            upload: 25,
            emotion: EmotionMode::Field,
            log: LogConfig::default(),
        }
    }
}

impl EmotionMode {
    /// Whether responses carry `emotion` objects.
    #[must_use]
    pub const fn field(self) -> bool {
        matches!(self, Self::Field | Self::Tag)
    }

    /// Whether text carries inline `[label]` prefixes.
    #[must_use]
    pub const fn tags(self) -> bool {
        matches!(self, Self::Tag)
    }
}

impl std::str::FromStr for EmotionMode {
    type Err = String;

    /// `field`, `tag` or `off`; `true`/`false` are accepted as `field`/`off`.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "field" | "true" => Ok(Self::Field),
            "tag" => Ok(Self::Tag),
            "off" | "false" | "none" => Ok(Self::Off),
            other => Err(format!("unsupported emotion {other:?}; use field, tag or off")),
        }
    }
}
