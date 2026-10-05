use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Wake-word detector: sherpa-onnx open-vocabulary keyword spotting (any phrase, no training) or an
/// openWakeWord classifier trained for one phrase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WwBackend {
    #[default]
    Off,
    Kws,
    Oww,
}

/// `keyword` is free text for `kws` (measure `threshold`/`boost` with `make wake`; one word fires
/// too often, use two or more) and a trained
/// classifier name for `oww`. A detection silences the detector for `cooldown`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WwConfig {
    pub backend: WwBackend,
    pub keyword: String,
    pub threshold: Option<f32>,
    pub boost: Option<f32>,
    #[serde(with = "humantime_serde")]
    pub cooldown: Duration,
    pub threads: u16,
}

impl Default for WwConfig {
    fn default() -> Self {
        Self {
            backend: WwBackend::Off,
            keyword: "hey eager".to_owned(),
            threshold: None,
            boost: None,
            cooldown: Duration::from_secs(2),
            threads: 1,
        }
    }
}

impl WwConfig {
    pub const OWW_KEYWORDS: &'static [&'static str] = &["hey_jarvis"];
    const KWS_BOOST: f32 = 3.0;

    /// Detection threshold in (0, 1]; lower triggers more easily.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold.unwrap_or(match self.backend {
            WwBackend::Off | WwBackend::Kws => 0.1,
            WwBackend::Oww => 0.5,
        })
    }

    #[must_use]
    pub fn boost(&self) -> f32 {
        self.boost.unwrap_or(Self::KWS_BOOST)
    }

    /// Manifest id of the selected backend's model, if any.
    #[must_use]
    pub fn model(&self) -> Option<String> {
        match self.backend {
            WwBackend::Off => None,
            WwBackend::Kws => Some("kws-gigaspeech".to_owned()),
            WwBackend::Oww => Some(format!("oww-{}", self.keyword.replace('_', "-"))),
        }
    }
}
