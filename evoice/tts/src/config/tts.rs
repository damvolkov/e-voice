use e_voice_core::config::ops::OpsConfig;
use e_voice_core::schema::lang::Lang;
use serde::{Deserialize, Deserializer, Serialize};

use crate::config::api::ApiConfig;
use crate::config::synth::SynthConfig;
use crate::config::text::TextConfig;

/// Text-to-speech: the default language and voice, its API, text segmentation, the backend and its
/// internal tooling (`ops.data` also holds learned voices under `voices/`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TtsConfig {
    pub lang: Lang,
    /// An empty value (e.g. an unset compose variable) means none.
    #[serde(deserialize_with = "TtsConfig::voice")]
    pub voice: Option<String>,
    pub api: ApiConfig,
    pub text: TextConfig,
    pub synth: SynthConfig,
    pub ops: OpsConfig,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            lang: Lang::Es,
            voice: None,
            api: ApiConfig::default(),
            text: TextConfig::default(),
            synth: SynthConfig::default(),
            ops: OpsConfig {
                data: "data/tts".into(),
                manifest: "evoice/tts/models.toml".into(),
                ..OpsConfig::default()
            },
        }
    }
}

impl TtsConfig {
    fn voice<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
        Ok(Option::<String>::deserialize(deserializer)?.filter(|voice| !voice.is_empty()))
    }
}
