use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// How installed models are checked before serving: the install stamp only, or every file re-hashed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelsVerify {
    #[default]
    Stamp,
    Full,
}

/// Internal tooling: the STT data root (pipeline weights in `models/`, tool weights and the sherpa
/// toolkit in `ops/`), the pinned model manifest, and how strictly installs are verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpsConfig {
    pub data: PathBuf,
    pub manifest: PathBuf,
    pub verify: ModelsVerify,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/stt"),
            manifest: PathBuf::from("stt/models.toml"),
            verify: ModelsVerify::Stamp,
        }
    }
}
