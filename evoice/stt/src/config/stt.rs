use e_voice_core::config::ops::OpsConfig;
use e_voice_core::schema::lang::Lang;
use serde::{Deserialize, Serialize};

use crate::config::api::ApiConfig;
use crate::config::pipeline::PipelineConfig;

/// Speech-to-text: the default language, its API, the pipeline and its internal tooling.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SttConfig {
    pub lang: Lang,
    pub api: ApiConfig,
    pub pipeline: PipelineConfig,
    pub ops: OpsConfig,
}
