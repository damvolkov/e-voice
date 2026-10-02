use serde::{Deserialize, Serialize};

use crate::config::ops::OpsConfig;
use crate::config::pipeline::PipelineConfig;
use crate::schema::lang::Lang;

/// Speech-to-text: the default language, the pipeline and its internal tooling.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SttConfig {
    pub lang: Lang,
    pub pipeline: PipelineConfig,
    pub ops: OpsConfig,
}
