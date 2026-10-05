use serde::{Deserialize, Serialize};

/// How streamed text becomes sentences: shorter ones merge up to `min` chars, longer ones are cut at
/// `max` (clause marks first, then spaces).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextConfig {
    pub min: usize,
    pub max: usize,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self { min: 12, max: 80 }
    }
}
