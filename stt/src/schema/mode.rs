use serde::{Deserialize, Serialize};

/// How a node consumes a segment: incrementally while speech lasts, or whole once it ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Streaming,
    Batch,
}
