use serde::{Deserialize, Serialize};

/// Why a node produced no result for a segment.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", content = "detail", rename_all = "lowercase")]
pub enum NodeError {
    #[error("deadline exceeded")]
    Timeout,
    #[error("segment stayed open beyond the stall limit")]
    Stalled,
    #[error("dropped under load")]
    Overload,
    #[error("cancelled")]
    Cancelled,
    #[error("backend failure: {0}")]
    Backend(String),
}

/// Why a backend or one of its sessions could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BackendError {
    #[error("model {0} is not in the manifest")]
    Unknown(String),
    #[error("model file not found: {0}")]
    Missing(std::path::PathBuf),
    #[error("{0} rejected its configuration")]
    Load(&'static str),
}
