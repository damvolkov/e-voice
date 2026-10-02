use tokio_util::sync::CancellationToken;

use crate::config::server::EmotionMode;
use crate::core::runtime::Runtime;
use crate::schema::lang::Lang;
use crate::workflow::runner::Runner;

/// Shared by every request: the runner over the loaded nodes, request defaults (`lang`, `emotion`),
/// the selectable file engines (default first) `/v1/models` advertises, the upload limit in bytes, and the shutdown signal
/// every stream obeys.
#[derive(Debug, Clone)]
pub struct AppState {
    pub runner: Runner,
    pub lang: Lang,
    pub emotion: EmotionMode,
    pub models: Vec<String>,
    pub upload: usize,
    pub runtime: Runtime,
    pub shutdown: CancellationToken,
}
