use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::api::state::AppState;
use crate::core::settings::Settings;
use crate::workflow::nodes::{Nodes, NodesError};
use crate::workflow::runner::Runner;

/// Gateway startup: shared node startup, then the request-wide state.
#[derive(Debug)]
pub struct Lifespan;

impl Lifespan {
    /// # Errors
    /// Node startup failed.
    pub async fn start(settings: &Settings) -> Result<AppState, NodesError> {
        let (nodes, runtime) = Nodes::start(settings).await?;
        tracing::info!(models = ?settings.models(), mode = ?nodes.asr.mode(), "lifespan.ready");
        Ok(AppState {
            runner: Runner::new(Arc::new(nodes), settings.stt.pipeline.clone()),
            lang: settings.stt.lang,
            emotion: settings.stt.api.emotion,
            models: settings
                .stt
                .pipeline
                .asr
                .selectable()
                .into_iter()
                .map(|engine| engine.name().to_owned())
                .collect(),
            upload: settings.stt.api.upload.saturating_mul(1 << 20),
            runtime,
            shutdown: CancellationToken::new(),
        })
    }
}
