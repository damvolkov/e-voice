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
        let asr = &settings.stt.pipeline.asr;
        let models = [Some(asr.model()), asr.offline_model()]
            .into_iter()
            .flatten()
            .map(str::to_owned);
        Ok(AppState {
            runner: Runner::new(Arc::new(nodes), settings.stt.pipeline.clone()),
            lang: settings.stt.lang,
            emotion: settings.server.emotion,
            models: models.fold(Vec::new(), |mut models, model| {
                if !models.contains(&model) {
                    models.push(model);
                }
                models
            }),
            upload: settings.server.upload.saturating_mul(1 << 20),
            runtime,
            shutdown: CancellationToken::new(),
        })
    }
}
