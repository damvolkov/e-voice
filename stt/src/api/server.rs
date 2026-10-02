use std::net::SocketAddr;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;

use tower_http::cors::CorsLayer;

use crate::api::deepgram::listen::{prerecorded, streaming};
use crate::api::docs::ApiDoc;
use crate::api::elevenlabs::scribe::scribe;
use crate::api::health::health;
use crate::api::lifespan::Lifespan;
use crate::api::native::stream::stream;
use crate::api::openai::models::models;
use crate::api::openai::realtime::realtime;
use crate::api::openai::transcriptions::transcriptions;
use crate::api::state::AppState;
use crate::core::settings::Settings;
use crate::workflow::nodes::NodesError;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Nodes(#[from] NodesError),
    #[error("cannot bind or serve: {0}")]
    Io(#[from] std::io::Error),
}

/// HTTP/WebSocket gateway; SIGINT/SIGTERM cancel every stream, which close after finalizing.
#[derive(Debug)]
pub struct Server;

impl Server {
    /// Every route: native stream, OpenAI (REST, SSE, Realtime), Deepgram (REST, live), ElevenLabs
    /// Scribe, health, `/openapi.json` and the `/docs` reference; CORS is open for browser clients.
    pub fn router(state: AppState) -> Router {
        let upload = DefaultBodyLimit::max(state.upload);
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/audio/transcriptions", post(transcriptions).layer(upload))
            .route("/v1/realtime", get(realtime))
            .route("/v1/listen", post(prerecorded).layer(upload).get(streaming))
            .route("/v1/speech-to-text", post(scribe).layer(upload))
            .route("/v1/stream", get(stream))
            .merge(ApiDoc::router())
            .layer(CorsLayer::permissive())
            .with_state(state)
    }

    /// # Errors
    /// Startup failed, or the address cannot be bound.
    pub async fn serve(settings: &Settings) -> Result<(), ServerError> {
        let state = Lifespan::start(settings).await?;
        let shutdown = state.shutdown.clone();
        let app = Self::router(state);
        let address = SocketAddr::new(settings.server.host, settings.server.port);
        let listener = TcpListener::bind(address).await?;
        tracing::info!(%address, "server.listening");
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let interrupt = tokio::signal::ctrl_c();
                let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
                let term = async move {
                    match terminate.as_mut() {
                        Some(signal) => signal.recv().await,
                        None => std::future::pending().await,
                    }
                };
                tokio::select! {
                    _ = interrupt => {}
                    _ = term => {}
                }
                tracing::info!("server.stopping");
                shutdown.cancel();
            })
            .await?;
        Ok(())
    }
}
