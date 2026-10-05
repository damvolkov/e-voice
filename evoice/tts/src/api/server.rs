use std::net::SocketAddr;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;

use crate::api::docs::ApiDoc;
use crate::api::health::health;
use crate::api::lifespan::{Lifespan, LifespanError};
use crate::api::native::stream::stream;
use crate::api::openai::models::models;
use crate::api::openai::speech::speech;
use crate::api::state::AppState;
use crate::api::voices;
use crate::core::settings::Settings;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Lifespan(#[from] LifespanError),
    #[error("cannot bind or serve: {0}")]
    Io(#[from] std::io::Error),
}

/// HTTP/WebSocket gateway; SIGINT/SIGTERM cancel every stream.
#[derive(Debug)]
pub struct Server;

impl Server {
    /// Every route: native stream, OpenAI speech and models, voices, health, `/openapi.json` and the
    /// `/docs` reference; CORS is open for browser clients.
    pub fn router(state: AppState) -> Router {
        let upload = DefaultBodyLimit::max(state.upload);
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/audio/speech", post(speech))
            .route("/v1/stream", get(stream))
            .route("/v1/voices", get(voices::list).post(voices::create).layer(upload))
            .route("/v1/voices/{voice_id}", delete(voices::remove))
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
        let address = SocketAddr::new(settings.server.host, settings.tts.api.port);
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
