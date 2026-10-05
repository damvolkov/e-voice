use axum::Router;
use utoipa::OpenApi;
use utoipa_scalar::{Scalar, Servable};

use crate::api::state::AppState;

/// The whole HTTP and WebSocket surface of the TTS service.
#[derive(Debug, OpenApi)]
#[openapi(
    info(
        title = "e-voice TTS",
        description = "CPU-first streaming text-to-speech. Text in (whole or token by token), audio out while it is generated: every backend streams frame by frame. Voices are learned from a few seconds of audio (`/v1/voices`) and used by id in every endpoint.",
        license(name = "MIT")
    ),
    paths(
        crate::api::health::health,
        crate::api::openai::models::models,
        crate::api::openai::speech::speech,
        crate::api::native::stream::stream,
        crate::api::voices::list,
        crate::api::voices::create,
        crate::api::voices::remove,
    ),
    components(schemas(crate::api::native::stream::ClientMessage, crate::api::native::stream::ServerMessage)),
    tags(
        (name = "native", description = "e-voice streaming synthesis over WebSocket"),
        (name = "openai", description = "OpenAI Audio API: speech (streamed bytes or SSE)"),
        (name = "voices", description = "Voices learned from audio clips"),
        (name = "service", description = "Health"),
    )
)]
pub struct ApiDoc;

impl ApiDoc {
    /// `/openapi.json` and the Scalar reference at `/docs`.
    pub fn router() -> Router<AppState> {
        let spec = Self::openapi();
        let json = spec.clone();
        Router::new()
            .route(
                "/openapi.json",
                axum::routing::get(move || async move { axum::Json(json) }),
            )
            .merge(Scalar::with_url("/docs", spec))
    }
}
