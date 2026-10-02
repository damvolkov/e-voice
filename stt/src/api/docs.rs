use axum::Router;
use utoipa::OpenApi;
use utoipa_scalar::{Scalar, Servable};

use crate::api::state::AppState;

/// The whole HTTP and WebSocket surface: native, OpenAI, Deepgram and ElevenLabs compatible.
#[derive(Debug, OpenApi)]
#[openapi(
    info(
        title = "e-voice",
        description = "CPU-first speech-to-text. One pipeline (wake word → VAD → ASR ∥ SER) behind native, OpenAI, Deepgram and ElevenLabs compatible endpoints. Emotion is an extension every endpoint can drop with `emotion=off`.",
        license(name = "MIT")
    ),
    paths(
        crate::api::health::health,
        crate::api::openai::models::models,
        crate::api::openai::transcriptions::transcriptions,
        crate::api::openai::realtime::realtime,
        crate::api::deepgram::listen::prerecorded,
        crate::api::deepgram::listen::streaming,
        crate::api::elevenlabs::scribe::scribe,
        crate::api::native::stream::stream,
    ),
    tags(
        (name = "native", description = "e-voice events over WebSocket (`view=struct|flat`)"),
        (name = "openai", description = "OpenAI Audio API: REST transcriptions (+SSE) and Realtime transcription"),
        (name = "deepgram", description = "Deepgram `/v1/listen`: prerecorded and live"),
        (name = "elevenlabs", description = "ElevenLabs Scribe `/v1/speech-to-text`"),
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
