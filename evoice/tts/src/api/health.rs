use axum::Json;
use axum::extract::State;
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::state::AppState;

#[derive(Debug, Serialize, ToSchema)]
pub struct Health {
    status: &'static str,
    version: &'static str,
    onnxruntime: &'static str,
    rate: u32,
}

/// Liveness, the service version (its release tag), the onnxruntime loaded and the output rate.
#[utoipa::path(get, path = "/health", tag = "service", responses((status = 200, body = Health)))]
pub async fn health(State(state): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("E_VOICE_VERSION"),
        onnxruntime: state.runtime.onnxruntime,
        rate: state.runner.caps().rate,
    })
}
