use axum::Json;
use axum::extract::State;
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::state::AppState;

#[derive(Debug, Serialize, ToSchema)]
pub struct Health {
    status: &'static str,
    sherpa: &'static str,
    onnxruntime: &'static str,
}

/// Liveness and the native runtime versions loaded.
#[utoipa::path(get, path = "/health", tag = "service", responses((status = 200, body = Health)))]
pub async fn health(State(state): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok",
        sherpa: state.runtime.sherpa,
        onnxruntime: state.runtime.onnxruntime,
    })
}
