use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::api::state::AppState;

/// The loaded ASR models, live first; transcription endpoints accept any `model` value.
#[utoipa::path(get, path = "/v1/models", tag = "openai", responses((status = 200, description = "OpenAI model list", body = Object)))]
pub async fn models(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = state
        .models
        .iter()
        .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "e-voice"}))
        .collect();
    Json(json!({"object": "list", "data": data}))
}
