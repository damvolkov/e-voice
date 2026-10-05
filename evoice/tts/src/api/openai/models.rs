use axum::Json;
use axum::extract::State;
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::state::AppState;

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

/// The loaded synthesis models (one per language).
#[utoipa::path(get, path = "/v1/models", tag = "openai", responses((status = 200, body = ModelList)))]
pub async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: state
            .models
            .iter()
            .map(|id| ModelCard {
                id: id.clone(),
                object: "model",
                owned_by: "e-voice",
            })
            .collect(),
    })
}
