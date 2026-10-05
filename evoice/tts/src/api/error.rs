use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use e_voice_core::schema::error::BackendError;
use serde::Serialize;
use utoipa::ToSchema;

use crate::core::voices::VoiceError;
use crate::schema::error::TrainError;
use crate::workflow::runner::RunnerError;

/// Error body in the OpenAI shape: `{"error": {"message", "type"}}`.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Voice(#[from] VoiceError),
    #[error(transparent)]
    Train(#[from] TrainError),
    #[error(transparent)]
    Runner(#[from] RunnerError),
}

impl ApiError {
    #[must_use]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::Invalid(_) | Self::Voice(VoiceError::Invalid(_)) | Self::Train(TrainError::Empty) => {
                StatusCode::BAD_REQUEST
            }
            Self::Voice(VoiceError::Unknown(_)) => StatusCode::NOT_FOUND,
            Self::Train(TrainError::Unsupported) => StatusCode::NOT_IMPLEMENTED,
            Self::Runner(RunnerError::Backend(BackendError::Busy(_))) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Runner(RunnerError::Backend(BackendError::Load(_))) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Voice(VoiceError::Io(_)) | Self::Train(TrainError::Backend(_)) | Self::Runner(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let kind = if status.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let body = ErrorBody {
            error: ErrorDetail {
                message: self.to_string(),
                kind,
            },
        };
        (status, Json(body)).into_response()
    }
}
