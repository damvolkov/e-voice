use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use e_voice_core::audio::AudioFile;
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::schema::audio::Audio;

#[derive(Debug, Serialize, ToSchema)]
pub struct VoiceList {
    voices: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VoiceCreated {
    voice_id: String,
    /// Seconds of audio the voice was learned from.
    seconds: f64,
}

/// Learned voices, usable as `voice` in every endpoint.
///
/// # Errors
/// The voices directory cannot be read.
#[utoipa::path(get, path = "/v1/voices", tag = "voices", responses((status = 200, body = VoiceList)))]
pub async fn list(State(state): State<AppState>) -> Result<Json<VoiceList>, ApiError> {
    let voices = Arc::clone(&state.voices);
    let ids = tokio::task::spawn_blocking(move || voices.list())
        .await
        .map_err(|error| ApiError::Invalid(error.to_string()))??;
    Ok(Json(VoiceList { voices: ids }))
}

/// Learns a voice from one or more clips of the same speaker (multipart: `voice_id`, optional `text`
/// with what the clips say, then `file` parts: wav, mp3, m4a, flac, ogg, webm). Clean speech, 5-30 s
/// in total; anything past 30 s is ignored. The transcript enables in-context cloning on backends that
/// use it (Qwen3, NeuTTS). Replaces a voice with the same id.
///
/// # Errors
/// A malformed form, undecodable audio, a backend that cannot learn voices, or a failed write.
#[utoipa::path(
    post,
    path = "/v1/voices",
    tag = "voices",
    request_body(content_type = "multipart/form-data", description = "`voice_id`, optional `text` (transcript), one or more `file` parts"),
    responses(
        (status = 201, body = VoiceCreated),
        (status = 400, body = crate::api::error::ErrorBody),
        (status = 501, description = "the backend cannot learn voices", body = crate::api::error::ErrorBody)
    )
)]
pub async fn create(
    State(state): State<AppState>,
    mut form: Multipart,
) -> Result<(StatusCode, Json<VoiceCreated>), ApiError> {
    let invalid = |error: &dyn std::fmt::Display| ApiError::Invalid(error.to_string());
    let (mut id, mut text, mut files) = (None, None, Vec::new());
    while let Some(field) = form.next_field().await.map_err(|error| invalid(&error))? {
        match field.name() {
            Some("voice_id" | "id" | "name") => id = Some(field.text().await.map_err(|error| invalid(&error))?),
            Some("text" | "transcript") => text = Some(field.text().await.map_err(|error| invalid(&error))?),
            Some("file" | "files" | "files[]") => {
                let extension = field
                    .file_name()
                    .and_then(|name| name.rsplit_once('.'))
                    .map(|(_, extension)| extension.to_ascii_lowercase());
                files.push((
                    field.bytes().await.map_err(|error| invalid(&error))?.to_vec(),
                    extension,
                ));
            }
            _ => {}
        }
    }
    let id = id.ok_or_else(|| invalid(&"missing voice_id"))?;
    (!files.is_empty())
        .then_some(())
        .ok_or_else(|| invalid(&"missing file"))?;
    let (rate, synth, voices) = (
        state.runner.caps().rate,
        Arc::clone(state.runner.synth()),
        Arc::clone(&state.voices),
    );
    let learned = id.clone();
    let seconds = tokio::task::spawn_blocking(move || {
        let clips = files
            .into_iter()
            .map(|(bytes, extension)| AudioFile::decode(bytes, extension.as_deref(), rate).map(Audio::from))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ApiError::Invalid(error.to_string()))?;
        let seconds = clips.iter().map(|clip| clip.duration(rate).as_secs_f64()).sum::<f64>();
        let voice = synth.train(&clips, text.as_deref())?;
        voices.put(&learned, &voice)?;
        Ok::<_, ApiError>(seconds)
    })
    .await
    .map_err(|error| invalid(&error))??;
    tracing::info!(voice = %id, seconds, "voices.learned");
    Ok((StatusCode::CREATED, Json(VoiceCreated { voice_id: id, seconds })))
}

/// Forgets a learned voice.
///
/// # Errors
/// An invalid or unknown id.
#[utoipa::path(
    delete,
    path = "/v1/voices/{voice_id}",
    tag = "voices",
    params(("voice_id" = String, Path)),
    responses((status = 204), (status = 404, body = crate::api::error::ErrorBody))
)]
pub async fn remove(State(state): State<AppState>, Path(id): Path<String>) -> Result<StatusCode, ApiError> {
    let voices = Arc::clone(&state.voices);
    tokio::task::spawn_blocking(move || voices.remove(&id))
        .await
        .map_err(|error| ApiError::Invalid(error.to_string()))??;
    Ok(StatusCode::NO_CONTENT)
}
