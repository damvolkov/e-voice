use std::convert::Infallible;

use axum::extract::{Multipart, State};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::stream;

use crate::api::batch::Batch;
use crate::api::openai::formats::{
    OpenaiError, OpenaiForm, OpenaiFormat, TranscriptionForm, TranscriptionJson, TranscriptionVerbose,
};
use crate::api::state::AppState;
use crate::schema::event::Event;
use crate::schema::transcript::Transcript;

/// OpenAI-compatible transcription. The file runs through the same pipeline as live streams,
/// losslessly; with `stream=true` every final arrives as a `transcript.text.delta` event.
///
/// # Errors
/// 400 for a bad form, language, format, emotion mode or undecodable file; 413 over the upload
/// limit; 500 when the pipeline fails or any segment fails.
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    tag = "openai",
    request_body(content = TranscriptionForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "`json` (default)", body = TranscriptionJson),
        (status = 200, description = "`verbose_json`", body = TranscriptionVerbose),
        (status = 200, description = "`text`, `srt` or `vtt`", body = String, content_type = "text/plain"),
        (status = 200, description = "`stream=true`: `transcript.text.delta` … `transcript.text.done`", content_type = "text/event-stream"),
        (status = 400, description = "Invalid request (OpenAI error shape)"),
        (status = 413, description = "Upload over the limit"),
    )
)]
pub async fn transcriptions(State(state): State<AppState>, multipart: Multipart) -> Result<Response, OpenaiError> {
    let form = OpenaiForm::parse(multipart, state.lang, state.emotion).await?;
    tracing::info!(lang = ?form.lang, model = ?form.model, format = ?form.format, stream = form.stream, "openai.transcription");
    let (lang, tags) = (form.lang, form.emotion.tags());
    if form.stream {
        let (events, seconds) = Batch::stream(&state, form.file.to_vec(), form.extension, lang).await?;
        let deltas = stream::unfold(
            (events, String::new(), false),
            move |(mut events, mut text, done)| async move {
                if done {
                    return None;
                }
                loop {
                    let (next, last) = match events.recv().await {
                        Some(Event::Final(segment)) if segment.error.is_none() && !segment.text.is_empty() => {
                            let piece = Transcript {
                                lang,
                                samples: 0,
                                segments: vec![segment],
                            }
                            .text(tags);
                            let delta = if text.is_empty() { piece } else { format!(" {piece}") };
                            text.push_str(&delta);
                            (OpenaiFormat::delta(&delta), false)
                        }
                        Some(Event::Final(segment)) => match segment.error {
                            Some(error) => (OpenaiFormat::failure(&error.to_string()), true),
                            None => continue,
                        },
                        Some(Event::Closed) => (OpenaiFormat::done(&text, seconds), true),
                        Some(Event::Wake(_) | Event::Speech(_) | Event::Partial(_)) => continue,
                        None => (OpenaiFormat::failure("transcription ended unexpectedly"), true),
                    };
                    return Some((Ok::<SseEvent, Infallible>(next), (events, text, last)));
                }
            },
        );
        return Ok(Sse::new(deltas).keep_alive(KeepAlive::default()).into_response());
    }
    let transcript = Batch::transcribe(&state, form.file.to_vec(), form.extension, lang).await?;
    Ok(form.format.render(&transcript, form.emotion))
}
