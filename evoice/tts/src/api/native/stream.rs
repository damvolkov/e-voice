use axum::extract::ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::Response;
use e_voice_core::schema::error::NodeError;
use e_voice_core::schema::lang::Lang;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use utoipa::{IntoParams, ToSchema};

use crate::api::error::ApiError;
use crate::api::state::AppState;
use crate::core::encode::AudioFormat;
use crate::schema::event::Event;
use crate::workflow::runner::Request;
use crate::workflow::synth::base::SynthSession;

const NORMAL: u16 = 1000;
const UNSUPPORTED: u16 = 1003;
const INTERNAL: u16 = 1011;

/// Stream options; a missing `voice` uses the default one.
#[derive(Debug, Deserialize, IntoParams)]
pub struct StreamQuery {
    lang: Option<String>,
    voice: Option<String>,
    /// `pcm` (s16le, default) or `f32` binary frames.
    format: Option<String>,
}

/// Client → server text frames.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClientMessage {
    /// Any amount of text, e.g. one LLM token; sentences are cut server-side.
    Text { text: String },
    /// Speak what is buffered even without a sentence end.
    Flush,
    /// Barge-in: stop the current sentence and drop everything queued.
    Cancel,
    /// Speak the rest, then close.
    Close,
}

/// Server → client text frames; audio travels as binary frames between `start` and `end`.
#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ServerMessage {
    Ready {
        rate: u32,
        format: AudioFormat,
        lang: Lang,
        voice: Option<String>,
    },
    Start {
        sentence: u64,
        text: String,
    },
    End {
        sentence: u64,
        error: Option<NodeError>,
    },
    Error {
        message: String,
    },
    Closed,
}

/// Native streaming synthesis: send text as it arrives, receive PCM as it is generated.
///
/// Text frames in: `{"type":"text","text":"…"}`, `flush`, `cancel` (barge-in), `close`. Out: `ready`,
/// then per sentence `start`, binary audio frames (one per 80 ms), `end`; `closed` last. The voice is
/// primed before the upgrade, so a busy server or an unknown voice fails as plain HTTP.
///
/// # Errors
/// An unsupported format or language, an unreadable default voice, or no free worker.
#[utoipa::path(
    get,
    path = "/v1/stream",
    tag = "native",
    params(StreamQuery),
    responses(
        (status = 101, description = "WebSocket upgrade"),
        (status = 503, description = "every worker is busy", body = crate::api::error::ErrorBody)
    )
)]
pub async fn stream(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let format = match query.format.as_deref().unwrap_or("pcm") {
        "pcm" => AudioFormat::Pcm,
        "f32" => AudioFormat::F32,
        other => {
            return Err(ApiError::Invalid(format!(
                "unsupported format {other:?}; use pcm or f32"
            )));
        }
    };
    let lang = query
        .lang
        .as_deref()
        .map(str::parse::<Lang>)
        .transpose()
        .map_err(ApiError::Invalid)?
        .unwrap_or(state.lang);
    let named = query
        .voice
        .clone()
        .filter(|id| state.voices.get(id).is_ok())
        .or_else(|| state.voice.clone());
    let voice = state.voice(query.voice).await?;
    let session = state.runner.open(lang, voice).await?;
    let ready = ServerMessage::Ready {
        rate: state.runner.caps().rate,
        format,
        lang,
        voice: named,
    };
    Ok(ws.on_upgrade(move |socket| serve(state, socket, session, format, ready)))
}

async fn serve(
    state: AppState,
    socket: WebSocket,
    session: Box<dyn SynthSession>,
    format: AudioFormat,
    ready: ServerMessage,
) {
    let json =
        |message: &ServerMessage| Message::Text(Utf8Bytes::from(serde_json::to_string(message).unwrap_or_default()));
    let (mut sink, mut source) = socket.split();
    let (requests, inbox) = mpsc::channel(64);
    let (outbox, mut events) = mpsc::channel(256);
    let cancel = state.shutdown.child_token();
    let runner = state.runner.clone();
    let token = cancel.clone();
    let run = tokio::spawn(async move { runner.run(session, inbox, outbox, token).await });
    let mut code = NORMAL;
    if sink.send(json(&ready)).await.is_err() {
        cancel.cancel();
        return;
    }
    loop {
        tokio::select! {
            message = source.next() => match message {
                Some(Ok(Message::Text(text))) => match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(message) => {
                        let request = match message {
                            ClientMessage::Text { text } => Request::Text(text),
                            ClientMessage::Flush => Request::Flush,
                            ClientMessage::Cancel => Request::Cancel,
                            ClientMessage::Close => Request::Close,
                        };
                        requests.send(request).await.ok();
                    }
                    Err(error) => {
                        sink.send(json(&ServerMessage::Error { message: error.to_string() })).await.ok();
                    }
                },
                Some(Ok(Message::Binary(_))) => {
                    code = UNSUPPORTED;
                    break;
                }
                Some(Ok(Message::Close(_)) | Err(_)) | None => {
                    cancel.cancel();
                    break;
                }
                Some(Ok(_)) => {}
            },
            event = events.recv() => match event {
                Some(Event::Audio { audio, .. }) => {
                    if sink.send(Message::Binary(format.samples(&audio).into())).await.is_err() {
                        cancel.cancel();
                        break;
                    }
                }
                Some(Event::Start { sentence, text }) => {
                    sink.send(json(&ServerMessage::Start { sentence: sentence.0, text })).await.ok();
                }
                Some(Event::End { sentence, error }) => {
                    sink.send(json(&ServerMessage::End { sentence: sentence.0, error })).await.ok();
                }
                Some(Event::Closed) => {
                    sink.send(json(&ServerMessage::Closed)).await.ok();
                    break;
                }
                None => {
                    code = if run.is_finished() { NORMAL } else { INTERNAL };
                    break;
                }
            },
        }
    }
    cancel.cancel();
    let frame = CloseFrame {
        code,
        reason: Utf8Bytes::from_static(""),
    };
    sink.send(Message::Close(Some(frame))).await.ok();
}
