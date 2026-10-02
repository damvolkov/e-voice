use axum::Json;
use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use utoipa::{IntoParams, ToSchema};

use crate::api::batch::{Batch, BatchError};
use crate::api::live::{Inbound, LiveOutcome, LiveProtocol, LiveStream};
use crate::api::state::AppState;
use crate::config::server::EmotionMode;
use crate::core::audio::AudioEncoding;
use crate::schema::emotion::Emotion;
use crate::schema::event::{Event, FinalEvent, SpeechState};
use crate::schema::lang::Lang;
use crate::schema::transcript::Transcript;

/// Error body in the shape Deepgram clients parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepgramError {
    pub status: StatusCode,
    pub message: String,
}

impl IntoResponse for DeepgramError {
    fn into_response(self) -> Response {
        let code = if self.status.is_client_error() {
            "Bad Request"
        } else {
            "Internal Server Error"
        };
        let body = json!({"err_code": code, "err_msg": self.message, "request_id": uuid::Uuid::new_v4().to_string()});
        (self.status, Json(body)).into_response()
    }
}

impl From<BatchError> for DeepgramError {
    fn from(error: BatchError) -> Self {
        match error {
            BatchError::Decode(message) | BatchError::Model(message) => Self {
                status: StatusCode::BAD_REQUEST,
                message,
            },
            BatchError::Failed(message) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message,
            },
        }
    }
}

/// `/v1/listen` query: Deepgram's names; options that do not apply (`punctuate`, `smart_format`,
/// `diarize`, …) are accepted and ignored.
#[derive(Debug, Default, Deserialize, IntoParams)]
#[serde(default)]
pub struct ListenQuery {
    /// A loaded engine name or model id selects it (prerecorded); any other value (`nova-2`) uses the
    /// default.
    pub model: Option<String>,
    /// `es` or `en` (regions such as `es-419` are accepted).
    pub language: Option<String>,
    /// Streaming only: `linear16` (default) or `pcm_f32le`.
    pub encoding: Option<String>,
    /// Streaming only: sample rate of the raw audio, 16000 by default.
    pub sample_rate: Option<u32>,
    /// Streaming only: send `is_final: false` results from partials (default true).
    pub interim_results: Option<bool>,
    /// `field`, `tag` or `off` for the `emotion` extension.
    pub emotion: Option<String>,
}

impl ListenQuery {
    fn parse(&self, state: &AppState) -> Result<(Lang, EmotionMode), DeepgramError> {
        let invalid = |message: String| DeepgramError {
            status: StatusCode::BAD_REQUEST,
            message,
        };
        let lang = self
            .language
            .as_deref()
            .map(str::parse)
            .transpose()
            .map_err(invalid)?
            .unwrap_or(state.lang);
        let emotion = self
            .emotion
            .as_deref()
            .map(str::parse)
            .transpose()
            .map_err(invalid)?
            .unwrap_or(state.emotion);
        Ok((lang, emotion))
    }
}

/// One alternative of a Deepgram channel.
#[derive(Debug, Serialize, ToSchema)]
pub struct ListenAlternative {
    pub transcript: String,
    pub confidence: f64,
    pub words: Vec<Value>,
}

/// Prerecorded response: `results.channels[0].alternatives[0].transcript` plus utterances.
#[derive(Debug, Serialize, ToSchema)]
pub struct ListenResponse {
    #[schema(value_type = Object)]
    pub metadata: Value,
    #[schema(value_type = Object)]
    pub results: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<Emotion>,
}

/// Deepgram live streaming over the shared pipeline: raw audio frames in; `Results` (interim from
/// partials, final per segment), `SpeechStarted`, `UtteranceEnd` and a closing `Metadata` out.
#[derive(Debug)]
pub struct Deepgram {
    request: String,
    emotion: EmotionMode,
    interim: bool,
    seconds: f64,
}

impl Deepgram {
    // ##### PRIVATE #####

    fn render_results(&self, start: f64, end: f64, transcript: &str, done: Option<&FinalEvent>) -> Message {
        let mut body = json!({
            "type": "Results",
            "channel_index": [0, 1],
            "duration": (end - start).max(0.0),
            "start": start,
            "is_final": done.is_some(),
            "speech_final": done.is_some(),
            "from_finalize": false,
            "channel": {"alternatives": [{"transcript": transcript, "confidence": 1.0, "words": []}]},
            "metadata": {"request_id": self.request},
        });
        if let (Some(done), Some(object)) = (done.filter(|_| self.emotion.field()), body.as_object_mut()) {
            object.insert("emotion".to_owned(), json!(done.emotion));
        }
        Message::Text(body.to_string().into())
    }

    // ##########################################################

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(emotion: EmotionMode, interim: bool) -> Self {
        Self {
            request: uuid::Uuid::new_v4().to_string(),
            emotion,
            interim,
            seconds: 0.0,
        }
    }

    /// The prerecorded JSON for a finished transcript.
    #[must_use]
    pub fn render(transcript: &Transcript, model: &str, emotion: EmotionMode) -> ListenResponse {
        let text = transcript.text(emotion.tags());
        let utterances: Vec<Value> = transcript
            .segments
            .iter()
            .filter(|segment| !segment.text.is_empty())
            .map(|segment| {
                let single = Transcript {
                    segments: vec![segment.clone()],
                    ..transcript.clone()
                };
                let mut utterance = json!({
                    "start": Transcript::seconds(segment.span.start),
                    "end": Transcript::seconds(segment.span.end),
                    "confidence": 1.0,
                    "channel": 0,
                    "transcript": single.text(emotion.tags()),
                    "words": [],
                    "id": uuid::Uuid::new_v4().to_string(),
                });
                if let Some(object) = utterance.as_object_mut().filter(|_| emotion.field()) {
                    object.insert("emotion".to_owned(), json!(segment.emotion));
                }
                utterance
            })
            .collect();
        ListenResponse {
            metadata: json!({
                "request_id": uuid::Uuid::new_v4().to_string(),
                "duration": transcript.duration(),
                "channels": 1,
                "models": [model],
            }),
            results: json!({
                "channels": [{"alternatives": [{"transcript": text, "confidence": 1.0, "words": []}], "detected_language": transcript.lang.locale().split('-').next()}],
                "utterances": utterances,
            }),
            emotion: emotion.field().then(|| transcript.emotion()),
        }
    }
}

impl LiveProtocol for Deepgram {
    fn greet(&mut self) -> Vec<Message> {
        Vec::new()
    }

    fn parse(&mut self, message: Message) -> Vec<Inbound> {
        match message {
            Message::Binary(bytes) => vec![Inbound::Audio(bytes.to_vec())],
            Message::Text(text) => {
                let kind = serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|body| body.get("type").and_then(Value::as_str).map(str::to_owned));
                match kind.as_deref() {
                    Some("CloseStream") => vec![Inbound::End],
                    _ => Vec::new(),
                }
            }
            Message::Close(_) => vec![Inbound::Close],
            Message::Ping(_) | Message::Pong(_) => Vec::new(),
        }
    }

    fn render(&mut self, event: &Event) -> Vec<Message> {
        match event {
            Event::Speech(speech) => {
                let at = Transcript::seconds(speech.at);
                self.seconds = self.seconds.max(at);
                match speech.state {
                    SpeechState::Started => {
                        let body = json!({"type": "SpeechStarted", "channel": [0], "timestamp": at});
                        vec![Message::Text(body.to_string().into())]
                    }
                    SpeechState::Stopped => Vec::new(),
                }
            }
            Event::Partial(partial) if self.interim => {
                vec![self.render_results(self.seconds, self.seconds, &partial.text, None)]
            }
            Event::Final(done) => {
                let text = Transcript {
                    lang: done.lang,
                    samples: 0,
                    segments: vec![done.clone()],
                }
                .text(self.emotion.tags());
                let (start, end) = (Transcript::seconds(done.span.start), Transcript::seconds(done.span.end));
                self.seconds = self.seconds.max(end);
                let close = json!({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": end});
                vec![
                    self.render_results(start, end, &text, Some(done)),
                    Message::Text(close.to_string().into()),
                ]
            }
            Event::Partial(_) | Event::Wake(_) | Event::Closed => Vec::new(),
        }
    }

    fn farewell(&mut self, outcome: &LiveOutcome) -> Vec<Message> {
        let body = json!({"type": "Metadata", "request_id": self.request, "duration": outcome.seconds, "channels": 1});
        vec![Message::Text(body.to_string().into())]
    }
}

/// Deepgram-compatible prerecorded transcription: the raw audio file is the request body.
///
/// # Errors
/// 400 for an unsupported language or emotion mode or undecodable audio; 500 when the pipeline fails.
#[utoipa::path(
    post,
    path = "/v1/listen",
    tag = "deepgram",
    params(ListenQuery),
    request_body(content = Vec<u8>, content_type = "audio/*", description = "wav, mp3, m4a, flac, ogg, opus or webm"),
    responses((status = 200, body = ListenResponse), (status = 400, description = "Invalid request (Deepgram error shape)"))
)]
pub async fn prerecorded(
    State(state): State<AppState>,
    Query(query): Query<ListenQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<ListenResponse>, DeepgramError> {
    let (lang, emotion) = query.parse(&state)?;
    let extension = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|mime| mime.split('/').nth(1))
        .map(|subtype| {
            subtype
                .trim_start_matches("x-")
                .split(';')
                .next()
                .unwrap_or_default()
                .to_owned()
        });
    let transcript = Batch::transcribe(&state, query.model.as_deref(), body.to_vec(), extension, lang).await?;
    let model = query
        .model
        .clone()
        .or_else(|| state.models.last().cloned())
        .unwrap_or_default();
    Ok(Json(Deepgram::render(&transcript, &model, emotion)))
}

/// Deepgram-compatible live streaming WebSocket.
///
/// # Errors
/// 400 for an unsupported language, emotion mode or encoding.
#[utoipa::path(
    get,
    path = "/v1/listen",
    tag = "deepgram",
    params(ListenQuery),
    responses(
        (status = 101, description = "WebSocket upgrade. Client: binary audio, `{\"type\":\"CloseStream\"}` to finish, `KeepAlive`/`Finalize` accepted. Server: `Results` (`is_final` false from partials, true per segment), `SpeechStarted`, `UtteranceEnd`, `Metadata` before closing."),
        (status = 400, description = "Invalid request (Deepgram error shape)"),
    )
)]
pub async fn streaming(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<ListenQuery>,
) -> Result<Response, DeepgramError> {
    let (lang, emotion) = query.parse(&state)?;
    let encoding = match query.encoding.as_deref().unwrap_or("linear16") {
        "linear16" => AudioEncoding::S16le,
        "pcm_f32le" => AudioEncoding::F32le,
        other => {
            return Err(DeepgramError {
                status: StatusCode::BAD_REQUEST,
                message: format!("unsupported encoding {other:?}; use linear16 or pcm_f32le"),
            });
        }
    };
    let rate = query.sample_rate.unwrap_or(16_000);
    let interim = query.interim_results.unwrap_or(true);
    Ok(ws
        .on_upgrade(move |socket| {
            LiveStream::new(lang, rate, encoding).serve(state, socket, Deepgram::new(emotion, interim))
        })
        .into_response())
}
