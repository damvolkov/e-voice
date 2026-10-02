use std::collections::HashMap;

use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::Deserialize;
use serde_json::{Value, json};
use utoipa::IntoParams;

use crate::api::live::{Inbound, LiveOutcome, LiveProtocol, LiveStream, NORMAL};
use crate::api::openai::formats::OpenaiError;
use crate::api::state::AppState;
use crate::config::server::EmotionMode;
use crate::core::audio::AudioEncoding;
use crate::schema::emotion::EmotionLabel;
use crate::schema::event::{Event, SpeechState};
use crate::schema::lang::Lang;
use crate::schema::segment::SegmentId;
use crate::schema::transcript::Transcript;

const RATE: u32 = 24_000;

/// `GET /v1/realtime?intent=transcription&language=es&emotion=field`
#[derive(Debug, Deserialize, IntoParams)]
pub struct RealtimeQuery {
    /// `transcription` (any other intent is served as transcription too).
    pub intent: Option<String>,
    /// `es` or `en`; a later `session.update` may still change it before audio starts.
    pub language: Option<String>,
    /// Accepted for compatibility.
    pub model: Option<String>,
    /// `field`, `tag` or `off` for the `emotion` extension on `completed` events.
    pub emotion: Option<String>,
}

/// OpenAI Realtime transcription over the shared pipeline: base64 `pcm16` appended through
/// `input_audio_buffer.append`, server VAD always on, `delta` events from streaming partials and one
/// `completed` per segment. GA (`session.*`) and beta (`transcription_session.*`) names are both served.
#[derive(Debug)]
pub struct Realtime {
    session: String,
    emotion: EmotionMode,
    lang: Lang,
    rate: u32,
    events: u64,
    sent: HashMap<SegmentId, String>,
    previous: Option<String>,
}

impl Realtime {
    // ##### PRIVATE #####

    fn common_frame(&mut self, kind: &str, mut body: Value) -> Message {
        self.events = self.events.saturating_add(1);
        if let Some(object) = body.as_object_mut() {
            object.insert("type".to_owned(), json!(kind));
            object.insert("event_id".to_owned(), json!(format!("event_{}", self.events)));
        }
        Message::Text(body.to_string().into())
    }

    fn common_item(&self, segment: SegmentId) -> String {
        format!("item_{}_{}", self.session, segment.0)
    }

    fn common_session(&self) -> Value {
        json!({
            "id": self.session,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "e-voice", "language": self.lang.locale().split('-').next()},
            "turn_detection": {"type": "server_vad"},
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": self.rate}, "turn_detection": {"type": "server_vad"}}},
        })
    }

    fn parse_update(&mut self, kind: &str, body: &Value) -> Vec<Inbound> {
        let session = body.get("session").cloned().unwrap_or(Value::Null);
        let language = session
            .pointer("/audio/input/transcription/language")
            .or_else(|| session.pointer("/input_audio_transcription/language"))
            .and_then(Value::as_str);
        let rate = session
            .pointer("/audio/input/format/rate")
            .and_then(Value::as_u64)
            .and_then(|rate| u32::try_from(rate).ok());
        let mut steps = Vec::new();
        match language.map(str::parse::<Lang>) {
            Some(Ok(lang)) => self.lang = lang,
            Some(Err(message)) => {
                let error = json!({"error": {"type": "invalid_request_error", "code": "unsupported_language", "message": message}});
                steps.push(Inbound::Reply(self.common_frame("error", error)));
            }
            None => {}
        }
        self.rate = rate.unwrap_or(self.rate);
        steps.push(Inbound::Configure {
            lang: Some(self.lang),
            rate: Some(self.rate),
            encoding: None,
        });
        let updated = kind.replace(".update", ".updated");
        let reply = self.common_session();
        steps.push(Inbound::Reply(self.common_frame(&updated, json!({"session": reply}))));
        steps
    }

    // ##########################################################

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(lang: Lang, emotion: EmotionMode) -> Self {
        let session = uuid::Uuid::new_v4().simple().to_string();
        Self {
            session,
            emotion,
            lang,
            rate: RATE,
            events: 0,
            sent: HashMap::new(),
            previous: None,
        }
    }
}

impl LiveProtocol for Realtime {
    fn greet(&mut self) -> Vec<Message> {
        let session = self.common_session();
        vec![
            self.common_frame("session.created", json!({"session": session.clone()})),
            self.common_frame("transcription_session.created", json!({"session": session})),
        ]
    }

    fn parse(&mut self, message: Message) -> Vec<Inbound> {
        let text = match message {
            Message::Text(text) => text,
            Message::Binary(bytes) => return vec![Inbound::Audio(bytes.to_vec())],
            Message::Close(_) => return vec![Inbound::Close],
            Message::Ping(_) | Message::Pong(_) => return Vec::new(),
        };
        let Ok(body) = serde_json::from_str::<Value>(&text) else {
            let error = json!({"error": {"type": "invalid_request_error", "message": "frames must be JSON events"}});
            return vec![Inbound::Reply(self.common_frame("error", error))];
        };
        let kind = body.get("type").and_then(Value::as_str).unwrap_or_default().to_owned();
        match kind.as_str() {
            "session.update" | "transcription_session.update" => self.parse_update(&kind, &body),
            "input_audio_buffer.append" => match body
                .get("audio")
                .and_then(Value::as_str)
                .map(|audio| STANDARD.decode(audio))
            {
                Some(Ok(bytes)) => vec![Inbound::Audio(bytes)],
                Some(Err(_)) | None => {
                    let error =
                        json!({"error": {"type": "invalid_request_error", "message": "audio must be base64 pcm16"}});
                    vec![Inbound::Reply(self.common_frame("error", error))]
                }
            },
            "input_audio_buffer.commit" => vec![Inbound::Reply(
                self.common_frame("input_audio_buffer.committed", json!({})),
            )],
            "input_audio_buffer.clear" => vec![Inbound::Reply(
                self.common_frame("input_audio_buffer.cleared", json!({})),
            )],
            "session.close" | "transcription_session.close" => vec![Inbound::End],
            _ => Vec::new(),
        }
    }

    fn render(&mut self, event: &Event) -> Vec<Message> {
        match event {
            Event::Speech(speech) => {
                let item = self.common_item(speech.segment);
                let ms = speech.at / 16;
                match speech.state {
                    SpeechState::Started => {
                        vec![self.common_frame(
                            "input_audio_buffer.speech_started",
                            json!({"audio_start_ms": ms, "item_id": item}),
                        )]
                    }
                    SpeechState::Stopped => {
                        let previous = self.previous.replace(item.clone());
                        vec![
                            self.common_frame(
                                "input_audio_buffer.speech_stopped",
                                json!({"audio_end_ms": ms, "item_id": item}),
                            ),
                            self.common_frame(
                                "input_audio_buffer.committed",
                                json!({"item_id": item, "previous_item_id": previous}),
                            ),
                        ]
                    }
                }
            }
            Event::Partial(partial) => {
                let sent = self.sent.entry(partial.segment).or_default();
                let delta = partial
                    .text
                    .strip_prefix(sent.as_str())
                    .filter(|delta| !delta.is_empty())
                    .map(str::to_owned);
                match delta {
                    Some(delta) => {
                        sent.clone_from(&partial.text);
                        let item = self.common_item(partial.segment);
                        vec![self.common_frame(
                            "conversation.item.input_audio_transcription.delta",
                            json!({"item_id": item, "content_index": 0, "delta": delta}),
                        )]
                    }
                    None => Vec::new(),
                }
            }
            Event::Final(done) => {
                self.sent.remove(&done.segment);
                let item = self.common_item(done.segment);
                match &done.error {
                    Some(error) => vec![self.common_frame(
                        "conversation.item.input_audio_transcription.failed",
                        json!({"item_id": item, "content_index": 0, "error": {"type": "server_error", "message": error.to_string()}}),
                    )],
                    None => {
                        let label = format!("[{}] ", Transcript::tag(done.emotion.label));
                        let tagged = self.emotion.tags() && done.emotion.label != EmotionLabel::Unknown;
                        let transcript = if tagged { format!("{label}{}", done.text) } else { done.text.clone() };
                        let mut body = json!({"item_id": item, "content_index": 0, "transcript": transcript});
                        if let Some(object) = body.as_object_mut().filter(|_| self.emotion.field()) {
                            object.insert("emotion".to_owned(), json!(done.emotion));
                        }
                        vec![self.common_frame("conversation.item.input_audio_transcription.completed", body)]
                    }
                }
            }
            Event::Wake(_) | Event::Closed => Vec::new(),
        }
    }

    fn farewell(&mut self, outcome: &LiveOutcome) -> Vec<Message> {
        match outcome.code {
            NORMAL => Vec::new(),
            code => {
                let error =
                    json!({"error": {"type": "server_error", "message": format!("stream closed with code {code}")}});
                vec![self.common_frame("error", error)]
            }
        }
    }
}

/// OpenAI Realtime-compatible transcription WebSocket.
///
/// # Errors
/// 400 when `language` or `emotion` is not supported.
#[utoipa::path(
    get,
    path = "/v1/realtime",
    tag = "openai",
    params(RealtimeQuery),
    responses(
        (status = 101, description = "WebSocket upgrade. Client: `session.update`, `input_audio_buffer.append` (base64 pcm16, 24 kHz unless the session sets `audio.input.format.rate`), `input_audio_buffer.commit`/`clear`. Server: `session.created`, `input_audio_buffer.speech_started`/`speech_stopped`/`committed`, `conversation.item.input_audio_transcription.delta`/`completed`/`failed`, `error`."),
        (status = 400, description = "Unsupported language or emotion mode (OpenAI error shape)"),
    )
)]
pub async fn realtime(
    ws: WebSocketUpgrade,
    Query(query): Query<RealtimeQuery>,
    State(state): State<AppState>,
) -> Result<Response, OpenaiError> {
    let lang = match query.language.as_deref() {
        Some(code) => code
            .parse()
            .map_err(|error: String| OpenaiError::invalid(error, Some("language")))?,
        None => state.lang,
    };
    let emotion = match query.emotion.as_deref() {
        Some(mode) => mode
            .parse()
            .map_err(|error: String| OpenaiError::invalid(error, Some("emotion")))?,
        None => state.emotion,
    };
    tracing::info!(?lang, intent = ?query.intent, model = ?query.model, "openai.realtime");
    Ok(ws
        .on_upgrade(move |socket| {
            LiveStream::new(lang, RATE, AudioEncoding::S16le).serve(state, socket, Realtime::new(lang, emotion))
        })
        .into_response())
}
