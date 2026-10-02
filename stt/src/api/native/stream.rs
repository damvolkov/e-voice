use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use utoipa::IntoParams;

use crate::api::live::{Inbound, LiveOutcome, LiveProtocol, LiveStream};
use crate::api::state::AppState;
use crate::config::server::EmotionMode;
use crate::core::audio::AudioEncoding;
use crate::schema::audio::RATE;
use crate::schema::emotion::EmotionLabel;
use crate::schema::event::Event;
use crate::schema::lang::Lang;
use crate::schema::transcript::Transcript;

/// What the native stream sends: every event as JSON, or only each final's text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamView {
    #[default]
    Struct,
    Flat,
}

/// `GET /v1/stream?lang=es&rate=48000&encoding=s16le&view=struct&emotion=field`
#[derive(Debug, Deserialize, IntoParams)]
pub struct StreamQuery {
    /// `es` or `en`; the server default when omitted.
    pub lang: Option<Lang>,
    /// Sample rate of the binary frames, 16000 by default.
    pub rate: Option<u32>,
    /// `s16le` (default) or `f32le`, mono.
    #[param(value_type = Option<String>)]
    pub encoding: Option<AudioEncoding>,
    /// `struct` (default): one JSON event per frame. `flat`: one text frame per final.
    #[param(value_type = Option<String>)]
    pub view: Option<StreamView>,
    /// `field`, `tag` or `off`; the server default when omitted.
    pub emotion: Option<String>,
}

/// Client → server text frames; binary frames carry PCM.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum StreamControl {
    End,
}

/// The native protocol: the pipeline's own events (`wake`, `speech`, `partial`, `final`, `closed`).
#[derive(Debug, Clone, Copy)]
pub struct Native {
    view: StreamView,
    emotion: EmotionMode,
}

impl Native {
    #[must_use]
    pub const fn new(view: StreamView, emotion: EmotionMode) -> Self {
        Self { view, emotion }
    }
}

impl LiveProtocol for Native {
    fn greet(&mut self) -> Vec<Message> {
        Vec::new()
    }

    fn parse(&mut self, message: Message) -> Vec<Inbound> {
        match message {
            Message::Binary(bytes) => vec![Inbound::Audio(bytes.to_vec())],
            Message::Text(text) => match serde_json::from_str::<StreamControl>(&text) {
                Ok(StreamControl::End) => vec![Inbound::End],
                Err(_) => Vec::new(),
            },
            Message::Close(_) => vec![Inbound::Close],
            Message::Ping(_) | Message::Pong(_) => Vec::new(),
        }
    }

    fn render(&mut self, event: &Event) -> Vec<Message> {
        match (self.view, event) {
            (StreamView::Flat, Event::Final(done)) => {
                let text = Transcript {
                    lang: done.lang,
                    samples: 0,
                    segments: vec![done.clone()],
                }
                .text(self.emotion.tags());
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![Message::Text(text.into())]
                }
            }
            (StreamView::Flat, _) => Vec::new(),
            (StreamView::Struct, event) => {
                let mut body = serde_json::to_value(event).unwrap_or(Value::Null);
                if let Event::Final(done) = event
                    && let Some(object) = body.as_object_mut()
                {
                    {
                        if self.emotion.tags() && done.emotion.label != EmotionLabel::Unknown {
                            object.insert(
                                "text".to_owned(),
                                json!(format!("[{}] {}", Transcript::tag(done.emotion.label), done.text)),
                            );
                        }
                        if !self.emotion.field() {
                            object.remove("emotion");
                        }
                    }
                }
                vec![Message::Text(body.to_string().into())]
            }
        }
    }

    fn farewell(&mut self, _outcome: &LiveOutcome) -> Vec<Message> {
        Vec::new()
    }
}

/// Native realtime transcription: binary PCM frames in; `{"type":"end"}` drains pending segments and
/// closes normally after `closed`; dropping the socket cancels them.
///
/// # Errors
/// 400 for an unsupported emotion mode.
#[utoipa::path(
    get,
    path = "/v1/stream",
    tag = "native",
    params(StreamQuery),
    responses(
        (status = 101, description = "WebSocket upgrade. `view=struct`: one JSON `Event` per frame (`wake`, `speech`, `partial`, `final`, `closed`). `view=flat`: one text frame per final. Close 1000 on a drained end, 1003 for unsupported audio parameters, 1011 on failure.", body = Event),
        (status = 400, description = "Unsupported emotion mode"),
    )
)]
pub async fn stream(ws: WebSocketUpgrade, Query(query): Query<StreamQuery>, State(state): State<AppState>) -> Response {
    let emotion = match query.emotion.as_deref().map(str::parse::<EmotionMode>).transpose() {
        Ok(emotion) => emotion.unwrap_or(state.emotion),
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };
    let lang = query.lang.unwrap_or(state.lang);
    let (rate, encoding) = (query.rate.unwrap_or(RATE), query.encoding.unwrap_or_default());
    let protocol = Native::new(query.view.unwrap_or_default(), emotion);
    ws.on_upgrade(move |socket| LiveStream::new(lang, rate, encoding).serve(state, socket, protocol))
}
