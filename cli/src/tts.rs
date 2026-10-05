use std::fmt::Write;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Args;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use crate::speaker::{Speaker, SpeakerError};

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error(transparent)]
    Speaker(#[from] SpeakerError),
    #[error("wav: {0}")]
    Wav(#[from] hound::Error),
    #[error("websocket: {0}")]
    Socket(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("server: {0}")]
    Server(String),
}

/// Streaming synthesis through `/v1/stream`: audio plays while it is generated.
///
/// With TEXT, speaks it and exits. Without, every stdin line is sent as it is typed (and flushed);
/// an empty line barges in (cancels what is playing); end of input closes.
#[derive(Debug, Args)]
pub struct TtsArgs {
    text: Option<String>,
    /// TTS gateway base URL.
    #[arg(long, default_value = "ws://127.0.0.1:5600")]
    url: String,
    #[arg(long)]
    lang: Option<String>,
    /// Learned voice id; the server default when omitted.
    #[arg(long)]
    voice: Option<String>,
    /// Also save the received audio to this WAV file.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Do not play (with --out: just record).
    #[arg(long)]
    mute: bool,
}

impl TtsArgs {
    // ##### PRIVATE #####

    fn run_lines(lines: mpsc::Sender<Option<String>>) {
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                if lines.blocking_send(Some(line)).is_err() {
                    return;
                }
            }
            lines.blocking_send(None).ok();
        });
    }

    // ##### PUBLIC #####

    /// # Errors
    /// The gateway refused the stream, or audio output / the WAV file failed.
    pub async fn run(self) -> Result<(), TtsError> {
        let mut url = format!("{}/v1/stream?format=f32", self.url.trim_end_matches('/'));
        for (key, value) in [("lang", &self.lang), ("voice", &self.voice)] {
            if let Some(value) = value {
                write!(url, "&{key}={value}").ok();
            }
        }
        let (socket, _) = tokio_tungstenite::connect_async(url.as_str()).await?;
        let (mut sink, mut source) = socket.split();
        let speaker = if self.mute { None } else { Some(Speaker::open()?) };
        let (lines, mut typed) = mpsc::channel::<Option<String>>(16);
        match self.text.clone() {
            Some(text) => {
                lines.send(Some(text)).await.ok();
                lines.send(None).await.ok();
            }
            None => Self::run_lines(lines.clone()),
        }
        let (mut rate, mut recorder, start) = (24_000_u32, None, Instant::now());
        let mut first: Option<Duration> = None;
        loop {
            tokio::select! {
                line = typed.recv() => {
                    let message = match line.flatten() {
                        Some(text) if text.trim().is_empty() => {
                            speaker.iter().for_each(Speaker::clear);
                            json!({ "type": "cancel" })
                        }
                        Some(text) => {
                            sink.send(Message::text(json!({ "type": "text", "text": format!("{text} ") }).to_string())).await?;
                            json!({ "type": "flush" })
                        }
                        None => json!({ "type": "close" }),
                    };
                    sink.send(Message::text(message.to_string())).await?;
                }
                message = source.next() => match message {
                    Some(Ok(Message::Binary(bytes))) => {
                        first.get_or_insert_with(|| start.elapsed());
                        let samples: Vec<f32> = bytes
                            .chunks_exact(4)
                            .filter_map(|chunk| chunk.try_into().ok().map(f32::from_le_bytes))
                            .collect();
                        if let Some(speaker) = &speaker {
                            speaker.push(&samples, rate);
                        }
                        if let Some(writer) = recorder.as_mut() {
                            samples.iter().try_for_each(|sample| hound::WavWriter::write_sample(writer, *sample))?;
                        }
                    }
                    Some(Ok(Message::Text(text))) => {
                        let event: Value = serde_json::from_str(&text).unwrap_or_default();
                        let field = |key: &str| event.get(key).cloned().unwrap_or_default();
                        match field("type").as_str().unwrap_or_default() {
                            "ready" => {
                                rate = field("rate").as_u64().and_then(|rate| u32::try_from(rate).ok()).unwrap_or(rate);
                                recorder = self
                                    .out
                                    .as_ref()
                                    .map(|path| {
                                        let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
                                        hound::WavWriter::create(path, spec)
                                    })
                                    .transpose()?;
                                eprintln!("{DIM}ready · {} Hz · voice {}{RESET}", rate, field("voice"));
                            }
                            "start" => eprintln!("{DIM}▶ {}{RESET}", field("text").as_str().unwrap_or_default()),
                            "end" if !field("error").is_null() => eprintln!("{DIM}■ {}{RESET}", field("error")),
                            "error" => eprintln!("ecli: {}", field("message")),
                            "closed" => break,
                            _ => {}
                        }
                    }
                    Some(Ok(Message::Close(frame))) => {
                        return Err(TtsError::Server(frame.map(|frame| frame.reason.to_string()).unwrap_or_default()));
                    }
                    Some(Err(error)) => return Err(error.into()),
                    None => break,
                    Some(Ok(_)) => {}
                },
            }
        }
        eprintln!("{DIM}first audio {} ms{RESET}", first.unwrap_or_default().as_millis());
        if let Some(writer) = recorder {
            writer.finalize()?;
        }
        while speaker.as_ref().is_some_and(|speaker| speaker.pending() > 0.0) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(())
    }
}
