use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use clap::Args;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::mic::{Mic, MicError};
use crate::wav::Wav;

const CLEAR: &str = "\r\x1b[2K";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
type Sink = futures_util::stream::SplitSink<Socket, Message>;
type Source = futures_util::stream::SplitStream<Socket>;
type Recorder = hound::WavWriter<std::io::BufWriter<std::fs::File>>;
type Tap = (std::sync::Arc<std::sync::atomic::AtomicU32>, String);

#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error(transparent)]
    Mic(#[from] MicError),
    #[error("wav: {0}")]
    Wav(#[from] hound::Error),
    #[error("websocket: {0}")]
    Socket(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("server closed the stream: {0}")]
    Closed(String),
}

/// Live transcription from the microphone (or a WAV) through `/v1/stream`.
#[derive(Debug, Args)]
#[command(group = clap::ArgGroup::new("output").args(["flat", "structured"]))]
pub struct SttArgs {
    /// Plain text: finals on stdout, the live partial redrawn on stderr (default).
    #[arg(long)]
    flat: bool,
    /// Every event as one JSON line: wake, partial, final (with emotion and spans), closed.
    #[arg(long = "struct")]
    structured: bool,
    /// Gateway base URL.
    #[arg(long, default_value = "ws://127.0.0.1:5500")]
    url: String,
    /// Spoken language; the server default when omitted.
    #[arg(long)]
    lang: Option<String>,
    /// Input device whose name contains this text; the system default when omitted.
    #[arg(long)]
    device: Option<String>,
    /// Replay this WAV in real time instead of opening the microphone.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Also save exactly what is sent (mono s16le at the source rate) to this WAV file.
    #[arg(long)]
    record: Option<PathBuf>,
}

impl SttArgs {
    // ##### PRIVATE #####

    /// Prints one server event; returns whether a partial now occupies the status line.
    fn run_render(&self, event: &Value, partial: bool, start: Instant) -> bool {
        let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();
        let text = event.get("text").and_then(Value::as_str).unwrap_or_default();
        let shown = match (self.structured, kind) {
            (true, _) => {
                eprint!("{CLEAR}");
                println!("{event}");
                false
            }
            (false, "partial") => {
                eprint!("{CLEAR}{DIM}{text}{RESET}");
                true
            }
            (false, "final") => {
                eprint!("{CLEAR}");
                if !text.is_empty() {
                    println!("{text}");
                    let end = event.pointer("/span/end").and_then(Value::as_f64).unwrap_or_default() / 16_000.0;
                    let lag = start.elapsed().as_secs_f64() - end;
                    let emotion = event
                        .pointer("/emotion/label")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    eprintln!("{DIM}  ↳ {lag:.2} s after speech ended · {emotion}{RESET}");
                }
                false
            }
            (false, "wake") => {
                eprintln!("{CLEAR}{DIM}· wake{RESET}");
                false
            }
            (false, _) => partial,
        };
        std::io::stderr().flush().ok();
        shown
    }

    fn run_meter(peak: f32) -> String {
        let db = 20.0 * peak.max(1e-6).log10();
        let filled = (((db + 60.0) / 3.0).clamp(0.0, 20.0)) as usize;
        format!(
            "{CLEAR}{DIM}mic {}{} {db:>4.0} dB{RESET}",
            "▮".repeat(filled),
            "▯".repeat(20usize.saturating_sub(filled))
        )
    }

    fn run_recorder(&self, rate: u32) -> Result<Option<Recorder>, hound::Error> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        self.record
            .as_ref()
            .map(|path| hound::WavWriter::create(path, spec))
            .transpose()
    }

    /// Forwards captured chunks until the source ends or Ctrl+C, then asks the server to drain.
    /// Returns the samples sent and their peak.
    fn run_send(
        mut chunks: mpsc::Receiver<Vec<u8>>,
        mut sink: Sink,
        mut recorder: Option<Recorder>,
    ) -> JoinHandle<Result<(usize, i32), tokio_tungstenite::tungstenite::Error>> {
        tokio::spawn(async move {
            let (mut sent, mut peak) = (0usize, 0i32);
            loop {
                let bytes = tokio::select! {
                    chunk = chunks.recv() => match chunk {
                        Some(bytes) => bytes,
                        None => break,
                    },
                    _ = tokio::signal::ctrl_c() => break,
                };
                let samples: Vec<i16> = bytes
                    .chunks_exact(2)
                    .filter_map(|pair| pair.try_into().ok())
                    .map(i16::from_le_bytes)
                    .collect();
                sent = sent.saturating_add(samples.len());
                peak = samples
                    .iter()
                    .fold(peak, |peak, sample| peak.max(i32::from(*sample).abs()));
                if let Some(writer) = recorder.as_mut() {
                    for sample in &samples {
                        writer.write_sample(*sample).ok();
                    }
                }
                sink.send(Message::Binary(bytes.into())).await?;
            }
            sink.send(Message::Text(r#"{"type":"end"}"#.into())).await?;
            if let Some(writer) = recorder {
                writer.finalize().ok();
            }
            Ok((sent, peak))
        })
    }

    /// Renders server events until the socket closes, drawing the input meter between them.
    async fn run_listen(&self, mut source: Source, tap: Option<Tap>, start: Instant) -> Result<(), SttError> {
        let metered = tap.is_some() && std::io::stderr().is_terminal();
        let mut tick = tokio::time::interval(Duration::from_millis(200));
        let (mut partial, mut quiet, mut warned) = (false, 0u32, false);
        loop {
            let message = tokio::select! {
                message = source.next() => message,
                _ = tick.tick(), if tap.is_some() => {
                    let Some((level, name)) = &tap else { continue };
                    let peak = f32::from_bits(level.load(Ordering::Relaxed));
                    quiet = if peak < 1e-4 { quiet.saturating_add(1) } else { 0 };
                    if quiet >= 15 && !warned {
                        warned = true;
                        eprintln!("{CLEAR}ecli: {name:?} has been digitally silent for 3 s; list inputs with `ecli devices` and pick one with --device");
                    }
                    if metered && !partial {
                        eprint!("{}", Self::run_meter(peak));
                        std::io::stderr().flush().ok();
                    }
                    continue;
                }
            };
            let Some(message) = message else { return Ok(()) };
            match message? {
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap_or(Value::String(text.to_string()));
                    partial = self.run_render(&event, partial, start);
                }
                Message::Close(Some(frame)) if u16::from(frame.code) != 1000 => {
                    return Err(SttError::Closed(format!("{} {}", u16::from(frame.code), frame.reason)));
                }
                Message::Close(_) => return Ok(()),
                Message::Binary(_) | Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            }
        }
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// The audio source or the connection fails, or the server closes with an error.
    pub async fn run(self) -> Result<(), SttError> {
        let (chunks_tx, chunks_rx) = mpsc::channel::<Vec<u8>>(64);
        let (mic, rate) = match &self.wav {
            Some(path) => (None, Wav::open(path, chunks_tx)?),
            None => {
                let mic = Mic::open(self.device.as_deref(), chunks_tx)?;
                eprintln!("{DIM}ecli · {} · {} Hz · Ctrl+C to finish{RESET}", mic.name, mic.rate);
                if mic.name.starts_with("Monitor of") {
                    eprintln!(
                        "ecli: {:?} is a playback monitor (what your speakers play), not a microphone; pick one with --device (see `ecli devices`)",
                        mic.name
                    );
                }
                let rate = mic.rate;
                (Some(mic), rate)
            }
        };
        let tap = mic
            .as_ref()
            .map(|mic| (std::sync::Arc::clone(&mic.level), mic.name.clone()));
        let lang = self
            .lang
            .as_ref()
            .map(|lang| format!("&lang={lang}"))
            .unwrap_or_default();
        let url = format!(
            "{}/v1/stream?rate={rate}&encoding=s16le{lang}",
            self.url.trim_end_matches('/')
        );
        let (socket, _) = tokio_tungstenite::connect_async(url.as_str()).await?;
        let (sink, source) = socket.split();
        let mut chunks_rx = chunks_rx;
        if mic.is_some() {
            while chunks_rx.try_recv().is_ok() {}
        }
        let start = Instant::now();
        let sender = Self::run_send(chunks_rx, sink, self.run_recorder(rate)?);
        let outcome = self.run_listen(source, tap, start).await;
        eprint!("{CLEAR}");
        if let Ok(Ok((sent, peak))) = sender.await {
            let level = 20.0 * (peak.max(1) as f32 / 32_768.0).log10();
            eprintln!(
                "{DIM}ecli · sent {:.1} s · peak {level:.1} dBFS{RESET}",
                sent as f32 / rate as f32
            );
        }
        outcome
    }
}
