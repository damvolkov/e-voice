use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use e_voice_core::audio::{AudioError, AudioFile};
use e_voice_core::config::ops::ModelsVerify;
use e_voice_core::logger::{Logger, LoggerError};
use e_voice_core::models::{ModelError, ModelStore};
use e_voice_core::schema::lang::Lang;
use e_voice_core::settings::SettingsError;
use e_voice_tts::api::docs::ApiDoc;
use e_voice_tts::api::lifespan::{Lifespan, LifespanError};
use e_voice_tts::api::server::{Server, ServerError};
use e_voice_tts::core::encode::AudioFormat;
use e_voice_tts::core::settings::Settings;
use e_voice_tts::core::voices::{VoiceError, VoiceStore};
use e_voice_tts::ops::bench::{Bench, BenchError};
use e_voice_tts::ops::similarity::Similarity;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::schema::error::TrainError;
use e_voice_tts::schema::event::Event;
use e_voice_tts::workflow::runner::{Request, RunnerError};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use utoipa::OpenApi;

/// Manifest id of the speaker-verification model `bench --reference` scores with.
const SPEAKER: &str = "speaker-resnet34";

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error(transparent)]
    Logger(#[from] LoggerError),
    #[error(transparent)]
    Models(#[from] ModelError),
    #[error(transparent)]
    Server(#[from] ServerError),
    #[error(transparent)]
    Lifespan(#[from] LifespanError),
    #[error(transparent)]
    Voice(#[from] VoiceError),
    #[error(transparent)]
    Train(#[from] TrainError),
    #[error(transparent)]
    Backend(#[from] e_voice_core::schema::error::BackendError),
    #[error(transparent)]
    Runner(#[from] RunnerError),
    #[error(transparent)]
    Audio(#[from] AudioError),
    #[error(transparent)]
    Bench(#[from] BenchError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Debug, Subcommand)]
enum VoiceCommand {
    /// Learn a voice from audio files of one speaker (5-30 s of clean speech in total).
    Add {
        id: String,
        files: Vec<PathBuf>,
        /// What the clips say: enables in-context cloning (Qwen3, NeuTTS).
        #[arg(long)]
        text: Option<String>,
    },
    /// List learned voices.
    List,
    /// Forget a voice.
    Rm { id: String },
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Serve the HTTP and WebSocket gateway.
    Serve,
    /// Install models: the given ids, every manifest entry with `--all`, else those the settings use.
    Pull {
        ids: Vec<String>,
        #[arg(long, conflicts_with = "ids")]
        all: bool,
    },
    /// Check installed models offline (all when no id is given).
    Verify {
        ids: Vec<String>,
        #[arg(long)]
        full: bool,
    },
    /// Learned voices: the training tool.
    Voice {
        #[command(subcommand)]
        command: VoiceCommand,
    },
    /// Speak a text into a WAV file (streams through the same runner as the server).
    Say {
        text: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long)]
        lang: Option<Lang>,
    },
    /// Synthesize every text of a JSON-lines manifest; one result line per item, summary last.
    Bench {
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Items synthesized concurrently.
        #[arg(long, default_value_t = 1)]
        streams: usize,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        voice: Option<String>,
        /// Also write each item's audio and an STT manifest into `<out>.wavs/`: `e-voice bench` on it
        /// measures intelligibility (the round trip).
        #[arg(long)]
        wavs: bool,
        /// Score speaker similarity of every item against this clip (installs nothing: `pull speaker-resnet34`).
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    /// Print the `OpenAPI` document (what `/openapi.json` serves).
    Openapi,
}

/// CPU-first streaming text-to-speech service.
#[derive(Debug, Parser)]
#[command(version = env!("E_VOICE_VERSION"))]
struct Cli {
    /// Settings file; without it, `evoice.toml` is read if present.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: CliCommand,
}

impl Cli {
    // ##### PRIVATE #####

    async fn run_voice(settings: &Settings, command: VoiceCommand) -> Result<(), MainError> {
        let voices = VoiceStore::open(&settings.tts.ops.data)?;
        match command {
            VoiceCommand::List => voices.list()?.iter().for_each(|id| println!("{id}")),
            VoiceCommand::Rm { id } => voices.remove(&id)?,
            VoiceCommand::Add { id, files, text } => {
                VoiceStore::valid(&id)
                    .then_some(())
                    .ok_or_else(|| VoiceError::Invalid(id.clone()))?;
                let (runner, _) = Lifespan::runner(settings).await?;
                let (synth, rate) = (std::sync::Arc::clone(runner.synth()), runner.caps().rate);
                let voice = tokio::task::spawn_blocking(move || {
                    let clips = files
                        .iter()
                        .map(|path| {
                            let extension = path.extension().and_then(|extension| extension.to_str());
                            Ok(Audio::from(AudioFile::decode(std::fs::read(path)?, extension, rate)?))
                        })
                        .collect::<Result<Vec<_>, MainError>>()?;
                    let seconds: f64 = clips.iter().map(|clip| clip.duration(rate).as_secs_f64()).sum();
                    tracing::info!(clips = clips.len(), seconds, "voice.training");
                    Ok::<_, MainError>(synth.train(&clips, text.as_deref())?)
                })
                .await??;
                voices.put(&id, &voice)?;
                tracing::info!(voice = %id, bytes = voice.len(), "voice.learned");
            }
        }
        Ok(())
    }

    async fn run_say(
        settings: &Settings,
        text: String,
        out: &std::path::Path,
        voice: Option<String>,
        lang: Option<Lang>,
    ) -> Result<(), MainError> {
        let (runner, _) = Lifespan::runner(settings).await?;
        let voices = VoiceStore::open(&settings.tts.ops.data)?;
        let voice = voice
            .or_else(|| settings.tts.voice.clone())
            .map(|id| voices.get(&id))
            .transpose()?;
        let session = runner.open(lang.unwrap_or(settings.tts.lang), voice).await?;
        let (requests, inbox) = mpsc::channel(2);
        let (outbox, mut events) = mpsc::channel(64);
        requests.send(Request::Text(text)).await.ok();
        requests.send(Request::Close).await.ok();
        let task = tokio::spawn({
            let runner = runner.clone();
            async move { runner.run(session, inbox, outbox, CancellationToken::new()).await }
        });
        let rate = runner.caps().rate;
        let mut bytes = AudioFormat::Wav.header(rate);
        let start = std::time::Instant::now();
        let (mut first, mut samples) = (None, 0_usize);
        while let Some(event) = events.recv().await {
            if let Event::Audio { audio, .. } = event {
                first.get_or_insert_with(|| start.elapsed());
                samples = samples.saturating_add(audio.len());
                bytes.extend(AudioFormat::Wav.samples(&audio));
            }
        }
        task.await??;
        let data = u32::try_from(samples.saturating_mul(2)).unwrap_or(u32::MAX);
        bytes.splice(4..8, data.saturating_add(36).to_le_bytes());
        bytes.splice(40..44, data.to_le_bytes());
        std::fs::write(out, bytes)?;
        let seconds = Audio::from(vec![0.0; samples]).duration(rate).as_secs_f64();
        tracing::info!(
            out = %out.display(),
            seconds,
            first_ms = first.unwrap_or_default().as_millis(),
            wall_s = start.elapsed().as_secs_f64(),
            "say.done"
        );
        Ok(())
    }

    // ##### PUBLIC #####

    async fn run(self) -> Result<(), MainError> {
        let settings = Settings::load(self.config.as_deref())?;
        let _logger = Logger::init(&settings.server.log)?;
        match self.command {
            CliCommand::Serve => Server::serve(&settings).await?,
            CliCommand::Pull { ids, all } => {
                let ids = match (ids.is_empty(), all) {
                    (false, _) | (true, true) => ids,
                    (true, false) => settings.models(),
                };
                ModelStore::open(&settings.tts.ops)?.pull(&ids).await?;
            }
            CliCommand::Verify { ids, full } => {
                let mode = if full {
                    ModelsVerify::Full
                } else {
                    settings.tts.ops.verify
                };
                ModelStore::open(&settings.tts.ops)?.verify(&ids, mode).await?;
                tracing::info!(?mode, "models.verified");
            }
            CliCommand::Voice { command } => Self::run_voice(&settings, command).await?,
            CliCommand::Say { text, out, voice, lang } => Self::run_say(&settings, text, &out, voice, lang).await?,
            CliCommand::Bench {
                manifest,
                out,
                streams,
                limit,
                voice,
                wavs,
                reference,
            } => {
                let (runner, _) = Lifespan::runner(&settings).await?;
                let similarity = reference
                    .map(|path| {
                        let extension = path.extension().and_then(|extension| extension.to_str());
                        let clip = AudioFile::decode(std::fs::read(&path)?, extension, runner.caps().rate)?;
                        let dir = ModelStore::open(&settings.tts.ops)?.dir(SPEAKER)?;
                        Ok::<_, MainError>(Similarity::open(&dir, &clip, runner.caps().rate)?)
                    })
                    .transpose()?;
                let voices = VoiceStore::open(&settings.tts.ops.data)?;
                let voice = voice
                    .or_else(|| settings.tts.voice.clone())
                    .map(|id| voices.get(&id))
                    .transpose()?;
                let bench = Bench {
                    runner,
                    voice,
                    streams,
                    wavs,
                    similarity,
                    settings: serde_json::to_value(&settings.tts).unwrap_or_default(),
                };
                let summary = bench.run(&manifest, &out, limit).await?;
                tracing::info!(
                    items = summary.items,
                    audio_s = summary.audio_s,
                    throughput_x = summary.throughput_x,
                    ttfa_p50_ms = summary.ttfa_p50_ms,
                    rtf_p50 = summary.rtf_p50,
                    similarity = ?summary.similarity_mean,
                    out = %out.display(),
                    "bench.done"
                );
            }
            CliCommand::Openapi => println!("{}", ApiDoc::openapi().to_pretty_json()?),
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match Cli::parse().run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("e-voice-tts: {error}");
            ExitCode::FAILURE
        }
    }
}
