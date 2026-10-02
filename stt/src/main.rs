use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use e_voice_stt::api::server::{Server, ServerError};
use e_voice_stt::config::ops::ModelsVerify;
use e_voice_stt::core::logger::{Logger, LoggerError};
use e_voice_stt::core::models::{ModelError, ModelStore};
use e_voice_stt::core::runtime::{Runtime, RuntimeError};
use e_voice_stt::core::settings::{Settings, SettingsError};
use e_voice_stt::ops::bench::manifest::{Manifest, ManifestError};
use e_voice_stt::ops::bench::suite::{Suite, SuiteError, SuiteMode};
use e_voice_stt::ops::wake::{Wake, WakeError};
use e_voice_stt::schema::error::BackendError;
use e_voice_stt::workflow::nodes::{Nodes, NodesError};
use e_voice_stt::workflow::runner::Runner;

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error(transparent)]
    Logger(#[from] LoggerError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Models(#[from] ModelError),
    #[error(transparent)]
    Server(#[from] ServerError),
    #[error(transparent)]
    Nodes(#[from] NodesError),
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Suite(#[from] SuiteError),
    #[error(transparent)]
    Wake(#[from] WakeError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Serve the HTTP and WebSocket gateway.
    Serve,
    /// Install models: the given ids, every manifest entry with `--all`, else those the settings use.
    Pull {
        ids: Vec<String>,
        /// Every model in the manifest, including test fixtures.
        #[arg(long, conflicts_with = "ids")]
        all: bool,
    },
    /// Run a JSON-lines manifest through the configured pipeline; one result line per item.
    Bench {
        manifest: PathBuf,
        /// Results file (JSON lines, summary last).
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "file")]
        mode: SuiteMode,
        /// Items processed concurrently.
        #[arg(long, default_value_t = 1)]
        streams: usize,
        /// Only the first N items.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Calibrate a wake phrase for the kws backend: sweep threshold × boost over synthetic and real speech.
    Wake {
        phrase: String,
        /// Real-speech clips per negative dataset (FLEURS, when fetched with `make datasets`).
        #[arg(long, default_value_t = 60)]
        limit: usize,
    },
    /// Check installed models offline (all when no id is given).
    Verify {
        ids: Vec<String>,
        /// Re-hash every file instead of trusting the install stamp.
        #[arg(long)]
        full: bool,
    },
}

/// CPU-first, OpenAI-compatible speech service.
#[derive(Debug, Parser)]
#[command(version)]
struct Cli {
    /// Settings file; without it, `evoice.toml` is read if present.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: CliCommand,
}

impl Cli {
    async fn run(self) -> Result<(), MainError> {
        let settings = Settings::load(self.config.as_deref())?;
        let _logger = Logger::init(&settings.server.log)?;
        let runtime = Runtime::probe()?;
        tracing::info!(
            sherpa = runtime.sherpa,
            onnxruntime = runtime.onnxruntime,
            git = runtime.git,
            "runtime.loaded"
        );
        let store = ModelStore::open(&settings.stt.ops)?;
        match self.command {
            CliCommand::Serve => Server::serve(&settings).await?,
            CliCommand::Wake { phrase, limit } => {
                let wake = Wake::prepare(&store, &phrase, &Wake::manifests(&settings.stt.ops.data), limit)?;
                tracing::info!(
                    phrase,
                    positives = wake.positives(),
                    negative_s = wake.negative_s,
                    "wake.sweep"
                );
                let mut trials = wake.sweep()?;
                trials.sort_by(|a, b| {
                    b.recall
                        .total_cmp(&a.recall)
                        .then(a.false_accepts.cmp(&b.false_accepts))
                });
                println!("threshold  boost  recall  false accepts  per hour");
                for trial in trials.iter().take(12) {
                    println!(
                        "{:>9.2}  {:>5.1}  {:>6.2}  {:>13}  {:>8.2}",
                        trial.threshold, trial.boost, trial.recall, trial.false_accepts, trial.per_hour
                    );
                }
                if let Some(best) = Wake::pick(&trials) {
                    let voices: Vec<String> = best
                        .voices
                        .iter()
                        .map(|(voice, rate)| format!("{voice} {:.0}%", rate * 100.0))
                        .collect();
                    println!("\nrecall per voice: {}", voices.join(" · "));
                    println!(
                        "\n[stt.pipeline.ww]\nbackend = \"kws\"\nkeyword = \"{phrase}\"\nthreshold = {}\nboost = {}\n# recall {:.0}% on {} synthetic utterances · {} false accepts in {:.0} min of other speech",
                        best.threshold,
                        best.boost,
                        best.recall * 100.0,
                        wake.positives(),
                        best.false_accepts,
                        wake.negative_s / 60.0
                    );
                }
            }
            CliCommand::Bench {
                manifest,
                out,
                mode,
                streams,
                limit,
            } => {
                let mut manifest = Manifest::load(&manifest)?;
                manifest.items.truncate(limit.unwrap_or(usize::MAX));
                let (nodes, _) = Nodes::start(&settings).await?;
                let suite = Suite {
                    runner: Runner::new(Arc::new(nodes), settings.stt.pipeline.clone()),
                    mode,
                    streams,
                    settings: serde_json::to_value(&settings).unwrap_or_default(),
                };
                let summary = suite.run(manifest, &out).await?;
                tracing::info!(
                    items = summary.items,
                    audio_s = summary.audio_s,
                    wall_s = summary.wall_s,
                    throughput_x = summary.throughput_x,
                    cpu_per_audio_s = summary.cpu_per_audio_s,
                    peak_rss_mb = summary.peak_rss_mb,
                    out = %out.display(),
                    "bench.done"
                );
            }
            CliCommand::Pull { ids, all } => {
                let ids = match (ids.is_empty(), all) {
                    (false, _) | (true, true) => ids,
                    (true, false) => settings.models(),
                };
                store.pull(&ids).await?;
            }
            CliCommand::Verify { ids, full } => {
                let mode = if full {
                    ModelsVerify::Full
                } else {
                    settings.stt.ops.verify
                };
                store.verify(&ids, mode).await?;
                tracing::info!(?mode, "models.verified");
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match Cli::parse().run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("e-voice: {error}");
            ExitCode::FAILURE
        }
    }
}
