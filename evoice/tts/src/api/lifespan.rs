use std::sync::Arc;

use e_voice_core::models::{ModelError, ModelStore};
use e_voice_core::runtime::{Runtime, RuntimeError};
use e_voice_core::schema::error::BackendError;
use e_voice_core::schema::lang::Lang;
use tokio_util::sync::CancellationToken;

use crate::api::state::AppState;
use crate::core::settings::Settings;
use crate::core::voices::{VoiceError, VoiceStore};
use crate::workflow::runner::Runner;
use crate::workflow::synth::registry::SynthRegistry;

#[derive(Debug, thiserror::Error)]
pub enum LifespanError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Models(#[from] ModelError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Voices(#[from] VoiceError),
    #[error("backend startup panicked")]
    Join,
}

/// Service startup: models verified offline, the backend built once, the voice store opened, and
/// the default voice primed once per language so the first request pays no encoding.
#[derive(Debug)]
pub struct Lifespan;

impl Lifespan {
    /// The runner over the configured backend (also used by `say` and `bench`).
    ///
    /// # Errors
    /// Runtime binding, a missing or stale model, or a backend refusing its files.
    pub async fn runner(settings: &Settings) -> Result<(Runner, Runtime), LifespanError> {
        let runtime = Runtime::probe()?;
        let store = ModelStore::open(&settings.tts.ops)?;
        store.verify(&settings.models(), settings.tts.ops.verify).await?;
        let config = settings.tts.synth.clone();
        let synth = tokio::task::spawn_blocking(move || SynthRegistry::build(&config, &store))
            .await
            .map_err(|_| LifespanError::Join)??;
        Ok((Runner::new(synth, settings.tts.text), runtime))
    }

    /// # Errors
    /// Startup of the runner or the voice store failed.
    pub async fn start(settings: &Settings) -> Result<AppState, LifespanError> {
        let (runner, runtime) = Self::runner(settings).await?;
        let voices = Arc::new(VoiceStore::open(&settings.tts.ops.data)?);
        let default = settings.tts.voice.as_deref().map(|id| voices.get(id)).transpose()?;
        if let Some(voice) = default {
            let warm = std::time::Instant::now();
            for lang in Lang::ALL {
                drop(runner.open(lang, Some(voice.clone())).await);
            }
            tracing::info!(voice = ?settings.tts.voice, ms = warm.elapsed().as_millis(), "lifespan.warm");
        }
        tracing::info!(models = ?settings.models(), voice = ?settings.tts.voice, "lifespan.ready");
        Ok(AppState {
            runner,
            voices,
            lang: settings.tts.lang,
            voice: settings.tts.voice.clone(),
            models: settings.models(),
            upload: settings.tts.api.upload.saturating_mul(1 << 20),
            runtime,
            shutdown: CancellationToken::new(),
        })
    }
}
