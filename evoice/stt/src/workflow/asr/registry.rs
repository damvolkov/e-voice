use std::path::PathBuf;
use std::sync::Arc;

use e_voice_core::models::ModelStore;
use e_voice_core::schema::error::BackendError;

use crate::config::asr::{AsrBackend as AsrChoice, AsrConfig, AsrEngine};
use crate::schema::mode::Mode;
use crate::workflow::asr::base::{BatchAsr, StreamingAsr};
use crate::workflow::asr::canary::CanaryAsr;
use crate::workflow::asr::cohere::CohereAsr;
use crate::workflow::asr::kroko::KrokoAsr;
use crate::workflow::asr::nemotron::NemotronAsr;
use crate::workflow::asr::parakeet::ParakeetAsr;
use crate::workflow::asr::whisper::WhisperAsr;

/// A built ASR backend; the variant fixes how the session drives it.
#[derive(Debug, Clone)]
pub enum AsrBackend {
    Streaming(Arc<dyn StreamingAsr>),
    Batch(Arc<dyn BatchAsr>),
}

impl AsrBackend {
    #[must_use]
    pub const fn mode(&self) -> Mode {
        match self {
            Self::Streaming(_) => Mode::Streaming,
            Self::Batch(_) => Mode::Batch,
        }
    }
}

/// Configuration → ASR backends.
#[derive(Debug)]
pub struct AsrRegistry;

impl AsrRegistry {
    // ##### PRIVATE #####

    fn common_dir(store: &ModelStore, model: &str) -> Result<PathBuf, BackendError> {
        store.dir(model).map_err(|_| BackendError::Unknown(model.to_owned()))
    }

    fn common_engine(engine: AsrEngine, store: &ModelStore, threads: u16) -> Result<AsrBackend, BackendError> {
        let dir = Self::common_dir(store, engine.model())?;
        let batch: Arc<dyn BatchAsr> = match engine {
            AsrEngine::Parakeet => Arc::new(ParakeetAsr::new(&dir, threads)?),
            AsrEngine::Canary => Arc::new(CanaryAsr::new(&dir, threads)?),
            AsrEngine::Cohere => Arc::new(CohereAsr::new(&dir, threads)?),
            AsrEngine::Whisper => Arc::new(WhisperAsr::new(&dir, threads)?),
        };
        Ok(AsrBackend::Batch(batch))
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// The live backend.
    ///
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &AsrConfig, store: &ModelStore) -> Result<AsrBackend, BackendError> {
        let streaming: Arc<dyn StreamingAsr> = match (config.backend, config.backend.engine()) {
            (_, Some(engine)) => return Self::common_engine(engine, store, config.threads),
            (AsrChoice::Kroko, None) => {
                let [es, en] = AsrConfig::KROKO.map(|model| Self::common_dir(store, model));
                Arc::new(KrokoAsr::new(&es?, &en?, config)?)
            }
            (_, None) => Arc::new(NemotronAsr::new(&Self::common_dir(store, config.nemotron())?, config)?),
        };
        Ok(AsrBackend::Streaming(streaming))
    }

    /// The file backend, when it differs from the live one; `None` means files reuse the live backend.
    ///
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn offline(config: &AsrConfig, store: &ModelStore) -> Result<Option<AsrBackend>, BackendError> {
        config
            .offline()
            .map(|engine| Self::common_engine(engine, store, config.offline.threads))
            .transpose()
    }

    /// Request-selectable engines beyond the live and default file backends.
    ///
    /// # Errors
    /// A model is not in the manifest, or a backend cannot be built from its files.
    pub fn extra(config: &AsrConfig, store: &ModelStore) -> Result<Vec<(AsrEngine, AsrBackend)>, BackendError> {
        config
            .extra()
            .into_iter()
            .map(|engine| Ok((engine, Self::common_engine(engine, store, config.offline.threads)?)))
            .collect()
    }
}
