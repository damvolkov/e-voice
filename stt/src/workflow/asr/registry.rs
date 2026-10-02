use std::sync::Arc;

use crate::config::asr::{AsrBackend as AsrChoice, AsrConfig, OfflineBackend};
use crate::core::models::ModelStore;
use crate::schema::error::BackendError;
use crate::schema::mode::Mode;
use crate::workflow::asr::base::{BatchAsr, StreamingAsr};
use crate::workflow::asr::nemotron::NemotronAsr;
use crate::workflow::asr::parakeet::ParakeetAsr;

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

    fn common_parakeet(store: &ModelStore, threads: u16) -> Result<AsrBackend, BackendError> {
        let dir = store
            .dir(AsrConfig::PARAKEET)
            .map_err(|_| BackendError::Unknown(AsrConfig::PARAKEET.to_owned()))?;
        Ok(AsrBackend::Batch(Arc::new(ParakeetAsr::new(&dir, threads)?)))
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// The live backend.
    ///
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &AsrConfig, store: &ModelStore) -> Result<AsrBackend, BackendError> {
        match config.backend {
            AsrChoice::Nemotron => {
                let dir = store
                    .dir(config.model())
                    .map_err(|_| BackendError::Unknown(config.model().to_owned()))?;
                Ok(AsrBackend::Streaming(Arc::new(NemotronAsr::new(&dir, config)?)))
            }
            AsrChoice::Parakeet => Self::common_parakeet(store, config.threads),
        }
    }

    /// The file backend, when it differs from the live one; `None` means files reuse the live backend.
    ///
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn offline(config: &AsrConfig, store: &ModelStore) -> Result<Option<AsrBackend>, BackendError> {
        match (config.offline.backend, config.backend) {
            (OfflineBackend::Live, _) | (OfflineBackend::Parakeet, AsrChoice::Parakeet) => Ok(None),
            (OfflineBackend::Parakeet, AsrChoice::Nemotron) => {
                Self::common_parakeet(store, config.offline.threads).map(Some)
            }
        }
    }
}
