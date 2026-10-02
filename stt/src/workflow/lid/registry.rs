use std::sync::Arc;

use crate::config::lid::{LidBackend, LidConfig};
use crate::core::models::ModelStore;
use crate::schema::error::BackendError;
use crate::workflow::lid::base::Lid;
use crate::workflow::lid::whisper::WhisperLid;

/// Configuration → language identifier; `None` when the node is off.
#[derive(Debug)]
pub struct LidRegistry;

impl LidRegistry {
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &LidConfig, store: &ModelStore) -> Result<Option<Arc<dyn Lid>>, BackendError> {
        let Some(model) = config.model() else {
            return Ok(None);
        };
        let dir = store.dir(model).map_err(|_| BackendError::Unknown(model.to_owned()))?;
        let lid: Arc<dyn Lid> = match config.backend {
            LidBackend::Off | LidBackend::Whisper => Arc::new(WhisperLid::new(&dir, config.threads)?),
        };
        Ok(Some(lid))
    }
}
