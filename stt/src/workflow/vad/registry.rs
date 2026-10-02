use std::sync::Arc;

use crate::config::vad::{VadBackend, VadConfig};
use crate::core::models::ModelStore;
use crate::schema::error::BackendError;
use crate::workflow::vad::base::Vad;
use crate::workflow::vad::silero::SileroVad;
use crate::workflow::vad::ten::TenVad;

/// Configuration → VAD backend.
#[derive(Debug)]
pub struct VadRegistry;

impl VadRegistry {
    /// # Errors
    /// The model id is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &VadConfig, store: &ModelStore) -> Result<Arc<dyn Vad>, BackendError> {
        let dir = store
            .dir(config.model())
            .map_err(|_| BackendError::Unknown(config.model().to_owned()))?;
        let vad = match config.backend {
            VadBackend::Silero => SileroVad::build(&dir, config)?,
            VadBackend::Ten => TenVad::build(&dir, config)?,
        };
        Ok(Arc::new(vad))
    }
}
