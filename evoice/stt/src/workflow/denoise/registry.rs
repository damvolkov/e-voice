use std::sync::Arc;

use e_voice_core::models::ModelStore;
use e_voice_core::schema::error::BackendError;

use crate::config::denoise::{DenoiseBackend, DenoiseConfig};
use crate::workflow::denoise::base::Denoise;
use crate::workflow::denoise::gtcrn::GtcrnDenoise;

/// Configuration → speech enhancer; `None` when the node is off.
#[derive(Debug)]
pub struct DenoiseRegistry;

impl DenoiseRegistry {
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &DenoiseConfig, store: &ModelStore) -> Result<Option<Arc<dyn Denoise>>, BackendError> {
        let Some(model) = config.model() else {
            return Ok(None);
        };
        let dir = store.dir(model).map_err(|_| BackendError::Unknown(model.to_owned()))?;
        let denoise: Arc<dyn Denoise> = match config.backend {
            DenoiseBackend::Off | DenoiseBackend::Gtcrn => Arc::new(GtcrnDenoise::new(&dir, config.threads)?),
        };
        Ok(Some(denoise))
    }
}
