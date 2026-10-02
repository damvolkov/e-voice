use std::sync::Arc;

use crate::config::ser::{SerBackend, SerConfig};
use crate::core::models::ModelStore;
use crate::schema::error::BackendError;
use crate::workflow::ser::base::Ser;
use crate::workflow::ser::emotion2vec::Emotion2vecSer;

/// Configuration → SER backend, or none when disabled.
#[derive(Debug)]
pub struct SerRegistry;

impl SerRegistry {
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &SerConfig, store: &ModelStore) -> Result<Option<Arc<dyn Ser>>, BackendError> {
        let Some(model) = config.model() else {
            return Ok(None);
        };
        let dir = store.dir(model).map_err(|_| BackendError::Unknown(model.to_owned()))?;
        match config.backend {
            SerBackend::Off => Ok(None),
            SerBackend::Emotion2vec => Ok(Some(Arc::new(Emotion2vecSer::new(&dir, config.threads, model)?))),
        }
    }
}
