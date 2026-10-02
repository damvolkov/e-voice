use std::sync::Arc;

use crate::config::ww::{WwBackend, WwConfig};
use crate::core::models::ModelStore;
use crate::schema::error::BackendError;
use crate::workflow::ww::base::Ww;
use crate::workflow::ww::kws::KwsWw;
use crate::workflow::ww::oww::OwwWw;

/// Configuration → wake-word backend, or none when disabled.
#[derive(Debug)]
pub struct WwRegistry;

impl WwRegistry {
    /// # Errors
    /// The model is not in the manifest, or the backend cannot be built from its files.
    pub fn build(config: &WwConfig, store: &ModelStore) -> Result<Option<Arc<dyn Ww>>, BackendError> {
        let Some(model) = config.model() else {
            return Ok(None);
        };
        let dir = store.dir(&model).map_err(|_| BackendError::Unknown(model.clone()))?;
        match config.backend {
            WwBackend::Off => Ok(None),
            WwBackend::Kws => Ok(Some(Arc::new(KwsWw::new(&dir, config)?))),
            WwBackend::Oww => Ok(Some(Arc::new(OwwWw::new(&dir, config)?))),
        }
    }
}
