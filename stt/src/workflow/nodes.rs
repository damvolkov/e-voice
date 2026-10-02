use std::sync::Arc;

use crate::config::asr::AsrEngine;
use crate::core::models::{ModelError, ModelStore};
use crate::core::runtime::{Runtime, RuntimeError};
use crate::core::settings::Settings;
use crate::schema::error::BackendError;
use crate::workflow::asr::registry::{AsrBackend, AsrRegistry};
use crate::workflow::denoise::base::Denoise;
use crate::workflow::denoise::registry::DenoiseRegistry;
use crate::workflow::lid::base::Lid;
use crate::workflow::lid::registry::LidRegistry;
use crate::workflow::ser::base::Ser;
use crate::workflow::ser::registry::SerRegistry;
use crate::workflow::vad::base::Vad;
use crate::workflow::vad::registry::VadRegistry;
use crate::workflow::ww::base::Ww;
use crate::workflow::ww::registry::WwRegistry;

#[derive(Debug, thiserror::Error)]
pub enum NodesError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Models(#[from] ModelError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

/// Every backend the pipeline runs, built once and shared read-only by all streams.
/// `offline`, when set, replaces `asr` for uploaded files; `extra` holds the further engines a
/// request may select.
#[derive(Debug, Clone)]
pub struct Nodes {
    pub denoise: Option<Arc<dyn Denoise>>,
    pub ww: Option<Arc<dyn Ww>>,
    pub vad: Arc<dyn Vad>,
    pub lid: Option<Arc<dyn Lid>>,
    pub asr: AsrBackend,
    pub offline: Option<AsrBackend>,
    pub extra: Vec<(AsrEngine, AsrBackend)>,
    pub ser: Option<Arc<dyn Ser>>,
}

impl Nodes {
    /// # Errors
    /// Any configured backend cannot be built.
    pub fn build(settings: &Settings, store: &ModelStore) -> Result<Self, BackendError> {
        let pipeline = &settings.stt.pipeline;
        Ok(Self {
            denoise: DenoiseRegistry::build(&pipeline.denoise, store)?,
            ww: WwRegistry::build(&pipeline.ww, store)?,
            vad: VadRegistry::build(&pipeline.vad, store)?,
            lid: LidRegistry::build(&pipeline.lid, store)?,
            asr: AsrRegistry::build(&pipeline.asr, store)?,
            offline: AsrRegistry::offline(&pipeline.asr, store)?,
            extra: AsrRegistry::extra(&pipeline.asr, store)?,
            ser: SerRegistry::build(&pipeline.ser, store)?,
        })
    }

    /// Startup order every entry point shares: native runtime → configured models verified offline →
    /// backends built once, off the async threads.
    ///
    /// # Errors
    /// The runtime cannot load, a configured model is missing or corrupt, or a backend fails to build.
    pub async fn start(settings: &Settings) -> Result<(Self, Runtime), NodesError> {
        let runtime = Runtime::probe()?;
        let store = ModelStore::open(&settings.stt.ops)?;
        store.verify(&settings.models(), settings.stt.ops.verify).await?;
        let settings = settings.clone();
        let nodes = tokio::task::spawn_blocking(move || Self::build(&settings, &store))
            .await
            .map_err(ModelError::Join)??;
        Ok((nodes, runtime))
    }
}
