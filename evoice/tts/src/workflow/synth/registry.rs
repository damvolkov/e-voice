use std::sync::Arc;

use e_voice_core::models::ModelStore;
use e_voice_core::schema::error::BackendError;
use e_voice_core::schema::lang::Lang;

use crate::config::synth::{SynthBackend, SynthConfig};
use crate::workflow::synth::base::Synth;
use crate::workflow::synth::neutts::{NeuttsOptions, NeuttsSynth};
use crate::workflow::synth::pocket::{PocketOptions, PocketSynth};
use crate::workflow::synth::qwen3::{Qwen3Options, Qwen3Synth};

/// Maps the configured backend to a built, shared synthesizer.
#[derive(Debug)]
pub struct SynthRegistry;

impl SynthRegistry {
    // ##### PRIVATE #####

    fn build_dir(store: &ModelStore, model: &str) -> Result<std::path::PathBuf, BackendError> {
        store.dir(model).map_err(|_| BackendError::Unknown(model.to_owned()))
    }

    fn build_dirs(
        config: &SynthConfig,
        store: &ModelStore,
    ) -> Result<Vec<(Lang, String, std::path::PathBuf)>, BackendError> {
        Lang::ALL
            .iter()
            .map(|lang| {
                let model = config.model(*lang);
                Self::build_dir(store, model).map(|dir| (*lang, model.to_owned(), dir))
            })
            .collect()
    }

    // ##### PUBLIC #####

    /// Blocking: loads every graph. Call from a blocking context.
    ///
    /// # Errors
    /// A model missing from the store, or the backend refusing its files.
    pub fn build(config: &SynthConfig, store: &ModelStore) -> Result<Arc<dyn Synth>, BackendError> {
        match config.backend {
            SynthBackend::Neutts => {
                let dirs = Self::build_dirs(config, store)?;
                let models: Vec<_> = dirs
                    .iter()
                    .map(|(lang, model, dir)| (*lang, model.clone(), dir.as_path()))
                    .collect();
                let options = NeuttsOptions {
                    threads: config.threads,
                    workers: config.workers,
                    temperature: config.temperature.unwrap_or(1.0),
                    espeak: config.espeak.clone(),
                };
                let encoder = Self::build_dir(store, SynthConfig::ENCODER)?;
                Ok(Arc::new(NeuttsSynth::new(&models, &encoder, options)?))
            }
            SynthBackend::Pocket => {
                let dirs = Self::build_dirs(config, store)?;
                let models: Vec<_> = dirs
                    .iter()
                    .map(|(lang, model, dir)| (*lang, model.clone(), dir.as_path()))
                    .collect();
                let options = PocketOptions {
                    threads: config.threads,
                    workers: config.workers,
                    temperature: config.temperature.unwrap_or(0.3),
                    steps: config.steps,
                    quantized: config.quantized,
                };
                Ok(Arc::new(PocketSynth::new(&models, options)?))
            }
            SynthBackend::Qwen3 => {
                let options = Qwen3Options {
                    threads: config.threads,
                    workers: config.workers,
                    temperature: config.temperature.unwrap_or(0.9),
                    context: 50,
                    first: 4,
                    chunk: 16,
                };
                let dir = Self::build_dir(store, config.model(Lang::Es))?;
                Ok(Arc::new(Qwen3Synth::new(&dir, options)?))
            }
        }
    }
}
