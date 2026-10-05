use std::sync::Arc;

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::lang::Lang;
use tokio_util::sync::CancellationToken;

use crate::core::voices::{VoiceError, VoiceStore};
use crate::schema::voice::VoiceState;
use crate::workflow::runner::Runner;

/// Shared by every request: the runner over the loaded backend, the voice store, request defaults
/// (`lang`, `voice`), the model ids `/v1/models` advertises, the upload limit in bytes, and the
/// shutdown signal every stream obeys.
#[derive(Debug, Clone)]
pub struct AppState {
    pub runner: Runner,
    pub voices: Arc<VoiceStore>,
    pub lang: Lang,
    pub voice: Option<String>,
    pub models: Vec<String>,
    pub upload: usize,
    pub runtime: Runtime,
    pub shutdown: CancellationToken,
}

impl AppState {
    /// The requested voice if stored; otherwise the configured default (OpenAI preset names such as
    /// `alloy` therefore speak with it); `None` is the backend's built-in voice.
    ///
    /// # Errors
    /// The default voice is configured but cannot be read.
    pub async fn voice(&self, requested: Option<String>) -> Result<Option<VoiceState>, VoiceError> {
        let (voices, default) = (Arc::clone(&self.voices), self.voice.clone());
        tokio::task::spawn_blocking(move || {
            let stored = requested.and_then(|id| voices.get(&id).ok());
            match (stored, default) {
                (Some(state), _) => Ok(Some(state)),
                (None, Some(id)) => voices.get(&id).map(Some),
                (None, None) => Ok(None),
            }
        })
        .await
        .map_err(|error| VoiceError::Io(std::io::Error::other(error)))?
    }
}
