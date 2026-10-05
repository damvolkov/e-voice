use std::fmt::{self, Debug};
use std::path::Path;

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use sherpa_onnx::{
    SpokenLanguageIdentification, SpokenLanguageIdentificationConfig, SpokenLanguageIdentificationWhisperConfig,
};

use crate::schema::audio::RATE;
use crate::workflow::lid::base::Lid;
use crate::workflow::parts::Parts;

const SAMPLE_RATE: i32 = RATE.cast_signed();

/// Whisper's language token from its multilingual encoder–decoder (tiny: 39M parameters).
pub struct WhisperLid {
    identifier: SpokenLanguageIdentification,
}

impl Debug for WhisperLid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WhisperLid").finish_non_exhaustive()
    }
}

impl WhisperLid {
    /// # Errors
    /// Model files missing from `dir`, or the runtime rejecting them.
    pub fn new(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let config = SpokenLanguageIdentificationConfig {
            whisper: SpokenLanguageIdentificationWhisperConfig {
                encoder: Some(Parts::onnx(dir, "encoder")?),
                decoder: Some(Parts::onnx(dir, "decoder")?),
                tail_paddings: -1,
            },
            num_threads: i32::from(threads),
            debug: false,
            provider: Some(Runtime::provider()),
        };
        let identifier = SpokenLanguageIdentification::create(&config).ok_or(BackendError::Load("whisper lid"))?;
        Ok(Self { identifier })
    }
}

impl Lid for WhisperLid {
    fn identify(&self, audio: &[f32]) -> Result<Option<Lang>, NodeError> {
        let stream = self.identifier.create_stream();
        stream.accept_waveform(SAMPLE_RATE, audio);
        self.identifier
            .compute(&stream)
            .map(|result| result.lang.parse().ok())
            .ok_or_else(|| NodeError::Backend("whisper lid produced no result".to_owned()))
    }
}
