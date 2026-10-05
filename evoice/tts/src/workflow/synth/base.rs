use std::fmt::Debug;

use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;

use crate::schema::audio::Audio;
use crate::schema::error::TrainError;
use crate::schema::voice::VoiceState;

/// Longest first chunk: what bounds the time to first audio on top of the backend's prefill.
pub const FIRST_LIMIT_MS: u64 = 500;

/// Longest chunk afterwards; codecs that decode with left context (Qwen3) voice several frames at once.
pub const CHUNK_LIMIT_MS: u64 = 2_000;

/// What a backend offers; callers read it instead of the backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynthCaps {
    pub rate: u32,
    pub langs: &'static [Lang],
}

/// Shared synthesis backend that opens one session per stream.
///
/// Only real streaming backends qualify: audio leaves while the sentence is still being generated —
/// a first chunk of at most [`FIRST_LIMIT_MS`], then chunks of at most [`CHUNK_LIMIT_MS`]. A backend
/// that renders a sentence whole is rejected.
pub trait Synth: Send + Sync + Debug {
    fn caps(&self) -> SynthCaps;

    /// `None` speaks with the backend's built-in voice.
    ///
    /// # Errors
    /// No worker was free, or the voice belongs to another backend or model.
    fn open(&self, lang: Lang, voice: Option<&VoiceState>) -> Result<Box<dyn SynthSession>, BackendError>;

    /// Learns a voice from mono clips at [`SynthCaps::rate`]; `text`, what the clips say, lets
    /// backends that clone in context (Qwen3, NeuTTS) use the reference codes too.
    ///
    /// # Errors
    /// The backend cannot learn voices, or the clips hold no usable speech.
    fn train(&self, clips: &[Audio], text: Option<&str>) -> Result<VoiceState, TrainError> {
        let _ = (clips, text);
        Err(TrainError::Unsupported)
    }
}

/// One stream's synthesis state, holding the leased worker until dropped.
pub trait SynthSession: Send {
    /// Lazily generates one sentence: each `next` computes and returns the following chunk.
    /// Dropping the iterator stops generation; no work may outlive it.
    fn speak<'a>(&'a mut self, sentence: &'a str) -> Box<dyn Iterator<Item = Result<Audio, NodeError>> + Send + 'a>;
}
