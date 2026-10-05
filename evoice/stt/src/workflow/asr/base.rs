use std::fmt::Debug;

use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;

/// Shared streaming ASR backend that opens one decoding session per segment.
pub trait StreamingAsr: Send + Sync + Debug {
    /// # Errors
    /// The runtime refused to create a stream.
    fn open(&self, lang: Lang) -> Result<Box<dyn AsrSession>, BackendError>;
}

/// One segment decoded incrementally from 16 kHz mono chunks of any size.
pub trait AsrSession: Send {
    /// Feeds audio; returns the hypothesis only when it changed.
    fn push(&mut self, audio: &[f32]) -> Option<String>;

    /// # Errors
    /// The runtime produced no result.
    fn finish(self: Box<Self>) -> Result<String, NodeError>;
}

/// Shared offline ASR backend that transcribes a whole 16 kHz mono segment at once.
pub trait BatchAsr: Send + Sync + Debug {
    /// # Errors
    /// The runtime produced no result.
    fn transcribe(&self, lang: Lang, audio: &[f32]) -> Result<String, NodeError>;
}
