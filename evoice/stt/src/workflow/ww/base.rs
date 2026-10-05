use std::fmt::Debug;

use e_voice_core::schema::error::{BackendError, NodeError};

use crate::schema::event::WakeEvent;

/// Shared wake-word backend that opens one stateful detector per stream.
pub trait Ww: Send + Sync + Debug {
    /// # Errors
    /// The runtime refused to build a detector.
    fn open(&self) -> Result<Box<dyn WwSession>, BackendError>;
}

/// Per-stream detector over 16 kHz mono audio, fed in chunks of any size.
pub trait WwSession: Send {
    /// Returns the first detection completed by this audio, if any.
    ///
    /// # Errors
    /// The runtime failed while scoring.
    fn push(&mut self, audio: &[f32]) -> Result<Option<WakeEvent>, NodeError>;
}
