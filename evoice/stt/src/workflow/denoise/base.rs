use std::fmt::Debug;

use e_voice_core::schema::error::BackendError;

/// Shared speech enhancement backend that opens one stateful session per stream.
pub trait Denoise: Send + Sync + Debug {
    /// # Errors
    /// The runtime refused to create a session.
    fn open(&self) -> Result<Box<dyn DenoiseSession>, BackendError>;
}

/// One stream's enhancer: 16 kHz mono in, the same rate out, delayed by the model's frame.
pub trait DenoiseSession: Send {
    fn push(&mut self, audio: &[f32]) -> Vec<f32>;

    /// Whatever the model still buffers.
    fn flush(&mut self) -> Vec<f32>;
}
