use std::fmt::Debug;

use e_voice_core::schema::error::BackendError;

use crate::schema::audio::Audio;
use crate::schema::segment::SegmentSpan;

/// Speech boundaries in stream samples; `Start.at` is an estimate, `End.span` is exact and
/// `End.audio` covers it plus whatever padding the backend adds on each side.
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    Start { at: u64 },
    End { span: SegmentSpan, audio: Audio },
}

/// Shared, read-only VAD backend that opens one stateful session per stream.
pub trait Vad: Send + Sync + Debug {
    /// # Errors
    /// The runtime refused to build a detector.
    fn open(&self) -> Result<Box<dyn VadSession>, BackendError>;
}

/// Per-stream segmenter over 16 kHz mono audio, fed in chunks of any size.
pub trait VadSession: Send {
    fn push(&mut self, audio: &[f32]) -> Vec<VadEvent>;
    fn flush(&mut self) -> Vec<VadEvent>;
}
