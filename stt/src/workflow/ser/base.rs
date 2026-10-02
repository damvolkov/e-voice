use std::fmt::Debug;

use crate::schema::emotion::Emotion;
use crate::schema::error::NodeError;

/// Shared SER backend classifying one whole 16 kHz mono segment.
pub trait Ser: Send + Sync + Debug {
    /// # Errors
    /// Segment too short for the model, or the runtime failing.
    fn classify(&self, audio: &[f32]) -> Result<Emotion, NodeError>;
}
