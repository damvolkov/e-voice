use std::fmt::Debug;

use crate::schema::error::NodeError;
use crate::schema::lang::Lang;

/// Shared spoken language identification over a whole 16 kHz mono segment.
pub trait Lid: Send + Sync + Debug {
    /// The supported language spoken in `audio`; `None` when it is none of them.
    ///
    /// # Errors
    /// The runtime produced no result.
    fn identify(&self, audio: &[f32]) -> Result<Option<Lang>, NodeError>;
}
