use std::path::Path;

use crate::schema::error::BackendError;
use crate::workflow::parts::Parts;

/// File set of an exported RNN-T/TDT transducer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transducer {
    pub encoder: String,
    pub decoder: String,
    pub joiner: String,
    pub tokens: String,
}

impl Transducer {
    /// # Errors
    /// A part or `tokens.txt` is missing from `dir`.
    pub fn locate(dir: &Path) -> Result<Self, BackendError> {
        Ok(Self {
            encoder: Parts::onnx(dir, "encoder")?,
            decoder: Parts::onnx(dir, "decoder")?,
            joiner: Parts::onnx(dir, "joiner")?,
            tokens: Parts::tokens(dir)?,
        })
    }
}
