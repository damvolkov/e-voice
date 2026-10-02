use std::path::{Path, PathBuf};

use crate::schema::error::BackendError;

/// File set of an exported RNN-T/TDT transducer; int8 weights win when both precisions are present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transducer {
    pub encoder: String,
    pub decoder: String,
    pub joiner: String,
    pub tokens: String,
}

impl Transducer {
    // ##### PRIVATE #####

    fn locate_part(dir: &Path, part: &str) -> Result<String, BackendError> {
        [format!("{part}.int8.onnx"), format!("{part}.onnx")]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file())
            .map(|path| path.display().to_string())
            .ok_or_else(|| BackendError::Missing(dir.join(format!("{part}.onnx"))))
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// A part or `tokens.txt` is missing from `dir`.
    pub fn locate(dir: &Path) -> Result<Self, BackendError> {
        let tokens: PathBuf = dir.join("tokens.txt");
        tokens
            .is_file()
            .then_some(())
            .ok_or_else(|| BackendError::Missing(tokens.clone()))?;
        Ok(Self {
            encoder: Self::locate_part(dir, "encoder")?,
            decoder: Self::locate_part(dir, "decoder")?,
            joiner: Self::locate_part(dir, "joiner")?,
            tokens: tokens.display().to_string(),
        })
    }
}
