use std::path::Path;

use e_voice_core::schema::error::BackendError;

/// Files of an exported model directory, found by suffix (`encoder.onnx`, `tiny-encoder.onnx`, …);
/// int8 weights win when both precisions are present. Shared by every node.
#[derive(Debug)]
pub struct Parts;

impl Parts {
    // ##### PRIVATE #####

    fn locate_suffix(dir: &Path, suffix: &str) -> Option<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.ends_with(suffix))
            .collect();
        names.sort_unstable();
        names.first().map(|name| dir.join(name).display().to_string())
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// The `part` weights (`encoder`, `decoder`, `joiner`, …), int8 first; an empty `part` matches
    /// any ONNX file (single-graph models).
    ///
    /// # Errors
    /// No file of that part in `dir`.
    pub fn onnx(dir: &Path, part: &str) -> Result<String, BackendError> {
        Self::locate_suffix(dir, &format!("{part}.int8.onnx"))
            .or_else(|| Self::locate_suffix(dir, &format!("{part}.onnx")))
            .ok_or_else(|| BackendError::Missing(dir.join(format!("{part}.onnx"))))
    }

    /// # Errors
    /// No `tokens.txt` in `dir`.
    pub fn tokens(dir: &Path) -> Result<String, BackendError> {
        Self::locate_suffix(dir, "tokens.txt").ok_or_else(|| BackendError::Missing(dir.join("tokens.txt")))
    }
}
