use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{PoisonError, RwLock};

use crate::schema::voice::VoiceState;

const EXTENSION: &str = "safetensors";
const LONGEST: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("voice id must be 1-{LONGEST} chars of a-z, 0-9, '-' or '_': {0:?}")]
    Invalid(String),
    #[error("no voice {0:?}")]
    Unknown(String),
    #[error("voice store: {0}")]
    Io(#[from] std::io::Error),
}

/// Learned voices on disk (`<data>/voices/<id>.safetensors`), cached in memory once read.
/// Blocking file I/O: call from a blocking context.
#[derive(Debug)]
pub struct VoiceStore {
    dir: PathBuf,
    cache: RwLock<HashMap<String, VoiceState>>,
}

impl VoiceStore {
    // ##### PRIVATE #####

    fn path(&self, id: &str) -> Result<PathBuf, VoiceError> {
        Self::valid(id)
            .then(|| self.dir.join(format!("{id}.{EXTENSION}")))
            .ok_or_else(|| VoiceError::Invalid(id.to_owned()))
    }

    // ##### PUBLIC #####

    /// # Errors
    /// The voices directory cannot be created.
    pub fn open(data: &Path) -> Result<Self, VoiceError> {
        let dir = data.join("voices");
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            cache: RwLock::new(HashMap::new()),
        })
    }

    #[must_use]
    pub fn valid(id: &str) -> bool {
        (1..=LONGEST).contains(&id.len())
            && id
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_')
    }

    /// # Errors
    /// Invalid id, no such voice, or an unreadable file.
    pub fn get(&self, id: &str) -> Result<VoiceState, VoiceError> {
        let path = self.path(id)?;
        let cached = self
            .cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned();
        cached.map_or_else(
            || {
                let state = VoiceState::from(std::fs::read(&path).map_err(|error| match error.kind() {
                    std::io::ErrorKind::NotFound => VoiceError::Unknown(id.to_owned()),
                    _ => VoiceError::Io(error),
                })?);
                self.cache
                    .write()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(id.to_owned(), state.clone());
                Ok(state)
            },
            Ok,
        )
    }

    /// Writes atomically (temp file, then rename), replacing any voice with the same id.
    ///
    /// # Errors
    /// Invalid id or a failed write.
    pub fn put(&self, id: &str, state: &VoiceState) -> Result<(), VoiceError> {
        let path = self.path(id)?;
        let staging = path.with_extension("staging");
        std::fs::write(&staging, &**state)?;
        std::fs::rename(&staging, &path)?;
        self.cache
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id.to_owned(), state.clone());
        Ok(())
    }

    /// # Errors
    /// Invalid id, no such voice, or a failed delete.
    pub fn remove(&self, id: &str) -> Result<(), VoiceError> {
        let path = self.path(id)?;
        self.cache.write().unwrap_or_else(PoisonError::into_inner).remove(id);
        std::fs::remove_file(&path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => VoiceError::Unknown(id.to_owned()),
            _ => VoiceError::Io(error),
        })
    }

    /// Voice ids, sorted.
    ///
    /// # Errors
    /// The voices directory cannot be read.
    pub fn list(&self) -> Result<Vec<String>, VoiceError> {
        let mut ids: Vec<String> = std::fs::read_dir(&self.dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == EXTENSION))
            .filter_map(|path| path.file_stem().and_then(|stem| stem.to_str()).map(str::to_owned))
            .filter(|id| Self::valid(id))
            .collect();
        ids.sort_unstable();
        Ok(ids)
    }
}
