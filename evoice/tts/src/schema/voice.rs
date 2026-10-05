use std::ops::Deref;
use std::sync::Arc;

/// A learned voice, opaque outside the backend that built it; only that backend can open it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceState(Arc<[u8]>);

impl From<Vec<u8>> for VoiceState {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes.into())
    }
}

impl Deref for VoiceState {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}
