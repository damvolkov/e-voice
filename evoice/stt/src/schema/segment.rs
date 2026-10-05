use serde::{Deserialize, Serialize};

use crate::schema::audio::Audio;

/// Monotonic per-stream segment identifier, starting at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(transparent)]
#[schema(value_type = u64)]
pub struct SegmentId(pub u64);

/// Segment bounds in samples from the start of the stream, end exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SegmentSpan {
    pub start: u64,
    pub end: u64,
}

impl SegmentSpan {
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// One closed span of speech and its audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub id: SegmentId,
    pub span: SegmentSpan,
    pub audio: Audio,
}
