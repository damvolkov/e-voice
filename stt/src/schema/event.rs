use serde::{Deserialize, Serialize};

use crate::schema::emotion::Emotion;
use crate::schema::error::NodeError;
use crate::schema::lang::Lang;
use crate::schema::segment::{SegmentId, SegmentSpan};

/// Everything a stream emits to its consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Event {
    Wake(WakeEvent),
    Speech(SpeechEvent),
    Partial(PartialEvent),
    Final(FinalEvent),
    Closed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct WakeEvent {
    pub keyword: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SpeechState {
    Started,
    Stopped,
}

/// Voice activity of one segment: `at` is the stream sample where speech started (estimated) or
/// stopped (exact).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SpeechEvent {
    pub segment: SegmentId,
    pub state: SpeechState,
    pub at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PartialEvent {
    pub segment: SegmentId,
    pub text: String,
}

/// Exactly one per segment; `error` is set only when the required ASR node failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct FinalEvent {
    pub segment: SegmentId,
    pub span: SegmentSpan,
    pub lang: Lang,
    pub text: String,
    pub emotion: Emotion,
    pub error: Option<NodeError>,
}
