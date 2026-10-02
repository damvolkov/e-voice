use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Closed emotion vocabulary; `Unknown` is also the fallback when SER is late, short or failed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize, utoipa::ToSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum EmotionLabel {
    Angry,
    Disgusted,
    Fearful,
    Happy,
    Neutral,
    Other,
    Sad,
    Surprised,
    #[default]
    Unknown,
}

/// Emotion of one segment: winning label, per-label scores and the model that produced them.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Emotion {
    pub label: EmotionLabel,
    #[schema(value_type = std::collections::HashMap<String, f32>)]
    pub scores: BTreeMap<EmotionLabel, f32>,
    pub model: Option<String>,
}
