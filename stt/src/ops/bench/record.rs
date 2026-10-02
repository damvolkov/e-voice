use serde::Serialize;

use crate::schema::emotion::{Emotion, EmotionLabel};
use crate::schema::error::NodeError;
use crate::schema::lang::Lang;

/// One recognized segment; latencies are set in live mode only.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecordSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub emotion: Emotion,
    pub error: Option<NodeError>,
    pub final_ms: Option<f64>,
    pub partial_ms: Option<f64>,
}

/// One manifest item through the pipeline: references, hypothesis and timing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Record {
    pub id: String,
    pub lang: Lang,
    pub mode: &'static str,
    pub audio_s: f64,
    pub wall_s: f64,
    pub rtf: f64,
    pub text: String,
    pub emotion: Emotion,
    pub ref_text: Option<String>,
    pub ref_emotion: Option<EmotionLabel>,
    pub segments: Vec<RecordSegment>,
    pub failure: Option<String>,
}

/// Last line of a bench run: what ran and what it cost.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecordSummary {
    pub kind: &'static str,
    pub mode: &'static str,
    pub items: usize,
    pub streams: usize,
    pub audio_s: f64,
    pub wall_s: f64,
    pub cpu_s: f64,
    pub cpu_per_audio_s: f64,
    pub throughput_x: f64,
    pub peak_rss_mb: f64,
    pub settings: serde_json::Value,
}
