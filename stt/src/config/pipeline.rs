use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::asr::AsrConfig;
use crate::config::ser::SerConfig;
use crate::config::vad::VadConfig;
use crate::config::ww::WwConfig;

/// What happens to a new segment when `pending` live segments already exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Overload {
    #[default]
    Reject,
    Evict,
}

/// When an open wake-word gate falls back to listening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateClose {
    Utterance,
    #[default]
    Window,
    Session,
}

/// Wake-word gate policy; ignored while `ww.backend = "off"`. `idle` closes an open gate without speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GateConfig {
    pub close: GateClose,
    #[serde(with = "humantime_serde")]
    pub idle: Duration,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            close: GateClose::Window,
            idle: Duration::from_secs(8),
        }
    }
}

/// Input gain ahead of every node: peak target in dBFS, maximum boost in dB (0 disables), and how
/// fast the tracked peak decays.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GainConfig {
    pub peak: f32,
    pub max: f32,
    #[serde(with = "humantime_serde")]
    pub release: Duration,
}

impl Default for GainConfig {
    fn default() -> Self {
        Self {
            peak: -6.0,
            max: 40.0,
            release: Duration::from_secs(5),
        }
    }
}

/// The STT pipeline: per-stream policy, then one table per node. `jobs` bounds segment work across
/// all streams; `preroll` is audio kept so streaming ASR starts at the estimated onset of speech.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineConfig {
    pub pending: usize,
    pub overload: Overload,
    #[serde(with = "humantime_serde")]
    pub stall: Duration,
    #[serde(with = "humantime_serde")]
    pub tick: Duration,
    #[serde(with = "humantime_serde")]
    pub preroll: Duration,
    pub jobs: usize,
    pub gate: GateConfig,
    pub gain: GainConfig,
    pub ww: WwConfig,
    pub vad: VadConfig,
    pub asr: AsrConfig,
    pub ser: SerConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            pending: 4,
            overload: Overload::Reject,
            stall: Duration::from_secs(60),
            tick: Duration::from_millis(50),
            preroll: Duration::from_secs(2),
            jobs: 8,
            gate: GateConfig::default(),
            gain: GainConfig::default(),
            ww: WwConfig::default(),
            vad: VadConfig::default(),
            asr: AsrConfig::default(),
            ser: SerConfig::default(),
        }
    }
}
