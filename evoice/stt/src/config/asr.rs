use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Live ASR. Streaming backends (`nemotron`, `kroko`) emit partials while speech lasts; batch
/// backends transcribe each segment once it ends (no partials). Kroko is CC-BY-SA for hobby and
/// research use: benchmarked, never a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrBackend {
    #[default]
    Nemotron,
    Kroko,
    Parakeet,
    Canary,
    Cohere,
    Whisper,
}

/// Batch ASR engines, shared by live segments and uploaded files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrEngine {
    /// Parakeet TDT v3: 25 European languages, identifies the language itself.
    Parakeet,
    /// Canary 180M flash: en, es, de, fr; the language is given per segment.
    Canary,
    /// Cohere Transcribe: 14 languages, the accuracy ceiling at several times the CPU.
    Cohere,
    /// Whisper large-v3 turbo: identifies the language itself.
    Whisper,
}

/// Nemotron encoder chunk: shorter refreshes partials sooner, costs more CPU and accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NemotronChunk {
    #[serde(rename = "160ms")]
    Ms160,
    #[serde(rename = "560ms")]
    Ms560,
    #[default]
    #[serde(rename = "1120ms")]
    Ms1120,
}

/// ASR for uploaded files: a batch engine, or `live` to reuse the live backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OfflineBackend {
    #[default]
    Parakeet,
    Canary,
    Cohere,
    Whisper,
    Live,
}

/// `backend` transcribes uploads by default; `choices` are further engines loaded at startup that a
/// request selects by naming one in its model field (`model`, `model_id`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OfflineConfig {
    pub backend: OfflineBackend,
    pub threads: u16,
    pub choices: Vec<AsrEngine>,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            backend: OfflineBackend::Parakeet,
            threads: 4,
            choices: Vec::new(),
        }
    }
}

/// `chunk` applies to Nemotron; `lead` (silence fed before speech for encoder context) and `tail`
/// (silence appended to flush the last chunk) to the streaming backends. `deadline` counts from the
/// end of speech.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrConfig {
    pub backend: AsrBackend,
    pub chunk: Option<NemotronChunk>,
    #[serde(with = "humantime_serde")]
    pub lead: Option<Duration>,
    #[serde(with = "humantime_serde")]
    pub tail: Option<Duration>,
    pub threads: u16,
    #[serde(with = "humantime_serde")]
    pub deadline: Duration,
    pub offline: OfflineConfig,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            backend: AsrBackend::Nemotron,
            chunk: None,
            lead: None,
            tail: None,
            threads: 4,
            deadline: Duration::from_secs(15),
            offline: OfflineConfig::default(),
        }
    }
}

impl AsrBackend {
    /// The batch engine behind this backend; `None` for streaming backends.
    #[must_use]
    pub const fn engine(self) -> Option<AsrEngine> {
        match self {
            Self::Nemotron | Self::Kroko => None,
            Self::Parakeet => Some(AsrEngine::Parakeet),
            Self::Canary => Some(AsrEngine::Canary),
            Self::Cohere => Some(AsrEngine::Cohere),
            Self::Whisper => Some(AsrEngine::Whisper),
        }
    }
}

impl OfflineBackend {
    /// The batch engine for files; `None` reuses the live backend.
    #[must_use]
    pub const fn engine(self) -> Option<AsrEngine> {
        match self {
            Self::Live => None,
            Self::Parakeet => Some(AsrEngine::Parakeet),
            Self::Canary => Some(AsrEngine::Canary),
            Self::Cohere => Some(AsrEngine::Cohere),
            Self::Whisper => Some(AsrEngine::Whisper),
        }
    }
}

impl AsrEngine {
    pub const ALL: [Self; 4] = [Self::Parakeet, Self::Canary, Self::Cohere, Self::Whisper];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet",
            Self::Canary => "canary",
            Self::Cohere => "cohere",
            Self::Whisper => "whisper",
        }
    }

    /// The engine a request's model field names, by engine name or model id (case-insensitive);
    /// `None` for anything else (`whisper-1`, `nova-2`, …), which means the default engine.
    #[must_use]
    pub fn named(model: &str) -> Option<Self> {
        let model = model.trim();
        Self::ALL
            .into_iter()
            .find(|engine| model.eq_ignore_ascii_case(engine.name()) || model.eq_ignore_ascii_case(engine.model()))
    }

    /// Manifest id of the engine's model.
    #[must_use]
    pub const fn model(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet-v3-int8",
            Self::Canary => "canary-180m-flash-int8",
            Self::Cohere => "cohere-transcribe-int8",
            Self::Whisper => "whisper-turbo",
        }
    }
}

impl AsrConfig {
    pub const KROKO: [&'static str; 2] = ["kroko-es", "kroko-en"];

    #[must_use]
    pub fn chunk(&self) -> NemotronChunk {
        self.chunk.unwrap_or_default()
    }

    #[must_use]
    pub fn lead(&self) -> Duration {
        self.lead.unwrap_or(Duration::from_millis(300))
    }

    #[must_use]
    pub fn tail(&self) -> Duration {
        self.tail.unwrap_or(Duration::from_millis(1200))
    }

    /// Manifest id of the Nemotron model for the configured chunk.
    #[must_use]
    pub fn nemotron(&self) -> &'static str {
        match self.chunk() {
            NemotronChunk::Ms160 => "nemotron-3.5-160ms-int8",
            NemotronChunk::Ms560 => "nemotron-3.5-560ms-int8",
            NemotronChunk::Ms1120 => "nemotron-3.5-1120ms-int8",
        }
    }

    /// Whether the live backend streams partials.
    #[must_use]
    pub const fn streaming(&self) -> bool {
        self.backend.engine().is_none()
    }

    /// Manifest ids of the live backend's models.
    #[must_use]
    pub fn live(&self) -> Vec<&'static str> {
        match self.backend {
            AsrBackend::Nemotron => vec![self.nemotron()],
            AsrBackend::Kroko => Self::KROKO.to_vec(),
            backend => backend.engine().map(AsrEngine::model).into_iter().collect(),
        }
    }

    /// The file engine when it differs from the live backend; `None` means files reuse it.
    #[must_use]
    pub fn offline(&self) -> Option<AsrEngine> {
        self.offline
            .backend
            .engine()
            .filter(|engine| self.backend.engine() != Some(*engine))
    }

    /// The engine uploads use by default: the offline one, else the live backend when it is batch.
    #[must_use]
    pub fn file(&self) -> Option<AsrEngine> {
        self.offline.backend.engine().or_else(|| self.backend.engine())
    }

    /// Engines a request may select, default first, without duplicates.
    #[must_use]
    pub fn selectable(&self) -> Vec<AsrEngine> {
        let mut engines: Vec<AsrEngine> = self.file().into_iter().collect();
        for engine in &self.offline.choices {
            if !engines.contains(engine) {
                engines.push(*engine);
            }
        }
        engines
    }

    /// Choice engines that need their own instance: neither the live backend nor the default file engine.
    #[must_use]
    pub fn extra(&self) -> Vec<AsrEngine> {
        self.selectable()
            .into_iter()
            .filter(|engine| Some(*engine) != self.backend.engine() && Some(*engine) != self.offline())
            .collect()
    }

    /// Manifest ids of every ASR model the configuration runs, live first.
    #[must_use]
    pub fn models(&self) -> Vec<&'static str> {
        let mut models = self.live();
        models.extend(self.offline().map(AsrEngine::model));
        models.extend(self.extra().into_iter().map(AsrEngine::model));
        models
    }
}
