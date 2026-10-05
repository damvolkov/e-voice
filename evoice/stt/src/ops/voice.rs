use std::fmt::{self, Debug};
use std::path::{Path, PathBuf};

use e_voice_core::audio::{AudioEncoding, AudioIngest};
use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::BackendError;
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig};

use crate::schema::audio::RATE;

/// A Piper voice used by internal tools to synthesize test speech, deterministically (sampling noise
/// is pinned near zero; sherpa treats an exact 0 as "use the model default") and at 16 kHz.
pub struct Voice {
    tts: OfflineTts,
    pub name: String,
}

impl Debug for Voice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Voice")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl Voice {
    /// # Errors
    /// The voice files are missing or the runtime rejects them.
    pub fn open(dir: &Path) -> Result<Self, BackendError> {
        let model = std::fs::read_dir(dir)
            .map_err(|_| BackendError::Missing(dir.to_path_buf()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find(|path| path.extension().is_some_and(|extension| extension == "onnx"))
            .ok_or_else(|| BackendError::Missing(dir.join("*.onnx")))?;
        let path = |name: &str| -> Option<String> { Some(dir.join(name).display().to_string()) };
        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(model.display().to_string()),
                    tokens: path("tokens.txt"),
                    data_dir: path("espeak-ng-data"),
                    noise_scale: 1e-6,
                    noise_scale_w: 1e-6,
                    ..OfflineTtsVitsModelConfig::default()
                },
                num_threads: 2,
                provider: Some(Runtime::provider()),
                ..OfflineTtsModelConfig::default()
            },
            ..OfflineTtsConfig::default()
        };
        let tts = OfflineTts::create(&config).ok_or(BackendError::Load("piper"))?;
        let name = dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("voice")
            .to_owned();
        Ok(Self { tts, name })
    }

    /// `text` spoken at `speed` (1.0 is natural), as 16 kHz mono samples.
    ///
    /// # Errors
    /// Synthesis or resampling failed.
    pub fn speak(&self, text: &str, speed: f32) -> Result<Vec<f32>, BackendError> {
        let generation = GenerationConfig {
            speed,
            ..GenerationConfig::default()
        };
        let audio = self
            .tts
            .generate_with_config(text, &generation, None::<fn(&[f32], f32) -> bool>)
            .ok_or(BackendError::Load("piper synthesis"))?;
        let rate = u32::try_from(audio.sample_rate()).map_err(|_| BackendError::Load("piper rate"))?;
        let mut ingest =
            AudioIngest::new(rate, RATE, AudioEncoding::F32le).map_err(|_| BackendError::Load("piper rate"))?;
        let mut samples = ingest
            .feed(audio.samples())
            .map_err(|_| BackendError::Load("piper resample"))?;
        samples.extend(ingest.flush().map_err(|_| BackendError::Load("piper resample"))?);
        Ok(samples)
    }

    /// Manifest ids of the voices tools use.
    pub const IDS: [&'static str; 3] = ["piper-en-amy-low", "piper-en-lessac-medium", "piper-en-ryan-medium"];

    #[must_use]
    pub fn dirs(resolve: impl Fn(&str) -> Option<PathBuf>) -> Vec<PathBuf> {
        Self::IDS.iter().filter_map(|id| resolve(id)).collect()
    }
}
