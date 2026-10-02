use std::path::Path;
use std::time::Duration;

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use serde::{Deserialize, Serialize};

use crate::config::asr::AsrBackend;
use crate::config::server::ServerConfig;
use crate::config::stt::SttConfig;
use crate::config::tts::TtsConfig;
use crate::config::ww::{WwBackend, WwConfig};

pub const DEFAULT_PATH: &str = "evoice.toml";
pub const ENV_PREFIX: &str = "EVOICE_";

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("invalid settings: {0}")]
    Load(#[from] Box<figment::Error>),
    #[error("invalid settings: `{key}` {reason}")]
    Invalid { key: &'static str, reason: String },
}

/// The whole service contract: serde defaults, then the TOML file, then
/// `EVOICE_<SECTION>__<KEY>` overrides (e.g. `EVOICE_STT__PIPELINE__ASR__CHUNK=560ms`). Every closed
/// choice is an enum and every table rejects unknown keys; values are then range-checked, so a
/// configuration that cannot run never starts.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub server: ServerConfig,
    pub stt: SttConfig,
    pub tts: TtsConfig,
}

impl Settings {
    // ##### PRIVATE #####

    fn validate_rules(&self) -> [(bool, &'static str, String); 14] {
        let pipeline = &self.stt.pipeline;
        let (ww, vad, asr, ser) = (&pipeline.ww, &pipeline.vad, &pipeline.asr, &pipeline.ser);
        let nemotron_only =
            asr.backend == AsrBackend::Parakeet && (asr.chunk.is_some() || asr.lead.is_some() || asr.tail.is_some());
        let kws_only = ww.backend == WwBackend::Oww && ww.boost.is_some();
        let unit = |value: f32| value > 0.0 && value <= 1.0;
        let threads = [ww.threads, vad.threads, asr.threads, asr.offline.threads, ser.threads];
        [
            (
                self.server.upload >= 1,
                "server.upload",
                "must be at least 1 MiB".to_owned(),
            ),
            (
                pipeline.pending >= 1,
                "stt.pipeline.pending",
                "must be at least 1".to_owned(),
            ),
            (pipeline.jobs >= 1, "stt.pipeline.jobs", "must be at least 1".to_owned()),
            (
                pipeline.tick > Duration::ZERO,
                "stt.pipeline.tick",
                "must be positive".to_owned(),
            ),
            (
                pipeline.gain.peak <= 0.0 && (0.0..=60.0).contains(&pipeline.gain.max),
                "stt.pipeline.gain",
                "needs peak <= 0 dBFS and 0 <= max <= 60 dB".to_owned(),
            ),
            (
                unit(ww.threshold()),
                "stt.pipeline.ww.threshold",
                "must be in (0, 1]".to_owned(),
            ),
            (
                !kws_only,
                "stt.pipeline.ww.boost",
                "applies to backend \"kws\" only".to_owned(),
            ),
            (
                ww.boost.is_none_or(|boost| boost > 0.0),
                "stt.pipeline.ww.boost",
                "must be positive".to_owned(),
            ),
            (
                Self::validate_keyword(ww),
                "stt.pipeline.ww.keyword",
                Self::validate_keyword_reason(ww),
            ),
            (
                unit(vad.threshold) && vad.threshold < 1.0,
                "stt.pipeline.vad.threshold",
                "must be in (0, 1)".to_owned(),
            ),
            (
                vad.min_silence > Duration::ZERO && vad.min_speech > Duration::ZERO && vad.max_speech > vad.min_speech,
                "stt.pipeline.vad",
                "needs min_silence > 0, min_speech > 0 and max_speech > min_speech".to_owned(),
            ),
            (
                !nemotron_only,
                "stt.pipeline.asr",
                "chunk, lead and tail apply to backend \"nemotron\" only".to_owned(),
            ),
            (
                asr.deadline > Duration::ZERO && ser.deadline > Duration::ZERO,
                "stt.pipeline",
                "deadlines must be positive".to_owned(),
            ),
            (
                threads.iter().all(|threads| *threads >= 1),
                "stt.pipeline.*.threads",
                "must be at least 1".to_owned(),
            ),
        ]
    }

    fn validate_keyword(ww: &WwConfig) -> bool {
        match ww.backend {
            WwBackend::Off => true,
            WwBackend::Kws => {
                !ww.keyword.trim().is_empty()
                    && ww
                        .keyword
                        .chars()
                        .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '\'')
            }
            WwBackend::Oww => WwConfig::OWW_KEYWORDS.contains(&ww.keyword.as_str()),
        }
    }

    fn validate_keyword_reason(ww: &WwConfig) -> String {
        match ww.backend {
            WwBackend::Off | WwBackend::Kws => "must be English letters, spaces or apostrophes".to_owned(),
            WwBackend::Oww => format!(
                "has no trained openWakeWord classifier; available: {:?}",
                WwConfig::OWW_KEYWORDS
            ),
        }
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// An explicit `path` must exist; without one, `evoice.toml` is read only if present.
    ///
    /// # Errors
    /// Missing explicit file, malformed TOML, unknown keys or backends, ill-typed or out-of-range values.
    pub fn load(path: Option<&Path>) -> Result<Self, SettingsError> {
        let file = path.map_or_else(|| Toml::file(DEFAULT_PATH), Toml::file_exact);
        let settings: Self = Figment::from(file)
            .merge(Env::prefixed(ENV_PREFIX).split("__"))
            .extract()
            .map_err(|error| SettingsError::Load(Box::new(error)))?;
        settings.validate()?;
        Ok(settings)
    }

    /// # Errors
    /// The first rule the values break.
    pub fn validate(&self) -> Result<(), SettingsError> {
        self.validate_rules()
            .into_iter()
            .find(|(ok, ..)| !ok)
            .map_or(Ok(()), |(_, key, reason)| Err(SettingsError::Invalid { key, reason }))
    }

    /// Manifest ids of every model the configured pipeline loads.
    #[must_use]
    pub fn models(&self) -> Vec<String> {
        let pipeline = &self.stt.pipeline;
        let ww = pipeline.ww.model();
        let live = Some(pipeline.asr.model().to_owned());
        let offline = pipeline
            .asr
            .offline_model()
            .filter(|model| *model != pipeline.asr.model())
            .map(str::to_owned);
        let ser = pipeline.ser.model().map(str::to_owned);
        ww.into_iter()
            .chain(
                [Some(pipeline.vad.model().to_owned()), live, offline, ser]
                    .into_iter()
                    .flatten(),
            )
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::result_large_err)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use figment::Jail;

    use crate::config::asr::{AsrBackend, NemotronChunk};
    use crate::config::log::LogFormat;
    use crate::config::vad::VadBackend;
    use crate::config::ww::WwBackend;
    use crate::core::settings::{Settings, SettingsError};

    fn invalid(toml: &str) -> Option<&'static str> {
        let mut key = None;
        Jail::expect_with(|jail| {
            jail.create_file("evoice.toml", toml)?;
            key = match Settings::load(None) {
                Err(SettingsError::Invalid { key, .. }) => Some(key),
                Err(SettingsError::Load(_)) => Some("load"),
                Ok(_) => None,
            };
            Ok(())
        });
        key
    }

    #[test]
    fn test_load_defaults_without_file() {
        Jail::expect_with(|_| {
            let settings = Settings::load(None).unwrap();
            assert_eq!(settings, Settings::default());
            assert_eq!(
                settings.models(),
                [
                    "silero-vad",
                    "nemotron-3.5-1120ms-int8",
                    "parakeet-v3-int8",
                    "emotion2vec-plus-base"
                ]
            );
            Ok(())
        });
    }

    #[test]
    fn test_load_file_then_env_overrides() {
        Jail::expect_with(|jail| {
            jail.create_file(
                "evoice.toml",
                "[server]\nport = 6000\nlog = { format = \"json\" }\n[stt.pipeline.asr]\nchunk = \"160ms\"\n",
            )?;
            jail.set_env("EVOICE_SERVER__PORT", "7000");
            jail.set_env("EVOICE_STT__OPS__DATA", "/data/stt");
            jail.set_env("EVOICE_STT__PIPELINE__VAD__BACKEND", "ten");
            let settings = Settings::load(None).unwrap();
            assert_eq!(settings.server.port, 7000);
            assert_eq!(settings.server.log.format, LogFormat::Json);
            assert_eq!(settings.stt.ops.data, PathBuf::from("/data/stt"));
            assert_eq!(settings.stt.pipeline.vad.backend, VadBackend::Ten);
            assert_eq!(settings.stt.pipeline.asr.chunk(), NemotronChunk::Ms160);
            assert_eq!(settings.models()[..2], ["ten-vad", "nemotron-3.5-160ms-int8"]);
            Ok(())
        });
    }

    #[test]
    fn test_backends_select_models() {
        Jail::expect_with(|jail| {
            jail.create_file(
                "evoice.toml",
                "[stt.pipeline.ww]\nbackend = \"kws\"\n[stt.pipeline.asr]\nbackend = \"parakeet\"\noffline = { backend = \"live\" }\n[stt.pipeline.ser]\nbackend = \"off\"\n",
            )?;
            let settings = Settings::load(None).unwrap();
            assert_eq!(settings.stt.pipeline.ww.backend, WwBackend::Kws);
            assert_eq!(settings.stt.pipeline.asr.backend, AsrBackend::Parakeet);
            assert_eq!(settings.models(), ["kws-gigaspeech", "silero-vad", "parakeet-v3-int8"]);
            Ok(())
        });
    }

    #[test]
    fn test_rejects_unknown_backends_keys_and_sections() {
        assert_eq!(invalid("[stt.pipeline.asr]\nbackend = \"whisper\"\n"), Some("load"));
        assert_eq!(invalid("[stt.pipeline.asr]\nchunk = \"300ms\"\n"), Some("load"));
        assert_eq!(invalid("[stt.pipeline.vad]\nmodel = \"silero-vad\"\n"), Some("load"));
        assert_eq!(invalid("[tts]\nvoice = \"x\"\n"), Some("load"));
        assert_eq!(invalid("[server]\nprot = 1\n"), Some("load"));
    }

    #[test]
    fn test_rejects_values_that_cannot_run() {
        assert_eq!(invalid("[stt.pipeline]\npending = 0\n"), Some("stt.pipeline.pending"));
        assert_eq!(
            invalid("[stt.pipeline.ww]\nthreshold = 1.5\n"),
            Some("stt.pipeline.ww.threshold")
        );
        assert_eq!(
            invalid("[stt.pipeline.ww]\nbackend = \"oww\"\nkeyword = \"eager\"\n"),
            Some("stt.pipeline.ww.keyword")
        );
        assert_eq!(
            invalid("[stt.pipeline.ww]\nbackend = \"oww\"\nkeyword = \"hey_jarvis\"\nboost = 2.0\n"),
            Some("stt.pipeline.ww.boost")
        );
        assert_eq!(
            invalid("[stt.pipeline.ww]\nbackend = \"kws\"\nkeyword = \"hey-1\"\n"),
            Some("stt.pipeline.ww.keyword")
        );
        assert_eq!(
            invalid("[stt.pipeline.asr]\nbackend = \"parakeet\"\nchunk = \"560ms\"\n"),
            Some("stt.pipeline.asr")
        );
        assert_eq!(
            invalid("[stt.pipeline.vad]\nmax_speech = \"100ms\"\n"),
            Some("stt.pipeline.vad")
        );
        assert_eq!(
            invalid("[stt.pipeline.ser]\nthreads = 0\n"),
            Some("stt.pipeline.*.threads")
        );
        assert_eq!(invalid("[stt.pipeline.gain]\nmax = 90.0\n"), Some("stt.pipeline.gain"));
        assert_eq!(
            invalid("[stt.pipeline.ww]\nbackend = \"kws\"\nkeyword = \"hey eager\"\nboost = 2.0\n"),
            None
        );
    }

    #[test]
    fn test_example_file_loads_with_defaults() {
        let example = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../evoice.example.toml");
        Jail::expect_with(|_| {
            let settings = Settings::load(Some(&example)).unwrap();
            assert_eq!(settings.stt.pipeline.asr.deadline, Duration::from_secs(15));
            assert_eq!(settings, Settings::default());
            Ok(())
        });
    }

    #[test]
    fn test_load_rejects_missing_explicit_file() {
        Jail::expect_with(|_| {
            assert!(Settings::load(Some(Path::new("absent.toml"))).is_err());
            Ok(())
        });
    }
}
