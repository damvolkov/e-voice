use std::path::Path;

use e_voice_core::config::server::ServerConfig;
use e_voice_core::settings::{Foreign, Settings as Loader, SettingsError};
use serde::{Deserialize, Serialize};

use crate::config::tts::TtsConfig;
use crate::core::voices::VoiceStore;

/// The TTS service contract over the shared `evoice.toml` (e.g. `EVOICE_TTS__SYNTH__WORKERS=8`);
/// `[stt]` belongs to the STT service. Values are range-checked, so a configuration that cannot run
/// never starts.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub server: ServerConfig,
    pub stt: Foreign,
    pub tts: TtsConfig,
}

impl Settings {
    // ##### PRIVATE #####

    fn validate_rules(&self) -> [(bool, &'static str, &'static str); 6] {
        let (tts, synth) = (&self.tts, &self.tts.synth);
        [
            (tts.api.upload >= 1, "tts.api.upload", "must be at least 1 MiB"),
            (
                tts.text.max >= 8 && tts.text.min < tts.text.max,
                "tts.text",
                "needs max >= 8 and min < max",
            ),
            (
                synth.threads >= 1 && synth.workers >= 1 && synth.steps >= 1,
                "tts.synth",
                "threads, workers and steps must be at least 1",
            ),
            (
                synth
                    .temperature
                    .is_none_or(|temperature| (0.0..=2.0).contains(&temperature)),
                "tts.synth.temperature",
                "must be in [0, 2]",
            ),
            (
                tts.voice.as_deref().is_none_or(VoiceStore::valid),
                "tts.voice",
                "must be 1-64 chars of a-z, 0-9, '-' or '_'",
            ),
            (tts.api.port > 0, "tts.api.port", "must be positive"),
        ]
    }

    // ##### PUBLIC #####

    /// An explicit `path` must exist; without one, `evoice.toml` is read only if present.
    ///
    /// # Errors
    /// Missing explicit file, malformed TOML, unknown keys or backends, ill-typed or out-of-range values.
    pub fn load(path: Option<&Path>) -> Result<Self, SettingsError> {
        let settings: Self = Loader::load(path)?;
        settings.validate()?;
        Ok(settings)
    }

    /// # Errors
    /// The first rule the values break.
    pub fn validate(&self) -> Result<(), SettingsError> {
        self.validate_rules()
            .into_iter()
            .find(|(ok, ..)| !ok)
            .map_or(Ok(()), |(_, key, reason)| {
                Err(SettingsError::Invalid {
                    key,
                    reason: reason.to_owned(),
                })
            })
    }

    /// Manifest ids of every model the configured backend loads.
    #[must_use]
    pub fn models(&self) -> Vec<String> {
        self.tts.synth.models()
    }
}

#[cfg(test)]
#[allow(clippy::result_large_err)]
mod tests {
    use std::path::PathBuf;

    use e_voice_core::settings::SettingsError;
    use figment::Jail;

    use crate::config::synth::SynthSize;
    use crate::core::settings::Settings;

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
    fn test_example_file_loads_with_defaults() {
        let example = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../evoice.example.toml");
        Jail::expect_with(|_| {
            let settings = Settings::load(Some(&example)).unwrap();
            let defaults = Settings::default();
            assert_eq!((settings.server, settings.tts), (defaults.server, defaults.tts));
            Ok(())
        });
    }

    #[test]
    fn test_env_overrides_and_models_follow_size() {
        Jail::expect_with(|jail| {
            jail.set_env("EVOICE_TTS__SYNTH__SIZE", "large");
            jail.set_env("EVOICE_TTS__API__PORT", "7600");
            jail.set_env("EVOICE_TTS__VOICE", "");
            let settings = Settings::load(None).unwrap();
            assert_eq!(settings.tts.synth.size, SynthSize::Large);
            assert_eq!(settings.tts.api.port, 7600);
            assert_eq!(settings.tts.voice, None);
            assert_eq!(settings.models(), ["pocket-es-24l", "pocket-en"]);
            Ok(())
        });
    }

    #[test]
    fn test_rejects_what_cannot_run() {
        assert_eq!(invalid("[tts.text]\nmin = 90\nmax = 80\n"), Some("tts.text"));
        assert_eq!(invalid("[tts.synth]\nworkers = 0\n"), Some("tts.synth"));
        assert_eq!(
            invalid("[tts.synth]\ntemperature = 3.0\n"),
            Some("tts.synth.temperature")
        );
        assert_eq!(invalid("[tts]\nvoice = \"Not Valid\"\n"), Some("tts.voice"));
        assert_eq!(invalid("[tts.synth]\nbackend = \"kokoro\"\n"), Some("load"));
        assert_eq!(invalid("[tts]\nspeed = 1.0\n"), Some("load"));
        assert_eq!(invalid("[stt.pipeline]\npending = 4\n"), None);
    }
}
