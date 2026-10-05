use std::collections::BTreeMap;
use std::path::Path;

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use figment::value::Value;
use serde::de::DeserializeOwned;

pub const DEFAULT_PATH: &str = "evoice.toml";
pub const ENV_PREFIX: &str = "EVOICE_";

/// Another service's section of the shared file: kept, never interpreted.
pub type Foreign = BTreeMap<String, Value>;

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("invalid settings: {0}")]
    Load(#[from] Box<figment::Error>),
    #[error("invalid settings: `{key}` {reason}")]
    Invalid { key: &'static str, reason: String },
}

/// One `evoice.toml` for every service: serde defaults, then the TOML file, then
/// `EVOICE_<SECTION>__<KEY>` overrides. Each service types its own section and the shared `[server]`,
/// and carries the others as [`Foreign`], so a typo in any top-level key is still rejected.
#[derive(Debug)]
pub struct Settings;

impl Settings {
    /// An explicit `path` must exist; without one, `evoice.toml` is read only if present.
    ///
    /// # Errors
    /// Missing explicit file, malformed TOML, unknown keys or ill-typed values.
    pub fn load<S: DeserializeOwned>(path: Option<&Path>) -> Result<S, SettingsError> {
        let file = path.map_or_else(|| Toml::file(DEFAULT_PATH), Toml::file_exact);
        Figment::from(file)
            .merge(Env::prefixed(ENV_PREFIX).split("__"))
            .extract()
            .map_err(|error| SettingsError::Load(Box::new(error)))
    }
}
