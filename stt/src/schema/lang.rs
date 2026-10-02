use serde::{Deserialize, Serialize};

/// Spoken language, fixed by configuration per stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    Es,
    En,
}

impl Lang {
    /// English name, as reported by `verbose_json` transcriptions.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Es => "spanish",
            Self::En => "english",
        }
    }

    #[must_use]
    pub const fn locale(self) -> &'static str {
        match self {
            Self::Es => "es-ES",
            Self::En => "en-US",
        }
    }
}

impl std::str::FromStr for Lang {
    type Err = String;

    /// ISO-639-1 code, optionally with a region (`es`, `es-ES`, `en_US`), case-insensitive.
    fn from_str(code: &str) -> Result<Self, Self::Err> {
        let base = code.split(['-', '_']).next().unwrap_or_default().to_ascii_lowercase();
        match base.as_str() {
            "es" => Ok(Self::Es),
            "en" => Ok(Self::En),
            _ => Err(format!("unsupported language {code:?}; supported: es, en")),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::schema::lang::Lang;

    #[test]
    fn test_parse_accepts_codes_and_regions_only_for_supported_languages() {
        assert_eq!("es".parse(), Ok(Lang::Es));
        assert_eq!("EN-us".parse(), Ok(Lang::En));
        assert_eq!("es_ES".parse(), Ok(Lang::Es));
        assert!("fr".parse::<Lang>().is_err());
        assert!("".parse::<Lang>().is_err());
    }
}
