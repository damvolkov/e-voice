use std::path::{Path, PathBuf};

use e_voice_core::schema::lang::Lang;
use serde::Deserialize;

use crate::schema::emotion::EmotionLabel;

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("cannot read manifest: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest line {line}: {error}")]
    Line { line: usize, error: serde_json::Error },
}

/// One evaluation sample: audio path relative to the manifest, plus optional references.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestItem {
    pub id: String,
    pub audio: PathBuf,
    pub lang: Lang,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub emotion: Option<EmotionLabel>,
}

/// JSON Lines of [`ManifestItem`], as written by the `eval` fetchers.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub items: Vec<ManifestItem>,
}

impl Manifest {
    /// Audio paths come back resolved against the manifest's directory.
    ///
    /// # Errors
    /// Unreadable file or a malformed line.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let items = std::fs::read_to_string(path)?
            .lines()
            .enumerate()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(index, text)| {
                let mut item: ManifestItem = serde_json::from_str(text).map_err(|error| ManifestError::Line {
                    line: index.saturating_add(1),
                    error,
                })?;
                item.audio = base.join(&item.audio);
                Ok(item)
            })
            .collect::<Result<_, ManifestError>>()?;
        Ok(Self { items })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use e_voice_core::schema::lang::Lang;

    use crate::ops::bench::manifest::{Manifest, ManifestError};
    use crate::schema::emotion::EmotionLabel;

    #[test]
    fn test_load_resolves_paths_and_optional_references() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.jsonl");
        let lines = "{\"id\":\"a\",\"audio\":\"wav/a.wav\",\"lang\":\"es\",\"text\":\"hola\"}\n\n{\"id\":\"b\",\"audio\":\"b.wav\",\"lang\":\"en\",\"emotion\":\"angry\"}\n";
        std::fs::write(&path, lines).unwrap();
        let manifest = Manifest::load(&path).unwrap();
        assert_eq!(manifest.items.len(), 2);
        assert_eq!(manifest.items[0].audio, dir.path().join(PathBuf::from("wav/a.wav")));
        assert_eq!(
            (manifest.items[0].lang, manifest.items[0].text.as_deref()),
            (Lang::Es, Some("hola"))
        );
        assert_eq!(
            (manifest.items[1].lang, manifest.items[1].emotion),
            (Lang::En, Some(EmotionLabel::Angry))
        );
    }

    #[test]
    fn test_load_reports_the_bad_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"a\",\"audio\":\"a.wav\",\"lang\":\"es\"}\n{\"id\":\"b\",\"lang\":\"fr\"}\n",
        )
        .unwrap();
        assert!(matches!(
            Manifest::load(&path),
            Err(ManifestError::Line { line: 2, .. })
        ));
    }
}
