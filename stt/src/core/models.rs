use std::collections::HashSet;
use std::fmt::Write;
use std::fs::{File, TryLockError};
use std::io::{BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use figment::Figment;
use figment::providers::{Format, Toml};
use futures_util::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use walkdir::WalkDir;

use crate::config::ops::{ModelsVerify, OpsConfig};

const STAMP: &str = ".stamp";
const STAGING: &str = ".staging";
const LOCK: &str = ".lock";
const ARCHIVE: &str = ".archive";
const CHUNK: usize = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("manifest: {0}")]
    Manifest(#[from] Box<figment::Error>),
    #[error("manifest entry {id}: {reason}")]
    Invalid { id: String, reason: &'static str },
    #[error("model {0} is not in the manifest")]
    Unknown(String),
    #[error("model {0} is not installed or its manifest entry changed; run `e-voice pull`")]
    Missing(String),
    #[error("model {id}: {path} does not match its recorded digest")]
    Corrupt { id: String, path: PathBuf },
    #[error("{url}: expected sha256 {expected}, got {actual}")]
    Digest {
        url: String,
        expected: String,
        actual: String,
    },
    #[error("download failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("another pull holds the models lock")]
    Busy,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("blocking task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ModelUnpack {
    #[serde(rename = "tar.bz2")]
    TarBz2,
}

/// One pinned artifact: a plain file stored at `path`, or an archive unpacked into `path`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    pub url: String,
    pub sha256: String,
    pub path: Option<PathBuf>,
    pub unpack: Option<ModelUnpack>,
    #[serde(default)]
    pub strip: usize,
}

impl ModelFile {
    fn target(&self) -> PathBuf {
        match (&self.path, self.unpack) {
            (Some(path), _) => path.clone(),
            (None, Some(_)) => PathBuf::new(),
            (None, None) => PathBuf::from(self.url.rsplit('/').next().unwrap_or_default()),
        }
    }
}

/// Who loads a model: the serving pipeline (`<data>/models`, the volume) or only internal tools and
/// tests (`<data>/ops/models`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelScope {
    #[default]
    Pipeline,
    Ops,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    #[serde(default)]
    pub scope: ModelScope,
    pub files: Vec<ModelFile>,
}

impl Model {
    fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        self.files.iter().for_each(|file| {
            let unpack = file.unpack.map_or("", |_| "tar.bz2");
            let line = format!(
                "{}\n{}\n{}\n{unpack}\n{}\n",
                file.url,
                file.sha256,
                file.target().display(),
                file.strip
            );
            hasher.update(line.as_bytes());
        });
        hex::encode(hasher.finalize())
    }

    fn invalid(&self, reason: &'static str) -> ModelError {
        ModelError::Invalid {
            id: self.id.clone(),
            reason,
        }
    }

    fn check(&self) -> Result<(), ModelError> {
        let id_ok = !self.id.is_empty()
            && !self.id.starts_with('.')
            && self
                .id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c));
        let digest_ok = |file: &ModelFile| {
            file.sha256.len() == 64
                && file
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        };
        let path_ok = |file: &ModelFile| {
            let target = file.target();
            target.components().all(|part| matches!(part, Component::Normal(_)))
                && (file.unpack.is_some() || target.components().next().is_some())
        };
        [
            (id_ok, "id must be lowercase [a-z0-9._-] and not start with a dot"),
            (!self.files.is_empty(), "no files"),
            (
                self.files.iter().all(digest_ok),
                "sha256 must be 64 lowercase hex chars",
            ),
            (
                self.files.iter().all(path_ok),
                "path must be relative and stay inside the model",
            ),
        ]
        .into_iter()
        .find(|(ok, _)| !ok)
        .map_or(Ok(()), |(_, reason)| Err(self.invalid(reason)))
    }
}

/// Every model the service may load, pinned by URL (commit-pinned for Hugging Face) and sha256.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    #[serde(default, rename = "model")]
    pub models: Vec<Model>,
}

impl ModelManifest {
    /// # Errors
    /// Unreadable or malformed manifest, invalid entry, or duplicated id.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let manifest: Self = Figment::from(Toml::file_exact(path)).extract().map_err(Box::new)?;
        manifest.models.iter().try_for_each(Model::check)?;
        let mut seen = HashSet::new();
        manifest
            .models
            .iter()
            .find(|model| !seen.insert(model.id.as_str()))
            .map_or(Ok(manifest.clone()), |model| Err(model.invalid("duplicated id")))
    }
}

/// Model weights under one data root: `<data>/models/<id>/` for the pipeline and
/// `<data>/ops/models/<id>/` for tools, installed atomically with a `.stamp` of file digests.
#[derive(Debug, Clone)]
pub struct ModelStore {
    data: PathBuf,
    manifest: ModelManifest,
    http: reqwest::Client,
}

impl ModelStore {
    // ##### PRIVATE #####

    fn common_dir(&self, model: &Model) -> PathBuf {
        match model.scope {
            ModelScope::Pipeline => self.data.join("models"),
            ModelScope::Ops => self.data.join("ops").join("models"),
        }
        .join(&model.id)
    }

    fn common_digest(path: &Path) -> std::io::Result<String> {
        let mut reader = BufReader::with_capacity(CHUNK, File::open(path)?);
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; CHUNK];
        loop {
            let read = reader.read(&mut buffer)?;
            let Some(chunk) = buffer.get(..read).filter(|chunk| !chunk.is_empty()) else {
                return Ok(hex::encode(hasher.finalize()));
            };
            hasher.update(chunk);
        }
    }

    fn common_stamp(dir: &Path) -> Option<(String, Vec<(String, PathBuf)>)> {
        let mut lines = BufReader::new(File::open(dir.join(STAMP)).ok()?).lines();
        let fingerprint = lines.next()?.ok()?.strip_prefix("fingerprint ")?.to_owned();
        let files = lines
            .map(|line| {
                let line = line.ok()?;
                let (digest, path) = line.split_once("  ")?;
                Some((digest.to_owned(), PathBuf::from(path)))
            })
            .collect::<Option<Vec<_>>>()?;
        Some((fingerprint, files))
    }

    fn select(&self, ids: &[String]) -> Result<Vec<&Model>, ModelError> {
        match ids {
            [] => Ok(self.manifest.models.iter().collect()),
            ids => ids
                .iter()
                .map(|id| {
                    self.manifest
                        .models
                        .iter()
                        .find(|model| &model.id == id)
                        .ok_or_else(|| ModelError::Unknown(id.clone()))
                })
                .collect(),
        }
    }

    fn verify_model(dir: &Path, model: &Model, mode: ModelsVerify) -> Result<(), ModelError> {
        let missing = || ModelError::Missing(model.id.clone());
        let (fingerprint, files) = Self::common_stamp(dir).ok_or_else(missing)?;
        (fingerprint == model.fingerprint()).then_some(()).ok_or_else(missing)?;
        match mode {
            ModelsVerify::Stamp => Ok(()),
            ModelsVerify::Full => files.into_iter().try_for_each(|(digest, path)| {
                let actual = Self::common_digest(&dir.join(&path)).ok();
                (actual.as_deref() == Some(digest.as_str()))
                    .then_some(())
                    .ok_or_else(|| ModelError::Corrupt {
                        id: model.id.clone(),
                        path,
                    })
            }),
        }
    }

    fn pull_lock(&self) -> Result<File, ModelError> {
        std::fs::create_dir_all(self.data.join(STAGING))?;
        let lock = File::create(self.data.join(LOCK))?;
        match lock.try_lock() {
            Ok(()) => Ok(lock),
            Err(TryLockError::WouldBlock) => Err(ModelError::Busy),
            Err(TryLockError::Error(error)) => Err(error.into()),
        }
    }

    async fn pull_fetch(&self, file: &ModelFile, dest: &Path) -> Result<(), ModelError> {
        let response = self.http.get(&file.url).send().await?.error_for_status()?;
        let mut out = tokio::fs::File::create(dest).await?;
        let mut hasher = Sha256::new();
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            hasher.update(&chunk);
            out.write_all(&chunk).await?;
        }
        out.sync_all().await?;
        let actual = hex::encode(hasher.finalize());
        (actual == file.sha256).then_some(()).ok_or_else(|| ModelError::Digest {
            url: file.url.clone(),
            expected: file.sha256.clone(),
            actual,
        })
    }

    fn pull_unpack(archive: &Path, dest: &Path, strip: usize) -> Result<Result<(), &'static str>, std::io::Error> {
        let mut entries = tar::Archive::new(bzip2::read::BzDecoder::new(BufReader::new(File::open(archive)?)));
        for entry in entries.entries()? {
            let mut entry = entry?;
            let relative: PathBuf = entry.path()?.components().skip(strip).collect();
            let safe = relative.components().all(|part| matches!(part, Component::Normal(_)));
            match (entry.header().entry_type(), safe, relative.as_os_str().is_empty()) {
                (_, _, true) => {}
                (_, false, false) => return Ok(Err("archive entry escapes the model directory")),
                (tar::EntryType::Directory, true, false) => std::fs::create_dir_all(dest.join(&relative))?,
                (tar::EntryType::Regular, true, false) => {
                    let target = dest.join(&relative);
                    target.parent().map(std::fs::create_dir_all).transpose()?;
                    entry.unpack(&target)?;
                }
                (_, true, false) => return Ok(Err("archive holds a link or special entry")),
            }
        }
        Ok(Ok(()))
    }

    fn pull_stamp(dir: &Path, fingerprint: &str) -> std::io::Result<()> {
        let mut stamp = format!("fingerprint {fingerprint}\n");
        let files = WalkDir::new(dir).sort_by_file_name().into_iter().filter(|entry| {
            entry
                .as_ref()
                .map_or(true, |entry| entry.file_type().is_file() && entry.file_name() != STAMP)
        });
        for entry in files {
            let entry = entry.map_err(std::io::Error::other)?;
            let relative = entry.path().strip_prefix(dir).map_err(std::io::Error::other)?;
            writeln!(stamp, "{}  {}", Self::common_digest(entry.path())?, relative.display())
                .map_err(std::io::Error::other)?;
        }
        std::fs::write(dir.join(STAMP), stamp)
    }

    async fn pull_model(&self, model: &Model) -> Result<(), ModelError> {
        let staging = tempfile::TempDir::new_in(self.data.join(STAGING))?;
        for file in &model.files {
            let target = staging.path().join(file.target());
            let fetched = match file.unpack {
                Some(_) => staging.path().join(ARCHIVE),
                None => target.clone(),
            };
            fetched.parent().map(std::fs::create_dir_all).transpose()?;
            tracing::info!(model = %model.id, url = %file.url, "model.fetch");
            self.pull_fetch(file, &fetched).await?;
            let strip = file.strip;
            let unpacked = match file.unpack {
                Some(ModelUnpack::TarBz2) => {
                    let archive = fetched.clone();
                    let unpacked =
                        tokio::task::spawn_blocking(move || Self::pull_unpack(&archive, &target, strip)).await??;
                    std::fs::remove_file(&fetched)?;
                    unpacked
                }
                None => Ok(()),
            };
            unpacked.map_err(|reason| model.invalid(reason))?;
        }
        let (dir, fingerprint) = (staging.path().to_path_buf(), model.fingerprint());
        tokio::task::spawn_blocking(move || Self::pull_stamp(&dir, &fingerprint)).await??;
        let installed = self.common_dir(model);
        installed.parent().map(std::fs::create_dir_all).transpose()?;
        match std::fs::remove_dir_all(&installed) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
            Ok(()) | Err(_) => {}
        }
        std::fs::rename(staging.keep(), &installed)?;
        tracing::info!(model = %model.id, "model.installed");
        Ok(())
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// Invalid manifest, or HTTP client construction failure.
    pub fn open(config: &OpsConfig) -> Result<Self, ModelError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(120))
            .user_agent(concat!("e-voice/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            data: config.data.clone(),
            manifest: ModelManifest::load(&config.manifest)?,
            http,
        })
    }

    /// Installed directory of a manifest model; serving code must [`ModelStore::verify`] first.
    ///
    /// # Errors
    /// The id is not in the manifest.
    pub fn dir(&self, id: &str) -> Result<PathBuf, ModelError> {
        self.manifest
            .models
            .iter()
            .find(|model| model.id == id)
            .map(|model| self.common_dir(model))
            .ok_or_else(|| ModelError::Unknown(id.to_owned()))
    }

    /// Checks `ids` (all models when empty) without touching the network.
    ///
    /// # Errors
    /// Unknown id, model missing or stale, or a file whose digest drifted (`Full` only).
    pub async fn verify(&self, ids: &[String], mode: ModelsVerify) -> Result<(), ModelError> {
        let models: Vec<(PathBuf, Model)> = self
            .select(ids)?
            .into_iter()
            .map(|model| (self.common_dir(model), model.clone()))
            .collect();
        tokio::task::spawn_blocking(move || {
            models
                .iter()
                .try_for_each(|(dir, model)| Self::verify_model(dir, model, mode))
        })
        .await?
    }

    /// Installs `ids` (all models when empty) that are missing or stale; already valid models are skipped.
    ///
    /// # Errors
    /// Unknown id, lock held by another pull, network failure, digest mismatch or unsafe archive.
    pub async fn pull(&self, ids: &[String]) -> Result<(), ModelError> {
        let _lock = self.pull_lock()?;
        for model in self.select(ids)? {
            let dir = self.common_dir(model);
            match Self::verify_model(&dir, model, ModelsVerify::Stamp) {
                Ok(()) => tracing::info!(model = %model.id, "model.present"),
                Err(_) => self.pull_model(model).await?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::Path;

    use sha2::{Digest, Sha256};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::config::ops::{ModelsVerify, OpsConfig};
    use crate::core::models::{ModelError, ModelManifest, ModelStore};

    fn sha(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn archive() -> Vec<u8> {
        let mut tar = tar::Builder::new(Vec::new());
        let body = b"tokens";
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, "pack-v1/sub/tokens.txt", &body[..])
            .unwrap();
        let mut bz = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
        bz.write_all(&tar.into_inner().unwrap()).unwrap();
        bz.finish().unwrap()
    }

    fn store(dir: &Path, manifest: &str) -> ModelStore {
        let manifest_path = dir.join("models.toml");
        std::fs::write(&manifest_path, manifest).unwrap();
        ModelStore::open(&OpsConfig {
            data: dir.join("data"),
            manifest: manifest_path,
            verify: ModelsVerify::Full,
        })
        .unwrap()
    }

    async fn served(server: &MockServer, route: &str, body: Vec<u8>, times: u64) {
        Mock::given(method("GET"))
            .and(path(route))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .expect(times)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn test_pull_installs_file_and_archive_then_verifies() {
        let (server, tmp) = (MockServer::start().await, tempfile::tempdir().unwrap());
        let tgz = archive();
        served(&server, "/vad.onnx", b"weights".to_vec(), 1).await;
        served(&server, "/pack.tar.bz2", tgz.clone(), 1).await;
        let manifest = format!(
            "[[model]]\nid = \"vad\"\nfiles = [{{ url = \"{0}/vad.onnx\", sha256 = \"{1}\" }}]\n\
             [[model]]\nid = \"asr\"\nfiles = [{{ url = \"{0}/pack.tar.bz2\", sha256 = \"{2}\", unpack = \"tar.bz2\", strip = 1 }}]\n",
            server.uri(),
            sha(b"weights"),
            sha(&tgz),
        );
        let store = store(tmp.path(), &manifest);
        store.pull(&[]).await.unwrap();
        store.pull(&[]).await.unwrap();
        store.verify(&[], ModelsVerify::Full).await.unwrap();
        assert_eq!(
            std::fs::read(store.dir("vad").unwrap().join("vad.onnx")).unwrap(),
            b"weights"
        );
        assert_eq!(
            std::fs::read(store.dir("asr").unwrap().join("sub/tokens.txt")).unwrap(),
            b"tokens"
        );
        assert!(!store.dir("asr").unwrap().join(".archive").exists());
    }

    #[tokio::test]
    async fn test_pull_rejects_digest_mismatch_without_installing() {
        let (server, tmp) = (MockServer::start().await, tempfile::tempdir().unwrap());
        served(&server, "/vad.onnx", b"tampered".to_vec(), 1).await;
        let manifest = format!(
            "[[model]]\nid = \"vad\"\nfiles = [{{ url = \"{}/vad.onnx\", sha256 = \"{}\" }}]\n",
            server.uri(),
            sha(b"weights"),
        );
        let store = store(tmp.path(), &manifest);
        assert!(matches!(store.pull(&[]).await, Err(ModelError::Digest { .. })));
        assert!(!store.dir("vad").unwrap().exists());
        assert!(matches!(
            store.verify(&[], ModelsVerify::Stamp).await,
            Err(ModelError::Missing(_))
        ));
    }

    #[tokio::test]
    async fn test_verify_full_detects_drift_that_stamp_misses() {
        let (server, tmp) = (MockServer::start().await, tempfile::tempdir().unwrap());
        served(&server, "/vad.onnx", b"weights".to_vec(), 1).await;
        let manifest = format!(
            "[[model]]\nid = \"vad\"\nfiles = [{{ url = \"{}/vad.onnx\", sha256 = \"{}\", path = \"silero.onnx\" }}]\n",
            server.uri(),
            sha(b"weights"),
        );
        let store = store(tmp.path(), &manifest);
        store.pull(&["vad".to_owned()]).await.unwrap();
        std::fs::write(store.dir("vad").unwrap().join("silero.onnx"), b"bitrot!").unwrap();
        store.verify(&[], ModelsVerify::Stamp).await.unwrap();
        let drifted = store.verify(&[], ModelsVerify::Full).await;
        assert!(matches!(drifted, Err(ModelError::Corrupt { path, .. }) if path == Path::new("silero.onnx")));
    }

    #[tokio::test]
    async fn test_pull_refuses_when_lock_is_held() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path(), "");
        std::fs::create_dir_all(tmp.path().join("data")).unwrap();
        let held = std::fs::File::create(tmp.path().join("data/.lock")).unwrap();
        held.lock().unwrap();
        assert!(matches!(store.pull(&[]).await, Err(ModelError::Busy)));
    }

    #[tokio::test]
    async fn test_select_rejects_unknown_id() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store(tmp.path(), "");
        assert!(matches!(
            store.verify(&["nope".to_owned()], ModelsVerify::Stamp).await,
            Err(ModelError::Unknown(_))
        ));
        assert!(matches!(store.dir("nope"), Err(ModelError::Unknown(_))));
    }

    #[test]
    fn test_manifest_rejects_unsafe_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let digest = "a".repeat(64);
        let cases = [
            format!("[[model]]\nid = \"../up\"\nfiles = [{{ url = \"http://x/a\", sha256 = \"{digest}\" }}]"),
            format!(
                "[[model]]\nid = \"m\"\nfiles = [{{ url = \"http://x/a\", sha256 = \"{digest}\", path = \"../a\" }}]"
            ),
            "[[model]]\nid = \"m\"\nfiles = [{ url = \"http://x/a\", sha256 = \"short\" }]".to_owned(),
            "[[model]]\nid = \"m\"\nfiles = []".to_owned(),
            format!(
                "[[model]]\nid = \"m\"\nfiles = [{{ url = \"http://x/a\", sha256 = \"{digest}\" }}]\n\
                 [[model]]\nid = \"m\"\nfiles = [{{ url = \"http://x/b\", sha256 = \"{digest}\" }}]"
            ),
        ];
        for case in cases {
            let manifest = tmp.path().join("models.toml");
            std::fs::write(&manifest, &case).unwrap();
            assert!(
                matches!(ModelManifest::load(&manifest), Err(ModelError::Invalid { .. })),
                "{case}"
            );
        }
    }
}
