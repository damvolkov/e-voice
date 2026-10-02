use std::fmt::{self, Debug};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sentencepiece::SentencePieceProcessor;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, OnlineStream, OnlineTransducerModelConfig};

use crate::config::ww::WwConfig;
use crate::core::runtime::Runtime;
use crate::schema::audio::{Audio, RATE};
use crate::schema::error::{BackendError, NodeError};
use crate::schema::event::WakeEvent;
use crate::workflow::ww::base::{Ww, WwSession};

const SAMPLE_RATE: i32 = RATE.cast_signed();

/// sherpa-onnx open-vocabulary keyword spotting (Zipformer transducer, GigaSpeech English). Any phrase
/// works without training: it is encoded into the model's BPE pieces at startup and matched with a
/// `boost` (how much the decoder favours it) and a `threshold` (how sure it must be).
pub struct KwsWw {
    spotter: Arc<KeywordSpotter>,
    keyword: String,
    cooldown: u64,
}

impl Debug for KwsWw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KwsWw")
            .field("keyword", &self.keyword)
            .field("cooldown", &self.cooldown)
            .finish_non_exhaustive()
    }
}

impl KwsWw {
    // ##### PRIVATE #####

    fn new_part(dir: &Path, part: &str) -> Result<String, BackendError> {
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|_| BackendError::Missing(dir.to_path_buf()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(part) && name.ends_with(".int8.onnx"))
            })
            .collect();
        found.sort();
        found
            .first()
            .map(|path| path.display().to_string())
            .ok_or_else(|| BackendError::Missing(dir.join(format!("{part}*.int8.onnx"))))
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// The keyword line sherpa expects: BPE pieces of the upper-cased phrase, then boost, threshold and
    /// the reported name, e.g. `▁E AGE R :1.5 #0.25 @eager`.
    ///
    /// # Errors
    /// `bpe.model` is missing or cannot encode the phrase.
    pub fn encode(dir: &Path, config: &WwConfig) -> Result<String, BackendError> {
        let bpe = dir.join("bpe.model");
        let model = SentencePieceProcessor::open(&bpe).map_err(|_| BackendError::Missing(bpe.clone()))?;
        let pieces = model
            .encode(&config.keyword.to_uppercase())
            .map_err(|_| BackendError::Load("kws keyword"))?;
        let tokens: Vec<String> = pieces.into_iter().map(|piece| piece.piece).collect();
        let name = config.keyword.trim().to_lowercase().replace(' ', "_");
        Ok(format!(
            "{} :{} #{} @{name}",
            tokens.join(" "),
            config.boost(),
            config.threshold()
        ))
    }

    /// # Errors
    /// Missing model files, an unencodable keyword, or the runtime rejecting the model.
    pub fn new(dir: &Path, config: &WwConfig) -> Result<Self, BackendError> {
        let mut sherpa = KeywordSpotterConfig::default();
        sherpa.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(Self::new_part(dir, "encoder")?),
            decoder: Some(Self::new_part(dir, "decoder")?),
            joiner: Some(Self::new_part(dir, "joiner")?),
        };
        let tokens = dir.join("tokens.txt");
        tokens
            .is_file()
            .then_some(())
            .ok_or_else(|| BackendError::Missing(tokens.clone()))?;
        sherpa.model_config.tokens = Some(tokens.display().to_string());
        sherpa.model_config.num_threads = i32::from(config.threads);
        sherpa.model_config.provider = Some(Runtime::provider());
        sherpa.keywords_buf = Some(Self::encode(dir, config)?);
        let spotter = KeywordSpotter::create(&sherpa).ok_or(BackendError::Load("kws"))?;
        Ok(Self {
            spotter: Arc::new(spotter),
            keyword: config.keyword.trim().to_lowercase().replace(' ', "_"),
            cooldown: Audio::length(config.cooldown),
        })
    }
}

impl Ww for KwsWw {
    fn open(&self) -> Result<Box<dyn WwSession>, BackendError> {
        Ok(Box::new(KwsWwSession {
            stream: self.spotter.create_stream(),
            spotter: Arc::clone(&self.spotter),
            keyword: self.keyword.clone(),
            cooldown: self.cooldown,
            pos: 0,
            quiet_until: 0,
        }))
    }
}

pub struct KwsWwSession {
    spotter: Arc<KeywordSpotter>,
    stream: OnlineStream,
    keyword: String,
    cooldown: u64,
    pos: u64,
    quiet_until: u64,
}

impl Debug for KwsWwSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KwsWwSession")
            .field("keyword", &self.keyword)
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

impl WwSession for KwsWwSession {
    fn push(&mut self, audio: &[f32]) -> Result<Option<WakeEvent>, NodeError> {
        self.stream.accept_waveform(SAMPLE_RATE, audio);
        self.pos = self.pos.saturating_add(audio.len() as u64);
        let mut detection = None;
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
            let spotted = self
                .spotter
                .get_result(&self.stream)
                .filter(|result| !result.keyword.is_empty());
            if spotted.is_some() {
                self.spotter.reset(&self.stream);
                let fresh = self.pos >= self.quiet_until;
                if fresh && detection.is_none() {
                    self.quiet_until = self.pos.saturating_add(self.cooldown);
                    detection = Some(WakeEvent {
                        keyword: self.keyword.clone(),
                        score: 1.0,
                    });
                }
            }
        }
        Ok(detection)
    }
}
