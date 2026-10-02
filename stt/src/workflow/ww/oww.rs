use std::collections::VecDeque;
use std::fmt::{self, Debug};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use ort::session::Session;
use ort::value::TensorRef;

use crate::config::ww::WwConfig;
use crate::core::runtime::Runtime;
use crate::schema::audio::Audio;
use crate::schema::error::{BackendError, NodeError};
use crate::schema::event::WakeEvent;
use crate::workflow::ww::base::{Ww, WwSession};

const MELSPEC: &str = "melspectrogram.onnx";
const EMBEDDING: &str = "embedding_model.onnx";
const CHUNK: usize = 1_280;
const CONTEXT: usize = 480;
const BANDS: usize = 32;
const WINDOW: usize = 76;
const FEATURES: usize = 96;
const HISTORY: usize = 16;
const SCALE: f32 = 32_767.0;

/// Three ONNX graphs shared by every stream; ort runs a session through `&mut`, so each is locked.
struct OwwModels {
    melspec: Mutex<Session>,
    embedding: Mutex<Session>,
    classifier: Mutex<Session>,
}

impl OwwModels {
    fn run(session: &Mutex<Session>, shape: [usize; 4], input: &[f32]) -> Result<Vec<f32>, NodeError> {
        let backend = |error: ort::Error| NodeError::Backend(format!("openwakeword: {error}"));
        let dims: Vec<usize> = shape.into_iter().filter(|dim| *dim > 0).collect();
        let tensor = TensorRef::from_array_view((dims, input)).map_err(backend)?;
        let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session.run(ort::inputs![tensor]).map_err(backend)?;
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| NodeError::Backend("openwakeword returned no output".to_owned()))?;
        Ok(output.try_extract_tensor::<f32>().map_err(backend)?.1.to_vec())
    }
}

/// openWakeWord pipeline: int16-scaled audio in 80 ms chunks → melspectrogram (`x/10 + 2`) →
/// 96-d embedding over the last 76 mel frames → keyword score over the last 16 embeddings.
/// Unlike the reference, history starts empty instead of random noise, so scores are deterministic
/// and none is produced until 16 embeddings exist.
pub struct OwwWw {
    models: Arc<OwwModels>,
    keyword: String,
    threshold: f32,
    cooldown: usize,
}

impl Debug for OwwWw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwwWw")
            .field("keyword", &self.keyword)
            .field("threshold", &self.threshold)
            .field("cooldown", &self.cooldown)
            .finish_non_exhaustive()
    }
}

impl OwwWw {
    /// # Errors
    /// Missing files or onnxruntime rejecting a graph.
    pub fn new(dir: &Path, config: &WwConfig) -> Result<Self, BackendError> {
        Runtime::probe().map_err(|_| BackendError::Load("onnxruntime"))?;
        let load = |name: &str| {
            let path = dir.join(name);
            path.is_file()
                .then_some(())
                .ok_or_else(|| BackendError::Missing(path.clone()))?;
            Session::builder()
                .map_err(|_| BackendError::Load("openwakeword session"))?
                .with_intra_threads(usize::from(config.threads))
                .map_err(|_| BackendError::Load("openwakeword threads"))?
                .with_intra_op_spinning(false)
                .map_err(|_| BackendError::Load("openwakeword threads"))?
                .commit_from_file(&path)
                .map_err(|_| BackendError::Load("openwakeword graph"))
        };
        let models = OwwModels {
            melspec: Mutex::new(load(MELSPEC)?),
            embedding: Mutex::new(load(EMBEDDING)?),
            classifier: Mutex::new(load(&format!("{}_v0.1.onnx", config.keyword))?),
        };
        let chunks = Audio::length(config.cooldown).checked_div(CHUNK as u64).unwrap_or(0);
        Ok(Self {
            models: Arc::new(models),
            keyword: config.keyword.clone(),
            threshold: config.threshold(),
            cooldown: usize::try_from(chunks).unwrap_or(usize::MAX),
        })
    }
}

impl Ww for OwwWw {
    fn open(&self) -> Result<Box<dyn WwSession>, BackendError> {
        Ok(Box::new(OwwWwSession {
            models: Arc::clone(&self.models),
            keyword: self.keyword.clone(),
            threshold: self.threshold,
            cooldown: self.cooldown,
            pending: Vec::with_capacity(CHUNK),
            raw: VecDeque::with_capacity(CHUNK + CONTEXT),
            mel: std::iter::repeat_n([1.0; BANDS], WINDOW).collect(),
            features: VecDeque::with_capacity(HISTORY),
            quiet: 0,
        }))
    }
}

pub struct OwwWwSession {
    models: Arc<OwwModels>,
    keyword: String,
    threshold: f32,
    cooldown: usize,
    pending: Vec<f32>,
    raw: VecDeque<f32>,
    mel: VecDeque<[f32; BANDS]>,
    features: VecDeque<Vec<f32>>,
    quiet: usize,
}

impl Debug for OwwWwSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwwWwSession")
            .field("keyword", &self.keyword)
            .field("quiet", &self.quiet)
            .finish_non_exhaustive()
    }
}

impl OwwWwSession {
    // ##### PRIVATE #####

    fn push_chunk(&mut self, chunk: &[f32]) -> Result<Option<f32>, NodeError> {
        self.raw.extend(
            chunk
                .iter()
                .map(|sample| (sample * SCALE).round().clamp(-SCALE - 1.0, SCALE)),
        );
        self.raw.drain(..self.raw.len().saturating_sub(CHUNK + CONTEXT));
        let audio: Vec<f32> = self.raw.iter().copied().collect();
        let spectrum = OwwModels::run(&self.models.melspec, [1, audio.len(), 0, 0], &audio)?;
        let frames = spectrum.chunks_exact(BANDS).map(|frame| {
            let mut bands = [0.0; BANDS];
            bands
                .iter_mut()
                .zip(frame)
                .for_each(|(band, value)| *band = value / 10.0 + 2.0);
            bands
        });
        self.mel.extend(frames);
        self.mel.drain(..self.mel.len().saturating_sub(WINDOW));
        let window: Vec<f32> = self.mel.iter().flatten().copied().collect();
        let embedding = OwwModels::run(&self.models.embedding, [1, WINDOW, BANDS, 1], &window)?;
        self.features.push_back(embedding);
        self.features.drain(..self.features.len().saturating_sub(HISTORY));
        let full = self.features.len() == HISTORY && self.features.iter().all(|feature| feature.len() == FEATURES);
        let history: Vec<f32> = self.features.iter().flatten().copied().collect();
        full.then(|| OwwModels::run(&self.models.classifier, [1, HISTORY, FEATURES, 0], &history))
            .transpose()
            .map(|scores| scores.and_then(|scores| scores.first().copied()))
    }

    fn push_detect(&mut self, score: Option<f32>) -> Option<WakeEvent> {
        let fired = self.quiet == 0 && score.is_some_and(|score| score >= self.threshold);
        self.quiet = if fired {
            self.cooldown
        } else {
            self.quiet.saturating_sub(1)
        };
        fired.then(|| WakeEvent {
            keyword: self.keyword.clone(),
            score: score.unwrap_or_default(),
        })
    }
}

impl WwSession for OwwWwSession {
    fn push(&mut self, audio: &[f32]) -> Result<Option<WakeEvent>, NodeError> {
        let mut pending = std::mem::take(&mut self.pending);
        pending.extend_from_slice(audio);
        let chunks = pending.chunks_exact(CHUNK);
        let rest = chunks.remainder().len();
        let mut detection = None;
        for chunk in chunks {
            let score = self.push_chunk(chunk)?;
            let wake = self.push_detect(score);
            detection = detection.or(wake);
        }
        pending.drain(..pending.len().saturating_sub(rest));
        self.pending = pending;
        Ok(detection)
    }
}
