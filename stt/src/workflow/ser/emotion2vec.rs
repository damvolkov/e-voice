use std::collections::BTreeMap;
use std::fmt::{self, Debug};
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use ort::session::Session;
use ort::value::TensorRef;
use serde::Deserialize;

use crate::core::runtime::Runtime;
use crate::schema::emotion::{Emotion, EmotionLabel};
use crate::schema::error::{BackendError, NodeError};
use crate::workflow::parts::Parts;
use crate::workflow::ser::base::Ser;

const HEAD: &str = "emotion2vec_head.json";
const MIN: usize = 1_600;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Head {
    labels: Vec<EmotionLabel>,
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl Head {
    fn dim(&self) -> Option<usize> {
        let dim = self.weight.first()?.len();
        let square = self.weight.iter().all(|row| row.len() == dim);
        let aligned = self.labels.len() == self.weight.len() && self.labels.len() == self.bias.len();
        (dim > 0 && square && aligned).then_some(dim)
    }
}

/// emotion2vec+ base: ONNX backbone (waveform normalization folded in), mean-pooled frames,
/// linear head and softmax. ort runs a session through `&mut`, so calls are serialized.
pub struct Emotion2vecSer {
    session: Mutex<Session>,
    head: Head,
    dim: usize,
    model: String,
}

impl Debug for Emotion2vecSer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Emotion2vecSer")
            .field("model", &self.model)
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl Emotion2vecSer {
    // ##### PRIVATE #####

    fn classify_score(&self, features: &[f32]) -> Option<Emotion> {
        let (pooled, count) =
            features
                .chunks_exact(self.dim)
                .fold((vec![0.0f32; self.dim], 0.0f32), |(mut sum, count), frame| {
                    sum.iter_mut().zip(frame).for_each(|(total, value)| *total += value);
                    (sum, count + 1.0)
                });
        let scale = 1.0 / count.max(1.0);
        let logits: Vec<f32> = self
            .head
            .weight
            .iter()
            .zip(&self.head.bias)
            .map(|(row, bias)| bias + scale * row.iter().zip(&pooled).map(|(w, x)| w * x).sum::<f32>())
            .collect();
        let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|logit| (logit - peak).exp()).collect();
        let total: f32 = exp.iter().sum();
        let scores: BTreeMap<EmotionLabel, f32> = self
            .head
            .labels
            .iter()
            .copied()
            .zip(exp.iter().map(|value| value / total))
            .collect();
        let label = scores
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(label, _)| *label)?;
        (count > 0.0).then(|| Emotion {
            label,
            scores,
            model: Some(self.model.clone()),
        })
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// Missing files, a malformed head, or onnxruntime rejecting the backbone.
    pub fn new(dir: &Path, threads: u16, model: &str) -> Result<Self, BackendError> {
        let (backbone, head) = (std::path::PathBuf::from(Parts::onnx(dir, "")?), dir.join(HEAD));
        Runtime::probe().map_err(|_| BackendError::Load("onnxruntime"))?;
        let raw = std::fs::read(&head).map_err(|_| BackendError::Missing(head.clone()))?;
        let head: Head = serde_json::from_slice(&raw).map_err(|_| BackendError::Load("emotion2vec head"))?;
        let dim = head.dim().ok_or(BackendError::Load("emotion2vec head"))?;
        let load = |_| BackendError::Load("emotion2vec backbone");
        let session = Session::builder()
            .map_err(load)?
            .with_intra_threads(usize::from(threads))
            .map_err(|_| BackendError::Load("emotion2vec threads"))?
            .with_intra_op_spinning(false)
            .map_err(|_| BackendError::Load("emotion2vec threads"))?
            .commit_from_file(&backbone)
            .map_err(load)?;
        Ok(Self {
            session: Mutex::new(session),
            head,
            dim,
            model: model.to_owned(),
        })
    }
}

impl Ser for Emotion2vecSer {
    fn classify(&self, audio: &[f32]) -> Result<Emotion, NodeError> {
        let backend = |error: ort::Error| NodeError::Backend(format!("emotion2vec: {error}"));
        let long = audio.len() >= MIN;
        long.then_some(())
            .ok_or_else(|| NodeError::Backend(format!("emotion2vec needs at least {MIN} samples")))?;
        let input = TensorRef::from_array_view(([1usize, audio.len()], audio)).map_err(backend)?;
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session.run(ort::inputs![input]).map_err(backend)?;
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| NodeError::Backend("emotion2vec returned no output".to_owned()))?;
        let (_, features) = output.try_extract_tensor::<f32>().map_err(backend)?;
        self.classify_score(features)
            .ok_or_else(|| NodeError::Backend("emotion2vec returned no frames".to_owned()))
    }
}
