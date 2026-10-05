use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{self, Debug};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;

use crate::schema::audio::Audio;
use crate::schema::error::TrainError;
use crate::schema::voice::VoiceState;
use crate::workflow::synth::base::{Synth, SynthCaps, SynthSession};
use crate::workflow::synth::pool::{Lease, Pool};
use crate::workflow::synth::prompt::Prompt;

const RATE: u32 = 24_000;
const EOS_THRESHOLD: f32 = -4.0;
const MIN_FRAMES: usize = 6;
const FADE: usize = 120;
const PROMPTS: usize = 64;
const STRIPPED: [char; 12] = ['"', '“', '”', '„', '«', '»', '(', ')', '[', ']', '¡', '¿'];
const CLOSERS: [char; 7] = ['"', '\'', '”', '’', ')', ']', '»'];
const ENDS: [char; 4] = ['.', '!', '?', '…'];
const SOFT_ENDS: [char; 6] = [',', ';', ':', '-', '–', '—'];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum StateDtype {
    Float32,
    Int64,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum StateFill {
    Zeros,
    Ones,
    Nan,
    Empty,
    Mixed,
}

/// One streaming state tensor of a graph, as `bundle.json` declares it.
#[derive(Debug, Clone, Deserialize)]
struct StateEntry {
    input_name: String,
    output_name: String,
    dtype: StateDtype,
    shape: Vec<i64>,
    fill: StateFill,
}

/// `bundle.json` written by the exporter (schema 2).
#[derive(Debug, Clone, Deserialize)]
struct BundleMeta {
    sample_rate: u32,
    samples_per_frame: usize,
    frame_rate: f32,
    latent_dim: usize,
    insert_bos_before_voice: bool,
    bos_before_voice_file: Option<String>,
    tokenizer_file: String,
    model_recommended_frames_after_eos: Option<usize>,
    max_token_per_chunk: usize,
    remove_semicolons: bool,
    pad_with_spaces_for_short_inputs: bool,
    flow_lm_state_manifest: Vec<StateEntry>,
    mimi_state_manifest: Vec<StateEntry>,
}

/// Generation knobs shared by every stream of a backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PocketOptions {
    pub threads: u16,
    pub workers: usize,
    pub temperature: f32,
    pub steps: u16,
    pub quantized: bool,
}

/// The five graphs of one model, owned by a single stream at a time.
struct Worker {
    text: Session,
    encoder: Session,
    main: Session,
    flow: Session,
    decoder: Session,
}

/// One language's model: metadata, tokenizer and its worker pool.
struct Bundle {
    model: String,
    meta: BundleMeta,
    tokenizer: SentencePieceProcessor,
    bos: Option<Vec<f32>>,
    pool: Arc<Pool<Worker>>,
}

/// Host copy of a state tensor, so a voice prefill can seed every sentence.
#[derive(Debug, Clone)]
enum HostTensor {
    F32(Vec<i64>, Vec<f32>),
    I64(Vec<i64>, Vec<i64>),
    Bool(Vec<i64>, Vec<bool>),
}

/// Kyutai Pocket TTS on our own autoregressive loop over the exported ONNX bundle: a causal
/// transformer proposes one Mimi latent per 80 ms frame, a one-step flow head samples it, and the
/// stateful Mimi decoder turns it into audio at once, so every frame is a chunk.
pub struct PocketSynth {
    bundles: BTreeMap<LangKey, Arc<Bundle>>,
    options: PocketOptions,
    prompts: Prompts,
}

/// Encoded voice prompts per (prompt hash, model).
type Prompts = Mutex<HashMap<(u64, LangKey), Arc<Vec<f32>>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct LangKey(u8);

struct PocketSession {
    bundle: Arc<Bundle>,
    worker: Lease<Worker>,
    voice: Vec<HostTensor>,
    options: PocketOptions,
    rng: StdRng,
}

/// One sentence in flight: its token chunks, and the chunk being generated frame by frame.
struct PocketStream<'a> {
    session: &'a mut PocketSession,
    chunks: VecDeque<(Vec<i64>, usize)>,
    current: Option<Generation>,
    first: bool,
}

struct Generation {
    flow: Vec<DynValue>,
    mimi: Vec<DynValue>,
    latent: Vec<f32>,
    step: usize,
    budget: usize,
    after: usize,
    eos: Option<usize>,
}

impl Debug for PocketSynth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let models: Vec<&str> = self.bundles.values().map(|bundle| bundle.model.as_str()).collect();
        f.debug_struct("PocketSynth")
            .field("models", &models)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl LangKey {
    const fn of(lang: Lang) -> Self {
        match lang {
            Lang::Es => Self(0),
            Lang::En => Self(1),
        }
    }
}

impl HostTensor {
    fn fresh(entry: &StateEntry) -> Self {
        let count = usize::try_from(entry.shape.iter().product::<i64>()).unwrap_or(0);
        let shape = entry.shape.clone();
        match (entry.dtype, entry.fill) {
            (StateDtype::Float32, StateFill::Nan) => Self::F32(shape, vec![f32::NAN; count]),
            (StateDtype::Float32, StateFill::Ones) => Self::F32(shape, vec![1.0; count]),
            (StateDtype::Float32, _) => Self::F32(shape, vec![0.0; count]),
            (StateDtype::Int64, StateFill::Ones) => Self::I64(shape, vec![1; count]),
            (StateDtype::Int64, _) => Self::I64(shape, vec![0; count]),
            (StateDtype::Bool, StateFill::Ones) => Self::Bool(shape, vec![true; count]),
            (StateDtype::Bool, _) => Self::Bool(shape, vec![false; count]),
        }
    }

    fn copy(value: &DynValue, entry: &StateEntry) -> Result<Self, ort::Error> {
        let shape = |dims: &ort::value::Shape| dims.iter().copied().collect::<Vec<i64>>();
        Ok(match entry.dtype {
            StateDtype::Float32 => value
                .try_extract_tensor::<f32>()
                .map(|(dims, data)| Self::F32(shape(dims), data.to_vec()))?,
            StateDtype::Int64 => value
                .try_extract_tensor::<i64>()
                .map(|(dims, data)| Self::I64(shape(dims), data.to_vec()))?,
            StateDtype::Bool => value
                .try_extract_tensor::<bool>()
                .map(|(dims, data)| Self::Bool(shape(dims), data.to_vec()))?,
        })
    }

    fn value(&self) -> Result<DynValue, ort::Error> {
        Ok(match self.clone() {
            Self::F32(shape, data) => Tensor::from_array((shape, data))?.into_dyn(),
            Self::I64(shape, data) => Tensor::from_array((shape, data))?.into_dyn(),
            Self::Bool(shape, data) => Tensor::from_array((shape, data))?.into_dyn(),
        })
    }
}

impl Worker {
    fn open(dir: &Path, threads: u16, quantized: bool) -> Result<Self, BackendError> {
        let graph = |name: &'static str, quantize: bool| {
            let file = dir.join(if quantize {
                format!("{name}_int8.onnx")
            } else {
                format!("{name}.onnx")
            });
            file.exists()
                .then_some(())
                .ok_or_else(|| BackendError::Missing(file.clone()))?;
            Session::builder()
                .map_err(|_| BackendError::Load(name))?
                .with_intra_threads(usize::from(threads))
                .map_err(|_| BackendError::Load(name))?
                .with_inter_threads(1)
                .map_err(|_| BackendError::Load(name))?
                .with_intra_op_spinning(false)
                .map_err(|_| BackendError::Load(name))?
                .commit_from_file(&file)
                .map_err(|_| BackendError::Load(name))
        };
        Ok(Self {
            text: graph("text_conditioner", false)?,
            encoder: graph("mimi_encoder", false)?,
            main: graph("flow_lm_main", quantized)?,
            flow: graph("flow_lm_flow", quantized)?,
            decoder: graph("mimi_decoder", quantized)?,
        })
    }
}

impl Bundle {
    fn open(model: &str, dir: &Path, options: PocketOptions) -> Result<Self, BackendError> {
        let path = dir.join("bundle.json");
        let raw = std::fs::read(&path).map_err(|_| BackendError::Missing(path.clone()))?;
        let meta: BundleMeta = serde_json::from_slice(&raw).map_err(|_| BackendError::Load("pocket bundle.json"))?;
        (meta.sample_rate == RATE)
            .then_some(())
            .ok_or(BackendError::Load("pocket sample rate"))?;
        let tokenizer_path = dir.join(&meta.tokenizer_file);
        let tokenizer =
            SentencePieceProcessor::open(&tokenizer_path).map_err(|_| BackendError::Missing(tokenizer_path.clone()))?;
        let bos = match (meta.insert_bos_before_voice, meta.bos_before_voice_file.as_deref()) {
            (true, Some(file)) => Some(Self::open_bos(&dir.join(file))?),
            _ => None,
        };
        let workers = (0..options.workers.max(1))
            .map(|_| Worker::open(dir, options.threads, options.quantized))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            model: model.to_owned(),
            meta,
            tokenizer,
            bos,
            pool: Pool::new(workers),
        })
    }

    fn open_bos(path: &Path) -> Result<Vec<f32>, BackendError> {
        let bytes = std::fs::read(path).map_err(|_| BackendError::Missing(path.to_path_buf()))?;
        npyz::NpyFile::new(&bytes[..])
            .and_then(npyz::NpyFile::into_vec::<f32>)
            .map_err(|_| BackendError::Load("pocket bos_before_voice"))
    }

    /// Upstream `prepare_text`: normalized punctuation, a capital first letter, a terminal mark.
    fn prepare(&self, text: &str) -> String {
        let cleaned: String = text
            .trim()
            .chars()
            .filter(|c| !STRIPPED.contains(c))
            .map(|c| match c {
                '’' | '‘' => '\'',
                ';' if self.meta.remove_semicolons => ',',
                _ => c,
            })
            .collect();
        let mut words = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        while let Some(at) = words
            .char_indices()
            .zip(words.chars().skip(1))
            .find(|((_, c), next)| ENDS.contains(c) && [',', ';', ':'].contains(next))
            .map(|((at, c), _)| at.saturating_add(c.len_utf8()))
        {
            words.remove(at);
        }
        let mut chars = words.chars();
        let mut text: String = chars
            .next()
            .map(|first| first.to_uppercase().chain(chars).collect())
            .unwrap_or_default();
        let trimmed = text.trim_end_matches(CLOSERS).len();
        text.truncate(trimmed);
        let last = text.chars().last();
        match last {
            Some(c) if ENDS.contains(&c) => {}
            Some(c) if SOFT_ENDS.contains(&c) => {
                text.pop();
                text.push('.');
            }
            _ => text.push('.'),
        }
        let short = text.split_whitespace().count() < 5 && self.meta.pad_with_spaces_for_short_inputs;
        if short { format!("        {text}") } else { text }
    }

    fn tokens(&self, text: &str) -> Vec<i64> {
        self.tokenizer
            .encode(text)
            .map(|pieces| pieces.iter().map(|piece| i64::from(piece.id)).collect())
            .unwrap_or_default()
    }

    /// Prepared token chunks of at most `max_token_per_chunk`, split between words, each with its
    /// word count (which sets how long generation runs past end-of-speech).
    fn chunks(&self, sentence: &str) -> VecDeque<(Vec<i64>, usize)> {
        let limit = self.meta.max_token_per_chunk.max(1);
        let mut chunks = VecDeque::new();
        let mut words: Vec<&str> = Vec::new();
        for word in sentence.split_whitespace() {
            let candidate = words.iter().copied().chain([word]).collect::<Vec<_>>().join(" ");
            let fits = words.is_empty() || self.tokens(&self.prepare(&candidate)).len() <= limit;
            if !fits {
                let text = self.prepare(&words.join(" "));
                chunks.push_back((self.tokens(&text), words.len()));
                words.clear();
            }
            words.push(word);
        }
        if !words.is_empty() {
            let text = self.prepare(&words.join(" "));
            chunks.push_back((self.tokens(&text), words.len()));
        }
        chunks
    }

    fn fresh(entries: &[StateEntry]) -> Vec<HostTensor> {
        entries.iter().map(HostTensor::fresh).collect()
    }

    /// Runs the backbone once; returns `(conditioning, eos_logit, state)`.
    fn main(
        &self,
        worker: &mut Worker,
        sequence: DynValue,
        text: DynValue,
        state: Vec<DynValue>,
    ) -> Result<(Vec<f32>, f32, Vec<DynValue>), ort::Error> {
        let manifest = &self.meta.flow_lm_state_manifest;
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            ("sequence".into(), sequence.into()),
            ("text_embeddings".into(), text.into()),
        ];
        inputs.extend(
            manifest
                .iter()
                .zip(state)
                .map(|(entry, value)| (Cow::Borrowed(entry.input_name.as_str()), value.into())),
        );
        let mut outputs = worker.main.run(inputs)?;
        let conditioning = outputs
            .get("conditioning")
            .map(|value| value.try_extract_tensor::<f32>().map(|(_, data)| data.to_vec()))
            .transpose()?
            .unwrap_or_default();
        let eos = outputs
            .get("eos_logit")
            .map(|value| value.try_extract_tensor::<f32>().map(|(_, data)| data.first().copied()))
            .transpose()?
            .flatten()
            .unwrap_or(f32::NEG_INFINITY);
        let state = manifest
            .iter()
            .map(|entry| outputs.remove(&entry.output_name))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ort::Error::new("flow_lm_main dropped a state output"))?;
        Ok((conditioning, eos, state))
    }

    /// Prefills the backbone with `latents` ([T, 1024]): the voice prompt.
    fn prefill(&self, worker: &mut Worker, latents: &[f32]) -> Result<Vec<HostTensor>, ort::Error> {
        let manifest = &self.meta.flow_lm_state_manifest;
        let state = Self::fresh(manifest)
            .iter()
            .map(HostTensor::value)
            .collect::<Result<Vec<_>, _>>()?;
        let frames = i64::try_from(latents.len() / 1024).unwrap_or(0);
        let sequence = Tensor::from_array((
            vec![1_i64, 0, i64::try_from(self.meta.latent_dim).unwrap_or(0)],
            Vec::<f32>::new(),
        ))?;
        let text = Tensor::from_array((vec![1_i64, frames, 1024], latents.to_vec()))?;
        let (_, _, state) = self.main(worker, sequence.into_dyn(), text.into_dyn(), state)?;
        manifest
            .iter()
            .zip(&state)
            .map(|(entry, value)| HostTensor::copy(value, entry))
            .collect()
    }

    /// Mimi-encodes a 24 kHz clip into the voice latents, BOS prepended when the graph lacks it.
    fn encode(&self, worker: &mut Worker, audio: &[f32]) -> Result<Vec<f32>, ort::Error> {
        let input = Tensor::from_array((vec![1_i64, 1, i64::try_from(audio.len()).unwrap_or(0)], audio.to_vec()))?;
        let outputs = worker.encoder.run(ort::inputs!["audio" => input])?;
        let (shape, latents) = outputs
            .get("latents")
            .ok_or_else(|| ort::Error::new("mimi_encoder returned no latents"))?
            .try_extract_tensor::<f32>()?;
        let frames = audio.len().div_ceil(self.meta.samples_per_frame);
        let embedded = shape.get(1).copied() == i64::try_from(frames.saturating_add(1)).ok();
        Ok(match (&self.bos, embedded) {
            (Some(bos), false) => bos.iter().chain(latents).copied().collect(),
            _ => latents.to_vec(),
        })
    }
}

impl PocketSynth {
    // ##### PRIVATE #####

    /// A small count as f32 (frame sizes, token counts): exact below 2^16.
    fn common_count(count: usize) -> f32 {
        f32::from(u16::try_from(count).unwrap_or(u16::MAX))
    }

    /// Mimi latents of a prompt for one model, encoded once and kept (up to [`PROMPTS`] entries).
    fn step_latents(
        &self,
        bundle: &Bundle,
        worker: &mut Worker,
        key: (u64, LangKey),
        audio: &[f32],
    ) -> Result<Arc<Vec<f32>>, BackendError> {
        let cached = self
            .prompts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .cloned();
        cached.map_or_else(
            || {
                let latents = Arc::new(
                    bundle
                        .encode(worker, audio)
                        .map_err(|_| BackendError::Load("pocket voice encoding"))?,
                );
                let mut prompts = self.prompts.lock().unwrap_or_else(PoisonError::into_inner);
                if prompts.len() >= PROMPTS {
                    prompts.clear();
                }
                prompts.insert(key, Arc::clone(&latents));
                Ok(latents)
            },
            Ok,
        )
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// `models` maps each served language to its model id and installed bundle directory.
    ///
    /// # Errors
    /// Missing bundle files, a malformed `bundle.json`, or onnxruntime rejecting a graph.
    pub fn new(models: &[(Lang, String, &Path)], options: PocketOptions) -> Result<Self, BackendError> {
        Runtime::probe().map_err(|_| BackendError::Load("onnxruntime"))?;
        let bundles = models
            .iter()
            .map(|(lang, model, dir)| Ok((LangKey::of(*lang), Arc::new(Bundle::open(model, dir, options)?))))
            .collect::<Result<BTreeMap<_, _>, BackendError>>()?;
        Ok(Self {
            bundles,
            options,
            prompts: Mutex::new(HashMap::new()),
        })
    }
}

impl Synth for PocketSynth {
    fn caps(&self) -> SynthCaps {
        SynthCaps {
            rate: RATE,
            langs: &Lang::ALL,
        }
    }

    fn open(&self, lang: Lang, voice: Option<&VoiceState>) -> Result<Box<dyn SynthSession>, BackendError> {
        let bundle = self
            .bundles
            .get(&LangKey::of(lang))
            .ok_or(BackendError::Load("pocket language"))?;
        let mut worker = bundle.pool.lease().ok_or(BackendError::Busy("pocket"))?;
        let prompt = voice.map(Prompt::decode).transpose()?;
        let voice = match prompt {
            Some(prompt) => {
                let key = (prompt.key(), LangKey::of(lang));
                let latents = self.step_latents(bundle, &mut worker, key, &prompt.audio)?;
                bundle
                    .prefill(&mut worker, &latents)
                    .map_err(|_| BackendError::Load("pocket voice prefill"))?
            }
            None => {
                return Err(BackendError::Load(
                    "pocket voice: Pocket needs one (learn it with `voice add`)",
                ));
            }
        };
        Ok(Box::new(PocketSession {
            bundle: Arc::clone(bundle),
            worker,
            voice,
            options: self.options,
            rng: StdRng::from_rng(&mut rand::rng()),
        }))
    }

    fn train(&self, clips: &[Audio], text: Option<&str>) -> Result<VoiceState, TrainError> {
        Prompt::prepare(clips, text)?.encode()
    }
}

impl PocketSession {
    fn sample(&mut self, dim: usize) -> Vec<f32> {
        let scale = self.options.temperature.max(0.0).sqrt();
        if scale > 0.0 {
            (0..dim)
                .map(|_| {
                    let draw: f32 = StandardNormal.sample(&mut self.rng);
                    scale * draw
                })
                .collect()
        } else {
            vec![0.0; dim]
        }
    }
}

impl SynthSession for PocketSession {
    fn speak<'a>(&'a mut self, sentence: &'a str) -> Box<dyn Iterator<Item = Result<Audio, NodeError>> + Send + 'a> {
        let chunks = self.bundle.chunks(sentence);
        Box::new(PocketStream {
            session: self,
            chunks,
            current: None,
            first: true,
        })
    }
}

impl PocketStream<'_> {
    fn start(&mut self, tokens: &[i64], words: usize) -> Result<Generation, ort::Error> {
        let session = &mut *self.session;
        let bundle = Arc::clone(&session.bundle);
        let meta = &bundle.meta;
        let ids = Tensor::from_array((vec![1_i64, i64::try_from(tokens.len()).unwrap_or(0)], tokens.to_vec()))?;
        let embeddings = session
            .worker
            .text
            .run(ort::inputs!["token_ids" => ids])?
            .remove("embeddings")
            .ok_or_else(|| ort::Error::new("text_conditioner returned no embeddings"))?;
        let state = session
            .voice
            .iter()
            .map(HostTensor::value)
            .collect::<Result<Vec<_>, _>>()?;
        let empty = Tensor::from_array((
            vec![1_i64, 0, i64::try_from(meta.latent_dim).unwrap_or(0)],
            Vec::<f32>::new(),
        ))?;
        let (_, _, flow) = bundle.main(&mut session.worker, empty.into_dyn(), embeddings, state)?;
        let mimi = Bundle::fresh(&meta.mimi_state_manifest)
            .iter()
            .map(HostTensor::value)
            .collect::<Result<Vec<_>, _>>()?;
        let seconds = PocketSynth::common_count(tokens.len()) / 3.0 + 2.0;
        // Upstream's budget: 3 tokens per second plus 2 s, in frames; small and positive.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let budget = (seconds * meta.frame_rate).ceil() as usize;
        Ok(Generation {
            flow,
            mimi,
            latent: vec![f32::NAN; meta.latent_dim],
            step: 0,
            budget,
            after: meta
                .model_recommended_frames_after_eos
                .unwrap_or(if words <= 4 { 5 } else { 3 }),
            eos: None,
        })
    }

    /// One frame: backbone step, flow sample, decoder. `None` once the chunk is finished.
    fn frame(&mut self, mut generation: Generation) -> Result<Option<(Generation, Vec<f32>)>, ort::Error> {
        let session = &mut *self.session;
        let bundle = Arc::clone(&session.bundle);
        let meta = &bundle.meta;
        let dim = i64::try_from(meta.latent_dim).unwrap_or(0);
        if generation.step >= generation.budget {
            return Ok(None);
        }
        let sequence = Tensor::from_array((vec![1_i64, 1, dim], std::mem::take(&mut generation.latent)))?;
        let empty = Tensor::from_array((vec![1_i64, 0, 1024], Vec::<f32>::new()))?;
        let (conditioning, eos, flow) = bundle.main(
            &mut session.worker,
            sequence.into_dyn(),
            empty.into_dyn(),
            std::mem::take(&mut generation.flow),
        )?;
        generation.flow = flow;
        let ended = generation.eos.is_none() && eos > EOS_THRESHOLD && generation.step >= MIN_FRAMES;
        generation.eos = ended.then_some(generation.step).or(generation.eos);
        if generation
            .eos
            .is_some_and(|at| generation.step >= at.saturating_add(generation.after))
        {
            return Ok(None);
        }
        let mut x = session.sample(meta.latent_dim);
        let steps = session.options.steps.max(1);
        for index in 0..steps {
            let (s, t) = (
                f32::from(index) / f32::from(steps),
                f32::from(index.saturating_add(1)) / f32::from(steps),
            );
            let inputs = ort::inputs![
                "c" => Tensor::from_array((vec![1_i64, 1024], conditioning.clone()))?,
                "s" => Tensor::from_array((vec![1_i64, 1], vec![s]))?,
                "t" => Tensor::from_array((vec![1_i64, 1], vec![t]))?,
                "x" => Tensor::from_array((vec![1_i64, dim], x.clone()))?,
            ];
            let outputs = session.worker.flow.run(inputs)?;
            let (_, direction) = outputs
                .get("flow_dir")
                .ok_or_else(|| ort::Error::new("flow_lm_flow returned no direction"))?
                .try_extract_tensor::<f32>()?;
            x.iter_mut()
                .zip(direction)
                .for_each(|(value, delta)| *value += delta / f32::from(steps));
        }
        let manifest = &meta.mimi_state_manifest;
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![(
            "latent".into(),
            Tensor::from_array((vec![1_i64, 1, dim], x.clone()))?.into_dyn().into(),
        )];
        inputs.extend(
            manifest
                .iter()
                .zip(std::mem::take(&mut generation.mimi))
                .map(|(entry, value)| (Cow::Borrowed(entry.input_name.as_str()), value.into())),
        );
        let mut outputs = session.worker.decoder.run(inputs)?;
        let pcm = outputs
            .get("audio_frame")
            .ok_or_else(|| ort::Error::new("mimi_decoder returned no audio"))?
            .try_extract_tensor::<f32>()?
            .1
            .to_vec();
        generation.mimi = manifest
            .iter()
            .map(|entry| outputs.remove(&entry.output_name))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ort::Error::new("mimi_decoder dropped a state output"))?;
        generation.latent = x;
        generation.step = generation.step.saturating_add(1);
        Ok(Some((generation, pcm)))
    }
}

impl Iterator for PocketStream<'_> {
    type Item = Result<Audio, NodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let fail = |error: ort::Error| NodeError::Backend(format!("pocket: {error}"));
        loop {
            let generation = match self.current.take() {
                Some(generation) => generation,
                None => {
                    let (tokens, words) = self.chunks.pop_front()?;
                    match self.start(&tokens, words) {
                        Ok(generation) => generation,
                        Err(error) => return Some(Err(fail(error))),
                    }
                }
            };
            match self.frame(generation) {
                Ok(Some((generation, mut pcm))) => {
                    self.current = Some(generation);
                    if std::mem::take(&mut self.first) {
                        pcm.iter_mut().take(FADE).enumerate().for_each(|(index, sample)| {
                            *sample *= PocketSynth::common_count(index) / PocketSynth::common_count(FADE);
                        });
                    }
                    return Some(Ok(Audio::from(pcm)));
                }
                Ok(None) => {}
                Err(error) => return Some(Err(fail(error))),
            }
        }
    }
}
