use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use realfft::RealFftPlanner;
use tokenizers::Tokenizer;

use crate::schema::audio::Audio;
use crate::schema::error::TrainError;
use crate::schema::voice::VoiceState;
use crate::workflow::synth::base::{Synth, SynthCaps, SynthSession};
use crate::workflow::synth::pool::{Lease, Pool};
use crate::workflow::synth::prompt::Prompt;

const RATE: u32 = 24_000;
const FRAME: usize = 1920;
const HIDDEN: usize = 1024;
const TEXT_HIDDEN: usize = 2048;
const GROUPS: usize = 16;
const CODES: usize = 2048;
const VOCAB: usize = 3072;
const LAYERS: i64 = 28;
const CP_LAYERS: i64 = 5;
const HEADS: i64 = 8;
const HEAD_DIM: i64 = 128;
const ENCODER_SAMPLES: usize = 240_000;
const PROMPTS: usize = 32;

const IM_START: u32 = 151_644;
const TTS_PAD: u32 = 151_671;
const TTS_BOS: u32 = 151_672;
const TTS_EOS: u32 = 151_673;
const CODEC_PAD: usize = 2148;
const CODEC_BOS: usize = 2149;
const CODEC_EOS: usize = 2150;
const CODEC_THINK: usize = 2154;
const CODEC_THINK_BOS: usize = 2156;
const CODEC_THINK_EOS: usize = 2157;

const TOP_K: usize = 50;
const PENALTY: f32 = 1.05;
const MIN_FRAMES: usize = 2;

const N_FFT: usize = 1024;
const HOP: usize = 256;
const MELS: usize = 128;
const MEL_MAX_HZ: f32 = 12_000.0;

/// Generation knobs shared by every stream; `context` and `chunk` shape the streaming vocoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3Options {
    pub threads: u16,
    pub workers: usize,
    pub temperature: f32,
    pub context: usize,
    pub first: usize,
    pub chunk: usize,
}

/// Embedding tables and the text projection, shared read-only by every worker.
struct Tables {
    text: Vec<f32>,
    fc1: (Vec<f32>, Vec<f32>),
    fc2: (Vec<f32>, Vec<f32>),
    codec: Vec<f32>,
    groups: Vec<Vec<f32>>,
    mel: Vec<f32>,
}

/// The six graphs, owned by one stream at a time.
struct Worker {
    prefill: Session,
    decode: Session,
    predictor: Session,
    vocoder: Vocoder,
    speaker: Session,
    encoder: Session,
}

/// The vocoder on its own thread, so a chunk is voiced while the talker generates the next frames.
/// One request in flight at a time keeps chunks in order.
struct Vocoder {
    requests: std::sync::mpsc::Sender<(Vec<i64>, usize, usize)>,
    results: std::sync::mpsc::Receiver<Result<Vec<f32>, String>>,
}

/// What a voice prompt becomes for this model: the x-vector, and — when its transcript is known and
/// the clip is at most 10 s (the speech tokenizer's window) — the reference codes and token ids for
/// in-context cloning. Measured on a 20 s prompt, the x-vector alone was both more faithful and
/// faster than in-context cloning, hence the limit.
#[derive(Debug, Clone)]
struct Voice {
    speaker: Vec<f32>,
    reference: Option<(Vec<[i64; GROUPS]>, Vec<u32>)>,
}

/// Qwen3-TTS 12Hz 0.6B Base: a talker transformer proposes the first codebook of each 80 ms frame, a
/// small code predictor fills the other 15, and the causal vocoder turns frames into audio in short
/// chunks with left context — streaming at frame granularity.
pub struct Qwen3Synth {
    tables: Arc<Tables>,
    tokenizer: Arc<Tokenizer>,
    pool: Arc<Pool<Worker>>,
    options: Qwen3Options,
    voices: Mutex<HashMap<u64, Arc<Voice>>>,
}

struct Qwen3Session {
    tables: Arc<Tables>,
    tokenizer: Arc<Tokenizer>,
    worker: Lease<Worker>,
    voice: Arc<Voice>,
    lang: Lang,
    options: Qwen3Options,
    rng: StdRng,
}

/// One sentence in flight: the talker's cache, frames generated and not yet voiced, and the vocoder's
/// left context.
struct Qwen3Stream<'a> {
    session: &'a mut Qwen3Session,
    text: String,
    started: bool,
    finished: bool,
    keys: Option<DynValue>,
    values: Option<DynValue>,
    past: i64,
    logits: Vec<f32>,
    hidden: Vec<f32>,
    trailing: Vec<Vec<f32>>,
    step: usize,
    limit: usize,
    seen: HashSet<usize>,
    frames: Vec<[i64; GROUPS]>,
    voiced: usize,
    chunks: usize,
    flight: bool,
}

impl Debug for Qwen3Synth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen3Synth")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl Tables {
    fn open(dir: &Path) -> Result<Self, BackendError> {
        let load = |name: &str| {
            let path = dir.join(format!("{name}.npy"));
            let bytes = std::fs::read(&path).map_err(|_| BackendError::Missing(path.clone()))?;
            npyz::NpyFile::new(&bytes[..])
                .and_then(npyz::NpyFile::into_vec::<f32>)
                .map_err(|_| BackendError::Load("qwen3 embeddings"))
        };
        Ok(Self {
            text: load("text_embedding")?,
            fc1: (load("text_projection_fc1_weight")?, load("text_projection_fc1_bias")?),
            fc2: (load("text_projection_fc2_weight")?, load("text_projection_fc2_bias")?),
            codec: load("talker_codec_embedding")?,
            groups: (0..GROUPS.saturating_sub(1))
                .map(|group| load(&format!("cp_codec_embedding_{group}")))
                .collect::<Result<Vec<_>, _>>()?,
            mel: Self::open_mel(),
        })
    }

    /// librosa `filters.mel(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)`: Slaney scale and
    /// area normalization, `[MELS, N_FFT / 2 + 1]` row-major.
    fn open_mel() -> Vec<f32> {
        let (f_sp, min_log_hz, min_log_mel, logstep) = (200.0_f64 / 3.0, 1000.0_f64, 15.0_f64, 6.4_f64.ln() / 27.0);
        let to_mel = |hz: f64| {
            if hz < min_log_hz {
                hz / f_sp
            } else {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            }
        };
        let to_hz = |mel: f64| {
            if mel < min_log_mel {
                mel * f_sp
            } else {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            }
        };
        let bins = N_FFT / 2 + 1;
        let top = to_mel(f64::from(MEL_MAX_HZ));
        let points: Vec<f64> = (0..MELS + 2)
            .map(|index| {
                to_hz(
                    top * f64::from(u32::try_from(index).unwrap_or(0))
                        / f64::from(u32::try_from(MELS + 1).unwrap_or(1)),
                )
            })
            .collect();
        let fft: Vec<f64> = (0..bins)
            .map(|bin| {
                f64::from(u32::try_from(bin).unwrap_or(0)) * f64::from(RATE)
                    / f64::from(u32::try_from(N_FFT).unwrap_or(1))
            })
            .collect();
        points
            .windows(3)
            .flat_map(|edge| {
                let (low, center, high) = (
                    edge.first().copied().unwrap_or(0.0),
                    edge.get(1).copied().unwrap_or(0.0),
                    edge.get(2).copied().unwrap_or(0.0),
                );
                let norm = 2.0 / (high - low);
                fft.iter().map(move |&hz| {
                    let rise = (hz - low) / (center - low);
                    let fall = (high - hz) / (high - center);
                    #[allow(clippy::cast_possible_truncation)]
                    let weight = (rise.min(fall).max(0.0) * norm) as f32;
                    weight
                })
            })
            .collect()
    }

    /// Official speaker-encoder features: reflect pad 384, STFT (1024 / 256, periodic Hann),
    /// magnitude, Slaney mel, natural log floored at 1e-5. `[frames, MELS]` row-major.
    fn mels(&self, audio: &[f32]) -> Vec<f32> {
        let pad = (N_FFT - HOP) / 2;
        let reflect = |index: isize| -> f32 {
            let len = isize::try_from(audio.len()).unwrap_or(isize::MAX);
            let mirrored = match index {
                index if index < 0 => index.saturating_neg(),
                index if index >= len => len.saturating_sub(2).saturating_sub(index.saturating_sub(len)),
                index => index,
            };
            usize::try_from(mirrored)
                .ok()
                .and_then(|at| audio.get(at))
                .copied()
                .unwrap_or(0.0)
        };
        let padded: Vec<f32> = (0_isize.saturating_sub(isize::try_from(pad).unwrap_or(0))
            ..isize::try_from(audio.len().saturating_add(pad)).unwrap_or(0))
            .map(reflect)
            .collect();
        let window: Vec<f32> = (0..N_FFT)
            .map(|n| {
                let phase = 2.0 * std::f32::consts::PI * f32::from(u16::try_from(n).unwrap_or(0))
                    / f32::from(u16::try_from(N_FFT).unwrap_or(1));
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let mut spectrum = fft.make_output_vec();
        let bins = N_FFT / 2 + 1;
        let frames = (padded.len().saturating_sub(N_FFT) / HOP).saturating_add(1);
        (0..frames)
            .flat_map(|frame| {
                let start = frame.saturating_mul(HOP);
                let mut input: Vec<f32> = padded
                    .iter()
                    .skip(start)
                    .take(N_FFT)
                    .zip(&window)
                    .map(|(sample, weight)| sample * weight)
                    .collect();
                input.resize(N_FFT, 0.0);
                fft.process(&mut input, &mut spectrum).ok();
                let magnitude: Vec<f32> = spectrum
                    .iter()
                    .map(|bin| (bin.re * bin.re + bin.im * bin.im + 1e-9).sqrt())
                    .collect();
                self.mel
                    .chunks_exact(bins)
                    .map(|filter| {
                        filter
                            .iter()
                            .zip(&magnitude)
                            .map(|(w, m)| w * m)
                            .sum::<f32>()
                            .max(1e-5)
                            .ln()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    /// `text_projection(text_embedding[ids])`: Linear(2048→2048) → `SiLU` → Linear(2048→1024).
    fn project(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        let rows = ids.len();
        let input: Vec<f32> = ids
            .iter()
            .flat_map(|&id| {
                let at = usize::try_from(id).unwrap_or(0).saturating_mul(TEXT_HIDDEN);
                self.text
                    .get(at..at.saturating_add(TEXT_HIDDEN))
                    .unwrap_or(&[])
                    .iter()
                    .copied()
            })
            .collect();
        let linear = |x: &[f32], (weight, bias): &(Vec<f32>, Vec<f32>), inner: usize, outer: usize| {
            let mut out: Vec<f32> = bias.iter().copied().cycle().take(rows.saturating_mul(outer)).collect();
            (x.len() == rows.saturating_mul(inner) && weight.len() == outer.saturating_mul(inner)).then(|| {
                out.chunks_exact_mut(outer)
                    .zip(x.chunks_exact(inner))
                    .for_each(|(row, input)| {
                        row.iter_mut()
                            .zip(weight.chunks_exact(inner))
                            .for_each(|(value, column)| {
                                *value += column.iter().zip(input).map(|(w, v)| w * v).sum::<f32>();
                            });
                    });
            });
            out
        };
        let hidden: Vec<f32> = linear(&input, &self.fc1, TEXT_HIDDEN, TEXT_HIDDEN)
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect();
        linear(&hidden, &self.fc2, TEXT_HIDDEN, HIDDEN)
            .chunks_exact(HIDDEN)
            .map(<[f32]>::to_vec)
            .collect()
    }

    fn codec(&self, id: usize) -> &[f32] {
        let at = id.saturating_mul(HIDDEN);
        self.codec.get(at..at.saturating_add(HIDDEN)).unwrap_or(&[])
    }

    fn group(&self, group: usize, id: usize) -> &[f32] {
        let at = id.saturating_mul(HIDDEN);
        self.groups
            .get(group)
            .and_then(|table| table.get(at..at.saturating_add(HIDDEN)))
            .unwrap_or(&[])
    }

    /// Embedding of a whole frame: codebook 0 through the talker table, 1..15 through the predictor's.
    fn frame(&self, codes: &[i64; GROUPS]) -> Vec<f32> {
        let mut sum = self.codec(usize::try_from(codes[0]).unwrap_or(0)).to_vec();
        for (group, &code) in codes.iter().skip(1).enumerate() {
            sum.iter_mut()
                .zip(self.group(group, usize::try_from(code).unwrap_or(0)))
                .for_each(|(total, value)| *total += value);
        }
        sum
    }
}

impl Vocoder {
    fn open(mut session: Session) -> Self {
        let (requests, inbox) = std::sync::mpsc::channel::<(Vec<i64>, usize, usize)>();
        let (outbox, results) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            while let Ok((codes, width, fresh)) = inbox.recv() {
                let audio = Self::run(&mut session, codes, width, fresh).map_err(|error| error.to_string());
                if outbox.send(audio).is_err() {
                    return;
                }
            }
        });
        Self { requests, results }
    }

    fn run(session: &mut Session, codes: Vec<i64>, width: usize, fresh: usize) -> Result<Vec<f32>, ort::Error> {
        let shape = vec![
            1_i64,
            i64::try_from(GROUPS).unwrap_or(0),
            i64::try_from(width).unwrap_or(0),
        ];
        let outputs = session.run(ort::inputs!["codes" => Tensor::from_array((shape, codes))?])?;
        let (_, wave) = outputs
            .get("waveform")
            .ok_or_else(|| ort::Error::new("vocoder returned no audio"))?
            .try_extract_tensor::<f32>()?;
        Ok(wave
            .get(wave.len().saturating_sub(fresh.saturating_mul(FRAME))..)
            .unwrap_or(&[])
            .to_vec())
    }
}

impl Worker {
    fn open(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let graph = |name: &'static str| {
            let file = dir.join(format!("{name}.onnx"));
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
            prefill: graph("talker_prefill")?,
            decode: graph("talker_decode")?,
            predictor: graph("code_predictor")?,
            vocoder: Vocoder::open(graph("vocoder")?),
            speaker: graph("speaker_encoder")?,
            encoder: graph("tokenizer_encoder")?,
        })
    }
}

impl Qwen3Synth {
    // ##### PRIVATE #####

    fn common_tensor(shape: Vec<i64>, data: Vec<f32>) -> Result<DynValue, ort::Error> {
        Ok(Tensor::from_array((shape, data))?.into_dyn())
    }

    fn common_ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .unwrap_or_default()
    }

    /// The x-vector and, with a transcript, the reference codes and ids, for one prompt.
    fn open_voice(&self, worker: &mut Worker, prompt: &Prompt) -> Result<Voice, ort::Error> {
        let mels = self.tables.mels(&prompt.audio);
        let frames = i64::try_from(mels.len() / MELS).unwrap_or(0);
        let outputs = worker.speaker.run(
            ort::inputs!["mels" => Tensor::from_array((vec![1, frames, i64::try_from(MELS).unwrap_or(0)], mels))?],
        )?;
        let speaker = outputs
            .get("speaker_embedding")
            .ok_or_else(|| ort::Error::new("speaker_encoder returned nothing"))?
            .try_extract_tensor::<f32>()?
            .1
            .to_vec();
        drop(outputs);
        let short = prompt.audio.len() <= ENCODER_SAMPLES;
        let reference = match (&prompt.text, short) {
            (Some(text), true) => {
                let mut padded = prompt.audio.clone();
                padded.resize(ENCODER_SAMPLES, 0.0);
                let outputs = worker.encoder.run(ort::inputs![
                    "waveform" => Tensor::from_array((vec![1_i64, i64::try_from(ENCODER_SAMPLES).unwrap_or(0)], padded))?
                ])?;
                let (shape, data) = outputs
                    .get("audio_codes")
                    .ok_or_else(|| ort::Error::new("tokenizer_encoder returned nothing"))?
                    .try_extract_tensor::<i64>()?;
                let width = usize::try_from(shape.get(2).copied().unwrap_or(0)).unwrap_or(0);
                let used = prompt.audio.len().div_ceil(FRAME).min(width);
                let codes = (0..used)
                    .map(|frame| {
                        let mut codes = [0_i64; GROUPS];
                        for (group, slot) in codes.iter_mut().enumerate() {
                            *slot = data
                                .get(group.saturating_mul(width).saturating_add(frame))
                                .copied()
                                .unwrap_or(0);
                        }
                        codes
                    })
                    .collect();
                let ids = Self::common_ids(&self.tokenizer, &format!("<|im_start|>assistant\n{text}<|im_end|>\n"));
                let content = ids.get(3..ids.len().saturating_sub(2)).unwrap_or(&[]).to_vec();
                Some((codes, content))
            }
            _ => None,
        };
        Ok(Voice { speaker, reference })
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// `dir` holds the installed model (`qwen3-tts`).
    ///
    /// # Errors
    /// Missing files, a malformed table or tokenizer, or onnxruntime rejecting a graph.
    pub fn new(dir: &Path, options: Qwen3Options) -> Result<Self, BackendError> {
        Runtime::probe().map_err(|_| BackendError::Load("onnxruntime"))?;
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|_| BackendError::Load("qwen3 tokenizer"))?;
        let workers = (0..options.workers.max(1))
            .map(|_| Worker::open(dir, options.threads))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            tables: Arc::new(Tables::open(dir)?),
            tokenizer: Arc::new(tokenizer),
            pool: Pool::new(workers),
            options,
            voices: Mutex::new(HashMap::new()),
        })
    }
}

impl Synth for Qwen3Synth {
    fn caps(&self) -> SynthCaps {
        SynthCaps {
            rate: RATE,
            langs: &Lang::ALL,
        }
    }

    fn open(&self, lang: Lang, voice: Option<&VoiceState>) -> Result<Box<dyn SynthSession>, BackendError> {
        let prompt = voice.map(Prompt::decode).transpose()?.ok_or(BackendError::Load(
            "qwen3 voice: Qwen3 Base clones a voice (learn one with `voice add`)",
        ))?;
        let mut worker = self.pool.lease().ok_or(BackendError::Busy("qwen3"))?;
        let key = prompt.key();
        let cached = self
            .voices
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .cloned();
        let voice = match cached {
            Some(voice) => voice,
            None => {
                let voice = Arc::new(
                    self.open_voice(&mut worker, &prompt)
                        .map_err(|_| BackendError::Load("qwen3 voice encoding"))?,
                );
                let mut voices = self.voices.lock().unwrap_or_else(PoisonError::into_inner);
                if voices.len() >= PROMPTS {
                    voices.clear();
                }
                voices.insert(key, Arc::clone(&voice));
                voice
            }
        };
        Ok(Box::new(Qwen3Session {
            tables: Arc::clone(&self.tables),
            tokenizer: Arc::clone(&self.tokenizer),
            worker,
            voice,
            lang,
            options: self.options,
            rng: StdRng::from_rng(&mut rand::rng()),
        }))
    }

    fn train(&self, clips: &[Audio], text: Option<&str>) -> Result<VoiceState, TrainError> {
        Prompt::prepare(clips, text)?.encode()
    }
}

impl Qwen3Session {
    /// Temperature, top-k, then a draw; `logits` are consumed.
    fn draw(&mut self, logits: &mut [f32]) -> usize {
        let temperature = self.options.temperature.max(1e-4);
        let mut order: Vec<usize> = (0..logits.len()).collect();
        let k = TOP_K.min(order.len());
        order.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            logits
                .get(*b)
                .copied()
                .unwrap_or(f32::MIN)
                .total_cmp(&logits.get(*a).copied().unwrap_or(f32::MIN))
        });
        order.truncate(k);
        let peak = order
            .iter()
            .filter_map(|&index| logits.get(index))
            .copied()
            .fold(f32::MIN, f32::max);
        let weights: Vec<f32> = order
            .iter()
            .map(|&index| ((logits.get(index).copied().unwrap_or(f32::MIN) - peak) / temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        let mut target = self.rng.random::<f32>() * total;
        order
            .iter()
            .zip(&weights)
            .find(|(_, weight)| {
                target -= **weight;
                target <= 0.0
            })
            .or_else(|| order.iter().zip(&weights).next_back())
            .map_or(0, |(index, _)| *index)
    }

    fn language(&self) -> usize {
        match self.lang {
            Lang::Es => 2054,
            Lang::En => 2050,
        }
    }
}

impl SynthSession for Qwen3Session {
    fn speak<'a>(&'a mut self, sentence: &'a str) -> Box<dyn Iterator<Item = Result<Audio, NodeError>> + Send + 'a> {
        Box::new(Qwen3Stream {
            session: self,
            text: sentence.to_owned(),
            started: false,
            finished: false,
            keys: None,
            values: None,
            past: 0,
            logits: Vec::new(),
            hidden: Vec::new(),
            trailing: Vec::new(),
            step: 0,
            limit: 0,
            seen: HashSet::new(),
            frames: Vec::new(),
            voiced: 0,
            chunks: 0,
            flight: false,
        })
    }
}

impl Qwen3Stream<'_> {
    /// The prompt (x-vector, or in-context with the reference codes) and the text tokens it holds;
    /// fills `trailing`, the text fed one per generated frame.
    fn start_prompt(&mut self) -> (Vec<Vec<f32>>, usize) {
        let session = &mut *self.session;
        let tables = Arc::clone(&session.tables);
        let ids = Qwen3Synth::common_ids(
            &session.tokenizer,
            &format!(
                "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
                self.text
            ),
        );
        let content = ids.get(3..ids.len().saturating_sub(5)).unwrap_or(&[]).to_vec();
        let add = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>();
        let specials = tables.project(&[TTS_BOS, TTS_EOS, TTS_PAD]);
        let (bos, eos, pad) = (
            specials.first().cloned().unwrap_or_default(),
            specials.get(1).cloned().unwrap_or_default(),
            specials.get(2).cloned().unwrap_or_default(),
        );
        let mut prompt: Vec<Vec<f32>> = tables.project(&[IM_START, 77_091, 198]);
        let tags = [CODEC_THINK, CODEC_THINK_BOS, session.language(), CODEC_THINK_EOS];
        prompt.extend(tags.iter().map(|&tag| add(&pad, tables.codec(tag))));
        prompt.push(add(&pad, &session.voice.speaker));
        prompt.push(add(&bos, tables.codec(CODEC_PAD)));
        let voice = Arc::clone(&session.voice);
        self.trailing = match &voice.reference {
            Some((codes, reference)) => {
                let mut text = tables.project(&[reference.as_slice(), content.as_slice()].concat());
                text.push(eos.clone());
                let mut spoken = vec![tables.codec(CODEC_BOS).to_vec()];
                spoken.extend(codes.iter().map(|frame| tables.frame(frame)));
                let joined = spoken.len();
                let rest = text.split_off(joined.min(text.len()));
                text.resize(joined, pad.clone());
                prompt.extend(text.iter().zip(&spoken).map(|(t, c)| add(t, c)));
                if rest.is_empty() { vec![pad.clone()] } else { rest }
            }
            None => {
                let first = tables.project(content.get(..1).unwrap_or(&[]));
                prompt.push(add(first.first().unwrap_or(&pad), tables.codec(CODEC_BOS)));
                let mut rest = tables.project(content.get(1..).unwrap_or(&[]));
                rest.push(eos.clone());
                rest
            }
        };
        self.trailing.push(pad);
        (prompt, content.len())
    }

    /// Builds the prompt and runs the prefill.
    fn start(&mut self) -> Result<(), ort::Error> {
        let (prompt, tokens) = self.start_prompt();
        let voice = Arc::clone(&self.session.voice);
        let session = &mut *self.session;
        let length = i64::try_from(prompt.len()).unwrap_or(0);
        let embeds: Vec<f32> = prompt.into_iter().flatten().collect();
        let positions: Vec<i64> = (0..3).flat_map(|_| 0..length).collect();
        let outputs = session.worker.prefill.run(ort::inputs![
            "inputs_embeds" => Tensor::from_array((vec![1, length, i64::try_from(HIDDEN).unwrap_or(0)], embeds))?,
            "attention_mask" => Tensor::from_array((vec![1, length], vec![1_i64; usize::try_from(length).unwrap_or(0)]))?,
            "position_ids" => Tensor::from_array((vec![3, 1, length], positions))?,
        ])?;
        let last = |name: &str, width: usize| -> Result<Vec<f32>, ort::Error> {
            let (_, data) = outputs
                .get(name)
                .ok_or_else(|| ort::Error::new("talker_prefill dropped an output"))?
                .try_extract_tensor::<f32>()?;
            Ok(data.get(data.len().saturating_sub(width)..).unwrap_or(&[]).to_vec())
        };
        self.logits = last("logits", VOCAB)?;
        self.hidden = last("hidden_states", HIDDEN)?;
        let stack = |prefix: &str| -> Result<Vec<f32>, ort::Error> {
            let mut all = Vec::new();
            for layer in 0..LAYERS {
                let (_, data) = outputs
                    .get(format!("{prefix}_{layer}"))
                    .ok_or_else(|| ort::Error::new("talker_prefill dropped a cache"))?
                    .try_extract_tensor::<f32>()?;
                all.extend_from_slice(data);
            }
            Ok(all)
        };
        let shape = vec![LAYERS, 1, HEADS, length, HEAD_DIM];
        self.keys = Some(Qwen3Synth::common_tensor(shape.clone(), stack("present_key")?)?);
        self.values = Some(Qwen3Synth::common_tensor(shape, stack("present_value")?)?);
        self.past = length;
        self.limit = tokens.saturating_mul(6).saturating_add(40);
        self.voiced = voice
            .reference
            .as_ref()
            .map_or(0, |(codes, _)| codes.len().min(session.options.context));
        self.frames = voice
            .reference
            .as_ref()
            .map(|(codes, _)| {
                codes
                    .iter()
                    .skip(codes.len().saturating_sub(self.voiced))
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        Ok(())
    }

    /// One frame: codebook 0 from the talker, 1..15 from the predictor, then the talker steps.
    /// `Ok(false)` when the talker ends the utterance.
    fn frame(&mut self) -> Result<bool, ort::Error> {
        let mut logits = std::mem::take(&mut self.logits);
        for (index, value) in logits.iter_mut().enumerate() {
            let banned = (index >= CODES && index != CODEC_EOS) || (index == CODEC_EOS && self.step < MIN_FRAMES);
            if banned {
                *value = f32::NEG_INFINITY;
            }
            if self.seen.contains(&index) {
                *value = if *value > 0.0 {
                    *value / PENALTY
                } else {
                    *value * PENALTY
                };
            }
        }
        let session = &mut *self.session;
        let first = session.draw(&mut logits);
        if first == CODEC_EOS || self.step >= self.limit {
            return Ok(false);
        }
        self.seen.insert(first);
        let codes = self.frame_predict(first)?;
        self.frames.push(codes);
        self.frame_step(&codes)?;
        Ok(true)
    }

    /// Codebooks 1..15 of a frame from the code predictor, conditioned on the talker's last state.
    fn frame_predict(&mut self, first: usize) -> Result<[i64; GROUPS], ort::Error> {
        let session = &mut *self.session;
        let tables = Arc::clone(&session.tables);
        let mut codes = [0_i64; GROUPS];
        codes[0] = i64::try_from(first).unwrap_or(0);
        let mut input = [self.hidden.as_slice(), tables.codec(first)].concat();
        let mut keys = Qwen3Synth::common_tensor(vec![CP_LAYERS, 1, HEADS, 0, HEAD_DIM], Vec::new())?;
        let mut values = Qwen3Synth::common_tensor(vec![CP_LAYERS, 1, HEADS, 0, HEAD_DIM], Vec::new())?;
        for group in 0..GROUPS.saturating_sub(1) {
            let length = i64::try_from(input.len() / HIDDEN).unwrap_or(0);
            let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
                (
                    "inputs_embeds".into(),
                    Qwen3Synth::common_tensor(
                        vec![1, length, i64::try_from(HIDDEN).unwrap_or(0)],
                        std::mem::take(&mut input),
                    )?
                    .into(),
                ),
                (
                    "generation_steps".into(),
                    Tensor::from_array((vec![1_i64], vec![i64::try_from(group).unwrap_or(0)]))?
                        .into_dyn()
                        .into(),
                ),
                ("past_keys".into(), keys.into()),
                ("past_values".into(), values.into()),
            ];
            let mut outputs = session.worker.predictor.run(inputs)?;
            let (_, all) = outputs
                .get("logits")
                .ok_or_else(|| ort::Error::new("code_predictor returned no logits"))?
                .try_extract_tensor::<f32>()?;
            let mut last = all.get(all.len().saturating_sub(CODES)..).unwrap_or(&[]).to_vec();
            keys = outputs
                .remove("present_keys")
                .ok_or_else(|| ort::Error::new("code_predictor dropped keys"))?;
            values = outputs
                .remove("present_values")
                .ok_or_else(|| ort::Error::new("code_predictor dropped values"))?;
            drop(outputs);
            let code = session.draw(&mut last);
            if let Some(slot) = codes.get_mut(group.saturating_add(1)) {
                *slot = i64::try_from(code).unwrap_or(0);
            }
            input = tables.group(group, code).to_vec();
        }
        Ok(codes)
    }

    /// Feeds the frame (plus the next text token) to the talker: next logits, state and cache.
    fn frame_step(&mut self, codes: &[i64; GROUPS]) -> Result<(), ort::Error> {
        let session = &mut *self.session;
        let mut next = session.tables.frame(codes);
        let text = self.trailing.get(self.step).or_else(|| self.trailing.last());
        if let Some(text) = text {
            next.iter_mut().zip(text).for_each(|(value, add)| *value += add);
        }
        let past = self.past;
        let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            (
                "inputs_embeds".into(),
                Qwen3Synth::common_tensor(vec![1, 1, i64::try_from(HIDDEN).unwrap_or(0)], next)?.into(),
            ),
            (
                "attention_mask".into(),
                Tensor::from_array((
                    vec![1, past.saturating_add(1)],
                    vec![1_i64; usize::try_from(past.saturating_add(1)).unwrap_or(0)],
                ))?
                .into_dyn()
                .into(),
            ),
            (
                "position_ids".into(),
                Tensor::from_array((vec![3_i64, 1, 1], vec![past; 3]))?
                    .into_dyn()
                    .into(),
            ),
            (
                "past_keys".into(),
                self.keys
                    .take()
                    .ok_or_else(|| ort::Error::new("talker cache lost"))?
                    .into(),
            ),
            (
                "past_values".into(),
                self.values
                    .take()
                    .ok_or_else(|| ort::Error::new("talker cache lost"))?
                    .into(),
            ),
        ];
        let mut outputs = session.worker.decode.run(inputs)?;
        let take = |name: &str, outputs: &ort::session::SessionOutputs<'_>| -> Result<Vec<f32>, ort::Error> {
            Ok(outputs
                .get(name)
                .ok_or_else(|| ort::Error::new("talker_decode dropped an output"))?
                .try_extract_tensor::<f32>()?
                .1
                .to_vec())
        };
        self.logits = take("logits", &outputs)?;
        self.hidden = take("hidden_states", &outputs)?;
        self.keys = outputs.remove("present_keys");
        self.values = outputs.remove("present_values");
        self.past = past.saturating_add(1);
        self.step = self.step.saturating_add(1);
        Ok(())
    }

    /// Sends the frames not yet heard to the vocoder, with up to `context` frames before them.
    fn submit(&mut self) {
        let context = self.session.options.context;
        let fresh = self.frames.len().saturating_sub(self.voiced);
        let window = self.frames.get(self.voiced.saturating_sub(context)..).unwrap_or(&[]);
        let width = window.len();
        let codes: Vec<i64> = (0..GROUPS)
            .flat_map(|group| window.iter().map(move |frame| frame.get(group).copied().unwrap_or(0)))
            .collect();
        self.flight = self.session.worker.vocoder.requests.send((codes, width, fresh)).is_ok();
        self.voiced = self.frames.len();
        let keep = context.saturating_add(1);
        if self.frames.len() > keep.saturating_mul(4) {
            let dropped = self.frames.len().saturating_sub(keep);
            self.frames.drain(..dropped);
            self.voiced = self.voiced.saturating_sub(dropped);
        }
        self.chunks = self.chunks.saturating_add(1);
    }

    /// The chunk in flight: waits for it when `wait`, else only if it is already done.
    fn collect(&mut self, wait: bool) -> Option<Result<Vec<f32>, String>> {
        let vocoder = &self.session.worker.vocoder;
        let result = if wait {
            vocoder.results.recv().ok()
        } else {
            vocoder.results.try_recv().ok()
        };
        self.flight = self.flight && result.is_none();
        result
    }
}

impl Drop for Qwen3Stream<'_> {
    fn drop(&mut self) {
        if self.flight {
            self.collect(true);
        }
    }
}

impl Iterator for Qwen3Stream<'_> {
    type Item = Result<Audio, NodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let fail = |error: ort::Error| NodeError::Backend(format!("qwen3: {error}"));
        if !self.started {
            self.started = true;
            if let Err(error) = self.start() {
                self.finished = true;
                return Some(Err(fail(error)));
            }
        }
        loop {
            if let Some(result) = self.flight.then(|| self.collect(false)).flatten() {
                return Some(result.map(Audio::from).map_err(NodeError::Backend));
            }
            let pending = self.frames.len().saturating_sub(self.voiced);
            let target = if self.chunks == 0 {
                self.session.options.first
            } else {
                self.session.options.chunk
            };
            if !self.flight && (pending >= target.max(1) || (self.finished && pending > 0)) {
                self.submit();
                continue;
            }
            if self.finished {
                return self
                    .flight
                    .then(|| self.collect(true))
                    .flatten()
                    .map(|result| result.map(Audio::from).map_err(NodeError::Backend));
            }
            match self.frame() {
                Ok(true) => {}
                Ok(false) => self.finished = true,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(fail(error)));
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::cast_precision_loss)]
mod tests {
    use std::path::PathBuf;

    use crate::workflow::synth::prompt::Prompt;
    use crate::workflow::synth::qwen3::{Qwen3Options, Qwen3Synth, Tables};

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/tts")
    }

    fn golden<T: npyz::Deserialize>(key: &str) -> Vec<T> {
        let file = std::fs::File::open(root().join("ops/exports/qwen3/golden.npz")).unwrap();
        let mut npz = npyz::npz::NpzArchive::new(std::io::BufReader::new(file)).unwrap();
        npz.by_name(key).unwrap().unwrap().into_vec::<T>().unwrap()
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    #[ignore = "requires the installed model and its golden: make qwen3"]
    fn test_preprocessing_matches_the_official_recipe() {
        let dir = root().join("models/qwen3-tts");
        let tables = Tables::open(&dir).unwrap();
        let mels = tables.mels(&golden::<f32>("audio"));
        assert!(worst(&mels, &golden::<f32>("mels")) < 1e-3);
        let ids: Vec<u32> = golden::<i64>("target")
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let text = "<|im_start|>assistant\nHola, esta es una prueba de la voz.<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(Qwen3Synth::common_ids(&tokenizer, text), ids);
        let projected: Vec<f32> = tables.project(&ids[..8]).into_iter().flatten().collect();
        assert!(worst(&projected, &golden::<f32>("projection")) < 1e-3);
    }

    #[test]
    #[ignore = "requires the installed model and its golden: make qwen3"]
    fn test_voice_matches_the_official_speaker_and_codes() {
        let dir = root().join("models/qwen3-tts");
        let options = Qwen3Options {
            threads: 4,
            workers: 1,
            temperature: 0.9,
            context: 50,
            first: 4,
            chunk: 16,
        };
        let synth = Qwen3Synth::new(&dir, options).unwrap();
        let prompt = Prompt {
            audio: golden::<f32>("audio"),
            text: Some("Esto es lo que se dice en la referencia.".to_owned()),
        };
        let mut worker = synth.pool.lease().unwrap();
        let voice = synth.open_voice(&mut worker, &prompt).unwrap();
        let speaker = golden::<f32>("speaker");
        let cosine = voice.speaker.iter().zip(&speaker).map(|(a, b)| a * b).sum::<f32>()
            / (voice.speaker.iter().map(|a| a * a).sum::<f32>().sqrt()
                * speaker.iter().map(|b| b * b).sum::<f32>().sqrt());
        assert!(cosine > 0.999, "{cosine}");
        let (codes, ids) = voice.reference.unwrap();
        let expected = golden::<i64>("codes");
        let frames = expected.len() / 16;
        assert_eq!(codes.len(), frames);
        let agree = (0..frames)
            .filter(|&frame| (0..16).all(|group| codes[frame][group] == expected[group * frames + frame]))
            .count();
        assert!(agree * 100 >= frames * 95, "{agree}/{frames}");
        let reference: Vec<u32> = golden::<i64>("ref")
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        assert_eq!(ids, reference[3..reference.len() - 2]);
    }
}
