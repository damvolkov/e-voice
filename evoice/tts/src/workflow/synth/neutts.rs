use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Debug};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use e_voice_core::audio::{AudioEncoding, AudioIngest};
use e_voice_core::runtime::Runtime;
use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tokenizers::Tokenizer;

use crate::schema::audio::Audio;
use crate::schema::error::TrainError;
use crate::schema::voice::VoiceState;
use crate::workflow::synth::base::{Synth, SynthCaps, SynthSession};
use crate::workflow::synth::pool::{Lease, Pool};
use crate::workflow::synth::prompt::Prompt;

const RATE: u32 = 24_000;
const ENCODER_RATE: u32 = 16_000;
const ENCODER_HOP: usize = 320;
const HOP: usize = 480;
const LAYERS: usize = 24;
const KV_HEADS: i64 = 3;
const HEAD_DIM: i64 = 64;
const CONTEXT: usize = 2048;
const END: usize = 128_261;
const SPEECH: usize = 128_262;
const SPEECH_ID: i64 = 128_262;
const CODEBOOK: usize = 65_536;
const TOP_K: usize = 50;
const MIN_CODES: usize = 50;
const CHUNK: usize = 25;
const LOOKAHEAD: usize = 5;
const LOOKBACK: usize = 50;
const OVERLAP: usize = 1;
const PROMPTS: usize = 32;
const PUNCTUATION: [char; 18] = [
    ';', ':', ',', '.', '!', '?', '¡', '¿', '—', '…', '"', '«', '»', '“', '”', '(', ')', '-',
];

/// Generation knobs shared by every stream; `espeak` is the espeak-ng binary that phonemizes text.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuttsOptions {
    pub threads: u16,
    pub workers: usize,
    pub temperature: f32,
    pub espeak: PathBuf,
}

/// Backbone and decoder of one language, owned by one stream at a time.
struct Worker {
    backbone: Session,
    decoder: Session,
}

/// One language's model: tokenizer, espeak voice and worker pool.
struct Bundle {
    model: String,
    tokenizer: Tokenizer,
    voice: &'static str,
    pool: Arc<Pool<Worker>>,
}

/// What a voice prompt becomes: its codes and its phonemized transcript.
#[derive(Debug, Clone)]
struct Voice {
    codes: Vec<u32>,
    phonemes: String,
}

/// NeuTTS Nano (neuphonic): a 24-layer Llama backbone writes `NeuCodec` speech tokens (50 Hz, one
/// codebook) after the phonemized reference and target text; the codec decoder voices them in
/// overlapping windows as they come. Cloning needs the reference transcript. Licence: NeuTTS Open
/// License 1.0 (free under $5M annual revenue, outputs included).
pub struct NeuttsSynth {
    bundles: BTreeMap<u8, Arc<Bundle>>,
    encoder: Mutex<Session>,
    options: NeuttsOptions,
    voices: Mutex<HashMap<(u64, u8), Arc<Voice>>>,
}

struct NeuttsSession {
    bundle: Arc<Bundle>,
    worker: Lease<Worker>,
    voice: Arc<Voice>,
    options: NeuttsOptions,
    rng: StdRng,
}

/// One sentence in flight: the backbone's cache, the codes so far (seeded with the reference), and
/// the overlap-add state of the decoded windows.
struct NeuttsStream<'a> {
    session: &'a mut NeuttsSession,
    text: String,
    started: bool,
    finished: bool,
    past: Vec<DynValue>,
    length: usize,
    logits: Vec<f32>,
    codes: Vec<u32>,
    generated: usize,
    decoded: usize,
    mixer: Mixer,
}

/// Linear overlap-add of fixed-stride windows with triangular weights (upstream
/// `_linear_overlap_add`), emitting each region as soon as no later window can touch it.
#[derive(Debug, Default)]
pub struct Mixer {
    sum: Vec<f32>,
    weight: Vec<f32>,
    offset: usize,
    emitted: usize,
}

impl Debug for NeuttsSynth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let models: Vec<&str> = self.bundles.values().map(|bundle| bundle.model.as_str()).collect();
        f.debug_struct("NeuttsSynth")
            .field("models", &models)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl Mixer {
    /// Adds a window at the current offset; returns what is now final. `last` flushes everything.
    pub fn push(&mut self, frame: &[f32], stride: usize, last: bool) -> Vec<f32> {
        let length = frame.len();
        let end = self.offset.saturating_add(length);
        if self.sum.len() < end {
            self.sum.resize(end, 0.0);
            self.weight.resize(end, 0.0);
        }
        let denominator = f32::from(u16::try_from(length.saturating_add(1)).unwrap_or(u16::MAX));
        for (index, sample) in frame.iter().enumerate() {
            let t = f32::from(u16::try_from(index.saturating_add(1)).unwrap_or(u16::MAX)) / denominator;
            let weight = 0.5 - (t - 0.5).abs();
            let at = self.offset.saturating_add(index);
            if let (Some(sum), Some(total)) = (self.sum.get_mut(at), self.weight.get_mut(at)) {
                *sum += weight * sample;
                *total += weight;
            }
        }
        self.offset = self.offset.saturating_add(stride);
        let ready = if last {
            self.sum.len()
        } else {
            self.offset.min(self.sum.len())
        };
        let out: Vec<f32> = self
            .sum
            .iter()
            .zip(&self.weight)
            .take(ready)
            .skip(self.emitted)
            .map(|(sum, weight)| if *weight > 0.0 { sum / weight } else { 0.0 })
            .collect();
        self.emitted = ready.max(self.emitted);
        out
    }
}

/// espeak-ng IPA with stress, punctuation kept in place (as `phonemizer` with
/// `preserve_punctuation=True`), language-switch flags removed, whitespace collapsed.
#[derive(Debug)]
pub struct Phonemizer;

impl Phonemizer {
    // ##### PRIVATE #####

    fn run_segment(espeak: &Path, voice: &str, text: &str) -> Result<String, std::io::Error> {
        let output = Command::new(espeak)
            .args(["-q", "--ipa", "-v", voice, "--", text])
            .output()?;
        let raw = String::from_utf8_lossy(&output.stdout);
        let mut clean = String::new();
        let mut flag = false;
        for c in raw.chars() {
            match c {
                '(' => flag = true,
                ')' if flag => flag = false,
                _ if flag => {}
                _ => clean.push(c),
            }
        }
        Ok(clean.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    // ##### PUBLIC #####

    /// Splits `text` into (words, punctuation) runs, in order.
    #[must_use]
    pub fn split(text: &str) -> Vec<(String, String)> {
        let mut runs: Vec<(String, String)> = vec![(String::new(), String::new())];
        for c in text.chars() {
            let marked = PUNCTUATION.contains(&c);
            match (marked, runs.last_mut()) {
                (true, Some(run)) => run.1.push(c),
                (false, Some(run)) if !run.1.is_empty() && !c.is_whitespace() => {
                    runs.push((c.to_string(), String::new()));
                }
                (false, Some(run)) if !run.1.is_empty() => run.1.push(c),
                (false, Some(run)) => run.0.push(c),
                (_, None) => {}
            }
        }
        runs
    }

    /// # Errors
    /// espeak-ng could not be run.
    pub fn run(espeak: &Path, voice: &str, text: &str) -> Result<String, std::io::Error> {
        let mut out = String::new();
        for (words, marks) in Self::split(text) {
            let words = words.trim();
            if !words.is_empty() {
                out.push_str(&Self::run_segment(espeak, voice, words)?);
            }
            out.push_str(marks.trim_end());
            if marks.ends_with(char::is_whitespace) || (!marks.is_empty() && !out.is_empty()) {
                out.push(' ');
            }
        }
        Ok(out.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

impl Worker {
    fn open(dir: &Path, threads: u16) -> Result<Self, BackendError> {
        let graph = |name: &'static str| NeuttsSynth::common_graph(&dir.join(format!("{name}.onnx")), threads, name);
        Ok(Self {
            backbone: graph("model")?,
            decoder: graph("decoder")?,
        })
    }
}

impl NeuttsSynth {
    // ##### PRIVATE #####

    fn common_graph(file: &Path, threads: u16, name: &'static str) -> Result<Session, BackendError> {
        file.exists()
            .then_some(())
            .ok_or_else(|| BackendError::Missing(file.to_path_buf()))?;
        Session::builder()
            .map_err(|_| BackendError::Load(name))?
            .with_intra_threads(usize::from(threads))
            .map_err(|_| BackendError::Load(name))?
            .with_inter_threads(1)
            .map_err(|_| BackendError::Load(name))?
            .with_intra_op_spinning(false)
            .map_err(|_| BackendError::Load(name))?
            .commit_from_file(file)
            .map_err(|_| BackendError::Load(name))
    }

    const fn common_key(lang: Lang) -> u8 {
        match lang {
            Lang::Es => 0,
            Lang::En => 1,
        }
    }

    /// Reference codes from the distilled `NeuCodec` encoder (16 kHz, padded to its 320-sample hop).
    fn open_codes(&self, audio: &[f32]) -> Result<Vec<u32>, BackendError> {
        let fail = |_| BackendError::Load("neutts reference encoding");
        let mut ingest = AudioIngest::new(RATE, ENCODER_RATE, AudioEncoding::F32le)
            .map_err(|_| BackendError::Load("neutts resampler"))?;
        let mut samples = ingest.feed(audio).map_err(|_| BackendError::Load("neutts resampler"))?;
        samples.extend(ingest.flush().map_err(|_| BackendError::Load("neutts resampler"))?);
        let padded = samples
            .len()
            .saturating_add(ENCODER_HOP.saturating_sub(samples.len() % ENCODER_HOP));
        samples.resize(padded, 0.0);
        let input = Tensor::from_array((vec![1_i64, 1, i64::try_from(padded).unwrap_or(0)], samples)).map_err(fail)?;
        let mut encoder = self.encoder.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = encoder.run(ort::inputs!["audio" => input]).map_err(fail)?;
        let (_, codes) = outputs
            .get("codes")
            .ok_or(BackendError::Load("neutts encoder output"))?
            .try_extract_tensor::<i32>()
            .map_err(fail)?;
        Ok(codes.iter().map(|&code| u32::try_from(code).unwrap_or(0)).collect())
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// `models` maps each served language to its model id and bundle directory; `encoder` holds the
    /// installed `neucodec-encoder`.
    ///
    /// # Errors
    /// Missing files, a malformed tokenizer, or onnxruntime rejecting a graph.
    pub fn new(models: &[(Lang, String, &Path)], encoder: &Path, options: NeuttsOptions) -> Result<Self, BackendError> {
        Runtime::probe().map_err(|_| BackendError::Load("onnxruntime"))?;
        let bundles = models
            .iter()
            .map(|(lang, model, dir)| {
                let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
                    .map_err(|_| BackendError::Load("neutts tokenizer"))?;
                let workers = (0..options.workers.max(1))
                    .map(|_| Worker::open(dir, options.threads))
                    .collect::<Result<Vec<_>, _>>()?;
                let voice = match lang {
                    Lang::Es => "es",
                    Lang::En => "en-us",
                };
                Ok((
                    Self::common_key(*lang),
                    Arc::new(Bundle {
                        model: model.clone(),
                        tokenizer,
                        voice,
                        pool: Pool::new(workers),
                    }),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, BackendError>>()?;
        let encoder = Self::common_graph(
            &encoder.join("distill_neucodec_encoder.onnx"),
            options.threads,
            "neucodec encoder",
        )?;
        Ok(Self {
            bundles,
            encoder: Mutex::new(encoder),
            options,
            voices: Mutex::new(HashMap::new()),
        })
    }
}

impl Synth for NeuttsSynth {
    fn caps(&self) -> SynthCaps {
        SynthCaps {
            rate: RATE,
            langs: &Lang::ALL,
        }
    }

    fn open(&self, lang: Lang, voice: Option<&VoiceState>) -> Result<Box<dyn SynthSession>, BackendError> {
        let prompt = voice.map(Prompt::decode).transpose()?.ok_or(BackendError::Load(
            "neutts voice: NeuTTS clones a voice (learn one with `voice add --text`)",
        ))?;
        let text = prompt.text.clone().ok_or(BackendError::Load(
            "neutts voice: NeuTTS needs the reference transcript (`voice add --text`)",
        ))?;
        let key = Self::common_key(lang);
        let bundle = self.bundles.get(&key).ok_or(BackendError::Load("neutts language"))?;
        let worker = bundle.pool.lease().ok_or(BackendError::Busy("neutts"))?;
        let cached = self
            .voices
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(prompt.key(), key))
            .cloned();
        let voice = match cached {
            Some(voice) => voice,
            None => {
                let phonemes = Phonemizer::run(&self.options.espeak, bundle.voice, &text)
                    .map_err(|_| BackendError::Load("neutts: espeak-ng is not runnable (install it: make system)"))?;
                let voice = Arc::new(Voice {
                    codes: self.open_codes(&prompt.audio)?,
                    phonemes,
                });
                let mut voices = self.voices.lock().unwrap_or_else(PoisonError::into_inner);
                if voices.len() >= PROMPTS {
                    voices.clear();
                }
                voices.insert((prompt.key(), key), Arc::clone(&voice));
                voice
            }
        };
        Ok(Box::new(NeuttsSession {
            bundle: Arc::clone(bundle),
            worker,
            voice,
            options: self.options.clone(),
            rng: StdRng::from_rng(&mut rand::rng()),
        }))
    }

    fn train(&self, clips: &[Audio], text: Option<&str>) -> Result<VoiceState, TrainError> {
        Prompt::prepare(clips, text)?.encode()
    }
}

impl NeuttsSession {
    /// END or a speech code, by temperature and top-k over the speech range only.
    fn draw(&mut self, logits: &[f32], allow_end: bool) -> Option<u32> {
        let window = logits.get(END..SPEECH.saturating_add(CODEBOOK)).unwrap_or(&[]);
        let temperature = self.options.temperature.max(1e-4);
        let mut order: Vec<usize> = (usize::from(!allow_end)..window.len()).collect();
        let k = TOP_K.min(order.len());
        if k == 0 {
            return None;
        }
        order.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            window
                .get(*b)
                .copied()
                .unwrap_or(f32::MIN)
                .total_cmp(&window.get(*a).copied().unwrap_or(f32::MIN))
        });
        order.truncate(k);
        let peak = order
            .iter()
            .filter_map(|&index| window.get(index))
            .copied()
            .fold(f32::MIN, f32::max);
        let weights: Vec<f32> = order
            .iter()
            .map(|&index| ((window.get(index).copied().unwrap_or(f32::MIN) - peak) / temperature).exp())
            .collect();
        let mut target = self.rng.random::<f32>() * weights.iter().sum::<f32>();
        let chosen = order
            .iter()
            .zip(&weights)
            .find(|(_, weight)| {
                target -= **weight;
                target <= 0.0
            })
            .or_else(|| order.iter().zip(&weights).next_back())
            .map_or(0, |(index, _)| *index);
        chosen.checked_sub(1).and_then(|code| u32::try_from(code).ok())
    }
}

impl SynthSession for NeuttsSession {
    fn speak<'a>(&'a mut self, sentence: &'a str) -> Box<dyn Iterator<Item = Result<Audio, NodeError>> + Send + 'a> {
        let codes = self.voice.codes.clone();
        let decoded = codes.len();
        Box::new(NeuttsStream {
            session: self,
            text: sentence.to_owned(),
            started: false,
            finished: false,
            past: Vec::new(),
            length: 0,
            logits: Vec::new(),
            codes,
            generated: 0,
            decoded,
            mixer: Mixer::default(),
        })
    }
}

impl NeuttsStream<'_> {
    /// Runs the backbone over `ids`; keeps the last logits and the cache.
    fn step(&mut self, ids: Vec<i64>) -> Result<(), ort::Error> {
        let count = ids.len();
        let total = self.length.saturating_add(count);
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            (
                "input_ids".into(),
                Tensor::from_array((vec![1_i64, i64::try_from(count).unwrap_or(0)], ids))?
                    .into_dyn()
                    .into(),
            ),
            (
                "attention_mask".into(),
                Tensor::from_array((vec![1_i64, i64::try_from(total).unwrap_or(0)], vec![1_i64; total]))?
                    .into_dyn()
                    .into(),
            ),
        ];
        let mut past = std::mem::take(&mut self.past).into_iter();
        for layer in 0..LAYERS {
            for kind in ["key", "value"] {
                let value = match past.next() {
                    Some(value) => value,
                    None => Tensor::from_array((vec![1_i64, KV_HEADS, 0, HEAD_DIM], Vec::<f32>::new()))?.into_dyn(),
                };
                inputs.push((format!("past_key_values.{layer}.{kind}").into(), value.into()));
            }
        }
        let mut outputs = self.session.worker.backbone.run(inputs)?;
        let (shape, logits) = outputs
            .get("logits")
            .ok_or_else(|| ort::Error::new("backbone returned no logits"))?
            .try_extract_tensor::<f32>()?;
        let vocab = usize::try_from(shape.get(2).copied().unwrap_or(0)).unwrap_or(0);
        self.logits = logits.get(logits.len().saturating_sub(vocab)..).unwrap_or(&[]).to_vec();
        for layer in 0..LAYERS {
            for kind in ["key", "value"] {
                let value = outputs
                    .remove(format!("present.{layer}.{kind}"))
                    .ok_or_else(|| ort::Error::new("backbone dropped a cache"))?;
                self.past.push(value);
            }
        }
        self.length = total;
        Ok(())
    }

    fn start(&mut self) -> Result<(), ort::Error> {
        let session = &mut *self.session;
        let phonemes = Phonemizer::run(&session.options.espeak, session.bundle.voice, &self.text)
            .map_err(|error| ort::Error::new(format!("espeak-ng: {error}")))?;
        let prompt = format!(
            "user: Convert the text to speech:<|TEXT_PROMPT_START|>{} {phonemes}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>",
            session.voice.phonemes
        );
        let encoding = session
            .bundle
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| ort::Error::new(error.to_string()))?;
        let ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| i64::from(id))
            .chain(
                session
                    .voice
                    .codes
                    .iter()
                    .map(|&code| i64::from(code).saturating_add(SPEECH_ID)),
            )
            .collect();
        self.step(ids)
    }

    /// Decodes the codes from `decoded` on (with look-back and look-ahead) into the next audio.
    fn decode(&mut self, last: bool) -> Result<Vec<f32>, ort::Error> {
        let pending = self.codes.len().saturating_sub(self.decoded);
        let back = LOOKBACK.saturating_add(OVERLAP);
        let start = self.decoded.saturating_sub(back);
        let end = if last {
            self.codes.len()
        } else {
            self.decoded
                .saturating_add(CHUNK)
                .saturating_add(LOOKAHEAD)
                .saturating_add(OVERLAP)
        };
        let window: Vec<i32> = self
            .codes
            .get(start..end.min(self.codes.len()))
            .unwrap_or(&[])
            .iter()
            .map(|&code| i32::try_from(code).unwrap_or(0))
            .collect();
        let width = i64::try_from(window.len()).unwrap_or(0);
        let outputs = self
            .session
            .worker
            .decoder
            .run(ort::inputs!["codes" => Tensor::from_array((vec![1_i64, 1, width], window))?])?;
        let (_, audio) = outputs
            .get("audio")
            .ok_or_else(|| ort::Error::new("decoder returned no audio"))?
            .try_extract_tensor::<f32>()?;
        let from = self
            .decoded
            .saturating_sub(start)
            .saturating_sub(OVERLAP)
            .saturating_mul(HOP);
        let span = if last {
            audio.len()
        } else {
            CHUNK.saturating_add(OVERLAP.saturating_mul(2)).saturating_mul(HOP)
        };
        let frame = audio
            .get(from..from.saturating_add(span).min(audio.len()))
            .unwrap_or(&[])
            .to_vec();
        self.decoded = self.decoded.saturating_add(if last { pending } else { CHUNK });
        Ok(self.mixer.push(&frame, CHUNK.saturating_mul(HOP), last))
    }
}

impl Iterator for NeuttsStream<'_> {
    type Item = Result<Audio, NodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let fail = |error: ort::Error| NodeError::Backend(format!("neutts: {error}"));
        if !self.started {
            self.started = true;
            if let Err(error) = self.start() {
                self.finished = true;
                return Some(Err(fail(error)));
            }
        }
        loop {
            let ready = self.codes.len().saturating_sub(self.decoded) >= CHUNK.saturating_add(LOOKAHEAD);
            if ready && !self.finished {
                return Some(self.decode(false).map(Audio::from).map_err(fail));
            }
            if self.finished {
                let pending = self.codes.len() > self.decoded;
                return pending.then(|| self.decode(true).map(Audio::from).map_err(fail));
            }
            let logits = std::mem::take(&mut self.logits);
            let full = self.length.saturating_add(1) >= CONTEXT;
            match self.session.draw(&logits, self.generated >= MIN_CODES) {
                Some(code) if !full => {
                    self.codes.push(code);
                    self.generated = self.generated.saturating_add(1);
                    if let Err(error) = self.step(vec![i64::from(code).saturating_add(SPEECH_ID)]) {
                        self.finished = true;
                        return Some(Err(fail(error)));
                    }
                }
                _ => self.finished = true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::workflow::synth::neutts::{Mixer, Phonemizer};

    #[test]
    fn test_split_keeps_punctuation_runs_in_order() {
        let runs = Phonemizer::split("¿Qué tal? Hoy, bien.");
        let joined: Vec<(&str, &str)> = runs
            .iter()
            .map(|(words, marks)| (words.as_str(), marks.as_str()))
            .collect();
        assert_eq!(joined, [("", "¿"), ("Qué tal", "? "), ("Hoy", ", "), ("bien", ".")]);
    }

    #[test]
    fn test_mixer_reconstructs_a_constant_signal_and_emits_every_sample_once() {
        let mut mixer = Mixer::default();
        let (stride, length) = (100, 140);
        let mut out = Vec::new();
        for index in 0..5 {
            out.extend(mixer.push(&vec![1.0; length], stride, index == 4));
        }
        assert_eq!(out.len(), 4 * stride + length);
        assert!(out.iter().all(|sample| (sample - 1.0).abs() < 1e-5));
    }
}
