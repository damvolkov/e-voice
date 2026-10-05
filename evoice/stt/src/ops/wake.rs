use std::path::{Path, PathBuf};

use e_voice_core::audio::AudioFile;
use e_voice_core::models::ModelStore;
use e_voice_core::schema::error::BackendError;

use crate::config::ww::{WwBackend, WwConfig};
use crate::ops::bench::manifest::Manifest;
use crate::ops::voice::Voice;
use crate::schema::audio::{Audio, RATE};
use crate::workflow::ww::base::Ww;
use crate::workflow::ww::kws::KwsWw;

const CARRIERS: [&str; 5] = ["{}.", "{}!", "Hey, {}.", "{}, are you there?", "Okay {}, let's start."];
const SPEEDS: [f32; 3] = [0.85, 1.0, 1.2];
const DECOYS: [&str; 6] = [
    "The eagle landed near the river.",
    "Either way, we leave at nine.",
    "Igor sent the report this morning.",
    "His ego got in the way.",
    "Regular updates arrive every day.",
    "Turn the music down a little.",
];
const THRESHOLDS: [f32; 8] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5];
const BOOSTS: [f32; 5] = [1.0, 1.5, 2.0, 3.0, 4.0];

#[derive(Debug, thiserror::Error)]
pub enum WakeError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("model {0} is not in the manifest or not installed; run `make setup ARGS=--all`")]
    Model(String),
}

/// One point of the sweep: how often the phrase fires when said, and how often it fires when not.
#[derive(Debug, Clone, PartialEq)]
pub struct WakeTrial {
    pub threshold: f32,
    pub boost: f32,
    pub recall: f64,
    pub voices: Vec<(String, f64)>,
    pub false_accepts: usize,
    pub per_hour: f64,
}

/// Enrollment for open-vocabulary keyword spotting: nothing is trained; the phrase is synthesized
/// with several voices, carriers and speeds (positives) and played against decoys and real speech
/// (negatives) for every threshold × boost, so the chosen pair is measured rather than guessed.
#[derive(Debug)]
pub struct Wake {
    pub phrase: String,
    dir: PathBuf,
    positives: Vec<(String, Vec<f32>)>,
    negatives: Vec<Vec<f32>>,
    pub negative_s: f64,
}

impl Wake {
    // ##### PRIVATE #####

    fn sweep_one(&self, threshold: f32, boost: f32) -> Result<WakeTrial, BackendError> {
        let config = WwConfig {
            backend: WwBackend::Kws,
            keyword: self.phrase.clone(),
            threshold: Some(threshold),
            boost: Some(boost),
            ..WwConfig::default()
        };
        let spotter = KwsWw::new(&self.dir, &config)?;
        let fired = |audio: &[f32]| -> Result<usize, BackendError> {
            let mut session = spotter.open()?;
            let mut hits = 0usize;
            for chunk in audio.chunks(1_600) {
                if session.push(chunk).map_err(|_| BackendError::Load("kws"))?.is_some() {
                    hits = hits.saturating_add(1);
                }
            }
            Ok(hits)
        };
        let heard: Vec<(String, usize)> = self
            .positives
            .iter()
            .map(|(voice, audio)| fired(audio).map(|hits| (voice.clone(), hits.min(1))))
            .collect::<Result<_, _>>()?;
        let detected: usize = heard.iter().map(|(_, hit)| hit).sum();
        let false_accepts = self
            .negatives
            .iter()
            .map(|audio| fired(audio))
            .sum::<Result<usize, _>>()?;
        let count = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let mut names: Vec<String> = heard.iter().map(|(voice, _)| voice.clone()).collect();
        names.dedup();
        let voices = names
            .into_iter()
            .map(|name| {
                let mine: Vec<usize> = heard
                    .iter()
                    .filter(|(voice, _)| *voice == name)
                    .map(|(_, hit)| *hit)
                    .collect();
                let rate = count(mine.iter().sum()) / count(mine.len()).max(1.0);
                (name, rate)
            })
            .collect();
        Ok(WakeTrial {
            threshold,
            boost,
            recall: count(detected) / count(self.positives.len()).max(1.0),
            voices,
            false_accepts,
            per_hour: count(false_accepts) * 3_600.0 / self.negative_s.max(1.0),
        })
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// Synthesizes positives and decoys and loads up to `limit` clips from each real-speech manifest.
    ///
    /// # Errors
    /// A needed model is missing, or synthesis fails.
    pub fn prepare(store: &ModelStore, phrase: &str, speech: &[PathBuf], limit: usize) -> Result<Self, WakeError> {
        let installed = |id: &str| store.dir(id).ok().filter(|dir| dir.is_dir());
        let dir = installed("kws-gigaspeech").ok_or_else(|| WakeError::Model("kws-gigaspeech".to_owned()))?;
        let voices: Vec<Voice> = Voice::dirs(installed)
            .iter()
            .map(|dir| Voice::open(dir))
            .collect::<Result<_, _>>()?;
        if voices.is_empty() {
            return Err(WakeError::Model(Voice::IDS.join(", ")));
        }
        let pad = vec![0.0f32; usize::try_from(Audio::length(std::time::Duration::from_millis(500))).unwrap_or(8_000)];
        let padded = |speech: Vec<f32>| [pad.clone(), speech, pad.clone()].concat();
        let mut positives = Vec::new();
        let mut negatives = Vec::new();
        for voice in &voices {
            for carrier in CARRIERS {
                for speed in SPEEDS {
                    positives.push((
                        voice.name.clone(),
                        padded(voice.speak(&carrier.replace("{}", phrase), speed)?),
                    ));
                }
            }
            for decoy in DECOYS {
                negatives.push(padded(voice.speak(decoy, 1.0)?));
            }
        }
        for manifest in speech.iter().filter(|path| path.is_file()) {
            let items = Manifest::load(manifest)
                .map(|manifest| manifest.items)
                .unwrap_or_default();
            for item in items.into_iter().take(limit) {
                let decoded = std::fs::read(&item.audio)
                    .ok()
                    .and_then(|bytes| AudioFile::decode(bytes, Some("wav"), RATE).ok());
                negatives.extend(decoded.map(padded));
            }
        }
        let samples: usize = negatives.iter().map(Vec::len).sum();
        let duration = f64::from(u32::try_from(samples).unwrap_or(u32::MAX)) / 16_000.0;
        Ok(Self {
            phrase: phrase.to_owned(),
            dir,
            positives,
            negatives,
            negative_s: duration,
        })
    }

    #[must_use]
    pub fn positives(&self) -> usize {
        self.positives.len()
    }

    /// Every threshold × boost, evaluated in parallel.
    ///
    /// # Errors
    /// The spotter could not be built for some pair.
    pub fn sweep(&self) -> Result<Vec<WakeTrial>, BackendError> {
        let grid: Vec<(f32, f32)> = THRESHOLDS
            .iter()
            .flat_map(|threshold| BOOSTS.iter().map(move |boost| (*threshold, *boost)))
            .collect();
        let workers = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
        let size = grid.len().div_ceil(workers.max(1)).max(1);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = grid
                .chunks(size)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|(threshold, boost)| self.sweep_one(*threshold, *boost))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            jobs.into_iter()
                .flat_map(|job| {
                    job.join()
                        .unwrap_or_else(|_| vec![Err(BackendError::Load("kws sweep"))])
                })
                .collect()
        })
    }

    /// Highest recall with the fewest false accepts; ties go to the stricter threshold, then the
    /// smaller boost.
    #[must_use]
    pub fn pick(trials: &[WakeTrial]) -> Option<WakeTrial> {
        let useful = trials.iter().any(|trial| trial.recall > 0.0);
        trials
            .iter()
            .filter(|trial| !useful || trial.recall > 0.0)
            .min_by(|a, b| {
                a.false_accepts
                    .cmp(&b.false_accepts)
                    .then(b.recall.total_cmp(&a.recall))
                    .then(b.threshold.total_cmp(&a.threshold))
                    .then(a.boost.total_cmp(&b.boost))
            })
            .cloned()
    }

    #[must_use]
    pub fn manifests(data: &Path) -> Vec<PathBuf> {
        ["fleurs-en", "fleurs-es"]
            .iter()
            .map(|set| data.join("ops/datasets").join(set).join("manifest.jsonl"))
            .collect()
    }
}
