use std::collections::VecDeque;
use std::fmt::{self, Debug};
use std::path::Path;
use std::time::Duration;

use sherpa_onnx::{VadModelConfig, VoiceActivityDetector};

use crate::config::vad::VadConfig;
use crate::schema::audio::Audio;
use crate::schema::error::BackendError;
use crate::schema::segment::SegmentSpan;
use crate::workflow::vad::base::{Vad, VadEvent, VadSession};

const RESET: u64 = 1 << 30;

/// sherpa-onnx segmenter shared by every VAD model family; brands only build its configuration.
#[derive(Debug, Clone)]
pub struct DetectorVad {
    config: VadModelConfig,
    window: usize,
    buffer: f32,
    lookback: u64,
    pad: u64,
    history: usize,
}

impl DetectorVad {
    /// `window` is the model frame in samples. Speech is assumed to begin `min_speech + window`
    /// before detection; enough history is kept to pad the longest possible segment.
    ///
    /// # Errors
    /// The model file does not exist.
    pub fn new(model: &Path, sherpa: VadModelConfig, window: usize, config: &VadConfig) -> Result<Self, BackendError> {
        model
            .is_file()
            .then_some(())
            .ok_or_else(|| BackendError::Missing(model.to_path_buf()))?;
        let window = window.max(1);
        let buffer = config.max_speech.saturating_add(Duration::from_secs(10)).as_secs_f32();
        let pad = Audio::length(config.pad);
        let span = Audio::length(config.max_speech.saturating_add(config.min_silence));
        let history = usize::try_from(span.saturating_add(pad.saturating_mul(2)))
            .unwrap_or(usize::MAX)
            .saturating_add(window.saturating_mul(2));
        let lookback = Audio::length(config.min_speech).saturating_add(window as u64);
        Ok(Self {
            config: sherpa,
            window,
            buffer,
            lookback,
            pad,
            history,
        })
    }
}

impl Vad for DetectorVad {
    fn open(&self) -> Result<Box<dyn VadSession>, BackendError> {
        let detector = VoiceActivityDetector::create(&self.config, self.buffer).ok_or(BackendError::Load("vad"))?;
        Ok(Box::new(DetectorSession {
            pad: self.pad,
            cap: self.history,
            history: VecDeque::new(),
            window: self.window,
            pending: Vec::with_capacity(self.window),
            detector,
            lookback: self.lookback,
            pos: 0,
            base: 0,
            speaking: false,
        }))
    }
}

/// One stream's detector, fed whole model windows only so segment bounds never depend on how the
/// caller chunks audio. Recent input is kept so each segment's audio carries `pad` on both sides. sherpa counts samples in `i32`, so the detector is reset while silent and
/// `base` carries the absolute offset, keeping spans valid on streams of any length.
pub struct DetectorSession {
    detector: VoiceActivityDetector,
    pad: u64,
    cap: usize,
    history: VecDeque<f32>,
    window: usize,
    pending: Vec<f32>,
    lookback: u64,
    pos: u64,
    base: u64,
    speaking: bool,
}

impl Debug for DetectorSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DetectorSession")
            .field("pos", &self.pos)
            .field("base", &self.base)
            .field("speaking", &self.speaking)
            .finish_non_exhaustive()
    }
}

impl DetectorSession {
    // ##### PRIVATE #####

    fn common_feed(&mut self, frame: &[f32]) {
        self.detector.accept_waveform(frame);
        self.pos = self.pos.saturating_add(frame.len() as u64);
        self.history.extend(frame);
        self.history.drain(..self.history.len().saturating_sub(self.cap));
    }

    fn common_drain(&mut self) -> Vec<VadEvent> {
        let (base, pad, pos) = (self.base, self.pad, self.pos);
        let first = pos.saturating_sub(self.history.len() as u64);
        let (detector, history) = (&self.detector, &self.history);
        std::iter::from_fn(|| {
            let segment = detector.front()?;
            detector.pop();
            let start = base.saturating_add(u64::try_from(segment.start()).unwrap_or(0));
            let end = start.saturating_add(segment.samples().len() as u64);
            let from = start.saturating_sub(pad).max(first);
            let to = end.saturating_add(pad).min(pos);
            let skip = usize::try_from(from.saturating_sub(first)).unwrap_or(usize::MAX);
            let take = usize::try_from(to.saturating_sub(from)).unwrap_or(0);
            let padded: Vec<f32> = history.iter().skip(skip).take(take).copied().collect();
            let audio = match (start >= first, padded.len() == take) {
                (true, true) => Audio::from(padded),
                (false, _) | (_, false) => Audio::from(segment.samples().to_vec()),
            };
            Some(VadEvent::End {
                span: SegmentSpan { start, end },
                audio,
            })
        })
        .collect()
    }

    fn push_window(&mut self, frame: &[f32], events: &mut Vec<VadEvent>) {
        self.common_feed(frame);
        let ended = self.common_drain();
        let detected = self.detector.detected();
        let started = detected && (!self.speaking || !ended.is_empty());
        let at = self.pos.saturating_sub(self.lookback).max(self.base);
        events.extend(ended);
        events.extend(started.then_some(VadEvent::Start { at }));
        self.speaking = detected;
        let idle = !self.speaking && self.detector.is_empty() && self.pos.saturating_sub(self.base) >= RESET;
        if idle {
            self.detector.reset();
            self.base = self.pos;
        }
    }
}

impl VadSession for DetectorSession {
    fn push(&mut self, audio: &[f32]) -> Vec<VadEvent> {
        let mut pending = std::mem::take(&mut self.pending);
        pending.extend_from_slice(audio);
        let mut events = Vec::new();
        let frames = pending.chunks_exact(self.window);
        let rest = frames.remainder().len();
        frames.for_each(|frame| self.push_window(frame, &mut events));
        pending.drain(..pending.len().saturating_sub(rest));
        self.pending = pending;
        events
    }

    fn flush(&mut self) -> Vec<VadEvent> {
        let rest = std::mem::take(&mut self.pending);
        self.common_feed(&rest);
        self.detector.flush();
        self.speaking = false;
        let events = self.common_drain();
        self.detector.reset();
        self.base = self.pos;
        events
    }
}
