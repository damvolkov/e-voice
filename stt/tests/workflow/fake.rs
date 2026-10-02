use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use e_voice_stt::schema::audio::Audio;
use e_voice_stt::schema::emotion::{Emotion, EmotionLabel};
use e_voice_stt::schema::error::{BackendError, NodeError};
use e_voice_stt::schema::event::WakeEvent;
use e_voice_stt::schema::lang::Lang;
use e_voice_stt::schema::segment::SegmentSpan;
use e_voice_stt::workflow::asr::base::{AsrSession, BatchAsr, StreamingAsr};
use e_voice_stt::workflow::asr::registry::AsrBackend;
use e_voice_stt::workflow::denoise::base::{Denoise, DenoiseSession};
use e_voice_stt::workflow::lid::base::Lid;
use e_voice_stt::workflow::nodes::Nodes;
use e_voice_stt::workflow::ser::base::Ser;
use e_voice_stt::workflow::vad::base::{Vad, VadEvent, VadSession};
use e_voice_stt::workflow::ww::base::{Ww, WwSession};

pub const SPEECH: f32 = 0.8;
pub const MARKER: f32 = 0.25;

#[derive(Debug)]
pub struct FakeVad;

#[derive(Default)]
struct FakeVadSession {
    pos: u64,
    speech: Option<(u64, Vec<f32>)>,
}

impl Vad for FakeVad {
    fn open(&self) -> Result<Box<dyn VadSession>, BackendError> {
        Ok(Box::new(FakeVadSession::default()))
    }
}

impl VadSession for FakeVadSession {
    fn push(&mut self, audio: &[f32]) -> Vec<VadEvent> {
        let mut events = Vec::new();
        for &sample in audio {
            match (self.speech.as_mut(), sample >= SPEECH) {
                (None, true) => {
                    events.push(VadEvent::Start { at: self.pos });
                    self.speech = Some((self.pos, vec![sample]));
                }
                (Some((_, buffer)), true) => buffer.push(sample),
                (Some(_), false) => events.extend(self.flush()),
                (None, false) => {}
            }
            self.pos += 1;
        }
        events
    }

    fn flush(&mut self) -> Vec<VadEvent> {
        self.speech
            .take()
            .map(|(start, buffer)| VadEvent::End {
                span: SegmentSpan {
                    start,
                    end: start + buffer.len() as u64,
                },
                audio: Audio::from(buffer),
            })
            .into_iter()
            .collect()
    }
}

#[derive(Debug)]
pub struct FakeStreaming;

struct FakeStreamingSession {
    lang: Lang,
    fed: usize,
}

impl StreamingAsr for FakeStreaming {
    fn open(&self, lang: Lang) -> Result<Box<dyn AsrSession>, BackendError> {
        Ok(Box::new(FakeStreamingSession { lang, fed: 0 }))
    }
}

impl AsrSession for FakeStreamingSession {
    fn push(&mut self, audio: &[f32]) -> Option<String> {
        self.fed += audio.len();
        Some(format!("{}", self.fed))
    }

    fn finish(self: Box<Self>) -> Result<String, NodeError> {
        Ok(format!("{:?}:{}", self.lang, self.fed))
    }
}

#[derive(Debug)]
pub struct FakeBatch {
    pub panic_on: usize,
    pub delay: Duration,
}

impl BatchAsr for FakeBatch {
    fn transcribe(&self, _lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        assert_ne!(audio.len(), self.panic_on, "injected panic");
        std::thread::sleep(self.delay);
        Ok(format!("batch:{}", audio.len()))
    }
}

/// A batch engine that answers `"<name>:<samples>"`, to tell engines apart.
#[derive(Debug)]
pub struct FakeNamed(pub &'static str);

impl BatchAsr for FakeNamed {
    fn transcribe(&self, _lang: Lang, audio: &[f32]) -> Result<String, NodeError> {
        Ok(format!("{}:{}", self.0, audio.len()))
    }
}

#[derive(Debug)]
pub struct FakeSer {
    pub delay: Duration,
}

impl Ser for FakeSer {
    fn classify(&self, _audio: &[f32]) -> Result<Emotion, NodeError> {
        std::thread::sleep(self.delay);
        Ok(Emotion {
            label: EmotionLabel::Happy,
            scores: BTreeMap::from([(EmotionLabel::Happy, 0.9), (EmotionLabel::Neutral, 0.1)]),
            model: Some("fake".into()),
        })
    }
}

#[derive(Debug)]
pub struct FakeWw;

struct FakeWwSession;

impl Ww for FakeWw {
    fn open(&self) -> Result<Box<dyn WwSession>, BackendError> {
        Ok(Box::new(FakeWwSession))
    }
}

impl WwSession for FakeWwSession {
    fn push(&mut self, audio: &[f32]) -> Result<Option<WakeEvent>, NodeError> {
        Ok(audio.contains(&MARKER).then(|| WakeEvent {
            keyword: "marker".into(),
            score: 1.0,
        }))
    }
}

/// Identifies English in segments of at least `english` samples, nothing in shorter ones.
#[derive(Debug)]
pub struct FakeLid {
    pub english: usize,
}

impl Lid for FakeLid {
    fn identify(&self, audio: &[f32]) -> Result<Option<Lang>, NodeError> {
        Ok((audio.len() >= self.english).then_some(Lang::En))
    }
}

/// Scales every sample by `gain` and holds back the last `hold` samples until flushed.
#[derive(Debug)]
pub struct FakeDenoise {
    pub gain: f32,
    pub hold: usize,
}

struct FakeDenoiseSession {
    gain: f32,
    hold: usize,
    held: Vec<f32>,
}

impl Denoise for FakeDenoise {
    fn open(&self) -> Result<Box<dyn DenoiseSession>, BackendError> {
        Ok(Box::new(FakeDenoiseSession {
            gain: self.gain,
            hold: self.hold,
            held: Vec::new(),
        }))
    }
}

impl DenoiseSession for FakeDenoiseSession {
    fn push(&mut self, audio: &[f32]) -> Vec<f32> {
        self.held.extend(audio.iter().map(|sample| sample * self.gain));
        let ready = self.held.len().saturating_sub(self.hold);
        self.held.drain(..ready).collect()
    }

    fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.held)
    }
}

pub fn nodes(asr: AsrBackend, ser: Option<Arc<dyn Ser>>, ww: Option<Arc<dyn Ww>>) -> Arc<Nodes> {
    Arc::new(Nodes {
        denoise: None,
        ww,
        vad: Arc::new(FakeVad),
        lid: None,
        asr,
        offline: None,
        extra: Vec::new(),
        ser,
    })
}

pub fn stratified(live: AsrBackend, offline: AsrBackend) -> Arc<Nodes> {
    Arc::new(Nodes {
        denoise: None,
        ww: None,
        vad: Arc::new(FakeVad),
        lid: None,
        asr: live,
        offline: Some(offline),
        extra: Vec::new(),
        ser: None,
    })
}

pub fn batch(panic_on: usize) -> AsrBackend {
    AsrBackend::Batch(Arc::new(FakeBatch {
        panic_on,
        delay: Duration::ZERO,
    }))
}

pub fn slow(delay_ms: u64) -> AsrBackend {
    AsrBackend::Batch(Arc::new(FakeBatch {
        panic_on: usize::MAX,
        delay: Duration::from_millis(delay_ms),
    }))
}

pub fn ser(delay_ms: u64) -> Arc<dyn Ser> {
    Arc::new(FakeSer {
        delay: Duration::from_millis(delay_ms),
    })
}
