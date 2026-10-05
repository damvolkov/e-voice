use std::time::Duration;

use e_voice_core::schema::error::{BackendError, NodeError};
use e_voice_core::schema::lang::Lang;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::schema::error::TrainError;
use e_voice_tts::schema::voice::VoiceState;
use e_voice_tts::workflow::synth::base::{Synth, SynthCaps, SynthSession};

pub const RATE: u32 = 24_000;
/// One 80 ms frame at [`RATE`], Pocket's granularity.
pub const FRAME: usize = 1920;
/// Characters spoken per frame.
pub const PACE: usize = 8;

/// Frame-by-frame generator: each `next` costs `delay`, like an autoregressive step.
#[derive(Debug)]
pub struct FakeSynth {
    pub delay: Duration,
}

/// Renders the whole sentence first, then slices it: the shape the contract forbids.
#[derive(Debug)]
pub struct FakeWhole {
    pub delay: Duration,
}

struct FakeSession {
    delay: Duration,
    whole: bool,
}

const CAPS: SynthCaps = SynthCaps {
    rate: RATE,
    langs: &Lang::ALL,
};

impl Synth for FakeSynth {
    fn caps(&self) -> SynthCaps {
        CAPS
    }

    fn open(&self, _lang: Lang, _voice: Option<&VoiceState>) -> Result<Box<dyn SynthSession>, BackendError> {
        Ok(Box::new(FakeSession {
            delay: self.delay,
            whole: false,
        }))
    }

    fn train(&self, clips: &[Audio], _text: Option<&str>) -> Result<VoiceState, TrainError> {
        let bytes: Vec<u8> = clips
            .iter()
            .flat_map(|clip| clip.iter())
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        (!bytes.is_empty()).then(|| bytes.into()).ok_or(TrainError::Empty)
    }
}

impl Synth for FakeWhole {
    fn caps(&self) -> SynthCaps {
        CAPS
    }

    fn open(&self, _lang: Lang, _voice: Option<&VoiceState>) -> Result<Box<dyn SynthSession>, BackendError> {
        Ok(Box::new(FakeSession {
            delay: self.delay,
            whole: true,
        }))
    }
}

impl SynthSession for FakeSession {
    fn speak<'a>(&'a mut self, sentence: &'a str) -> Box<dyn Iterator<Item = Result<Audio, NodeError>> + Send + 'a> {
        let frames = sentence.chars().count().div_ceil(PACE);
        let whole = self.delay * u32::try_from(frames).unwrap();
        let (upfront, step) = if self.whole {
            (whole, Duration::ZERO)
        } else {
            (Duration::ZERO, self.delay)
        };
        std::thread::sleep(upfront);
        Box::new((0..frames).map(move |_| {
            std::thread::sleep(step);
            Ok(Audio::from(vec![0.1; FRAME]))
        }))
    }
}
