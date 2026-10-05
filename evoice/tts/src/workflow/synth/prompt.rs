use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use e_voice_core::schema::error::BackendError;
use safetensors::{Dtype, SafeTensors};

use crate::schema::audio::Audio;
use crate::schema::error::TrainError;
use crate::schema::voice::VoiceState;

const AUDIO: &str = "prompt";
const TEXT: &str = "text";
const LIMIT_S: usize = 30;
const PAUSE_DB: f32 = 35.0;

/// A voice as every backend learns it: a cleaned clip at [`Prompt::RATE`] (≤ 30 s, trailing silence
/// trimmed) and, optionally, what is said in it. Backends derive their own features from it when a
/// stream opens, so one stored voice serves every backend and model.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    pub audio: Vec<f32>,
    pub text: Option<String>,
}

impl Prompt {
    // ##### PRIVATE #####

    fn common_count(count: usize) -> f32 {
        f32::from(u16::try_from(count).unwrap_or(u16::MAX))
    }

    /// Upstream Pocket `end_on_pause`: trailing 20 ms frames 35 dB under the loudest are dropped, the
    /// end fades over one frame, then 80 ms of silence follow.
    fn prepare_pause(mut audio: Vec<f32>) -> Vec<f32> {
        let frame = Self::FRAME;
        let rms: Vec<f32> = audio
            .chunks(frame)
            .map(|chunk| (chunk.iter().map(|x| x * x).sum::<f32>() / Self::common_count(chunk.len())).sqrt())
            .collect();
        let peak = rms.iter().copied().fold(0.0_f32, f32::max);
        let floor = peak * 10_f32.powf(-PAUSE_DB / 20.0);
        let keep = rms
            .iter()
            .rposition(|&level| level > floor)
            .map_or(0, |last| last.saturating_add(1));
        audio.truncate(keep.saturating_mul(frame).min(audio.len()));
        let fade = audio.len().min(frame);
        let start = audio.len().saturating_sub(fade);
        for (index, sample) in audio.iter_mut().skip(start).enumerate() {
            *sample *= 1.0 - Self::common_count(index) / Self::common_count(fade.max(1));
        }
        audio.extend(std::iter::repeat_n(0.0, Self::TAIL));
        audio
    }

    // ##########################################################

    // ##### PUBLIC #####

    pub const RATE: u32 = 24_000;
    const FRAME: usize = Self::RATE as usize / 50;
    const TAIL: usize = Self::RATE as usize * 2 / 25;

    /// Clips of one speaker at [`Prompt::RATE`], joined in order.
    ///
    /// # Errors
    /// No audible speech in the clips.
    pub fn prepare(clips: &[Audio], text: Option<&str>) -> Result<Self, TrainError> {
        let joined: Vec<f32> = clips
            .iter()
            .flat_map(|clip| clip.iter().copied())
            .take(LIMIT_S.saturating_mul(Self::RATE as usize))
            .collect();
        let audio = Self::prepare_pause(joined);
        (audio.len() > Self::TAIL).then_some(()).ok_or(TrainError::Empty)?;
        let text = text.map(str::trim).filter(|text| !text.is_empty()).map(str::to_owned);
        Ok(Self { audio, text })
    }

    /// # Errors
    /// The serializer failed.
    pub fn encode(&self) -> Result<VoiceState, TrainError> {
        let bytes: Vec<u8> = self.audio.iter().flat_map(|value| value.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![self.audio.len()], &bytes)
            .map_err(|error| TrainError::Backend(error.to_string()))?;
        let metadata = self
            .text
            .as_ref()
            .map(|text| HashMap::from([(TEXT.to_owned(), text.clone())]));
        safetensors::serialize([(AUDIO, view)], metadata)
            .map(VoiceState::from)
            .map_err(|error| TrainError::Backend(error.to_string()))
    }

    /// # Errors
    /// The voice was not written by [`Prompt::encode`].
    pub fn decode(voice: &VoiceState) -> Result<Self, BackendError> {
        let (_, metadata) = SafeTensors::read_metadata(voice).map_err(|_| BackendError::Load("voice"))?;
        let text = metadata
            .metadata()
            .as_ref()
            .and_then(|metadata| metadata.get(TEXT))
            .cloned();
        let tensors = SafeTensors::deserialize(voice).map_err(|_| BackendError::Load("voice"))?;
        let view = tensors.tensor(AUDIO).map_err(|_| BackendError::Load("voice prompt"))?;
        (view.dtype() == Dtype::F32)
            .then_some(())
            .ok_or(BackendError::Load("voice dtype"))?;
        let audio = view
            .data()
            .chunks_exact(4)
            .filter_map(|bytes| bytes.try_into().ok().map(f32::from_le_bytes))
            .collect();
        Ok(Self { audio, text })
    }

    /// Identity of the prompt for caches of derived features.
    #[must_use]
    pub fn key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.audio.iter().for_each(|sample| sample.to_bits().hash(&mut hasher));
        self.text.hash(&mut hasher);
        hasher.finish()
    }
}
