use std::fmt::{self, Debug};
use std::io::Cursor;
use std::sync::OnceLock;

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use serde::{Deserialize, Serialize};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as MediaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia_adapter_libopus::OpusDecoder;

const CHUNK: usize = 1_024;

/// Little-endian mono PCM layout of incoming bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioEncoding {
    #[default]
    S16le,
    F32le,
}

impl AudioEncoding {
    const fn width(self) -> usize {
        match self {
            Self::S16le => 2,
            Self::F32le => 4,
        }
    }

    fn decode(self, bytes: &[u8]) -> f32 {
        match (self, bytes) {
            (Self::S16le, &[a, b]) => f32::from(i16::from_le_bytes([a, b])) / 32_768.0,
            (Self::F32le, &[a, b, c, d]) => f32::from_le_bytes([a, b, c, d]).clamp(-1.0, 1.0),
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AudioError {
    #[error("unsupported sample rate {0}")]
    Rate(u32),
    #[error("resampler failed: {0}")]
    Resample(String),
    #[error("cannot decode audio: {0}")]
    Container(String),
}

/// Byte stream of any rate and encoding → mono f32 at a target rate, fed in pieces of any size.
/// Partial samples split across pieces are carried; resampling is FFT-based and streaming.
pub struct AudioIngest {
    encoding: AudioEncoding,
    carry: Vec<u8>,
    resampler: Option<Fft<f32>>,
    pending: Vec<f32>,
    out: Vec<f32>,
}

impl Debug for AudioIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioIngest")
            .field("encoding", &self.encoding)
            .field("resampling", &self.resampler.is_some())
            .finish_non_exhaustive()
    }
}

impl AudioIngest {
    // ##### PRIVATE #####

    fn push_resample(&mut self, last: bool) -> Result<Vec<f32>, AudioError> {
        let Some(resampler) = self.resampler.as_mut() else {
            return Ok(std::mem::take(&mut self.pending));
        };
        let fail = |error: &dyn fmt::Display| AudioError::Resample(error.to_string());
        let mut output = Vec::new();
        loop {
            let need = resampler.input_frames_next();
            let available = self.pending.len();
            let partial = match (available >= need, last && available > 0) {
                (true, _) => None,
                (false, true) => Some(available),
                (false, false) => break,
            };
            let mut input = std::mem::take(&mut self.pending);
            input.resize(available.max(need), 0.0);
            let source = InterleavedSlice::new(&input, 1, input.len()).map_err(|error| fail(&error))?;
            let capacity = self.out.len();
            let mut sink = InterleavedSlice::new_mut(&mut self.out, 1, capacity).map_err(|error| fail(&error))?;
            let indexing = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: partial,
                active_channels_mask: None,
            };
            let (used, written) = resampler
                .process_into_buffer(&source, &mut sink, Some(&indexing))
                .map_err(|error| fail(&error))?;
            output.extend_from_slice(self.out.get(..written).unwrap_or_default());
            input.truncate(available);
            input.drain(..used.min(available));
            self.pending = input;
            if partial.is_some() {
                break;
            }
        }
        Ok(output)
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// A zero rate, or a pair the resampler cannot convert between.
    pub fn new(rate: u32, target: u32, encoding: AudioEncoding) -> Result<Self, AudioError> {
        let resampler = match (rate, target) {
            (0, _) => return Err(AudioError::Rate(rate)),
            (_, 0) => return Err(AudioError::Rate(target)),
            _ if rate == target => None,
            _ => Some(
                Fft::<f32>::new(rate as usize, target as usize, CHUNK, 1, FixedSync::Input)
                    .map_err(|_| AudioError::Rate(rate))?,
            ),
        };
        let out = vec![0.0; resampler.as_ref().map_or(0, Resampler::output_frames_max)];
        Ok(Self {
            encoding,
            carry: Vec::new(),
            resampler,
            pending: Vec::new(),
            out,
        })
    }

    /// # Errors
    /// The resampler failed.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<f32>, AudioError> {
        let mut data = std::mem::take(&mut self.carry);
        data.extend_from_slice(bytes);
        let samples = data.chunks_exact(self.encoding.width());
        self.carry = samples.remainder().to_vec();
        let encoding = self.encoding;
        self.pending.extend(samples.map(|sample| encoding.decode(sample)));
        self.push_resample(false)
    }

    /// Same as [`AudioIngest::push`] for samples already decoded to f32 mono.
    ///
    /// # Errors
    /// The resampler failed.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        self.pending.extend_from_slice(samples);
        self.push_resample(false)
    }

    /// Resamples whatever input remains, padding the last chunk with silence.
    ///
    /// # Errors
    /// The resampler failed.
    pub fn flush(&mut self) -> Result<Vec<f32>, AudioError> {
        self.carry.clear();
        self.push_resample(true)
    }
}

/// Peak normalizer for quiet inputs: instant attack, exponential release, gain in `[1, max]` toward
/// a target peak. Applied sample by sample, so the result never depends on chunking; it only ever
/// boosts, so normal-level audio passes untouched once its first peak is seen (the rising edge of the
/// first wave after silence is lifted for a few milliseconds).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioGain {
    target: f32,
    max: f32,
    decay: f32,
    peak: f32,
}

impl AudioGain {
    /// `target_db` and `max_db` in dB (peak dBFS, maximum boost); `release` is the time for the tracked
    /// peak to fall by 1/e at `rate`. A `max_db` of 0 disables the stage.
    #[must_use]
    pub fn new(rate: u32, target_db: f32, max_db: f32, release: std::time::Duration) -> Self {
        let samples = release.as_secs_f32() * f32::from(u16::try_from(rate / 10).unwrap_or(u16::MAX)) * 10.0;
        Self {
            target: 10f32.powf(target_db / 20.0),
            max: 10f32.powf(max_db.max(0.0) / 20.0),
            decay: (-1.0 / samples.max(1.0)).exp(),
            peak: 0.0,
        }
    }

    #[must_use]
    pub fn enabled(&self) -> bool {
        self.max > 1.0
    }

    pub fn apply(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            self.peak = sample.abs().max(self.peak * self.decay);
            let gain = (self.target / self.peak.max(f32::MIN_POSITIVE)).clamp(1.0, self.max);
            *sample = (*sample * gain).clamp(-1.0, 1.0);
        }
    }
}

/// A whole uploaded file (wav, mp3, m4a/aac, flac, ogg/vorbis, ogg/opus, webm/opus) →
/// mono f32 at `rate`. The container is sniffed; `extension` is only a hint.
#[derive(Debug)]
pub struct AudioFile;

impl AudioFile {
    // ##### PRIVATE #####

    fn decode_codecs() -> &'static CodecRegistry {
        static CODECS: OnceLock<CodecRegistry> = OnceLock::new();
        CODECS.get_or_init(|| {
            let mut registry = CodecRegistry::new();
            symphonia::default::register_enabled_codecs(&mut registry);
            registry.register_audio_decoder::<OpusDecoder>();
            registry
        })
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// CPU-bound: call from a blocking context.
    ///
    /// # Errors
    /// Unknown container or codec, no audio track, a fatal decode error, or no audio at all.
    pub fn decode(bytes: Vec<u8>, extension: Option<&str>, target: u32) -> Result<Vec<f32>, AudioError> {
        let fail = |error: &dyn fmt::Display| AudioError::Container(error.to_string());
        let mut hint = Hint::new();
        if let Some(extension) = extension {
            hint.with_extension(extension);
        }
        let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), MediaSourceStreamOptions::default());
        let mut format = symphonia::default::get_probe()
            .probe(&hint, source, FormatOptions::default(), MetadataOptions::default())
            .map_err(|error| fail(&error))?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| fail(&"no audio track"))?;
        let (id, params) = (
            track.id,
            track.codec_params.as_ref().and_then(|params| params.audio()).cloned(),
        );
        let params = params.ok_or_else(|| fail(&"no audio codec parameters"))?;
        let mut decoder = Self::decode_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|error| fail(&error))?;
        let (mut ingest, mut output, mut interleaved) = (None::<AudioIngest>, Vec::new(), Vec::<f32>::new());
        loop {
            // Streamed containers (MediaRecorder WebM: unknown-size segments, no cues) end in an
            // unexpected EOF rather than a clean end of stream.
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) | Err(MediaError::ResetRequired) => break,
                Err(MediaError::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(fail(&error)),
            };
            if packet.track_id != id {
                continue;
            }
            let buffer = match decoder.decode(&packet) {
                Ok(buffer) => buffer,
                Err(MediaError::DecodeError(_) | MediaError::IoError(_)) => continue,
                Err(error) => return Err(fail(&error)),
            };
            let channels = buffer.spec().channels().count().max(1);
            let rate = buffer.spec().rate();
            buffer.copy_to_vec_interleaved(&mut interleaved);
            let scale = 1.0 / f32::from(u16::try_from(channels).unwrap_or(u16::MAX));
            let mono: Vec<f32> = interleaved
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() * scale)
                .collect();
            let ingest = match ingest.as_mut() {
                Some(ingest) => ingest,
                None => ingest.insert(AudioIngest::new(rate, target, AudioEncoding::F32le)?),
            };
            output.extend(ingest.feed(&mono)?);
        }
        let mut ingest = ingest.ok_or_else(|| fail(&"no decodable audio"))?;
        output.extend(ingest.flush()?);
        Ok(output)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use crate::audio::{AudioEncoding, AudioFile, AudioGain, AudioIngest};

    fn s16(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
    }

    #[test]
    fn test_native_rate_decodes_and_carries_split_samples() {
        let mut ingest = AudioIngest::new(16_000, 16_000, AudioEncoding::S16le).unwrap();
        let bytes = s16(&[16_384, -16_384, 0]);
        let mut out = ingest.push(&bytes[..3]).unwrap();
        out.extend(ingest.push(&bytes[3..]).unwrap());
        assert_eq!(out, [0.5, -0.5, 0.0]);
    }

    #[test]
    fn test_f32_is_clamped() {
        let mut ingest = AudioIngest::new(16_000, 16_000, AudioEncoding::F32le).unwrap();
        let bytes: Vec<u8> = [2.0f32, -0.25].iter().flat_map(|sample| sample.to_le_bytes()).collect();
        assert_eq!(ingest.push(&bytes).unwrap(), [1.0, -0.25]);
    }

    #[test]
    fn test_resampling_preserves_duration_and_tone() {
        let rate = 48_000u32;
        let tone: Vec<i16> = (0..rate)
            .map(|n| ((n as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 16_000.0) as i16)
            .collect();
        let mut ingest = AudioIngest::new(rate, 16_000, AudioEncoding::S16le).unwrap();
        let mut out: Vec<f32> = tone
            .chunks(777)
            .flat_map(|chunk| ingest.push(&s16(chunk)).unwrap())
            .collect();
        out.extend(ingest.flush().unwrap());
        assert!((15_500..=16_500).contains(&out.len()), "{}", out.len());
        let crossings = out[2_000..14_000]
            .windows(2)
            .filter(|pair| pair[0] < 0.0 && pair[1] >= 0.0)
            .count();
        assert!((325..=335).contains(&crossings), "{crossings}");
    }

    fn tone(amplitude: f32, samples: usize) -> Vec<f32> {
        (0..samples).map(|n| amplitude * (n as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_gain_lifts_quiet_audio_to_the_target_peak_within_the_cap() {
        let mut quiet = tone(0.006, 16_000);
        AudioGain::new(16_000, -6.0, 40.0, Duration::from_secs(5)).apply(&mut quiet);
        let peak = quiet[1_000..]
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!((0.45..=0.51).contains(&peak), "{peak}");
        let mut whisper = tone(0.000_1, 16_000);
        AudioGain::new(16_000, -6.0, 40.0, Duration::from_secs(5)).apply(&mut whisper);
        let capped = whisper[1_000..]
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!((0.0095..=0.0101).contains(&capped), "{capped}");
    }

    #[test]
    fn test_gain_never_attenuates_and_can_be_disabled() {
        let loud = tone(0.9, 4_000);
        let mut boosted = loud.clone();
        AudioGain::new(16_000, -6.0, 40.0, Duration::from_secs(5)).apply(&mut boosted);
        assert_eq!(boosted[100..], loud[100..]);
        let mut off = tone(0.003, 4_000);
        let stage = AudioGain::new(16_000, -6.0, 0.0, Duration::from_secs(5));
        assert!(!stage.enabled());
        let reference = off.clone();
        stage.clone().apply(&mut off);
        assert_eq!(off, reference);
    }

    #[test]
    fn test_gain_does_not_depend_on_chunking() {
        let signal = [tone(0.002, 8_000), tone(0.3, 4_000), tone(0.01, 8_000)].concat();
        let mut whole = signal.clone();
        AudioGain::new(16_000, -6.0, 40.0, Duration::from_secs(1)).apply(&mut whole);
        let mut stage = AudioGain::new(16_000, -6.0, 40.0, Duration::from_secs(1));
        let mut pieces = signal;
        pieces.chunks_mut(37).for_each(|chunk| stage.apply(chunk));
        assert_eq!(whole, pieces);
    }

    #[test]
    fn test_zero_rate_is_rejected() {
        assert!(AudioIngest::new(0, 16_000, AudioEncoding::S16le).is_err());
    }

    #[test]
    fn test_file_decodes_every_openai_format_to_16k_mono() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/resources/audio");
        let files = ["wav", "mp3", "m4a", "flac", "ogg", "opus", "webm"].map(|extension| format!("tone.{extension}"));
        // What browsers' MediaRecorder writes: unknown-size WebM without cues, fragmented MP4.
        let recorded = ["recorder.webm", "recorder.m4a"].map(str::to_owned);
        for extension in files.iter().chain(&recorded) {
            let bytes = std::fs::read(dir.join(extension)).unwrap();
            let samples = AudioFile::decode(bytes, None, 16_000).unwrap();
            assert!(
                (15_000..=17_500).contains(&samples.len()),
                "{extension}: {}",
                samples.len()
            );
            let middle = &samples[3_000..13_000];
            let crossings = middle.windows(2).filter(|pair| pair[0] < 0.0 && pair[1] >= 0.0).count();
            assert!((270..=280).contains(&crossings), "{extension}: {crossings} crossings");
        }
    }

    #[test]
    fn test_file_rejects_garbage() {
        assert!(AudioFile::decode(b"definitely not audio".to_vec(), Some("mp3"), 16_000).is_err());
    }
}
