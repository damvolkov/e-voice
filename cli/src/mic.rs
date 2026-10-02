use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Host, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use tokio::sync::mpsc;

#[derive(Debug, thiserror::Error)]
pub enum MicError {
    #[error("no input device{}; run `ecli devices`", .0.as_deref().map(|name| format!(" matching {name:?}")).unwrap_or_default())]
    Device(Option<String>),
    #[error("unsupported sample format {0}")]
    Format(SampleFormat),
    #[error("audio backend: {0}")]
    Backend(#[from] cpal::Error),
}

/// An input device (named, or the system default) captured as mono s16le at its own rate.
/// `level` holds the peak of the latest callback as `f32` bits, for a live meter.
/// The stream must stay alive on the thread that opened it.
pub struct Mic {
    _stream: Stream,
    pub rate: u32,
    pub name: String,
    pub level: Arc<AtomicU32>,
}

impl Mic {
    // ##### PRIVATE #####

    fn open_typed<T>(
        device: &cpal::Device,
        config: StreamConfig,
        chunks: mpsc::Sender<Vec<u8>>,
        level: Arc<AtomicU32>,
    ) -> Result<Stream, cpal::Error>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let channels = usize::from(config.channels).max(1);
        let scale = 1.0 / channels as f32;
        device.build_input_stream::<T, _, _>(
            config,
            move |data: &[T], _| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|frame| frame.iter().map(|sample| f32::from_sample(*sample)).sum::<f32>() * scale)
                    .collect();
                let peak = mono.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
                level.store(peak.to_bits(), Ordering::Relaxed);
                let bytes = mono
                    .into_iter()
                    .flat_map(|sample| ((sample.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes())
                    .collect();
                chunks.try_send(bytes).ok();
            },
            |error| eprintln!("ecli: microphone: {error}"),
            None,
        )
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// PulseAudio when present (served by PipeWire on modern Linux) names real sources; otherwise the
    /// platform default (ALSA, CoreAudio, WASAPI).
    #[must_use]
    pub fn host() -> Host {
        cpal::available_hosts()
            .into_iter()
            .find(|id| id.name() == "PulseAudio")
            .and_then(|id| cpal::host_from_id(id).ok())
            .unwrap_or_else(cpal::default_host)
    }

    /// `(name, is_default)` for every input device of [`Mic::host`].
    ///
    /// # Errors
    /// The audio backend cannot enumerate devices.
    pub fn devices() -> Result<Vec<(String, bool)>, MicError> {
        let host = Self::host();
        let default = host
            .default_input_device()
            .and_then(|device| device.description().ok())
            .map(|d| d.name().to_owned());
        Ok(host
            .input_devices()?
            .filter_map(|device| {
                device
                    .description()
                    .ok()
                    .map(|description| description.name().to_owned())
            })
            .map(|name| {
                let chosen = default.as_deref() == Some(name.as_str());
                (name, chosen)
            })
            .collect())
    }

    /// # Errors
    /// No matching device, an unsupported sample format, or the audio backend refusing the stream.
    pub fn open(name: Option<&str>, chunks: mpsc::Sender<Vec<u8>>) -> Result<Self, MicError> {
        let host = Self::host();
        let device = match name {
            None => host.default_input_device(),
            Some(wanted) => {
                let wanted = wanted.to_lowercase();
                host.input_devices()?.find(|device| {
                    device
                        .description()
                        .is_ok_and(|description| description.name().to_lowercase().contains(&wanted))
                })
            }
        }
        .ok_or_else(|| MicError::Device(name.map(str::to_owned)))?;
        let supported = device.default_input_config()?;
        let config = supported.config();
        let level = Arc::new(AtomicU32::new(0));
        let tap = Arc::clone(&level);
        let stream = match supported.sample_format() {
            SampleFormat::I16 => Self::open_typed::<i16>(&device, config, chunks, tap)?,
            SampleFormat::I32 => Self::open_typed::<i32>(&device, config, chunks, tap)?,
            SampleFormat::U16 => Self::open_typed::<u16>(&device, config, chunks, tap)?,
            SampleFormat::F32 => Self::open_typed::<f32>(&device, config, chunks, tap)?,
            other => return Err(MicError::Format(other)),
        };
        stream.play()?;
        let name = device
            .description()
            .map(|description| description.name().to_owned())
            .unwrap_or_default();
        Ok(Self {
            _stream: stream,
            rate: config.sample_rate,
            name,
            level,
        })
    }
}
