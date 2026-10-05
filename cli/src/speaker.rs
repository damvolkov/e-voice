use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::mic::Mic;

#[derive(Debug, thiserror::Error)]
pub enum SpeakerError {
    #[error("no output device")]
    Device,
    #[error("unsupported sample format {0}")]
    Format(SampleFormat),
    #[error("audio backend: {0}")]
    Backend(#[from] cpal::Error),
}

/// Mono samples queued at the device rate; the playback callback drains them.
#[derive(Debug, Default)]
struct Queue {
    samples: VecDeque<f32>,
    /// Fractional read position into the source and the previous source sample, for continuity.
    phase: f64,
    last: f32,
}

/// The default output device, fed mono f32 at any rate (resampled linearly: a tester, not a mixer).
/// The stream must stay alive on the thread that opened it.
pub struct Speaker {
    _stream: Stream,
    queue: Arc<Mutex<Queue>>,
    pub rate: u32,
}

impl Speaker {
    // ##### PRIVATE #####

    fn open_typed<T>(
        device: &cpal::Device,
        config: StreamConfig,
        queue: Arc<Mutex<Queue>>,
    ) -> Result<Stream, cpal::Error>
    where
        T: SizedSample + FromSample<f32>,
    {
        let channels = usize::from(config.channels).max(1);
        device.build_output_stream::<T, _, _>(
            config,
            move |data: &mut [T], _| {
                let mut queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
                for frame in data.chunks_mut(channels) {
                    let sample = T::from_sample(queue.samples.pop_front().unwrap_or(0.0));
                    for slot in frame.iter_mut() {
                        *slot = sample;
                    }
                }
            },
            |error| eprintln!("ecli: speaker: {error}"),
            None,
        )
    }

    // ##### PUBLIC #####

    /// # Errors
    /// No output device, an unsupported sample format, or the backend refusing the stream.
    pub fn open() -> Result<Self, SpeakerError> {
        let device = Mic::host().default_output_device().ok_or(SpeakerError::Device)?;
        let supported = device.default_output_config()?;
        let config = supported.config();
        let queue = Arc::new(Mutex::new(Queue::default()));
        let stream = match supported.sample_format() {
            SampleFormat::I16 => Self::open_typed::<i16>(&device, config, Arc::clone(&queue))?,
            SampleFormat::I32 => Self::open_typed::<i32>(&device, config, Arc::clone(&queue))?,
            SampleFormat::U16 => Self::open_typed::<u16>(&device, config, Arc::clone(&queue))?,
            SampleFormat::F32 => Self::open_typed::<f32>(&device, config, Arc::clone(&queue))?,
            other => return Err(SpeakerError::Format(other)),
        };
        stream.play()?;
        Ok(Self {
            _stream: stream,
            queue,
            rate: config.sample_rate,
        })
    }

    /// Queues `samples` recorded at `rate`.
    pub fn push(&self, samples: &[f32], rate: u32) {
        let step = f64::from(rate) / f64::from(self.rate.max(1));
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        let (mut phase, mut last) = (queue.phase, queue.last);
        for &sample in samples {
            while phase < 1.0 {
                let mixed = last + (sample - last) * phase as f32;
                queue.samples.push_back(mixed);
                phase += step;
            }
            phase -= 1.0;
            last = sample;
        }
        (queue.phase, queue.last) = (phase, last);
    }

    /// Drops whatever has not been played yet (barge-in).
    pub fn clear(&self) {
        self.queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .samples
            .clear();
    }

    /// Seconds still queued.
    #[must_use]
    pub fn pending(&self) -> f64 {
        let queued = self.queue.lock().unwrap_or_else(PoisonError::into_inner).samples.len();
        queued as f64 / f64::from(self.rate.max(1))
    }
}
