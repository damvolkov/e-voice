use std::path::Path;
use std::time::Duration;

use tokio::sync::mpsc;

/// A WAV file replayed in real time as mono s16le, for reproducible runs without a microphone.
#[derive(Debug)]
pub struct Wav;

impl Wav {
    /// Spawns the replay and returns the file's sample rate.
    ///
    /// # Errors
    /// The file cannot be read as WAV.
    pub fn open(path: &Path, chunks: mpsc::Sender<Vec<u8>>) -> Result<u32, hound::Error> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let channels = usize::from(spec.channels).max(1);
        let peak = match spec.sample_format {
            hound::SampleFormat::Float => 1.0,
            hound::SampleFormat::Int => f32::powi(2.0, i32::from(spec.bits_per_sample).saturating_sub(1)),
        };
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / peak))
                .collect::<Result<_, _>>()?,
        };
        let mono: Vec<u8> = samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .flat_map(|sample| ((sample.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes())
            .collect();
        let step = (spec.sample_rate as usize / 10).max(1).saturating_mul(2);
        tokio::spawn(async move {
            let mut pace = tokio::time::interval(Duration::from_millis(100));
            for chunk in mono.chunks(step) {
                pace.tick().await;
                if chunks.send(chunk.to_vec()).await.is_err() {
                    return;
                }
            }
        });
        Ok(spec.sample_rate)
    }
}
