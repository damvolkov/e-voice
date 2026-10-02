use tokio::sync::mpsc;

use crate::api::state::AppState;
use crate::core::audio::AudioFile;
use crate::schema::event::Event;
use crate::schema::lang::Lang;
use crate::schema::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchError {
    #[error("{0}")]
    Decode(String),
    #[error("{0}")]
    Failed(String),
}

/// An uploaded file through the lossless file intake: the path every REST protocol shares.
#[derive(Debug)]
pub struct Batch;

impl Batch {
    /// Container sniffing, decoding and resampling, off the async threads.
    ///
    /// # Errors
    /// The bytes are not decodable audio.
    pub async fn decode(bytes: Vec<u8>, extension: Option<String>) -> Result<Vec<f32>, BatchError> {
        tokio::task::spawn_blocking(move || AudioFile::decode(bytes, extension.as_deref()))
            .await
            .map_err(|error| BatchError::Failed(error.to_string()))?
            .map_err(|error| BatchError::Decode(error.to_string()))
    }

    /// The whole transcript; a failed segment fails the request rather than silently dropping text.
    ///
    /// # Errors
    /// Undecodable audio, a pipeline failure, or any segment failing.
    pub async fn transcribe(
        state: &AppState,
        bytes: Vec<u8>,
        extension: Option<String>,
        lang: Lang,
    ) -> Result<Transcript, BatchError> {
        let samples = Self::decode(bytes, extension).await?;
        tracing::info!(
            ?lang,
            seconds = Transcript::seconds(samples.len() as u64),
            "transcription.start"
        );
        let transcript = state
            .runner
            .transcribe(lang, samples)
            .await
            .map_err(|error| BatchError::Failed(error.to_string()))?;
        let failed: Vec<String> = transcript
            .segments
            .iter()
            .filter_map(|segment| segment.error.as_ref().map(ToString::to_string))
            .collect();
        match failed.as_slice() {
            [] => Ok(transcript),
            [first, ..] => Err(BatchError::Failed(format!(
                "{} segment(s) failed: {first}",
                failed.len()
            ))),
        }
    }

    /// Events of a decoded file as they are produced, for streaming responses; returns the audio
    /// length in seconds alongside.
    ///
    /// # Errors
    /// Undecodable audio.
    pub async fn stream(
        state: &AppState,
        bytes: Vec<u8>,
        extension: Option<String>,
        lang: Lang,
    ) -> Result<(mpsc::Receiver<Event>, f64), BatchError> {
        let samples = Self::decode(bytes, extension).await?;
        let seconds = Transcript::seconds(samples.len() as u64);
        Ok((state.runner.file(lang, samples), seconds))
    }
}
