use std::fmt::{self, Debug};
use std::path::Path;

use e_voice_core::audio::{AudioEncoding, AudioIngest};
use e_voice_core::schema::error::BackendError;
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

const RATE: u32 = 16_000;
const MODEL: &str = "wespeaker_en_voxceleb_resnet34_LM.onnx";

/// Speaker similarity against a reference clip: cosine of speaker-verification embeddings
/// (WeSpeaker ResNet34, `VoxCeleb`), the standard voice-cloning metric. 1 is the same voice.
pub struct Similarity {
    extractor: SpeakerEmbeddingExtractor,
    reference: Vec<f32>,
}

impl Debug for Similarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Similarity")
            .field("dim", &self.reference.len())
            .finish_non_exhaustive()
    }
}

impl Similarity {
    // ##### PRIVATE #####

    fn embed(extractor: &SpeakerEmbeddingExtractor, audio: &[f32], rate: u32) -> Option<Vec<f32>> {
        let mut ingest = AudioIngest::new(rate, RATE, AudioEncoding::F32le).ok()?;
        let mut samples = ingest.feed(audio).ok()?;
        samples.extend(ingest.flush().ok()?);
        let stream = extractor.create_stream()?;
        stream.accept_waveform(i32::try_from(RATE).ok()?, &samples);
        stream.input_finished();
        extractor
            .is_ready(&stream)
            .then(|| extractor.compute(&stream))
            .flatten()
    }

    // ##### PUBLIC #####

    /// `dir` holds the installed speaker model; `reference` is mono audio at `rate`.
    ///
    /// # Errors
    /// The model is missing or rejected, or the reference is too short to embed.
    pub fn open(dir: &Path, reference: &[f32], rate: u32) -> Result<Self, BackendError> {
        let model = dir.join(MODEL);
        model
            .exists()
            .then_some(())
            .ok_or_else(|| BackendError::Missing(model.clone()))?;
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.display().to_string()),
            num_threads: 4,
            ..SpeakerEmbeddingExtractorConfig::default()
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config).ok_or(BackendError::Load("speaker model"))?;
        let reference = Self::embed(&extractor, reference, rate).ok_or(BackendError::Load("speaker reference"))?;
        Ok(Self { extractor, reference })
    }

    /// Cosine similarity of `audio` (mono at `rate`) to the reference; `None` if too short.
    #[must_use]
    pub fn score(&self, audio: &[f32], rate: u32) -> Option<f64> {
        let embedding = Self::embed(&self.extractor, audio, rate)?;
        let dot: f32 = embedding.iter().zip(&self.reference).map(|(a, b)| a * b).sum();
        let norm = |vector: &[f32]| vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        let scale = norm(&embedding) * norm(&self.reference);
        (scale > 0.0).then(|| f64::from(dot / scale))
    }
}
