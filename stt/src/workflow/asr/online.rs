use std::fmt::{self, Debug};
use std::sync::Arc;

use sherpa_onnx::{OnlineRecognizer, OnlineStream};

use crate::schema::audio::RATE;
use crate::schema::error::NodeError;
use crate::workflow::asr::base::AsrSession;

const SAMPLE_RATE: i32 = RATE.cast_signed();

/// One segment on a sherpa online recognizer: `lead` silence is fed by the caller at open, `tail`
/// silence flushes the last chunk at finish.
pub struct OnlineSession {
    recognizer: Arc<OnlineRecognizer>,
    stream: OnlineStream,
    tail: usize,
    last: String,
}

impl Debug for OnlineSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnlineSession")
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl OnlineSession {
    // ##### PRIVATE #####

    fn common_decode(&self) -> Option<String> {
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
        self.recognizer
            .get_result(&self.stream)
            .map(|result| result.text.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// Opens a stream on `recognizer` with `lead` samples of silence already fed.
    #[must_use]
    pub fn open(
        recognizer: &Arc<OnlineRecognizer>,
        prepare: impl FnOnce(&OnlineStream),
        lead: usize,
        tail: usize,
    ) -> Self {
        let stream = recognizer.create_stream();
        prepare(&stream);
        stream.accept_waveform(SAMPLE_RATE, &vec![0.0; lead]);
        Self {
            recognizer: Arc::clone(recognizer),
            stream,
            tail,
            last: String::new(),
        }
    }
}

impl AsrSession for OnlineSession {
    fn push(&mut self, audio: &[f32]) -> Option<String> {
        self.stream.accept_waveform(SAMPLE_RATE, audio);
        let text = self.common_decode().filter(|text| *text != self.last)?;
        self.last.clone_from(&text);
        Some(text)
    }

    fn finish(self: Box<Self>) -> Result<String, NodeError> {
        self.stream.accept_waveform(SAMPLE_RATE, &vec![0.0; self.tail]);
        self.stream.input_finished();
        self.common_decode()
            .ok_or_else(|| NodeError::Backend("streaming asr produced no result".to_owned()))
    }
}
