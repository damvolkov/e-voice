use e_voice_core::schema::error::NodeError;

use crate::schema::audio::Audio;

/// Position of a sentence within its stream, starting at 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SentenceId(pub u64);

/// What a stream tells its client, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Start {
        sentence: SentenceId,
        text: String,
    },
    Audio {
        sentence: SentenceId,
        audio: Audio,
    },
    /// `error` is set when the sentence was cut short.
    End {
        sentence: SentenceId,
        error: Option<NodeError>,
    },
    Closed,
}
