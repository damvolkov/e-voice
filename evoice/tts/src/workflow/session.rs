use std::collections::VecDeque;
use std::collections::vec_deque::Drain;

use e_voice_core::schema::error::NodeError;

use crate::schema::audio::Audio;
use crate::schema::event::{Event, SentenceId};
use crate::workflow::text::Chunker;

/// Facts fed into a session: client requests and synthesis outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionInput {
    Text(String),
    /// Speak buffered text now, even if no sentence bound closed it.
    Flush,
    /// Barge-in: stop the current sentence, forget everything queued; the stream stays open.
    Cancel,
    /// The client is done: speak what remains, then close.
    Close,
    Audio {
        sentence: SentenceId,
        audio: Audio,
    },
    Done {
        sentence: SentenceId,
        result: Result<(), NodeError>,
    },
}

/// Work the runner must perform on the session's behalf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionCommand {
    Speak { sentence: SentenceId, text: String },
    Abort(SentenceId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionOutput {
    Command(SessionCommand),
    Event(Event),
}

/// One stream's synthesis as a sans-IO state machine: inputs in, commands and events out.
///
/// Guarantees: sentences are spoken one at a time, in order; every `Start` gets exactly one `End`;
/// no `Audio` outside its `Start`/`End`; one `Closed`, after `Close`, once nothing is left.
#[derive(Debug)]
pub struct Session {
    chunker: Chunker,
    queue: VecDeque<String>,
    current: Option<SentenceId>,
    next: u64,
    closing: bool,
    closed: bool,
    out: VecDeque<SessionOutput>,
}

impl Session {
    // ##### PRIVATE #####

    fn step_queue(&mut self, sentences: Vec<String>) {
        self.queue.extend(sentences);
        self.step_advance();
    }

    fn step_advance(&mut self) {
        let idle = self.current.is_none();
        match idle.then(|| self.queue.pop_front()).flatten() {
            Some(text) => {
                let sentence = SentenceId(self.next);
                self.next = self.next.saturating_add(1);
                self.current = Some(sentence);
                self.out.push_back(SessionOutput::Event(Event::Start {
                    sentence,
                    text: text.clone(),
                }));
                self.out
                    .push_back(SessionOutput::Command(SessionCommand::Speak { sentence, text }));
            }
            None if idle && self.closing && !self.closed => {
                self.closed = true;
                self.out.push_back(SessionOutput::Event(Event::Closed));
            }
            None => {}
        }
    }

    fn step_end(&mut self, sentence: SentenceId, error: Option<NodeError>) {
        self.current = None;
        self.out.push_back(SessionOutput::Event(Event::End { sentence, error }));
    }

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(chunker: Chunker) -> Self {
        Self {
            chunker,
            queue: VecDeque::new(),
            current: None,
            next: 0,
            closing: false,
            closed: false,
            out: VecDeque::new(),
        }
    }

    pub fn step(&mut self, input: SessionInput) {
        match input {
            _ if self.closed => {}
            SessionInput::Text(_) | SessionInput::Flush if self.closing => {}
            SessionInput::Text(delta) => {
                let sentences = self.chunker.push(&delta);
                self.step_queue(sentences);
            }
            SessionInput::Flush => {
                let sentences = self.chunker.flush();
                self.step_queue(sentences);
            }
            SessionInput::Close => {
                let sentences = self.chunker.flush();
                self.closing = true;
                self.step_queue(sentences);
            }
            SessionInput::Cancel => {
                self.chunker.clear();
                self.queue.clear();
                if let Some(sentence) = self.current {
                    self.out
                        .push_back(SessionOutput::Command(SessionCommand::Abort(sentence)));
                    self.step_end(sentence, Some(NodeError::Cancelled));
                }
                self.step_advance();
            }
            SessionInput::Audio { sentence, audio } if self.current == Some(sentence) => {
                self.out
                    .push_back(SessionOutput::Event(Event::Audio { sentence, audio }));
            }
            SessionInput::Done { sentence, result } if self.current == Some(sentence) => {
                self.step_end(sentence, result.err());
                self.step_advance();
            }
            SessionInput::Audio { .. } | SessionInput::Done { .. } => {}
        }
    }

    pub fn poll(&mut self) -> Drain<'_, SessionOutput> {
        self.out.drain(..)
    }

    #[must_use]
    pub const fn closed(&self) -> bool {
        self.closed
    }
}
