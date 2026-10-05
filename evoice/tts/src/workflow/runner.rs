use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use e_voice_core::schema::error::BackendError;
use e_voice_core::schema::lang::Lang;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::config::text::TextConfig;
use crate::schema::event::{Event, SentenceId};
use crate::schema::voice::VoiceState;
use crate::workflow::session::{Session, SessionCommand, SessionInput, SessionOutput};
use crate::workflow::synth::base::{Synth, SynthCaps, SynthSession};
use crate::workflow::text::Chunker;

/// What a client may ask of a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Text(String),
    Flush,
    Cancel,
    Close,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunnerError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("synthesis panicked")]
    Panicked,
}

/// A sentence being synthesized on the blocking pool; it hands the session back when done.
struct Job {
    handle: JoinHandle<Box<dyn SynthSession>>,
    abort: Arc<AtomicBool>,
}

/// Drives streams: each owns one backend session; a sentence is synthesized on the blocking pool,
/// chunk by chunk, and stops within one chunk of a barge-in or a vanished client.
#[derive(Debug, Clone)]
pub struct Runner {
    synth: Arc<dyn Synth>,
    text: TextConfig,
}

impl From<Request> for SessionInput {
    fn from(request: Request) -> Self {
        match request {
            Request::Text(text) => Self::Text(text),
            Request::Flush => Self::Flush,
            Request::Cancel => Self::Cancel,
            Request::Close => Self::Close,
        }
    }
}

impl Runner {
    // ##### PRIVATE #####

    fn run_job(
        mut session: Box<dyn SynthSession>,
        sentence: SentenceId,
        text: String,
        outputs: mpsc::Sender<SessionInput>,
    ) -> Job {
        let abort = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&abort);
        let handle = tokio::task::spawn_blocking(move || {
            let mut result = Ok(());
            for chunk in session.speak(&text) {
                let sent = match (stop.load(Ordering::Relaxed), chunk) {
                    (true, _) => false,
                    (false, Ok(audio)) => outputs.blocking_send(SessionInput::Audio { sentence, audio }).is_ok(),
                    (false, Err(error)) => {
                        result = Err(error);
                        false
                    }
                };
                if !sent {
                    break;
                }
            }
            outputs.blocking_send(SessionInput::Done { sentence, result }).ok();
            session
        });
        Job { handle, abort }
    }

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(synth: Arc<dyn Synth>, text: TextConfig) -> Self {
        Self { synth, text }
    }

    #[must_use]
    pub fn caps(&self) -> SynthCaps {
        self.synth.caps()
    }

    #[must_use]
    pub fn synth(&self) -> &Arc<dyn Synth> {
        &self.synth
    }

    /// Leases a worker and primes it with the voice, on the blocking pool.
    ///
    /// # Errors
    /// Every worker is busy, the language is not served, or the voice does not fit the model.
    pub async fn open(&self, lang: Lang, voice: Option<VoiceState>) -> Result<Box<dyn SynthSession>, RunnerError> {
        let synth = Arc::clone(&self.synth);
        tokio::task::spawn_blocking(move || synth.open(lang, voice.as_ref()))
            .await
            .map_err(|_| RunnerError::Panicked)?
            .map_err(RunnerError::from)
    }

    /// Streams `requests` through `session` until `Close` (or the request channel ends) and every
    /// sentence is spoken, or until `cancel`. Events leave in order; `Closed` is the last one sent.
    ///
    /// # Errors
    /// The backend panicked mid-sentence.
    pub async fn run(
        &self,
        session: Box<dyn SynthSession>,
        mut requests: mpsc::Receiver<Request>,
        events: mpsc::Sender<Event>,
        cancel: CancellationToken,
    ) -> Result<(), RunnerError> {
        let mut state = Session::new(Chunker::new(self.text.min, self.text.max));
        let (outputs, mut produced) = mpsc::channel::<SessionInput>(32);
        let mut idle = Some(session);
        let mut job: Option<Job> = None;
        let mut pending: Option<(SentenceId, String)> = None;
        let mut listening = true;
        loop {
            for output in state.poll().collect::<Vec<_>>() {
                match output {
                    SessionOutput::Command(SessionCommand::Speak { sentence, text }) => {
                        pending = Some((sentence, text));
                    }
                    SessionOutput::Command(SessionCommand::Abort(_)) => {
                        pending = None;
                        if let Some(job) = &job {
                            job.abort.store(true, Ordering::Relaxed);
                        }
                    }
                    SessionOutput::Event(event) => {
                        if events.send(event).await.is_err() {
                            if let Some(job) = &job {
                                job.abort.store(true, Ordering::Relaxed);
                            }
                            return Ok(());
                        }
                    }
                }
            }
            if state.closed() {
                return Ok(());
            }
            match (pending.take(), idle.take()) {
                (Some((sentence, text)), Some(session)) => {
                    job = Some(Self::run_job(session, sentence, text, outputs.clone()));
                }
                (waiting, session) => (pending, idle) = (waiting, session),
            }
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    if let Some(job) = &job {
                        job.abort.store(true, Ordering::Relaxed);
                    }
                    return Ok(());
                }
                Some(input) = produced.recv() => state.step(input),
                returned = async {
                    match job.as_mut() {
                        Some(job) => Some((&mut job.handle).await),
                        None => None,
                    }
                }, if job.is_some() => {
                    job = None;
                    idle = Some(returned.ok_or(RunnerError::Panicked)?.map_err(|_| RunnerError::Panicked)?);
                }
                request = requests.recv(), if listening => match request {
                    Some(request) => state.step(request.into()),
                    None => {
                        listening = false;
                        state.step(SessionInput::Close);
                    }
                },
            }
        }
    }
}
