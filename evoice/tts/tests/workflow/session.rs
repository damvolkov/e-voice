use e_voice_core::schema::error::NodeError;
use e_voice_tts::schema::audio::Audio;
use e_voice_tts::schema::event::{Event, SentenceId};
use e_voice_tts::workflow::session::{Session, SessionCommand, SessionInput, SessionOutput};
use e_voice_tts::workflow::text::Chunker;
use proptest::prelude::*;

fn session() -> Session {
    Session::new(Chunker::new(0, 200))
}

fn run(session: &mut Session, input: SessionInput) -> Vec<SessionOutput> {
    session.step(input);
    session.poll().collect()
}

fn speak(sentence: u64, text: &str) -> SessionOutput {
    SessionOutput::Command(SessionCommand::Speak {
        sentence: SentenceId(sentence),
        text: text.to_owned(),
    })
}

fn start(sentence: u64, text: &str) -> SessionOutput {
    SessionOutput::Event(Event::Start {
        sentence: SentenceId(sentence),
        text: text.to_owned(),
    })
}

fn end(sentence: u64, error: Option<NodeError>) -> SessionOutput {
    SessionOutput::Event(Event::End {
        sentence: SentenceId(sentence),
        error,
    })
}

fn done(sentence: u64) -> SessionInput {
    SessionInput::Done {
        sentence: SentenceId(sentence),
        result: Ok(()),
    }
}

#[test]
fn test_sentences_are_spoken_one_at_a_time_in_order() {
    let mut session = session();
    let out = run(
        &mut session,
        SessionInput::Text("Primera frase. Segunda frase. Tercera".into()),
    );
    assert_eq!(out, [start(0, "Primera frase."), speak(0, "Primera frase.")]);
    assert!(run(&mut session, SessionInput::Flush).is_empty());
    assert_eq!(
        run(&mut session, done(0)),
        [end(0, None), start(1, "Segunda frase."), speak(1, "Segunda frase.")]
    );
    assert_eq!(
        run(&mut session, done(1)),
        [end(1, None), start(2, "Tercera"), speak(2, "Tercera")]
    );
}

#[test]
fn test_audio_is_forwarded_only_for_the_current_sentence() {
    let mut session = session();
    run(&mut session, SessionInput::Text("Hola.".into()));
    run(&mut session, SessionInput::Flush);
    let audio = Audio::from(vec![0.1; 4]);
    let live = run(
        &mut session,
        SessionInput::Audio {
            sentence: SentenceId(0),
            audio: audio.clone(),
        },
    );
    assert_eq!(
        live,
        [SessionOutput::Event(Event::Audio {
            sentence: SentenceId(0),
            audio: audio.clone(),
        })]
    );
    let stale = run(
        &mut session,
        SessionInput::Audio {
            sentence: SentenceId(7),
            audio,
        },
    );
    assert!(stale.is_empty());
}

#[test]
fn test_cancel_aborts_the_current_sentence_and_drops_the_queue() {
    let mut session = session();
    run(&mut session, SessionInput::Text("Uno. Dos. Tres".into()));
    let out = run(&mut session, SessionInput::Cancel);
    assert_eq!(
        out,
        [
            SessionOutput::Command(SessionCommand::Abort(SentenceId(0))),
            end(0, Some(NodeError::Cancelled))
        ]
    );
    assert!(run(&mut session, done(0)).is_empty());
    assert!(run(&mut session, SessionInput::Flush).is_empty());
    let after = run(&mut session, SessionInput::Text("Sigo aquí.".into()));
    assert!(after.is_empty());
    assert_eq!(
        run(&mut session, SessionInput::Flush),
        [start(1, "Sigo aquí."), speak(1, "Sigo aquí.")]
    );
}

#[test]
fn test_close_speaks_the_rest_then_closes_once() {
    let mut session = session();
    run(&mut session, SessionInput::Text("Adiós".into()));
    assert_eq!(
        run(&mut session, SessionInput::Close),
        [start(0, "Adiós"), speak(0, "Adiós")]
    );
    assert!(run(&mut session, SessionInput::Text("tarde".into())).is_empty());
    assert_eq!(
        run(&mut session, done(0)),
        [end(0, None), SessionOutput::Event(Event::Closed)]
    );
    assert!(session.closed());
    assert!(run(&mut session, SessionInput::Close).is_empty());
}

#[test]
fn test_failed_sentence_ends_with_its_error_and_the_next_one_starts() {
    let mut session = session();
    run(&mut session, SessionInput::Text("Uno. Dos".into()));
    run(&mut session, SessionInput::Flush);
    let failure = NodeError::Backend("kv overflow".into());
    let out = run(
        &mut session,
        SessionInput::Done {
            sentence: SentenceId(0),
            result: Err(failure.clone()),
        },
    );
    assert_eq!(out, [end(0, Some(failure)), start(1, "Dos"), speak(1, "Dos")]);
}

#[derive(Debug, Clone)]
enum Action {
    Text(&'static str),
    Flush,
    Cancel,
    Close,
    Audio,
    Done(bool),
    Stale,
}

fn action() -> impl Strategy<Value = Action> {
    prop_oneof![
        prop::sample::select(vec!["Hola. ", "que tal", " estás? ", "Bien, gracias."]).prop_map(Action::Text),
        Just(Action::Flush),
        Just(Action::Cancel),
        Just(Action::Close),
        Just(Action::Audio),
        any::<bool>().prop_map(Action::Done),
        Just(Action::Stale),
    ]
}

proptest! {
    #[test]
    fn prop_events_keep_start_end_pairing_and_order(actions in prop::collection::vec(action(), 0..80)) {
        let mut session = session();
        let mut spoken: Option<SentenceId> = None;
        let mut open: Option<SentenceId> = None;
        let mut last: Option<SentenceId> = None;
        let mut closed = false;
        let tail = [Action::Close, Action::Done(true), Action::Done(true), Action::Done(true), Action::Done(true)];
        for action in actions.into_iter().chain(tail.into_iter().cycle().take(40)) {
            let current = spoken.unwrap_or(SentenceId(u64::MAX));
            session.step(match action {
                Action::Text(text) => SessionInput::Text(text.into()),
                Action::Flush => SessionInput::Flush,
                Action::Cancel => SessionInput::Cancel,
                Action::Close => SessionInput::Close,
                Action::Audio => SessionInput::Audio { sentence: current, audio: Audio::from(vec![0.0; 2]) },
                Action::Done(ok) => SessionInput::Done {
                    sentence: current,
                    result: ok.then_some(()).ok_or(NodeError::Backend("x".into())),
                },
                Action::Stale => SessionInput::Done { sentence: SentenceId(u64::MAX - 1), result: Ok(()) },
            });
            for output in session.poll() {
                prop_assert!(!closed, "output after Closed: {output:?}");
                match output {
                    SessionOutput::Command(SessionCommand::Speak { sentence, .. }) => spoken = Some(sentence),
                    SessionOutput::Command(SessionCommand::Abort(sentence)) => prop_assert_eq!(open, Some(sentence)),
                    SessionOutput::Event(Event::Start { sentence, .. }) => {
                        prop_assert_eq!(open, None);
                        prop_assert!(last.is_none_or(|last| sentence > last));
                        open = Some(sentence);
                        last = Some(sentence);
                    }
                    SessionOutput::Event(Event::Audio { sentence, .. }) => prop_assert_eq!(open, Some(sentence)),
                    SessionOutput::Event(Event::End { sentence, .. }) => {
                        prop_assert_eq!(open.take(), Some(sentence));
                    }
                    SessionOutput::Event(Event::Closed) => {
                        prop_assert_eq!(open, None);
                        closed = true;
                    }
                }
            }
        }
        prop_assert!(closed);
    }
}
