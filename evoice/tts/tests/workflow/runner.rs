use std::sync::Arc;
use std::time::{Duration, Instant};

use e_voice_core::schema::error::NodeError;
use e_voice_core::schema::lang::Lang;
use e_voice_tts::config::text::TextConfig;
use e_voice_tts::schema::event::{Event, SentenceId};
use e_voice_tts::workflow::runner::{Request, Runner};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::fake::{FRAME, FakeSynth};

fn runner(delay: u64) -> Runner {
    let text = TextConfig { min: 0, max: 200 };
    Runner::new(
        Arc::new(FakeSynth {
            delay: Duration::from_millis(delay),
        }),
        text,
    )
}

/// Starts a stream; returns the request sender and the event receiver.
async fn start(runner: &Runner, cancel: CancellationToken) -> (mpsc::Sender<Request>, mpsc::Receiver<Event>) {
    let session = runner.open(Lang::Es, None).await.unwrap();
    let (requests, inbox) = mpsc::channel(8);
    let (outbox, events) = mpsc::channel(64);
    let runner = runner.clone();
    tokio::spawn(async move { runner.run(session, inbox, outbox, cancel).await.unwrap() });
    (requests, events)
}

/// Collapses consecutive audio events into one `(sentence, frames)` marker.
async fn drain(events: &mut mpsc::Receiver<Event>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    while let Some(event) = events.recv().await {
        let line = match event {
            Event::Start { sentence, text } => format!("start {} {text}", sentence.0),
            Event::Audio { sentence, audio } => {
                assert_eq!(audio.len(), FRAME);
                format!("audio {}", sentence.0)
            }
            Event::End { sentence, error } => format!("end {} {error:?}", sentence.0),
            Event::Closed => "closed".to_owned(),
        };
        if out.last() != Some(&line) || !line.starts_with("audio") {
            out.push(line);
        }
    }
    out
}

#[tokio::test]
async fn test_run_speaks_every_sentence_in_order_then_closes() {
    let runner = runner(1);
    let (requests, mut events) = start(&runner, CancellationToken::new()).await;
    requests
        .send(Request::Text("Primera frase. Segunda".into()))
        .await
        .unwrap();
    requests.send(Request::Text(" frase.".into())).await.unwrap();
    requests.send(Request::Close).await.unwrap();
    assert_eq!(
        drain(&mut events).await,
        [
            "start 0 Primera frase.",
            "audio 0",
            "end 0 None",
            "start 1 Segunda frase.",
            "audio 1",
            "end 1 None",
            "closed"
        ]
    );
}

#[tokio::test]
async fn test_run_closes_when_requests_end() {
    let runner = runner(1);
    let (requests, mut events) = start(&runner, CancellationToken::new()).await;
    requests
        .send(Request::Text("Sin cierre explícito".into()))
        .await
        .unwrap();
    drop(requests);
    assert_eq!(drain(&mut events).await.last().map(String::as_str), Some("closed"));
}

#[tokio::test]
async fn test_cancel_stops_the_sentence_within_a_chunk_and_the_stream_continues() {
    let runner = runner(20);
    let (requests, mut events) = start(&runner, CancellationToken::new()).await;
    let long = "palabra ".repeat(80);
    requests.send(Request::Text(long)).await.unwrap();
    requests.send(Request::Flush).await.unwrap();
    while !matches!(events.recv().await, Some(Event::Audio { .. })) {}
    let barge = Instant::now();
    requests.send(Request::Cancel).await.unwrap();
    loop {
        match events.recv().await.unwrap() {
            Event::End { sentence, error } => {
                assert_eq!((sentence, error), (SentenceId(0), Some(NodeError::Cancelled)));
                break;
            }
            Event::Audio { sentence, .. } => assert_eq!(sentence, SentenceId(0)),
            other => panic!("unexpected {other:?}"),
        }
    }
    assert!(barge.elapsed() < Duration::from_millis(500), "{:?}", barge.elapsed());
    requests.send(Request::Text("Sigo aquí.".into())).await.unwrap();
    requests.send(Request::Close).await.unwrap();
    let rest = drain(&mut events).await;
    assert_eq!(rest.first().map(String::as_str), Some("start 1 Sigo aquí."));
    assert_eq!(rest.last().map(String::as_str), Some("closed"));
}

#[tokio::test]
async fn test_shutdown_ends_a_stream_mid_sentence() {
    let runner = runner(20);
    let cancel = CancellationToken::new();
    let (requests, mut events) = start(&runner, cancel.clone()).await;
    requests.send(Request::Text("palabra ".repeat(80))).await.unwrap();
    requests.send(Request::Flush).await.unwrap();
    while !matches!(events.recv().await, Some(Event::Audio { .. })) {}
    cancel.cancel();
    let started = Instant::now();
    while events.recv().await.is_some() {}
    assert!(started.elapsed() < Duration::from_millis(500));
}
