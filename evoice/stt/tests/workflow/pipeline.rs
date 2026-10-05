use std::sync::Arc;
use std::time::Duration;

use e_voice_core::schema::lang::Lang;
use e_voice_stt::config::asr::AsrBackend;
use e_voice_stt::config::pipeline::GateClose;
use e_voice_stt::config::ww::WwBackend;
use e_voice_stt::core::settings::Settings;
use e_voice_stt::schema::audio::Audio;
use e_voice_stt::schema::emotion::EmotionLabel;
use e_voice_stt::schema::event::{Event, FinalEvent};
use e_voice_stt::workflow::nodes::Nodes;
use e_voice_stt::workflow::runner::{Runner, RunnerIntake};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::fixture;

async fn transcribe(settings: Settings, signal: Vec<f32>) -> Vec<Event> {
    let nodes = Arc::new(Nodes::build(&settings, &fixture::store()).unwrap());
    let runner = Runner::new(nodes, settings.stt.pipeline.clone());
    let (audio_tx, audio_rx) = mpsc::channel(16);
    let (events_tx, mut events_rx) = mpsc::channel(256);
    let task = tokio::spawn(async move {
        runner
            .run(
                Lang::Es,
                RunnerIntake::Live,
                audio_rx,
                events_tx,
                CancellationToken::new(),
            )
            .await
    });
    for chunk in signal.chunks(1_600) {
        audio_tx.send(Audio::from(chunk.to_vec())).await.unwrap();
    }
    drop(audio_tx);
    let mut events = Vec::new();
    while let Some(event) = events_rx.recv().await {
        events.push(event);
    }
    tokio::time::timeout(Duration::from_secs(60), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    events
}

fn parakeet() -> Settings {
    let mut settings = Settings::default();
    settings.stt.pipeline.asr.backend = AsrBackend::Parakeet;
    settings
}

fn finals(events: &[Event]) -> Vec<&FinalEvent> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Final(done) => Some(done),
            _ => None,
        })
        .collect()
}

fn twice() -> Vec<f32> {
    let es = fixture::wav("parakeet-v3-int8", "es.wav");
    [
        fixture::silence(1.0),
        es.clone(),
        fixture::silence(2.0),
        es,
        fixture::silence(1.5),
    ]
    .concat()
}

fn assert_two_spanish_finals(events: &[Event]) {
    let done = finals(events);
    assert_eq!(done.len(), 2, "{events:?}");
    assert!(
        done.iter().all(|f| f.error.is_none() && f.text.contains("por tu país")),
        "{done:?}"
    );
    assert!(
        done.iter().all(|f| f.emotion.label == EmotionLabel::Neutral),
        "{done:?}"
    );
    assert!(done[0].span.end <= done[1].span.start);
    assert_eq!(events.last(), Some(&Event::Closed));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires installed models: make pull"]
async fn test_streaming_pipeline_end_to_end() {
    let events = transcribe(Settings::default(), twice()).await;
    assert_two_spanish_finals(&events);
    assert!(events.iter().any(|event| matches!(event, Event::Partial(_))));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires installed models: make pull"]
async fn test_batch_pipeline_end_to_end() {
    let settings = parakeet();
    let events = transcribe(settings, twice()).await;
    assert_two_spanish_finals(&events);
    assert!(
        finals(&events)
            .iter()
            .all(|f| f.text.starts_with("No preguntes qué puede hacer tu país por ti")
                && f.text.ends_with("por tu país.")),
        "{:?}",
        finals(&events)
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires installed models: make pull"]
async fn test_wake_word_gates_the_pipeline() {
    let mut settings = parakeet();
    settings.stt.pipeline.ww.backend = WwBackend::Oww;
    settings.stt.pipeline.ww.keyword = "hey_jarvis".to_owned();
    settings.stt.pipeline.gate.close = GateClose::Utterance;
    settings.stt.pipeline.gate.idle = Duration::from_secs(5);
    let en = fixture::wav("parakeet-v3-int8", "en.wav");
    let es = fixture::wav("parakeet-v3-int8", "es.wav");
    let signal = [
        fixture::silence(1.0),
        en,
        fixture::silence(1.0),
        fixture::speak("Hey Jarvis."),
        fixture::silence(0.5),
        es,
        fixture::silence(1.5),
    ]
    .concat();
    let events = transcribe(settings, signal).await;
    let done = finals(&events);
    assert!(events.iter().any(|event| matches!(event, Event::Wake(_))), "{events:?}");
    assert_eq!(done.len(), 1, "{done:?}");
    assert!(done[0].text.starts_with("No preguntes"), "{done:?}");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires installed models: make pull"]
async fn test_gain_rescues_a_quiet_microphone() {
    let settings = parakeet();
    let quiet: Vec<f32> = twice().into_iter().map(|sample| sample * 0.005).collect();
    let events = transcribe(settings, quiet).await;
    let done = finals(&events);
    assert_eq!(done.len(), 2, "{done:?}");
    assert!(done.iter().all(|f| f.text.starts_with("No preguntes")), "{done:?}");
}
