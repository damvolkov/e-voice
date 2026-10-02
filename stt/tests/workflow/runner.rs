use std::sync::Arc;
use std::time::Duration;

use e_voice_stt::config::asr::AsrConfig;
use e_voice_stt::config::pipeline::{GateClose, GateConfig, PipelineConfig};
use e_voice_stt::config::ser::SerConfig;
use e_voice_stt::schema::audio::Audio;
use e_voice_stt::schema::emotion::EmotionLabel;
use e_voice_stt::schema::error::NodeError;
use e_voice_stt::schema::event::{Event, FinalEvent};
use e_voice_stt::schema::lang::Lang;
use e_voice_stt::workflow::asr::registry::AsrBackend;
use e_voice_stt::workflow::runner::{Runner, RunnerIntake};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::fake::{
    FakeDenoise, FakeLid, FakeNamed, FakeStreaming, FakeWw, MARKER, SPEECH, batch, nodes, ser, slow, stratified,
};

fn config() -> PipelineConfig {
    PipelineConfig {
        tick: Duration::from_millis(10),
        asr: AsrConfig {
            deadline: Duration::from_millis(500),
            ..AsrConfig::default()
        },
        ser: SerConfig {
            deadline: Duration::from_millis(100),
            min: Duration::ZERO,
            ..SerConfig::default()
        },
        gate: GateConfig {
            close: GateClose::Utterance,
            idle: Duration::from_secs(5),
        },
        ..PipelineConfig::default()
    }
}

fn audio(parts: &[(f32, usize)]) -> Vec<f32> {
    parts
        .iter()
        .flat_map(|&(value, n)| std::iter::repeat_n(value, n))
        .collect()
}

async fn drive(runner: Runner, signal: Vec<f32>, cancel: Option<CancellationToken>) -> Vec<Event> {
    drive_with(runner, signal, cancel, RunnerIntake::Live).await
}

async fn drive_with(
    runner: Runner,
    signal: Vec<f32>,
    cancel: Option<CancellationToken>,
    intake: RunnerIntake,
) -> Vec<Event> {
    let (audio_tx, audio_rx) = mpsc::channel(8);
    let (events_tx, mut events_rx) = mpsc::channel(64);
    let token = cancel.clone().unwrap_or_default();
    let task = tokio::spawn(async move { runner.run(Lang::Es, intake, audio_rx, events_tx, token).await });
    for chunk in signal.chunks(1_600) {
        audio_tx.send(Audio::from(chunk.to_vec())).await.unwrap();
    }
    match cancel {
        Some(token) => {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token.cancel();
        }
        None => drop(audio_tx),
    }
    let mut events = Vec::new();
    while let Some(event) = events_rx.recv().await {
        events.push(event);
    }
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    events
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

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_utterances_finalize_in_order() {
    let runner = Runner::new(nodes(batch(0), Some(ser(0)), None), config());
    let signal = audio(&[
        (0.0, 1_600),
        (SPEECH, 8_000),
        (0.0, 4_000),
        (SPEECH, 4_000),
        (0.0, 1_600),
    ]);
    let events = drive(runner, signal, None).await;
    let texts: Vec<&str> = finals(&events).iter().map(|done| done.text.as_str()).collect();
    assert_eq!(texts, ["batch:8000", "batch:4000"]);
    assert!(
        finals(&events)
            .iter()
            .all(|done| done.emotion.label == EmotionLabel::Happy && done.error.is_none())
    );
    assert_eq!(events.last(), Some(&Event::Closed));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_feeds_from_speech_start_and_emits_partials() {
    let runner = Runner::new(
        nodes(AsrBackend::Streaming(Arc::new(FakeStreaming)), None, None),
        config(),
    );
    let signal = audio(&[(0.0, 3_200), (SPEECH, 8_000), (0.0, 3_200)]);
    let events = drive(runner, signal, None).await;
    let done = finals(&events);
    assert_eq!(done.len(), 1);
    let fed: usize = done[0].text.strip_prefix("Es:").unwrap().parse().unwrap();
    assert!((8_000..=8_000 + 1_600).contains(&fed), "{fed}");
    assert!(events.iter().any(|event| matches!(event, Event::Partial(_))));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_late_ser_falls_back_to_unknown() {
    let runner = Runner::new(nodes(batch(0), Some(ser(400)), None), config());
    let events = drive(runner, audio(&[(SPEECH, 4_000), (0.0, 1_600)]), None).await;
    let done = finals(&events);
    assert_eq!(
        (done[0].text.as_str(), done[0].emotion.label),
        ("batch:4000", EmotionLabel::Unknown)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_backend_panic_fails_only_its_segment() {
    let runner = Runner::new(nodes(batch(8_000), None, None), config());
    let signal = audio(&[(SPEECH, 8_000), (0.0, 3_200), (SPEECH, 4_000), (0.0, 1_600)]);
    let events = drive(runner, signal, None).await;
    let done = finals(&events);
    assert!(matches!(done[0].error, Some(NodeError::Backend(_))), "{:?}", done[0]);
    assert_eq!((done[1].text.as_str(), &done[1].error), ("batch:4000", &None));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_closes_open_segment() {
    let runner = Runner::new(
        nodes(AsrBackend::Streaming(Arc::new(FakeStreaming)), None, None),
        config(),
    );
    let events = drive(runner, audio(&[(SPEECH, 6_400)]), Some(CancellationToken::new())).await;
    assert_eq!(finals(&events)[0].error, Some(NodeError::Cancelled));
    assert_eq!(events.last(), Some(&Event::Closed));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_gate_ignores_speech_until_wake() {
    let runner = Runner::new(nodes(batch(0), None, Some(Arc::new(FakeWw))), config());
    let signal = audio(&[
        (SPEECH, 6_400),
        (0.0, 3_200),
        (MARKER, 1_600),
        (0.0, 1_600),
        (SPEECH, 4_800),
        (0.0, 3_200),
        (SPEECH, 6_400),
        (0.0, 1_600),
    ]);
    let events = drive(runner, signal, None).await;
    let texts: Vec<&str> = finals(&events).iter().map(|done| done.text.as_str()).collect();
    assert_eq!(texts, ["batch:4800"]);
    assert!(events.iter().any(|event| matches!(event, Event::Wake(_))));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runner_stops_when_consumer_leaves() {
    let runner = Runner::new(nodes(batch(0), None, None), config());
    let (audio_tx, audio_rx) = mpsc::channel(8);
    let (events_tx, events_rx) = mpsc::channel(1);
    drop(events_rx);
    let task = tokio::spawn(async move {
        runner
            .run(
                Lang::En,
                RunnerIntake::Live,
                audio_rx,
                events_tx,
                CancellationToken::new(),
            )
            .await
    });
    audio_tx
        .send(Audio::from(audio(&[(SPEECH, 4_000), (0.0, 1_600)])))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_file_intake_backpressures_instead_of_dropping() {
    let config = PipelineConfig {
        pending: 1,
        asr: AsrConfig {
            deadline: Duration::from_secs(5),
            ..AsrConfig::default()
        },
        ..config()
    };
    let utterances: Vec<(f32, usize)> = (0..5).flat_map(|_| [(SPEECH, 3_200), (0.0, 1_600)]).collect();
    let file = drive_with(
        Runner::new(nodes(slow(80), None, None), config.clone()),
        audio(&utterances),
        None,
        RunnerIntake::File,
    )
    .await;
    assert_eq!(finals(&file).len(), 5);
    assert!(finals(&file).iter().all(|done| done.error.is_none()), "{file:?}");
    let live = drive_with(
        Runner::new(nodes(slow(80), None, None), config),
        audio(&utterances),
        None,
        RunnerIntake::Live,
    )
    .await;
    assert!(
        finals(&live).iter().any(|done| done.error == Some(NodeError::Overload)),
        "{live:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_file_intake_ignores_the_gate() {
    let runner = Runner::new(nodes(batch(0), None, Some(Arc::new(FakeWw))), config());
    let events = drive_with(
        runner,
        audio(&[(SPEECH, 3_200), (0.0, 1_600)]),
        None,
        RunnerIntake::File,
    )
    .await;
    assert_eq!(finals(&events).len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_files_use_the_offline_asr_and_streams_the_live_one() {
    let nodes = stratified(AsrBackend::Streaming(Arc::new(FakeStreaming)), batch(0));
    let signal = audio(&[(SPEECH, 4_000), (0.0, 1_600)]);
    let file = drive_with(
        Runner::new(Arc::clone(&nodes), config()),
        signal.clone(),
        None,
        RunnerIntake::File,
    )
    .await;
    let live = drive_with(Runner::new(nodes, config()), signal, None, RunnerIntake::Live).await;
    assert_eq!(finals(&file)[0].text, "batch:4000");
    assert!(finals(&live)[0].text.starts_with("Es:"), "{live:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_identified_language_reaches_the_final_of_long_enough_segments() {
    let mut built = nodes(batch(0), None, None);
    Arc::get_mut(&mut built).unwrap().lid = Some(Arc::new(FakeLid { english: 6_000 }));
    let mut settings = config();
    settings.lid.min = Duration::from_millis(250);
    let signal = audio(&[
        (0.0, 1_600),
        (SPEECH, 8_000),
        (0.0, 4_000),
        (SPEECH, 4_400),
        (0.0, 1_600),
    ]);
    let events = drive(Runner::new(built, settings), signal, None).await;
    let langs: Vec<Lang> = finals(&events).iter().map(|done| done.lang).collect();
    assert_eq!(langs, [Lang::En, Lang::Es]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_denoise_runs_before_vad_and_flushes_its_tail() {
    let denoised = |gain: f32| {
        let mut built = nodes(batch(0), None, None);
        Arc::get_mut(&mut built).unwrap().denoise = Some(Arc::new(FakeDenoise { gain, hold: 800 }));
        Runner::new(built, config())
    };
    let signal = audio(&[(0.0, 1_600), (SPEECH, 8_000)]);
    let kept = drive(denoised(1.0), signal.clone(), None).await;
    assert_eq!(
        finals(&kept).iter().map(|done| done.text.as_str()).collect::<Vec<_>>(),
        ["batch:8000"]
    );
    let quiet = drive(denoised(0.5), signal, None).await;
    assert!(finals(&quiet).is_empty(), "{quiet:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_engine_swaps_the_file_backend_only_for_loaded_engines() {
    use e_voice_stt::config::asr::AsrEngine;

    let mut built = stratified(AsrBackend::Streaming(Arc::new(FakeStreaming)), batch(0));
    Arc::get_mut(&mut built).unwrap().extra =
        vec![(AsrEngine::Canary, AsrBackend::Batch(Arc::new(FakeNamed("canary"))))];
    let runner = Runner::new(built, config());
    let signal = audio(&[(0.0, 1_600), (SPEECH, 8_000), (0.0, 4_000)]);
    let text = |transcript: e_voice_stt::schema::transcript::Transcript| transcript.text(false);
    let default = runner.engine(AsrEngine::Parakeet).unwrap();
    assert_eq!(
        text(default.transcribe(Lang::Es, signal.clone()).await.unwrap()),
        "batch:8000"
    );
    let canary = runner.engine(AsrEngine::Canary).unwrap();
    assert_eq!(text(canary.transcribe(Lang::Es, signal).await.unwrap()), "canary:8000");
    assert!(runner.engine(AsrEngine::Cohere).is_none());
}
