use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

use crate::app::{self, exchange, pcm};
use crate::fake::{FakeStreaming, nodes, ser};

fn append(value: f32, samples: usize) -> Message {
    Message::Text(
        json!({"type": "input_audio_buffer.append", "audio": STANDARD.encode(pcm(value, samples))})
            .to_string()
            .into(),
    )
}

async fn base() -> String {
    let streaming = e_voice_stt::workflow::asr::registry::AsrBackend::Streaming(std::sync::Arc::new(FakeStreaming));
    let (address, _) = app::serve(nodes(streaming, Some(ser(0)), None), 1 << 20, false).await;
    format!("ws://{address}/v1/realtime?intent=transcription")
}

fn by_type<'a>(events: &'a [Value], kind: &str) -> Vec<&'a Value> {
    events.iter().filter(|event| event["type"] == kind).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_transcription_session_follows_openai_events() {
    let frames = vec![
        Message::Text(json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}, "transcription": {"language": "en"}}}}}).to_string().into()),
        append(0.0, 1_600),
        append(0.9, 8_000),
        append(0.0, 1_600),
        Message::Text(json!({"type": "session.close"}).to_string().into()),
    ];
    let (texts, code) = exchange(&base().await, frames).await;
    let events: Vec<Value> = texts.iter().map(|text| serde_json::from_str(text).unwrap()).collect();
    assert_eq!(by_type(&events, "session.created").len(), 1);
    assert_eq!(
        by_type(&events, "session.updated")[0]["session"]["audio"]["input"]["format"]["rate"],
        16000
    );
    let started = by_type(&events, "input_audio_buffer.speech_started");
    let stopped = by_type(&events, "input_audio_buffer.speech_stopped");
    assert_eq!(
        (
            started.len(),
            stopped.len(),
            by_type(&events, "input_audio_buffer.committed").len()
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (
            started[0]["audio_start_ms"].as_u64(),
            stopped[0]["audio_end_ms"].as_u64()
        ),
        (Some(100), Some(600))
    );
    let deltas: String = by_type(&events, "conversation.item.input_audio_transcription.delta")
        .iter()
        .map(|event| event["delta"].as_str().unwrap().to_owned())
        .collect();
    let completed = by_type(&events, "conversation.item.input_audio_transcription.completed");
    assert_eq!(completed.len(), 1);
    assert!(
        completed[0]["transcript"].as_str().unwrap().starts_with("En:"),
        "{completed:?}"
    );
    assert!(!deltas.is_empty());
    assert_eq!(completed[0]["item_id"], started[0]["item_id"]);
    assert_eq!(completed[0]["emotion"]["label"], "happy");
    assert_eq!(code, Some(1000));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unsupported_language_is_rejected_before_upgrade() {
    let refused = tokio_tungstenite::connect_async(format!("{}&language=fr", base().await)).await;
    assert!(refused.is_err());
}
