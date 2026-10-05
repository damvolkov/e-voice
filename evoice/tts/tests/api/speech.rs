use serde_json::json;

use crate::app::serve;
use crate::fake::{FRAME, PACE};

#[tokio::test]
async fn test_speech_streams_a_wav_by_default() {
    let app = serve(1).await;
    let input = "Hola mundo, esto es una prueba.";
    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/audio/speech", app.addr))
        .json(&json!({ "model": "tts-1", "input": input, "voice": "alloy" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "audio/wav");
    let body = response.bytes().await.unwrap();
    assert_eq!(&body[..4], b"RIFF");
    let frames = input.chars().count().div_ceil(PACE);
    assert_eq!(body.len(), 44 + frames * FRAME * 2);
}

#[tokio::test]
async fn test_speech_sse_sends_deltas_then_done() {
    let app = serve(1).await;
    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/audio/speech", app.addr))
        .json(&json!({ "model": "tts-1", "input": "Hola.", "voice": "alloy", "response_format": "pcm", "stream_format": "sse" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let body = response.text().await.unwrap();
    let events: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(events.len() >= 2);
    assert!(
        events[..events.len() - 1]
            .iter()
            .all(|event| event["type"] == "speech.audio.delta")
    );
    assert_eq!(events.last().unwrap()["type"], "speech.audio.done");
}

#[tokio::test]
async fn test_speech_rejects_compressed_formats_and_empty_input() {
    let app = serve(1).await;
    let client = reqwest::Client::new();
    let url = format!("http://{}/v1/audio/speech", app.addr);
    let mp3 = client
        .post(&url)
        .json(&json!({ "model": "tts-1", "input": "Hola.", "voice": "alloy", "response_format": "mp3" }))
        .send()
        .await
        .unwrap();
    assert_eq!(mp3.status(), 400);
    let body: serde_json::Value = mp3.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let empty = client
        .post(&url)
        .json(&json!({ "model": "tts-1", "input": "  ", "voice": "alloy" }))
        .send()
        .await
        .unwrap();
    assert_eq!(empty.status(), 400);
}

#[tokio::test]
async fn test_health_and_models_describe_the_service() {
    let app = serve(1).await;
    let client = reqwest::Client::new();
    let health: serde_json::Value = client
        .get(format!("http://{}/health", app.addr))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        (health["status"].as_str(), health["rate"].as_u64()),
        (Some("ok"), Some(24_000))
    );
    let models: serde_json::Value = client
        .get(format!("http://{}/v1/models", app.addr))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "fake-es");
    let docs = client
        .get(format!("http://{}/openapi.json", app.addr))
        .send()
        .await
        .unwrap();
    assert_eq!(docs.status(), 200);
}
