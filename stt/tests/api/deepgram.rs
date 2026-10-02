use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

use crate::app::{self, exchange, json, pcm, wav};
use crate::fake::{FakeStreaming, batch, nodes, ser};

const LOUD: f32 = 0.9;

#[tokio::test(flavor = "multi_thread")]
async fn test_prerecorded_matches_deepgram_shape() {
    let (address, _) = app::serve(nodes(batch(0), Some(ser(0)), None), 1 << 20, false).await;
    let body = wav(
        16_000,
        &[
            (0.0, 16_000),
            (LOUD, 8_000),
            (0.0, 16_000),
            (LOUD, 16_000),
            (0.0, 8_000),
        ],
    );
    let response = reqwest::Client::new()
        .post(format!(
            "http://{address}/v1/listen?model=nova-2&language=es&smart_format=true"
        ))
        .header("content-type", "audio/wav")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(
        body["results"]["channels"][0]["alternatives"][0]["transcript"],
        "batch:8000 batch:16000"
    );
    assert_eq!(body["results"]["utterances"].as_array().unwrap().len(), 2);
    assert_eq!(body["results"]["utterances"][1]["start"], 2.5);
    assert_eq!(body["metadata"]["duration"], 4.0);
    assert_eq!(body["metadata"]["models"][0], "nova-2");
    assert_eq!(body["emotion"]["label"], "happy");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_prerecorded_errors_follow_deepgram_shape() {
    let (address, _) = app::serve(nodes(batch(0), None, None), 1 << 20, false).await;
    let response = reqwest::Client::new()
        .post(format!("http://{address}/v1/listen?language=fr"))
        .body(wav(16_000, &[(LOUD, 1_600)]))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["err_code"], "Bad Request");
    assert!(body["err_msg"].as_str().unwrap().contains("unsupported language"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_live_streams_interim_and_final_results() {
    let streaming = e_voice_stt::workflow::asr::registry::AsrBackend::Streaming(std::sync::Arc::new(FakeStreaming));
    let (address, _) = app::serve(nodes(streaming, Some(ser(0)), None), 1 << 20, false).await;
    let frames = vec![
        Message::Binary(pcm(0.0, 1_600).into()),
        Message::Binary(pcm(LOUD, 8_000).into()),
        Message::Binary(pcm(0.0, 1_600).into()),
        Message::Text(r#"{"type":"KeepAlive"}"#.into()),
        Message::Text(r#"{"type":"CloseStream"}"#.into()),
    ];
    let (texts, code) = exchange(
        &format!("ws://{address}/v1/listen?encoding=linear16&sample_rate=16000&language=en&emotion=off"),
        frames,
    )
    .await;
    let events = json(&texts);
    let kinds: Vec<&str> = events.iter().filter_map(|event| event["type"].as_str()).collect();
    assert_eq!(kinds.first(), Some(&"SpeechStarted"), "{kinds:?}");
    assert_eq!(kinds.last(), Some(&"Metadata"));
    let ended = kinds.iter().position(|kind| *kind == "UtteranceEnd").unwrap();
    assert_eq!(
        kinds[ended - 1],
        "Results",
        "the final precedes UtteranceEnd: {kinds:?}"
    );
    assert_eq!(events[ended - 1]["is_final"], true);
    let results: Vec<&Value> = events.iter().filter(|event| event["type"] == "Results").collect();
    assert!(results.iter().any(|event| event["is_final"] == false));
    let finals: Vec<&&Value> = results.iter().filter(|event| event["is_final"] == true).collect();
    assert_eq!(finals.len(), 1);
    assert!(
        finals[0]["channel"]["alternatives"][0]["transcript"]
            .as_str()
            .unwrap()
            .starts_with("En:")
    );
    assert!(finals[0].get("emotion").is_none());
    assert_eq!(code, Some(1000));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_live_rejects_unknown_encoding() {
    let (address, _) = app::serve(nodes(batch(0), None, None), 1 << 20, false).await;
    assert!(
        tokio_tungstenite::connect_async(format!("ws://{address}/v1/listen?encoding=mulaw"))
            .await
            .is_err()
    );
}
