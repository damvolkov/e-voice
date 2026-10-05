use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

use crate::app::{self, exchange, json, pcm};
use crate::fake::{SPEECH, batch, nodes, ser};

async fn base() -> String {
    let (address, _) = app::serve(nodes(batch(0), Some(ser(0)), None), 1 << 20, false).await;
    format!("ws://{address}/v1/stream")
}

fn utterance() -> Vec<Message> {
    vec![
        Message::Binary(pcm(0.0, 1_600).into()),
        Message::Binary(pcm(0.9, 8_000).into()),
        Message::Binary(pcm(0.0, 1_600).into()),
        Message::Text(r#"{"type":"end"}"#.into()),
    ]
}

fn kinds(events: &[Value]) -> Vec<&str> {
    events.iter().filter_map(|event| event["type"].as_str()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_struct_view_streams_every_event_then_closes_normally() {
    let (texts, code) = exchange(&format!("{}?lang=en", base().await), utterance()).await;
    let events = json(&texts);
    assert_eq!(kinds(&events), ["speech", "speech", "final", "closed"], "{events:?}");
    assert_eq!(
        (events[0]["state"].as_str(), events[1]["state"].as_str()),
        (Some("started"), Some("stopped"))
    );
    assert_eq!(
        (&events[2]["text"], &events[2]["lang"], &events[2]["emotion"]["label"]),
        (&Value::from("batch:8000"), &Value::from("en"), &Value::from("happy"))
    );
    assert_eq!(code, Some(1000));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_flat_view_sends_only_final_text() {
    let (texts, code) = exchange(&format!("{}?view=flat&emotion=tag", base().await), utterance()).await;
    assert_eq!(texts, ["[happy] batch:8000"]);
    assert_eq!(code, Some(1000));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_emotion_off_drops_the_field() {
    let (texts, _) = exchange(&format!("{}?emotion=off", base().await), utterance()).await;
    let events = json(&texts);
    let done = events.iter().find(|event| event["type"] == "final").unwrap();
    assert!(done.get("emotion").is_none(), "{done}");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_split_samples_and_f32_encoding_are_accepted() {
    let bytes: Vec<u8> = std::iter::repeat_n(SPEECH.to_le_bytes(), 4_000).flatten().collect();
    let (left, right) = bytes.split_at(4_001);
    let frames = vec![
        Message::Binary(left.to_vec().into()),
        Message::Binary(right.to_vec().into()),
        Message::Text(r#"{"type":"end"}"#.into()),
    ];
    let (texts, _) = exchange(&format!("{}?encoding=f32le", base().await), frames).await;
    let events = json(&texts);
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "final" && event["text"] == "batch:4000"),
        "{events:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_parameters_are_refused() {
    let (texts, code) = exchange(
        &format!("{}?rate=0", base().await),
        vec![Message::Binary(pcm(0.9, 160).into())],
    )
    .await;
    assert!(texts.is_empty());
    assert_eq!(code, Some(1003));
    let refused = tokio_tungstenite::connect_async(format!("{}?emotion=loud", base().await)).await;
    assert!(refused.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_shutdown_finalizes_open_segment() {
    let (address, shutdown) = app::serve(nodes(batch(0), Some(ser(0)), None), 1 << 20, false).await;
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/v1/stream"))
        .await
        .unwrap();
    socket.send(Message::Binary(pcm(0.9, 4_000).into())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    shutdown.cancel();
    let mut finals = Vec::new();
    while let Some(Ok(Message::Text(text))) = socket.next().await {
        let event: Value = serde_json::from_str(&text).unwrap();
        if event["type"] != "speech" {
            finals.push((
                event["type"].as_str().unwrap().to_owned(),
                event["error"]["kind"].as_str().map(str::to_owned),
            ));
        }
    }
    assert_eq!(
        finals,
        [
            ("final".to_owned(), Some("cancelled".to_owned())),
            ("closed".to_owned(), None)
        ]
    );
}
