use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

use crate::app::serve;
use crate::fake::FRAME;

#[tokio::test]
async fn test_stream_speaks_streamed_text_and_closes() {
    let app = serve(1).await;
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{}/v1/stream?lang=es", app.addr))
        .await
        .unwrap();
    let ready: Value = match socket.next().await.unwrap().unwrap() {
        Message::Text(text) => serde_json::from_str(&text).unwrap(),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        (ready["type"].as_str(), ready["rate"].as_u64()),
        (Some("ready"), Some(24_000))
    );
    for token in ["Hola ", "mundo. ", "Adiós."] {
        socket
            .send(Message::text(json!({ "type": "text", "text": token }).to_string()))
            .await
            .unwrap();
    }
    socket
        .send(Message::text(json!({ "type": "close" }).to_string()))
        .await
        .unwrap();
    let (mut kinds, mut audio) = (Vec::new(), 0usize);
    while let Some(Ok(message)) = socket.next().await {
        match message {
            Message::Binary(bytes) => audio += bytes.len(),
            Message::Text(text) => kinds.push(
                serde_json::from_str::<Value>(&text).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            ),
            Message::Close(_) => break,
            _ => {}
        }
    }
    assert_eq!(kinds, ["start", "end", "start", "end", "closed"]);
    assert_eq!(audio % (FRAME * 2), 0);
    assert!(audio > 0);
}

#[tokio::test]
async fn test_stream_barge_in_cuts_the_sentence() {
    let app = serve(20).await;
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{}/v1/stream", app.addr))
        .await
        .unwrap();
    socket.next().await.unwrap().unwrap();
    let long = "palabra ".repeat(60);
    socket
        .send(Message::text(json!({ "type": "text", "text": long }).to_string()))
        .await
        .unwrap();
    socket
        .send(Message::text(json!({ "type": "flush" }).to_string()))
        .await
        .unwrap();
    while !matches!(socket.next().await, Some(Ok(Message::Binary(_)))) {}
    socket
        .send(Message::text(json!({ "type": "cancel" }).to_string()))
        .await
        .unwrap();
    let end = loop {
        if let Some(Ok(Message::Text(text))) = socket.next().await {
            let event: Value = serde_json::from_str(&text).unwrap();
            if event["type"] == "end" {
                break event;
            }
        }
    };
    assert_eq!(end["error"]["kind"], "cancelled");
}
