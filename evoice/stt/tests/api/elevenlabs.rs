use reqwest::multipart::{Form, Part};
use serde_json::Value;

use crate::app::{self, wav};
use crate::fake::{batch, nodes, ser};

#[tokio::test(flavor = "multi_thread")]
async fn test_scribe_matches_elevenlabs_shape() {
    let (address, _) = app::serve(nodes(batch(0), Some(ser(0)), None), 1 << 20, false).await;
    let file = wav(16_000, &[(0.0, 16_000), (0.9, 8_000), (0.0, 8_000)]);
    let form = Form::new()
        .text("model_id", "scribe_v1")
        .text("language_code", "spa")
        .part("file", Part::bytes(file).file_name("speech.wav"));
    let response = reqwest::Client::new()
        .post(format!("http://{address}/v1/speech-to-text"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(
        (body["text"].as_str(), body["language_code"].as_str()),
        (Some("batch:8000"), Some("es"))
    );
    assert_eq!(body["words"][0]["start"], 1.0);
    assert_eq!(body["emotion"]["label"], "happy");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_scribe_errors_follow_elevenlabs_shape() {
    let (address, _) = app::serve(nodes(batch(0), None, None), 1 << 20, false).await;
    let form = Form::new().text("model_id", "scribe_v1");
    let response = reqwest::Client::new()
        .post(format!("http://{address}/v1/speech-to-text"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["detail"]["status"], "invalid_request");
}
