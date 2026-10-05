use reqwest::multipart::{Form, Part};
use serde_json::Value;

use crate::app::{serve, wav};

#[tokio::test]
async fn test_voices_learn_list_and_forget() {
    let app = serve(1).await;
    let client = reqwest::Client::new();
    let url = format!("http://{}/v1/voices", app.addr);
    let form = Form::new()
        .text("voice_id", "damien")
        .part("file", Part::bytes(wav()).file_name("a.wav"))
        .part("file", Part::bytes(wav()).file_name("b.wav"));
    let created = client.post(&url).multipart(form).send().await.unwrap();
    assert_eq!(created.status(), 201);
    let body: Value = created.json().await.unwrap();
    assert_eq!(body["voice_id"], "damien");
    assert!((body["seconds"].as_f64().unwrap() - 2.0).abs() < 0.05);
    let listed: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
    assert_eq!(listed["voices"], serde_json::json!(["damien"]));
    let deleted = client.delete(format!("{url}/damien")).send().await.unwrap();
    assert_eq!(deleted.status(), 204);
    let missing = client.delete(format!("{url}/damien")).send().await.unwrap();
    assert_eq!(missing.status(), 404);
}

#[tokio::test]
async fn test_voices_reject_bad_ids_and_missing_audio() {
    let app = serve(1).await;
    let client = reqwest::Client::new();
    let url = format!("http://{}/v1/voices", app.addr);
    let bad = Form::new()
        .text("voice_id", "Bad Id!")
        .part("file", Part::bytes(wav()).file_name("a.wav"));
    assert_eq!(client.post(&url).multipart(bad).send().await.unwrap().status(), 400);
    let empty = Form::new().text("voice_id", "ok");
    assert_eq!(client.post(&url).multipart(empty).send().await.unwrap().status(), 400);
}
