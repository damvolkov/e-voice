use reqwest::multipart::{Form, Part};
use serde_json::Value;

use crate::app::{self, wav};
use crate::fake::{batch, nodes, ser};

const LOUD: f32 = 0.9;

type Case<'a> = (&'a [(&'a str, &'a str)], Option<Vec<u8>>, u16, Option<&'a str>);

fn speech() -> Vec<u8> {
    wav(
        16_000,
        &[
            (0.0, 16_000),
            (LOUD, 8_000),
            (0.0, 16_000),
            (LOUD, 16_000),
            (0.0, 8_000),
        ],
    )
}

async fn post(fields: &[(&str, &str)], file: Option<Vec<u8>>, upload: usize, tags: bool) -> reqwest::Response {
    let (address, _) = app::serve(nodes(batch(0), Some(ser(0)), None), upload, tags).await;
    let mut form = fields.iter().fold(Form::new(), |form, (name, value)| {
        form.text(name.to_string(), value.to_string())
    });
    if let Some(bytes) = file {
        form = form.part("file", Part::bytes(bytes).file_name("speech.wav"));
    }
    reqwest::Client::new()
        .post(format!("http://{address}/v1/audio/transcriptions"))
        .multipart(form)
        .send()
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_json_is_openai_shaped_with_emotion_extension() {
    let response = post(
        &[("model", "whisper-1"), ("language", "es")],
        Some(speech()),
        1 << 20,
        false,
    )
    .await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["text"], "batch:8000 batch:16000", "{body}");
    assert_eq!(body["emotion"]["label"], "happy", "{body}");
    assert_eq!(body["usage"], serde_json::json!({"type": "duration", "seconds": 4.0}));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_verbose_json_reports_segments_in_seconds() {
    let response = post(
        &[("model", "x"), ("response_format", "verbose_json")],
        Some(speech()),
        1 << 20,
        false,
    )
    .await;
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["task"], "transcribe");
    assert_eq!(body["language"], "spanish");
    assert_eq!(body["duration"], 4.0);
    let segments = body["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 2);
    assert_eq!(
        (
            segments[0]["id"].as_u64(),
            segments[0]["start"].as_f64(),
            segments[0]["end"].as_f64()
        ),
        (Some(0), Some(1.0), Some(1.5))
    );
    assert_eq!(segments[1]["text"], "batch:16000");
    assert_eq!(segments[1]["emotion"]["label"], "happy");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_text_srt_and_vtt_render_cues() {
    let text = post(&[("response_format", "text")], Some(speech()), 1 << 20, true)
        .await
        .text()
        .await
        .unwrap();
    assert_eq!(text, "[happy] batch:8000 [happy] batch:16000");
    let srt = post(&[("response_format", "srt")], Some(speech()), 1 << 20, false)
        .await
        .text()
        .await
        .unwrap();
    assert_eq!(
        srt,
        "1\n00:00:01,000 --> 00:00:01,500\nbatch:8000\n\n2\n00:00:02,500 --> 00:00:03,500\nbatch:16000\n\n"
    );
    let vtt = post(&[("response_format", "vtt")], Some(speech()), 1 << 20, false)
        .await
        .text()
        .await
        .unwrap();
    assert!(
        vtt.starts_with("WEBVTT\n\n00:00:01.000 --> 00:00:01.500\nbatch:8000\n"),
        "{vtt}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resampled_upload_keeps_timing() {
    let file = wav(48_000, &[(0.0, 48_000), (LOUD, 24_000), (0.0, 24_000)]);
    let body: Value = post(&[("response_format", "verbose_json")], Some(file), 1 << 20, false)
        .await
        .json()
        .await
        .unwrap();
    let start = body["segments"][0]["start"].as_f64().unwrap();
    assert!((start - 1.0).abs() < 0.02, "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_errors_follow_openai_shape() {
    let cases: [Case<'_>; 4] = [
        (&[("language", "fr")], Some(speech()), 400, Some("language")),
        (
            &[("response_format", "xml")],
            Some(speech()),
            400,
            Some("response_format"),
        ),
        (&[("model", "x")], None, 400, Some("file")),
        (&[], Some(b"not audio at all".to_vec()), 400, Some("file")),
    ];
    for (fields, file, status, param) in cases {
        let response = post(fields, file, 1 << 20, false).await;
        assert_eq!(response.status(), status, "{fields:?}");
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"].as_str(), param);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_upload_over_limit_is_rejected() {
    let response = post(&[], Some(speech()), 1_024, false).await;
    assert_eq!(response.status(), 413);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stream_emits_deltas_then_done() {
    let body = post(&[("stream", "true")], Some(speech()), 1 << 20, false)
        .await
        .text()
        .await
        .unwrap();
    let events: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let kinds: Vec<&str> = events.iter().map(|event| event["type"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        ["transcript.text.delta", "transcript.text.delta", "transcript.text.done"]
    );
    assert_eq!(
        (events[0]["delta"].as_str(), events[1]["delta"].as_str()),
        (Some("batch:8000"), Some(" batch:16000"))
    );
    assert_eq!(events[2]["text"], "batch:8000 batch:16000");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_emotion_parameter_selects_field_tag_or_off() {
    let tagged: Value = post(
        &[("model", "whisper-1"), ("emotion", "tag")],
        Some(speech()),
        1 << 20,
        false,
    )
    .await
    .json()
    .await
    .unwrap();
    assert!(tagged["text"].as_str().unwrap().starts_with("[happy] "));
    assert_eq!(tagged["emotion"]["label"], "happy");
    let off: Value = post(
        &[("model", "whisper-1"), ("emotion", "off")],
        Some(speech()),
        1 << 20,
        true,
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(off["text"], "batch:8000 batch:16000");
    assert!(off.get("emotion").is_none());
    let bad = post(
        &[("model", "whisper-1"), ("emotion", "loud")],
        Some(speech()),
        1 << 20,
        false,
    )
    .await;
    assert_eq!(bad.status(), 400);
    let body: Value = bad.json().await.unwrap();
    assert_eq!(body["error"]["param"], "emotion");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_field_selects_a_loaded_engine() {
    use e_voice_stt::config::asr::AsrEngine;
    use e_voice_stt::workflow::asr::registry::AsrBackend;

    use crate::fake::FakeNamed;

    let mut built = nodes(batch(0), None, None);
    std::sync::Arc::get_mut(&mut built).unwrap().extra = vec![(
        AsrEngine::Whisper,
        AsrBackend::Batch(std::sync::Arc::new(FakeNamed("whisper"))),
    )];
    let (address, _) = app::serve(built, 1 << 20, false).await;
    let ask = |model: &'static str| {
        let form = Form::new()
            .text("model", model)
            .text("response_format", "text")
            .part("file", Part::bytes(speech()).file_name("speech.wav"));
        reqwest::Client::new()
            .post(format!("http://{address}/v1/audio/transcriptions"))
            .multipart(form)
            .send()
    };
    assert_eq!(
        ask("whisper").await.unwrap().text().await.unwrap(),
        "whisper:8000 whisper:16000"
    );
    assert_eq!(
        ask("WHISPER-TURBO").await.unwrap().text().await.unwrap(),
        "whisper:8000 whisper:16000"
    );
    assert_eq!(
        ask("whisper-1").await.unwrap().text().await.unwrap(),
        "batch:8000 batch:16000"
    );
    let missing = ask("cohere").await.unwrap();
    assert_eq!(missing.status(), 400);
    let body: Value = missing.json().await.unwrap();
    assert_eq!(body["error"]["param"], "model");
}
