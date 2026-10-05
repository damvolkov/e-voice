use std::path::PathBuf;
use std::time::Duration;

use e_voice_stt::config::pipeline::PipelineConfig;
use e_voice_stt::ops::bench::manifest::Manifest;
use e_voice_stt::ops::bench::suite::{Suite, SuiteMode};
use e_voice_stt::workflow::runner::Runner;
use serde_json::Value;

use crate::app::wav;
use crate::fake::{batch, nodes, ser};

fn workspace(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("e-voice-suite-{name}-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("speech.wav"),
        wav(16_000, &[(0.0, 8_000), (0.9, 8_000), (0.0, 12_000)]),
    )
    .unwrap();
    std::fs::write(dir.join("broken.wav"), b"not audio").unwrap();
    let lines = [
        r#"{"id":"speech","audio":"speech.wav","lang":"en","text":"hello","emotion":"happy"}"#,
        r#"{"id":"broken","audio":"broken.wav","lang":"es"}"#,
    ];
    std::fs::write(dir.join("manifest.jsonl"), lines.join("\n")).unwrap();
    dir
}

async fn run(mode: SuiteMode) -> (Vec<Value>, PathBuf) {
    let dir = workspace(mode_name(mode));
    let config = PipelineConfig {
        tick: Duration::from_millis(10),
        ..PipelineConfig::default()
    };
    let suite = Suite {
        runner: Runner::new(nodes(batch(0), Some(ser(0)), None), config),
        mode,
        streams: 2,
        settings: serde_json::json!({"case": "fake"}),
    };
    let out = dir.join("results/run.jsonl");
    let summary = suite
        .run(Manifest::load(&dir.join("manifest.jsonl")).unwrap(), &out)
        .await
        .unwrap();
    assert_eq!((summary.items, summary.streams), (2, 2));
    let lines = std::fs::read_to_string(&out).unwrap();
    (
        lines.lines().map(|line| serde_json::from_str(line).unwrap()).collect(),
        dir,
    )
}

const fn mode_name(mode: SuiteMode) -> &'static str {
    match mode {
        SuiteMode::File => "file",
        SuiteMode::Live => "live",
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_file_mode_writes_one_record_per_item_and_a_summary() {
    let (records, dir) = run(SuiteMode::File).await;
    assert_eq!(records.len(), 3);
    let speech = records.iter().find(|record| record["id"] == "speech").unwrap();
    assert_eq!(
        (speech["text"].as_str(), speech["mode"].as_str()),
        (Some("batch:8000"), Some("file"))
    );
    assert_eq!(
        (speech["ref_text"].as_str(), speech["ref_emotion"].as_str()),
        (Some("hello"), Some("happy"))
    );
    assert_eq!(speech["segments"][0]["start"], 0.5);
    assert!(speech["segments"][0]["final_ms"].is_null());
    let broken = records.iter().find(|record| record["id"] == "broken").unwrap();
    assert!(broken["failure"].is_string());
    let summary = records.last().unwrap();
    assert_eq!(
        (summary["kind"].as_str(), summary["settings"]["case"].as_str()),
        (Some("summary"), Some("fake"))
    );
    assert_eq!(summary["audio_s"], 1.75);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_live_mode_times_finals_against_speech_end() {
    let (records, dir) = run(SuiteMode::Live).await;
    let speech = records.iter().find(|record| record["id"] == "speech").unwrap();
    assert_eq!(speech["mode"], "live");
    let segment = &speech["segments"][0];
    assert!(segment["final_ms"].as_f64().unwrap() > 0.0, "{segment}");
    assert!(speech["wall_s"].as_f64().unwrap() >= 1.5);
    std::fs::remove_dir_all(dir).unwrap();
}
