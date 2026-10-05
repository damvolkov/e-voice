use std::sync::Arc;
use std::time::Duration;

use e_voice_tts::config::text::TextConfig;
use e_voice_tts::ops::bench::Bench;
use e_voice_tts::workflow::runner::Runner;
use serde_json::Value;
use tempfile::TempDir;

use crate::fake::FakeSynth;

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn test_bench_records_items_writes_the_round_trip_manifest_and_summarizes() {
    let dir = TempDir::new().unwrap();
    let manifest = dir.path().join("manifest.jsonl");
    std::fs::write(
        &manifest,
        concat!(
            r#"{"id":"a","audio":"a.wav","lang":"es","text":"Hola mundo."}"#,
            "\n",
            r#"{"id":"b","lang":"en","text":"Hello there, friend."}"#,
            "\n",
            r#"{"id":"c","lang":"es"}"#,
            "\n"
        ),
    )
    .unwrap();
    let bench = Bench {
        runner: Runner::new(
            Arc::new(FakeSynth {
                delay: Duration::from_millis(1),
            }),
            TextConfig { min: 0, max: 80 },
        ),
        voice: None,
        streams: 2,
        wavs: true,
        similarity: None,
        settings: Value::Null,
    };
    let out = dir.path().join("run.jsonl");
    let summary = bench.run(&manifest, &out, None).await.unwrap();
    assert_eq!((summary.items, summary.failures), (2, 0));
    assert!(summary.audio_s > 0.0 && summary.throughput_x > 0.0 && summary.ttfa_p50_ms > 0.0);
    let records = lines(&out);
    assert_eq!(records.len(), 3);
    assert_eq!(records[2]["kind"], "summary");
    let trip = lines(&dir.path().join("run.wavs/manifest.jsonl"));
    assert_eq!(trip.len(), 2);
    assert_eq!(
        (trip[0]["audio"].as_str(), trip[0]["text"].as_str()),
        (Some("a.wav"), Some("Hola mundo."))
    );
    let wav = std::fs::read(dir.path().join("run.wavs/a.wav")).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    let limited = bench
        .run(&manifest, &dir.path().join("one.jsonl"), Some(1))
        .await
        .unwrap();
    assert_eq!(limited.items, 1);
}

#[tokio::test]
async fn test_bench_rejects_a_malformed_manifest() {
    let dir = TempDir::new().unwrap();
    let manifest = dir.path().join("bad.jsonl");
    std::fs::write(&manifest, "not json\n").unwrap();
    let bench = Bench {
        runner: Runner::new(Arc::new(FakeSynth { delay: Duration::ZERO }), TextConfig::default()),
        voice: None,
        streams: 1,
        wavs: false,
        similarity: None,
        settings: Value::Null,
    };
    assert!(bench.run(&manifest, &dir.path().join("out.jsonl"), None).await.is_err());
}
