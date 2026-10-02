use std::path::Path;

use e_voice_stt::config::ops::{ModelsVerify, OpsConfig};
use e_voice_stt::core::models::ModelStore;
use e_voice_stt::ops::wake::{Wake, WakeError, WakeTrial};

use crate::fixture;

fn trial(threshold: f32, boost: f32, recall: f64, false_accepts: usize) -> WakeTrial {
    WakeTrial {
        threshold,
        boost,
        recall,
        voices: Vec::new(),
        false_accepts,
        per_hour: 0.0,
    }
}

#[test]
fn test_pick_prefers_silence_then_recall_then_strictness() {
    let trials = [
        trial(0.1, 1.0, 0.9, 2),
        trial(0.1, 2.0, 0.7, 0),
        trial(0.2, 2.0, 0.7, 0),
        trial(0.2, 1.0, 0.7, 0),
        trial(0.5, 1.0, 0.0, 0),
    ];
    let chosen = Wake::pick(&trials).unwrap();
    assert_eq!((chosen.threshold, chosen.boost), (0.2, 1.0));
    assert_eq!(Wake::pick(&[trial(0.5, 1.0, 0.0, 0)]).unwrap().threshold, 0.5);
    assert!(Wake::pick(&[]).is_none());
}

#[test]
fn test_manifests_point_at_fleurs() {
    let found = Wake::manifests(Path::new("/data"));
    assert_eq!(found[0], Path::new("/data/ops/datasets/fleurs-en/manifest.jsonl"));
    assert_eq!(found.len(), 2);
}

#[test]
fn test_prepare_names_the_missing_model() {
    let config = OpsConfig {
        data: std::env::temp_dir().join("e-voice-wake-empty"),
        manifest: Path::new(env!("CARGO_MANIFEST_DIR")).join("models.toml"),
        verify: ModelsVerify::Stamp,
    };
    let store = ModelStore::open(&config).unwrap();
    let error = Wake::prepare(&store, "hey eager", &[], 0).unwrap_err();
    assert!(matches!(error, WakeError::Model(id) if id == "kws-gigaspeech"));
}

#[test]
#[ignore = "requires installed models: make wake"]
fn test_sweep_measures_every_pair_and_finds_a_usable_one() {
    let wake = Wake::prepare(&fixture::store(), "hey eager", &[], 0).unwrap();
    assert_eq!(wake.positives(), 3 * 5 * 3);
    assert!(wake.negative_s > 10.0);
    let trials = wake.sweep().unwrap();
    assert_eq!(trials.len(), 8 * 5);
    let chosen = Wake::pick(&trials).unwrap();
    assert!(chosen.recall > 0.3, "{chosen:?}");
    assert_eq!(chosen.voices.len(), 3);
}
