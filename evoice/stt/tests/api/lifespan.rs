use std::path::PathBuf;

use e_voice_core::config::ops::{ModelsVerify, OpsConfig};
use e_voice_core::models::ModelError;
use e_voice_stt::api::lifespan::Lifespan;
use e_voice_stt::core::settings::Settings;
use e_voice_stt::workflow::nodes::NodesError;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models.toml")
}

#[tokio::test(flavor = "multi_thread")]
async fn test_start_refuses_missing_models() {
    let empty = tempfile::tempdir().unwrap();
    let mut settings = Settings::default();
    settings.stt.ops = OpsConfig {
        data: empty.path().to_path_buf(),
        manifest: manifest(),
        verify: ModelsVerify::Stamp,
    };
    let refused = Lifespan::start(&settings).await;
    assert!(
        matches!(refused, Err(NodesError::Models(ModelError::Missing(_)))),
        "{refused:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires installed models: make pull"]
async fn test_start_loads_configured_nodes() {
    let mut settings = Settings::default();
    settings.stt.ops = OpsConfig {
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/stt"),
        manifest: manifest(),
        verify: ModelsVerify::Stamp,
    };
    let state = Lifespan::start(&settings).await.unwrap();
    assert_eq!(state.models, ["parakeet"]);
    settings.stt.pipeline.asr.offline.choices = vec![e_voice_stt::config::asr::AsrEngine::Whisper];
    let chosen = Lifespan::start(&settings).await.unwrap();
    assert_eq!(chosen.models, ["parakeet", "whisper"]);
    assert!(
        chosen
            .runner
            .engine(e_voice_stt::config::asr::AsrEngine::Whisper)
            .is_some()
    );
    assert_eq!(
        settings.models(),
        [
            "silero-vad",
            "nemotron-3.5-1120ms-int8",
            "parakeet-v3-int8",
            "whisper-turbo",
            "emotion2vec-plus-base"
        ]
    );
}
