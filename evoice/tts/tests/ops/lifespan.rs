use std::path::PathBuf;

use e_voice_core::audio::AudioFile;
use e_voice_core::config::ops::OpsConfig;
use e_voice_core::models::ModelStore;
use e_voice_core::schema::lang::Lang;
use e_voice_tts::api::lifespan::Lifespan;
use e_voice_tts::core::settings::Settings;
use e_voice_tts::ops::similarity::Similarity;
use e_voice_tts::workflow::runner::Request;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn settings() -> Settings {
    let mut settings = Settings::default();
    settings.tts.ops = OpsConfig {
        data: root().join("data/tts"),
        manifest: root().join("evoice/tts/models.toml"),
        ..settings.tts.ops
    };
    settings.tts.synth.workers = 1;
    settings
}

#[tokio::test]
#[ignore = "requires installed models: make pocket"]
async fn test_lifespan_builds_pocket_learns_a_voice_and_speaks_both_languages() {
    let settings = settings();
    let (runner, _) = Lifespan::runner(&settings).await.unwrap();
    let rate = runner.caps().rate;
    let clip = root().join("data/tts/ops/vendor/pocket-tts-onnx-export/pocket_tts/config/jos.wav");
    let audio = AudioFile::decode(std::fs::read(&clip).unwrap(), Some("wav"), rate).unwrap();
    let voice = runner.synth().train(&[audio.clone().into()], None).unwrap();
    for (lang, text) in [
        (Lang::Es, "Hola, esto es una prueba."),
        (Lang::En, "Hello, this is a test."),
    ] {
        let session = runner.open(lang, Some(voice.clone())).await.unwrap();
        let (requests, inbox) = mpsc::channel(2);
        let (outbox, mut events) = mpsc::channel(64);
        requests.send(Request::Text(text.to_owned())).await.unwrap();
        requests.send(Request::Close).await.unwrap();
        let run = runner.clone();
        tokio::spawn(async move { run.run(session, inbox, outbox, CancellationToken::new()).await });
        let mut samples = 0;
        while let Some(event) = events.recv().await {
            if let e_voice_tts::schema::event::Event::Audio { audio, .. } = event {
                samples += audio.len();
            }
        }
        assert!(samples > rate as usize / 2, "{lang:?}: {samples} samples");
    }
    let state = Lifespan::start(&settings).await.unwrap();
    assert_eq!(state.models, ["pocket-es", "pocket-en"]);
    let store = ModelStore::open(&settings.tts.ops).unwrap();
    let similarity = Similarity::open(&store.dir("speaker-resnet34").unwrap(), &audio, rate).unwrap();
    let same = similarity.score(&audio, rate).unwrap();
    assert!(same > 0.99, "{same}");
    assert!(similarity.score(&[0.0; 100], rate).is_none_or(|score| score < same));
}
