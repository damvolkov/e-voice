use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::lang::Lang;
use e_voice_tts::api::server::Server;
use e_voice_tts::api::state::AppState;
use e_voice_tts::config::text::TextConfig;
use e_voice_tts::core::voices::VoiceStore;
use e_voice_tts::workflow::runner::Runner;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use crate::fake::FakeSynth;

/// A server over the fake backend on an ephemeral port, its data dir kept alive with it.
pub struct App {
    pub addr: SocketAddr,
    _data: TempDir,
}

pub async fn serve(delay: u64) -> App {
    let data = TempDir::new().unwrap();
    let state = AppState {
        runner: Runner::new(
            Arc::new(FakeSynth {
                delay: Duration::from_millis(delay),
            }),
            TextConfig { min: 0, max: 200 },
        ),
        voices: Arc::new(VoiceStore::open(data.path()).unwrap()),
        lang: Lang::Es,
        voice: None,
        models: vec!["fake-es".to_owned(), "fake-en".to_owned()],
        upload: 1 << 20,
        runtime: Runtime {
            sherpa: "test",
            git: "test",
            onnxruntime: "test",
        },
        shutdown: CancellationToken::new(),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, Server::router(state)).await.unwrap() });
    App { addr, _data: data }
}

/// A one-second 16 kHz mono 16-bit WAV of a tone.
pub fn wav() -> Vec<u8> {
    let samples: Vec<i16> = (0..16_000).map(|i| ((i as f32 * 0.1).sin() * 8_000.0) as i16).collect();
    let data = (samples.len() * 2) as u32;
    let mut bytes = Vec::new();
    for part in [
        b"RIFF".as_slice(),
        &(36 + data).to_le_bytes(),
        b"WAVEfmt ",
        &16u32.to_le_bytes(),
        &1u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &16_000u32.to_le_bytes(),
        &32_000u32.to_le_bytes(),
        &2u16.to_le_bytes(),
        &16u16.to_le_bytes(),
        b"data",
        &data.to_le_bytes(),
    ] {
        bytes.extend_from_slice(part);
    }
    bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
    bytes
}
