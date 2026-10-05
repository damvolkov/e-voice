use std::sync::Arc;

use e_voice_core::runtime::Runtime;
use e_voice_core::schema::lang::Lang;
use e_voice_stt::api::server::Server;
use e_voice_stt::api::state::AppState;
use e_voice_stt::config::api::EmotionMode;
use e_voice_stt::config::pipeline::PipelineConfig;
use e_voice_stt::workflow::nodes::Nodes;
use e_voice_stt::workflow::runner::Runner;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Serves the real router over fake nodes on an ephemeral port; returns `host:port` and the shutdown token.
pub async fn serve(nodes: Arc<Nodes>, upload: usize, tags: bool) -> (String, CancellationToken) {
    let shutdown = CancellationToken::new();
    let state = AppState {
        runner: Runner::new(nodes, PipelineConfig::default()),
        lang: Lang::Es,
        emotion: if tags { EmotionMode::Tag } else { EmotionMode::Field },
        models: vec!["fake-asr".to_owned()],
        upload,
        runtime: Runtime::probe().unwrap(),
        shutdown: shutdown.clone(),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, Server::router(state)).await });
    (address.to_string(), shutdown)
}

/// Little-endian s16 PCM of one constant value.
pub fn pcm(value: f32, samples: usize) -> Vec<u8> {
    std::iter::repeat_n(((value * 32_767.0) as i16).to_le_bytes(), samples)
        .flatten()
        .collect()
}

/// A WAV file of constant segments, for the REST endpoints.
pub fn wav(rate: u32, parts: &[(f32, u32)]) -> Vec<u8> {
    let samples: Vec<u8> = parts.iter().flat_map(|&(value, n)| pcm(value, n as usize)).collect();
    let size = samples.len() as u32;
    let mut bytes = Vec::with_capacity(44 + samples.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(&samples);
    bytes
}

/// Sends `frames`, then gathers every text frame until the server closes; returns them with the close code.
pub async fn exchange(url: &str, frames: Vec<tokio_tungstenite::tungstenite::Message>) -> (Vec<String>, Option<u16>) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    let (mut socket, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    for frame in frames {
        socket.send(frame).await.unwrap();
    }
    let (mut texts, mut code) = (Vec::new(), None);
    let gather = async {
        while let Some(Ok(message)) = socket.next().await {
            match message {
                Message::Text(text) => texts.push(text.to_string()),
                Message::Close(frame) => {
                    code = frame.map(|frame| u16::from(frame.code));
                    break;
                }
                _ => {}
            }
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), gather)
        .await
        .unwrap();
    (texts, code)
}

/// Text frames parsed as JSON.
pub fn json(texts: &[String]) -> Vec<serde_json::Value> {
    texts.iter().map(|text| serde_json::from_str(text).unwrap()).collect()
}
