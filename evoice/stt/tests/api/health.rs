use crate::app;
use crate::fake::{batch, nodes};

async fn get(path: &str) -> String {
    let (address, _) = app::serve(nodes(batch(0), None, None), 1 << 20, false).await;
    let mut stream = tokio::net::TcpStream::connect(&address).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    tokio::io::AsyncWriteExt::write_all(&mut stream, request.as_bytes())
        .await
        .unwrap();
    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut stream, &mut response)
        .await
        .unwrap();
    response
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reports_loaded_runtime() {
    let response = get("/health").await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(
        response.ends_with(r#"{"status":"ok","version":"0.1.0","sherpa":"1.13.8","onnxruntime":"1.28.2"}"#),
        "{response}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_models_lists_the_loaded_asr() {
    let response = get("/v1/models").await;
    assert!(
        response.ends_with(
            r#"{"data":[{"created":0,"id":"fake-asr","object":"model","owned_by":"e-voice"}],"object":"list"}"#
        ),
        "{response}"
    );
}
