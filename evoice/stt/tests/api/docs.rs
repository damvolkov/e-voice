use serde_json::Value;

use crate::app;
use crate::fake::{batch, nodes};

#[tokio::test(flavor = "multi_thread")]
async fn test_openapi_lists_every_protocol_and_docs_render() {
    let (address, _) = app::serve(nodes(batch(0), None, None), 1 << 20, false).await;
    let spec: Value = reqwest::get(format!("http://{address}/openapi.json"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let paths: Vec<&String> = spec["paths"].as_object().unwrap().keys().collect();
    for path in [
        "/health",
        "/v1/models",
        "/v1/audio/transcriptions",
        "/v1/realtime",
        "/v1/listen",
        "/v1/speech-to-text",
        "/v1/stream",
    ] {
        assert!(
            paths.iter().any(|known| *known == path),
            "{path} missing from {paths:?}"
        );
    }
    assert!(spec["components"]["schemas"].get("Event").is_some());
    let page = reqwest::get(format!("http://{address}/docs")).await.unwrap();
    assert_eq!(page.status(), 200);
    assert!(page.text().await.unwrap().contains("openapi"));
}
