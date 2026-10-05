use std::sync::Arc;

use e_voice_stt::config::vad::{VadBackend, VadConfig};
use e_voice_stt::workflow::vad::base::Vad;
use e_voice_stt::workflow::vad::registry::VadRegistry;

use crate::fixture;
use crate::vad::suite;

fn vad() -> Arc<dyn Vad> {
    let config = VadConfig {
        backend: VadBackend::Silero,
        ..VadConfig::default()
    };
    VadRegistry::build(&config, &fixture::store()).unwrap()
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_dialogue_yields_ordered_exact_segments() {
    suite::dialogue_yields_ordered_exact_segments(&vad());
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_silence_yields_nothing() {
    suite::silence_yields_nothing(&vad());
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_segments_do_not_depend_on_chunking() {
    suite::segments_do_not_depend_on_chunking(&vad());
}

#[test]
#[ignore = "requires installed models: make pull"]
fn test_sessions_are_isolated() {
    suite::sessions_are_isolated(&vad());
}
