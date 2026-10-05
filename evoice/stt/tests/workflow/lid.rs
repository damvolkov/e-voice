use e_voice_core::schema::lang::Lang;
use e_voice_stt::config::lid::{LidBackend, LidConfig};
use e_voice_stt::workflow::lid::registry::LidRegistry;

use crate::fixture;

#[test]
fn test_off_builds_nothing() {
    assert!(
        LidRegistry::build(&LidConfig::default(), &fixture::store())
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "requires installed models: make setup ARGS=--all"]
fn test_whisper_identifies_supported_languages_only() {
    let config = LidConfig {
        backend: LidBackend::Whisper,
        ..LidConfig::default()
    };
    let lid = LidRegistry::build(&config, &fixture::store()).unwrap().unwrap();
    let identify = |name: &str| lid.identify(&fixture::wav("parakeet-v3-int8", name)).unwrap();
    assert_eq!(identify("es.wav"), Some(Lang::Es));
    assert_eq!(identify("en.wav"), Some(Lang::En));
    assert_eq!(identify("fr.wav"), None);
}
