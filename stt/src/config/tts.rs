use serde::{Deserialize, Serialize};

/// Reserved for text-to-speech; any key in `[tts]` is rejected until it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsConfig {}
