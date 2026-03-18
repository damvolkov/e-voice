"""STT value objects — ModelSpec and InferenceParams for whisper adapter."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Identity of a loaded model — same model_id on different devices = different instances."""

    model_id: str
    device: str = "cpu"
    compute_type: str = "auto"


@dataclass(frozen=True, slots=True)
class InferenceParams:
    """Unified inference params — eliminates 8-arg signatures."""

    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    word_timestamps: bool = False
    vad_filter: bool = False
    hotwords: str | None = None
