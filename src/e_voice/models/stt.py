"""STT domain types — backend-agnostic value objects for speech-to-text."""

from dataclasses import dataclass

##### DOMAIN TYPES #####


@dataclass(frozen=True, slots=True)
class Word:
    """Single word with timing — normalized across all STT backends."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class Span:
    """Timed segment of transcription — normalized across all STT backends."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()
    no_speech_prob: float = 0.0


@dataclass(frozen=True, slots=True)
class Transcript:
    """Complete transcription result — normalized across all STT backends."""

    spans: tuple[Span, ...]
    language: str = ""
    duration: float = 0.0

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans).strip()


##### SPECS #####


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Identity of a loaded STT model — same model_id on different devices = different instances."""

    model_id: str
    device: str = "cpu"
    compute_type: str = "auto"


##### INFERENCE PARAMS #####


@dataclass(frozen=True, slots=True)
class InferenceParams:
    """Backend-agnostic inference params. Backends ignore unsupported fields."""

    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    word_timestamps: bool = False
    vad_filter: bool = False
    hotwords: str | None = None
