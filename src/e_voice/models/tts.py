"""Pydantic models and value objects for Text-to-Speech API."""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from e_voice.core.settings import DeviceType

##### ADAPTER ENUMS #####


class OnnxProvider(StrEnum):
    """ONNX execution provider identifiers."""

    CUDA = "CUDAExecutionProvider"
    CPU = "CPUExecutionProvider"


class VoiceLang(StrEnum):
    """Voice prefix → BCP-47 language code."""

    A = "en-us"
    B = "en-gb"
    E = "es"
    F = "fr"
    H = "hi"
    I = "it"  # noqa: E741
    J = "ja"
    P = "pt-br"
    Z = "zh"


##### VOICE RESOLUTION #####

_VOICE_LANG_MAP: dict[str, str] = {member.name: member.value for member in VoiceLang}
_VALID_LANGS: frozenset[str] = frozenset(_VOICE_LANG_MAP.values())


def resolve_voice_lang(voice: str) -> str:
    """Resolve voice ID → BCP-47 language code from its prefix. O(1)."""
    if not voice:
        raise ValueError("Voice ID must not be empty.")
    prefix = voice[0].upper()
    if (lang := _VOICE_LANG_MAP.get(prefix)) is not None:
        return lang
    valid = ", ".join(sorted(_VOICE_LANG_MAP))
    raise ValueError(f"Unknown voice prefix '{voice[0]}'. Valid prefixes: {valid}")


##### ADAPTER VALUE OBJECTS #####


@dataclass(frozen=True, slots=True)
class TTSModelSpec:
    """Identity of a loaded TTS model+device pair."""

    model_id: str = "kokoro"
    device: DeviceType = DeviceType.CPU


@dataclass(frozen=True, slots=True)
class SynthesisParams:
    """Validated synthesis params passed to the TTS adapter."""

    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"


##### API ENUMS #####


class SpeechResponseFormat(StrEnum):
    """Supported audio output formats."""

    PCM = auto()
    MP3 = auto()
    WAV = auto()
    FLAC = auto()
    OPUS = auto()
    AAC = auto()


class StreamFormat(StrEnum):
    """How streaming audio is delivered."""

    AUDIO = auto()
    SSE = auto()


##### REQUEST MODELS #####


class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body (OpenAI-compatible)."""

    model: str = Field(default="kokoro", description="TTS model identifier.")
    input: str = Field(..., min_length=1, max_length=10_000, description="Text to synthesize.")
    voice: str = Field(
        default="af_heart",
        min_length=4,
        description="Voice ID — first char encodes language, second encodes gender (f/m).",
        examples=["af_heart", "ef_dora", "bm_george", "jf_alpha"],
    )
    response_format: SpeechResponseFormat = Field(
        default=SpeechResponseFormat.MP3,
        description="Output audio format.",
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed multiplier.")
    stream: bool = Field(default=True, description="Enable streaming delivery.")
    stream_format: StreamFormat = Field(default=StreamFormat.AUDIO, description="Streaming transport mode.")
    sample_rate: int | None = Field(default=None, ge=8000, le=48000, description="Custom output sample rate (Hz).")
    lang: str = Field(
        default="en-us",
        min_length=2,
        max_length=10,
        description="BCP-47 language code. Auto-inferred from voice prefix when not provided.",
        examples=["en-us", "es", "fr", "ja"],
    )

    @field_validator("voice")
    @classmethod
    def _validate_voice_prefix(cls, v: str) -> str:
        resolve_voice_lang(v)
        return v

    @field_validator("lang")
    @classmethod
    def _validate_lang_code(cls, v: str) -> str:
        if v not in _VALID_LANGS:
            raise ValueError(f"Unsupported language '{v}'. Valid: {', '.join(sorted(_VALID_LANGS))}")
        return v

    @model_validator(mode="after")
    def _resolve_and_validate_lang(self) -> Self:
        """Infer lang from voice prefix; reject explicit lang that conflicts with voice."""
        voice_lang = resolve_voice_lang(self.voice)
        if "lang" not in self.model_fields_set:
            self.lang = voice_lang
        elif self.lang != voice_lang:
            raise ValueError(
                f"Language '{self.lang}' conflicts with voice '{self.voice}' "
                f"(voice language: '{voice_lang}'). Pick a voice that matches."
            )
        return self


##### RESPONSE MODELS #####


class VoiceObject(BaseModel):
    """Single voice entry in the catalog."""

    id: str = Field(..., description="Unique voice identifier.", examples=["af_heart", "ef_dora"])
    name: str = Field(..., description="Display name.", examples=["af_heart"])
    language: str = Field(..., description="BCP-47 language code.", examples=["en-us", "es"])


class ListVoicesResponse(BaseModel):
    """Voice catalog, optionally filtered by language."""

    voices: list[VoiceObject] = Field(default_factory=list, description="Available voice entries.")
    object: Literal["list"] = "list"


class SpeechAudioDeltaEvent(BaseModel):
    """SSE event for streaming audio chunk."""

    type: Literal["speech.audio.delta"] = "speech.audio.delta"
    audio: str = Field(..., description="Base64-encoded PCM16 audio data.")


class SpeechAudioDoneEvent(BaseModel):
    """SSE event for streaming completion."""

    type: Literal["speech.audio.done"] = "speech.audio.done"
