"""TTS domain types — backend-agnostic value objects for text-to-speech."""

from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from e_voice.core.settings import DeviceType

##### DOMAIN TYPES #####

type AudioChunk = tuple[NDArray[np.float32], int]

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
class VoiceEntry:
    """Typed voice metadata — used by all backends to expose voices uniformly."""

    id: str
    language: str = "multilingual"
    cloned: bool = False


def parse_voice_filename(filename: str) -> VoiceEntry:
    """Parse a voice file stem like 'aiden_en' → VoiceEntry(id='aiden', language='en').

    Convention: {name}_{langcode}.ext — last underscore segment is the language.
    Falls back to 'multilingual' if no underscore found.
    """
    stem = filename.removesuffix(".pt").removesuffix(".bin")
    if "_" in stem:
        name, lang = stem.rsplit("_", 1)
        return VoiceEntry(id=name, language=lang, cloned=True)
    return VoiceEntry(id=stem, language="multilingual", cloned=True)


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
        min_length=1,
        description="Voice ID — backend-specific identifier.",
        examples=["af_heart", "serena", "tatan"],
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
        max_length=15,
        description="BCP-47 language code or language name.",
        examples=["en-us", "es", "english", "spanish"],
    )

    @model_validator(mode="after")
    def _resolve_lang_from_voice(self) -> Self:
        """Infer lang from Kokoro voice prefix if applicable; leave as-is otherwise."""
        if "lang" not in self.model_fields_set:
            with suppress(ValueError):
                self.lang = resolve_voice_lang(self.voice)
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
