"""Pydantic models and value objects for Text-to-Speech API."""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field

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


##### ADAPTER VALUE OBJECTS #####


@dataclass(frozen=True, slots=True)
class TTSModelSpec:
    """Identity of a loaded TTS model+device pair."""

    model_id: str = "kokoro"
    device: DeviceType = DeviceType.CPU


@dataclass(frozen=True, slots=True)
class SynthesisParams:
    """Unified synthesis params."""

    voice: str = "af_heart"
    speed: float = 1.0
    lang: str | None = None

    @property
    def resolved_lang(self) -> str:
        """Infer language from voice prefix if not explicitly set. O(1)."""
        if self.lang:
            return self.lang
        prefix = self.voice[0].upper() if self.voice else "A"
        try:
            return VoiceLang[prefix]
        except KeyError:
            return VoiceLang.A


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

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: SpeechResponseFormat = SpeechResponseFormat.MP3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = True
    stream_format: StreamFormat = StreamFormat.AUDIO
    sample_rate: int | None = Field(default=None, ge=8000, le=48000)
    lang: str = "en-us"


##### RESPONSE MODELS #####


class VoiceObject(BaseModel):
    """Single voice entry."""

    id: str
    name: str
    language: str


class ListVoicesResponse(BaseModel):
    """Voice listing response."""

    voices: list[VoiceObject]
    object: Literal["list"] = "list"


class SpeechAudioDeltaEvent(BaseModel):
    """SSE event for streaming audio chunk."""

    type: Literal["speech.audio.delta"] = "speech.audio.delta"
    audio: str


class SpeechAudioDoneEvent(BaseModel):
    """SSE event for streaming completion."""

    type: Literal["speech.audio.done"] = "speech.audio.done"
