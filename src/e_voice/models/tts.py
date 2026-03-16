"""Pydantic models for Text-to-Speech API — OpenAI-compatible."""

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field


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
