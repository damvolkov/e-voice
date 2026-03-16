"""Pydantic models for transcription/translation API — OpenAI-compatible."""

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field


class ResponseFormat(StrEnum):
    """Supported transcription output formats."""

    TEXT = auto()
    JSON = auto()
    VERBOSE_JSON = auto()
    SRT = auto()
    VTT = auto()


class TimestampGranularity(StrEnum):
    """Timestamp granularity levels."""

    SEGMENT = auto()
    WORD = auto()


##### REQUEST MODELS #####


class TranscriptionParams(BaseModel):
    """Form parameters for transcription request (parsed from multipart)."""

    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp_granularities: list[TimestampGranularity] = Field(default_factory=lambda: [TimestampGranularity.SEGMENT])
    stream: bool = False
    hotwords: str | None = None
    vad_filter: bool = False


class TranslationParams(BaseModel):
    """Form parameters for translation request."""

    model: str | None = None
    prompt: str | None = None
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    stream: bool = False
    vad_filter: bool = False


##### RESPONSE MODELS #####


class TranscriptionWord(BaseModel):
    """Word-level timestamp."""

    start: float
    end: float
    word: str
    probability: float = 0.0


class TranscriptionSegment(BaseModel):
    """Segment-level transcription result."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: list[TranscriptionWord] | None = None


class TranscriptionResponse(BaseModel):
    """Simple JSON transcription response (OpenAI-compatible)."""

    text: str


class TranscriptionVerboseResponse(BaseModel):
    """Verbose JSON transcription response with segments and words."""

    task: Literal["transcribe", "translate"] = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    words: list[TranscriptionWord] | None = None


##### MODEL MANAGEMENT #####


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    created: int = 0
    object: Literal["model"] = "model"
    owned_by: str = "whisper"


class ListModelsResponse(BaseModel):
    """OpenAI-compatible model list response."""

    data: list[ModelObject]
    object: Literal["list"] = "list"
