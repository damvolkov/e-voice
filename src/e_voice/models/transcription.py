"""Transcription response models and formatters — backend-agnostic.

Formatters work with domain types (Span, Transcript) — never library types.
All STT backends produce Transcript, and these formatters render to API formats.
"""

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field

from e_voice.core.audio import Audio
from e_voice.models.stt import Span, Transcript

##### ENUMS #####


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
    seek: int = 0
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
    owned_by: str = "system"


class ListModelsResponse(BaseModel):
    """OpenAI-compatible model list response."""

    data: list[ModelObject]
    object: Literal["list"] = "list"


##### FORMATTERS — Span-based (backend-agnostic) #####


def span_to_model(span: Span, index: int = 0, word_timestamps: bool = False) -> TranscriptionSegment:
    """Convert domain Span → API response model."""
    words = (
        [TranscriptionWord(start=w.start, end=w.end, word=w.text, probability=w.probability) for w in span.words]
        if word_timestamps and span.words
        else None
    )
    return TranscriptionSegment(
        id=index,
        start=span.start,
        end=span.end,
        text=span.text,
        no_speech_prob=span.no_speech_prob,
        words=words,
    )


def format_span(span: Span, fmt: str, index: int = 0) -> str:
    """Format a single Span for streaming SSE delivery."""
    match fmt:
        case "text":
            return span.text
        case "json":
            return TranscriptionResponse(text=span.text).model_dump_json()
        case "verbose_json":
            return span_to_model(span, index).model_dump_json()
        case "srt":
            start, end = Audio.format_timestamp(span.start), Audio.format_timestamp(span.end)
            return f"{index + 1}\n{start} --> {end}\n{span.text.strip()}\n"
        case "vtt":
            start, end = Audio.format_timestamp_vtt(span.start), Audio.format_timestamp_vtt(span.end)
            return f"{start} --> {end}\n{span.text.strip()}\n"
        case _:
            return span.text


def build_transcript_response(
    transcript: Transcript,
    fmt: str,
    *,
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> tuple[str, str]:
    """Build full batch response from a Transcript. Returns (body, content_type)."""
    match fmt:
        case "text":
            return transcript.text, "text/plain"
        case "json":
            return TranscriptionResponse(text=transcript.text).model_dump_json(), "application/json"
        case "verbose_json":
            seg_models = [span_to_model(s, i, word_timestamps) for i, s in enumerate(transcript.spans)]
            all_words = [w for s in seg_models if s.words for w in s.words] if word_timestamps else None
            resp = TranscriptionVerboseResponse(
                task=task,  # ty: ignore[invalid-argument-type]
                language=transcript.language,
                duration=transcript.duration,
                text=transcript.text,
                segments=seg_models,
                words=all_words,
            )
            return resp.model_dump_json(), "application/json"
        case "srt":
            body = "\n".join(format_span(s, "srt", i) for i, s in enumerate(transcript.spans))
            return body, "text/plain"
        case "vtt":
            body = "WEBVTT\n\n" + "\n".join(format_span(s, "vtt") for s in transcript.spans)
            return body, "text/plain"
        case _:
            return transcript.text, "text/plain"
