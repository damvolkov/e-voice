"""System API models — model download and management."""

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class ServiceType(StrEnum):
    """Available ML service types."""

    STT = auto()
    TTS = auto()


class DownloadRequest(BaseModel):
    """Request body for model download."""

    model: str = Field(description="Model ID (HuggingFace repo for STT, 'kokoro' for TTS)")
    service: ServiceType = Field(description="Target service: stt or tts")


class DownloadResponse(BaseModel):
    """Response after model download."""

    status: str
    service: str
    model: str
    path: str


class ModelEntry(BaseModel):
    """A single downloaded model entry."""

    id: str
    service: str
    path: str
    size_mb: float = 0.0


class ModelsListResponse(BaseModel):
    """Response listing all downloaded models by service."""

    stt: list[ModelEntry] = Field(default_factory=list)
    tts: list[ModelEntry] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
