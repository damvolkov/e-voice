"""Base adapter contracts — STTBackend and TTSBackend ABCs.

All consumers (API, WebSocket, controller, streaming) depend ONLY on these ABCs.
Library-specific types (faster_whisper.Segment, kokoro_onnx.Kokoro) NEVER leak
beyond the adapter boundary — adapters convert to domain types internally.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from e_voice.core.settings import DeviceType
from e_voice.models.error import BackendCapabilityError
from e_voice.models.stt import InferenceParams, ModelSpec, Span, Transcript
from e_voice.models.tts import AudioChunk, SynthesisParams, TTSModelSpec, VoiceEntry

##### STT BACKEND #####


class STTBackend(ABC):
    """Contract for any Speech-to-Text backend.

    Abstract methods MUST be implemented. Concrete defaults raise BackendCapabilityError
    for optional capabilities — override to enable.
    """

    # ── Lifecycle (abstract — MUST implement) ──

    @abstractmethod
    async def load(self, spec: ModelSpec | None = None) -> None: ...

    @abstractmethod
    async def unload(self, spec: ModelSpec | None = None) -> bool: ...

    @abstractmethod
    async def is_loaded(self, spec: ModelSpec | None = None) -> bool: ...

    @abstractmethod
    def loaded_models(self) -> list[ModelSpec]: ...

    # ── Inference (abstract — MUST implement) ──

    @abstractmethod
    async def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> Transcript: ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[Span, None]: ...

    # ── Capabilities (concrete default → BackendCapabilityError) ──

    @property
    def supported_devices(self) -> frozenset[DeviceType]:
        """Devices this backend supports. Empty = no device concept (cloud API)."""
        return frozenset()

    async def download(self, model_id: str) -> Path:
        raise BackendCapabilityError(f"{type(self).__name__} does not support local download")

    async def translate(
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> Transcript:
        raise BackendCapabilityError(f"{type(self).__name__} does not support translation")

    async def translate_stream(
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[Span, None]:
        raise BackendCapabilityError(f"{type(self).__name__} does not support streaming translation")
        yield  # unreachable — makes this a valid async generator  # noqa: RUF027


##### TTS BACKEND #####


class TTSBackend(ABC):
    """Contract for any Text-to-Speech backend.

    Abstract methods MUST be implemented. Concrete defaults raise BackendCapabilityError
    for optional capabilities — override to enable.
    """

    # ── Lifecycle (abstract — MUST implement) ──

    @abstractmethod
    async def load(self, spec: TTSModelSpec | None = None) -> None: ...

    @abstractmethod
    async def unload(self, spec: TTSModelSpec | None = None) -> bool: ...

    @abstractmethod
    async def is_loaded(self, spec: TTSModelSpec | None = None) -> bool: ...

    @abstractmethod
    def loaded_models(self) -> list[TTSModelSpec]: ...

    # ── Inference (abstract — MUST implement) ──

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        params: SynthesisParams | None = None,
    ) -> AudioChunk: ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        *,
        params: SynthesisParams | None = None,
    ) -> AsyncGenerator[AudioChunk, None]: ...

    @property
    @abstractmethod
    def voices(self) -> list[str]: ...

    @property
    @abstractmethod
    def voice_entries(self) -> list[VoiceEntry]: ...

    # ── Capabilities (concrete default → BackendCapabilityError) ──

    @property
    def supported_devices(self) -> frozenset[DeviceType]:
        return frozenset()

    @property
    def supports_voice_clone(self) -> bool:
        return False

    async def download(self, model_id: str) -> Path:
        raise BackendCapabilityError(f"{type(self).__name__} does not support local download")

    async def clone_voice(
        self,
        voice_id: str,
        ref_audio: Path,
        ref_text: str,
        *,
        language: str | None = None,
    ) -> str:
        """Clone a voice from reference audio. Returns the voice_id for synthesis."""
        raise BackendCapabilityError(f"{type(self).__name__} does not support voice cloning")
