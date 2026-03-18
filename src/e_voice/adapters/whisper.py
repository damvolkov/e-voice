"""Faster-whisper adapter — model lifecycle, transcription, translation, streaming."""

import asyncio
import functools
import threading
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from ovld import ovld

from e_voice.adapters.base import BaseModelAdapter
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.settings import ComputeType, DeviceType, STTConfig, VADConfig, resolve_compute_type
from e_voice.core.settings import settings as st
from e_voice.models.stt import InferenceParams, ModelSpec
from e_voice.models.transcription import (
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionVerboseResponse,
    TranscriptionWord,
)

type Task = Literal["transcribe", "translate"]

_SENTINEL = object()


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def format_segment(seg: Segment, fmt: Literal["text"], index: int = 0) -> str:
    """Plain text — just the text."""
    return seg.text


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def format_segment(seg: Segment, fmt: Literal["json"], index: int = 0) -> str:
    """JSON — minimal envelope."""
    return TranscriptionResponse(text=seg.text).model_dump_json()


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def format_segment(seg: Segment, fmt: Literal["verbose_json"], index: int = 0) -> str:
    """Verbose JSON — full segment model."""
    return segment_to_model(seg).model_dump_json()


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def format_segment(seg: Segment, fmt: Literal["srt"], index: int = 0) -> str:
    """SRT subtitle block."""
    start, end = Audio.format_timestamp(seg.start), Audio.format_timestamp(seg.end)
    return f"{index + 1}\n{start} --> {end}\n{seg.text.strip()}\n"


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def format_segment(seg: Segment, fmt: Literal["vtt"], index: int = 0) -> str:
    """VTT subtitle block."""
    start, end = Audio.format_timestamp_vtt(seg.start), Audio.format_timestamp_vtt(seg.end)
    return f"{start} --> {end}\n{seg.text.strip()}\n"


# ── ovld: Batch Response Dispatch ──────────────────────────────────────


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def build_response(
    segments: list[Segment],
    info: TranscriptionInfo,
    audio: np.ndarray,
    fmt: Literal["text"],
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> tuple[str, str]:
    return "".join(s.text for s in segments).strip(), "text/plain"


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def build_response(
    segments: list[Segment],
    info: TranscriptionInfo,
    audio: np.ndarray,
    fmt: Literal["json"],
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> tuple[str, str]:
    text = "".join(s.text for s in segments).strip()
    return TranscriptionResponse(text=text).model_dump_json(), "application/json"


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def build_response(
    segments: list[Segment],
    info: TranscriptionInfo,
    audio: np.ndarray,
    fmt: Literal["verbose_json"],
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> tuple[str, str]:
    text = "".join(s.text for s in segments).strip()
    seg_models = [segment_to_model(s, word_timestamps) for s in segments]
    all_words = [w for s in seg_models if s.words for w in s.words] if word_timestamps else None
    resp = TranscriptionVerboseResponse(
        task=task,  # ty: ignore[invalid-argument-type]
        language=info.language,
        duration=Audio.duration(audio),
        text=text,
        segments=seg_models,
        words=all_words,
    )
    return resp.model_dump_json(), "application/json"


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def build_response(
    segments: list[Segment],
    info: TranscriptionInfo,
    audio: np.ndarray,
    fmt: Literal["srt"],
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> tuple[str, str]:
    body = "\n".join(format_segment(s, "srt", i) for i, s in enumerate(segments))
    return body, "text/plain"


@ovld  # ty: ignore[invalid-overload,useless-overload-body]
def build_response(
    segments: list[Segment],
    info: TranscriptionInfo,
    audio: np.ndarray,
    fmt: Literal["vtt"],
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> tuple[str, str]:
    body = "WEBVTT\n\n" + "\n".join(format_segment(s, "vtt") for s in segments)
    return body, "text/plain"


# ── Pure Helpers ───────────────────────────────────────────────────────


def segment_to_model(seg: Segment, word_timestamps: bool = False) -> TranscriptionSegment:
    """Convert faster-whisper Segment → Pydantic model."""
    words = (
        [TranscriptionWord(start=w.start, end=w.end, word=w.word, probability=w.probability) for w in seg.words]
        if word_timestamps and seg.words
        else None
    )
    return TranscriptionSegment(
        id=seg.id,
        seek=seg.seek,
        start=seg.start,
        end=seg.end,
        text=seg.text,
        tokens=list(seg.tokens),
        temperature=seg.temperature or 0.0,
        avg_logprob=seg.avg_logprob,
        compression_ratio=seg.compression_ratio,
        no_speech_prob=seg.no_speech_prob,
        words=words,
    )


# ── Adapter ────────────────────────────────────────────────────────────


class WhisperAdapter(BaseModelAdapter):
    """Manages faster-whisper model registry and inference.

    Lifecycle:
        1. Lifespan creates adapter, calls `await adapter.load(spec)` per desired model+device.
        2. Stores adapter in `app.state.whisper`.
        3. Routers resolve via `adapter.transcribe(audio, spec=spec, params=params)`.
        4. Shutdown calls `await adapter.unload(spec)` or just lets GC handle it.
    """

    __slots__ = ("_config", "_gpu_lock", "_models", "_vad_config")

    def __init__(self, config: STTConfig | None = None, vad_config: VADConfig | None = None) -> None:
        self._models: dict[ModelSpec, WhisperModel] = {}
        self._config = config or st.stt
        self._vad_config = vad_config or st.vad
        self._gpu_lock = threading.Lock()

    # ── Model Lifecycle ────────────────────────────────────────────────

    async def load(self, spec: ModelSpec) -> None:
        """Load whisper model into memory (thread-offloaded). Idempotent."""
        if spec in self._models:
            return
        logger.info("🔄 MODEL_LOADING", extra={"model": spec.model_id, "device": spec.device})
        self._models[spec] = await asyncio.to_thread(self._create_model, spec)
        logger.info("✅ MODEL_LOADED", extra={"model": spec.model_id, "device": spec.device})

    async def unload(self, spec: ModelSpec) -> bool:
        """Unload model from memory."""
        if (model := self._models.pop(spec, None)) is not None:
            del model
            logger.info("🗑️ MODEL_UNLOADED", extra={"model": spec.model_id})
            return True
        return False

    async def is_loaded(self, spec: ModelSpec) -> bool:
        return spec in self._models

    def loaded_models(self) -> list[ModelSpec]:
        return list(self._models)

    async def download(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub (thread-offloaded)."""
        logger.info("⬇️ MODEL_DOWNLOADING", extra={"model": model_id})
        path = await asyncio.to_thread(
            functools.partial(
                snapshot_download,
                repo_id=model_id,
                repo_type="model",
                local_dir=str(st.MODELS_PATH / "stt" / model_id.replace("/", "--")),
                allow_patterns=st.stt.hf_allow_patterns,
            )
        )
        logger.info("✅ MODEL_DOWNLOADED", extra={"model": model_id, "path": path})
        return Path(str(path))

    # ── Batch Inference ────────────────────────────────────────────────

    async def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        spec: ModelSpec | None = None,
        params: InferenceParams | None = None,
        response_format: str | None = None,
    ) -> tuple[list[Segment], TranscriptionInfo] | tuple[str, str]:
        """Full transcription. Without response_format → raw (segments, info). With → (body, content_type)."""
        p = params or InferenceParams()
        segments, info = await self._run_batch(audio, task="transcribe", spec=spec, params=p)
        if response_format is None:
            return segments, info
        return build_response(segments, info, audio, response_format, p.word_timestamps)  # ty: ignore[no-matching-overload]

    async def translate(
        self,
        audio: NDArray[np.float32],
        *,
        spec: ModelSpec | None = None,
        params: InferenceParams | None = None,
        response_format: str | None = None,
    ) -> tuple[list[Segment], TranscriptionInfo] | tuple[str, str]:
        """Full translation. Without response_format → raw (segments, info). With → (body, content_type)."""
        p = params or InferenceParams()
        segments, info = await self._run_batch(audio, task="translate", spec=spec, params=p)
        if response_format is None:
            return segments, info
        return build_response(segments, info, audio, response_format, p.word_timestamps, "translate")  # ty: ignore[no-matching-overload]

    # ── Streaming Inference ────────────────────────────────────────────

    async def transcribe_stream(
        self,
        audio: NDArray[np.float32],
        *,
        spec: ModelSpec | None = None,
        params: InferenceParams | None = None,
        response_format: str = "text",
    ) -> AsyncGenerator[str]:
        """Yield formatted segments as decoded — true streaming for SSE."""
        idx = 0
        async for seg in self._run_stream(audio, task="transcribe", spec=spec, params=params or InferenceParams()):
            yield format_segment(seg, response_format, idx)  # ty: ignore[no-matching-overload]
            idx += 1

    async def translate_stream(
        self,
        audio: NDArray[np.float32],
        *,
        spec: ModelSpec | None = None,
        params: InferenceParams | None = None,
        response_format: str = "text",
    ) -> AsyncGenerator[str]:
        """Yield formatted translated segments as decoded — true streaming for SSE."""
        idx = 0
        async for seg in self._run_stream(audio, task="translate", spec=spec, params=params or InferenceParams()):
            yield format_segment(seg, response_format, idx)  # ty: ignore[no-matching-overload]
            idx += 1

    # ── Private: Inference Core ────────────────────────────────────────

    async def _run_batch(
        self,
        audio: NDArray[np.float32],
        *,
        task: Task,
        spec: ModelSpec | None,
        params: InferenceParams,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Materialize all segments in thread. O(n) segments, GPU-locked."""
        model = self._resolve(spec)

        def _produce() -> tuple[list[Segment], TranscriptionInfo]:
            with self._gpu_lock:
                gen, info = self._invoke(model, audio, task, params)
                return list(gen), info

        return await asyncio.to_thread(_produce)

    async def _run_stream(
        self,
        audio: NDArray[np.float32],
        *,
        task: Task,
        spec: ModelSpec | None,
        params: InferenceParams,
    ) -> AsyncGenerator[Segment]:
        """Queue-bridged streaming: thread produces → async generator consumes."""
        model = self._resolve(spec)
        queue: asyncio.Queue[Segment | object] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        errors: list[Exception] = []

        def _produce() -> None:
            try:
                with self._gpu_lock:
                    gen, _info = self._invoke(model, audio, task, params)
                    for seg in gen:
                        loop.call_soon_threadsafe(queue.put_nowait, seg)
            except Exception as exc:
                errors.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        producer = loop.run_in_executor(None, _produce)

        while (seg := await queue.get()) is not _SENTINEL:
            yield seg  # type: ignore[misc]

        await producer
        if errors:
            raise errors[0]

    def _invoke(
        self,
        model: WhisperModel,
        audio: NDArray[np.float32],
        task: Task,
        params: InferenceParams,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        """Call model.transcribe. Caller MUST hold _gpu_lock."""
        use_vad = params.vad_filter or self._vad_config.enabled
        return model.transcribe(
            audio,
            task=task,
            language=params.language or st.stt.default_language,
            initial_prompt=params.prompt,
            temperature=params.temperature,
            word_timestamps=params.word_timestamps,
            vad_filter=use_vad,
            vad_parameters=self._vad_config.to_dict() if use_vad else None,
            hotwords=params.hotwords,
            condition_on_previous_text=True,
        )

    # ── Private: Model Registry ────────────────────────────────────────

    def _create_model(self, spec: ModelSpec) -> WhisperModel:
        """Instantiate WhisperModel (CPU-bound, runs in thread)."""
        return WhisperModel(
            spec.model_id,
            device=spec.device,
            device_index=self._config.device_index,
            compute_type=str(resolve_compute_type(DeviceType(spec.device), ComputeType(spec.compute_type))),
            cpu_threads=self._config.cpu_threads,
            num_workers=self._config.num_workers,
            download_root=str(st.MODELS_PATH / "stt"),
        )

    def _resolve(self, spec: ModelSpec | None) -> WhisperModel:
        """Resolve spec → loaded WhisperModel. Falls back to default spec from config."""
        target = spec or ModelSpec(
            model_id=st.stt.model,
            device=self._config.device.value,
            compute_type=self._config.compute_type.value,
        )
        if target not in self._models:
            raise RuntimeError(f"Model {target!r} not loaded. Call load() first.")
        return self._models[target]
