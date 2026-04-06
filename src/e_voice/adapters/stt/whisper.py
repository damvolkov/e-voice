"""Faster-whisper adapter — implements STTBackend with local model lifecycle."""

import asyncio
import functools
import gc
import threading
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from huggingface_hub import snapshot_download
from numpy.typing import NDArray

from e_voice.adapters.base import STTBackend
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.settings import ComputeType, DeviceType, STTConfig, VADConfig, resolve_compute_type
from e_voice.core.settings import settings as st
from e_voice.models.stt import InferenceParams, ModelSpec, Span, Transcript, Word

type Task = Literal["transcribe", "translate"]

_SENTINEL = object()


##### SEGMENT → DOMAIN CONVERSION #####


def _segment_to_span(seg: Segment) -> Span:
    """Convert faster-whisper Segment → domain Span. Library types stay inside the adapter."""
    words = tuple(
        Word(text=w.word.strip(), start=w.start, end=w.end, probability=w.probability) for w in (seg.words or ())
    )
    return Span(
        text=seg.text,
        start=seg.start,
        end=seg.end,
        words=words,
        no_speech_prob=seg.no_speech_prob,
    )


def _to_transcript(segments: list[Segment], info: TranscriptionInfo, audio: NDArray[np.float32]) -> Transcript:
    """Convert faster-whisper results → domain Transcript."""
    return Transcript(
        spans=tuple(_segment_to_span(seg) for seg in segments),
        language=info.language,
        duration=Audio.duration(audio),
    )


##### ADAPTER #####


class WhisperAdapter(STTBackend):
    """Manages faster-whisper model registry and inference."""

    __slots__ = ("_config", "_gpu_lock", "_models", "_vad_config")

    def __init__(self, config: STTConfig | None = None, vad_config: VADConfig | None = None) -> None:
        self._models: dict[ModelSpec, WhisperModel] = {}
        self._config = config or st.stt
        self._vad_config = vad_config or st.vad
        self._gpu_lock = threading.Lock()

    # ── Capabilities ──────────────────────────────────────────────────

    @property
    def supported_devices(self) -> frozenset[DeviceType]:
        return frozenset({DeviceType.CPU, DeviceType.GPU})

    # ── Model Lifecycle ───────────────────────────────────────────────

    async def load(self, spec: ModelSpec | None = None) -> None:
        """Load whisper model into memory (thread-offloaded). Idempotent."""
        target = spec or ModelSpec(
            model_id=st.stt.model,
            device=self._config.device.value,
            compute_type=self._config.compute_type.value,
        )
        if target in self._models:
            return
        logger.info("🔄 MODEL_LOADING", extra={"model": target.model_id, "device": target.device})
        self._models[target] = await asyncio.to_thread(self._wa_create_model, target)
        logger.info("✅ MODEL_LOADED", extra={"model": target.model_id, "device": target.device})

    async def unload(self, spec: ModelSpec | None = None) -> bool:
        """Unload model and release VRAM."""
        target = spec or ModelSpec(
            model_id=st.stt.model,
            device=self._config.device.value,
            compute_type=self._config.compute_type.value,
        )
        if (model := self._models.pop(target, None)) is not None:
            del model
            gc.collect()
            logger.info("🗑️ MODEL_UNLOADED", extra={"model": target.model_id, "device": target.device})
            return True
        return False

    async def is_loaded(self, spec: ModelSpec | None = None) -> bool:
        target = spec or ModelSpec(
            model_id=st.stt.model,
            device=self._config.device.value,
            compute_type=self._config.compute_type.value,
        )
        return target in self._models

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
                local_dir=str(st.MODELS_PATH / "stt" / st.stt.backend / model_id.replace("/", "--")),
                allow_patterns=st.stt.hf_allow_patterns,
            )
        )
        logger.info("✅ MODEL_DOWNLOADED", extra={"model": model_id, "path": path})
        return Path(str(path))

    # ── Batch Inference ───────────────────────────────────────────────

    async def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> Transcript:
        """Full transcription → domain Transcript."""
        p = params or InferenceParams()
        segments, info = await self._wa_run_batch(audio, task="transcribe", params=p)
        return _to_transcript(segments, info, audio)

    async def translate(
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> Transcript:
        """Full translation → domain Transcript."""
        p = params or InferenceParams()
        segments, info = await self._wa_run_batch(audio, task="translate", params=p)
        return _to_transcript(segments, info, audio)

    # ── Streaming Inference ───────────────────────────────────────────

    async def transcribe_stream(  # ty: ignore[invalid-method-override]
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[Span, None]:
        """Yield domain Spans as they are produced — true streaming."""
        async for seg in self._wa_run_stream(audio, task="transcribe", params=params or InferenceParams()):
            yield _segment_to_span(seg)

    async def translate_stream(  # ty: ignore[invalid-method-override]
        self,
        audio: NDArray[np.float32],
        *,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[Span, None]:
        """Yield translated Spans — true streaming."""
        async for seg in self._wa_run_stream(audio, task="translate", params=params or InferenceParams()):
            yield _segment_to_span(seg)

    # ── Private: Inference Core ───────────────────────────────────────

    async def _wa_run_batch(
        self,
        audio: NDArray[np.float32],
        *,
        task: Task,
        params: InferenceParams,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Materialize all segments in thread. O(n) segments, GPU-locked."""
        model = self._wa_resolve()

        def _produce() -> tuple[list[Segment], TranscriptionInfo]:
            with self._gpu_lock:
                gen, info = self._wa_invoke(model, audio, task, params)
                return list(gen), info

        return await asyncio.to_thread(_produce)

    async def _wa_run_stream(
        self,
        audio: NDArray[np.float32],
        *,
        task: Task,
        params: InferenceParams,
    ) -> AsyncGenerator[Segment]:
        """Queue-bridged streaming: thread produces → async generator consumes."""
        model = self._wa_resolve()
        queue: asyncio.Queue[Segment | object] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        errors: list[Exception] = []

        def _produce() -> None:
            try:
                with self._gpu_lock:
                    gen, _info = self._wa_invoke(model, audio, task, params)
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

    def _wa_invoke(
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

    # ── Private: Model Registry ───────────────────────────────────────

    def _wa_create_model(self, spec: ModelSpec) -> WhisperModel:
        """Instantiate WhisperModel (CPU-bound, runs in thread)."""
        device = DeviceType(spec.device)
        return WhisperModel(
            spec.model_id,
            device=device.runtime,
            device_index=self._config.device_index,
            compute_type=str(resolve_compute_type(device, ComputeType(spec.compute_type))),
            cpu_threads=self._config.cpu_threads,
            num_workers=self._config.num_workers,
            download_root=str(st.MODELS_PATH / "stt" / st.stt.backend),
        )

    def _wa_resolve(self) -> WhisperModel:
        """Resolve current config → loaded WhisperModel."""
        target = ModelSpec(
            model_id=self._config.model,
            device=self._config.device.value,
            compute_type=self._config.compute_type.value,
        )
        if target not in self._models:
            raise RuntimeError(f"Model {target!r} not loaded. Call load() first.")
        return self._models[target]
