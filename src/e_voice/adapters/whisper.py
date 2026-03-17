"""Faster-whisper adapter — model lifecycle, transcription, translation, formatting."""

import asyncio
import threading
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from huggingface_hub import snapshot_download
from numpy.typing import NDArray

from e_voice.adapters.base import BaseModelAdapter
from e_voice.core.helpers import audio_duration, format_timestamp, format_timestamp_vtt
from e_voice.core.logger import logger
from e_voice.core.settings import STTConfig, VADConfig, resolve_compute_type
from e_voice.core.settings import settings as st
from e_voice.models.transcription import (
    ResponseFormat,
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionVerboseResponse,
    TranscriptionWord,
)

_HF_ALLOW_PATTERNS = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"]


class WhisperAdapter(BaseModelAdapter):
    """Manages faster-whisper model lifecycle and inference."""

    __slots__ = ("_models", "_config", "_vad_config", "_gpu_lock")

    def __init__(self, config: STTConfig | None = None, vad_config: VADConfig | None = None) -> None:
        self._models: dict[str, WhisperModel] = {}
        self._config = config or st.stt
        self._vad_config = vad_config or st.vad
        self._gpu_lock = threading.Lock()

    ##### MODEL LIFECYCLE #####

    async def load(self, model_id: str) -> None:
        """Load a whisper model into memory (thread-offloaded)."""
        if model_id in self._models:
            return

        logger.info("loading whisper model", model=model_id, device=self._config.device.value, step="MODEL")
        model = await asyncio.to_thread(self._lc_create_model, model_id)
        self._models[model_id] = model
        logger.info("whisper model loaded", model=model_id, step="MODEL")

    async def unload(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if (model := self._models.pop(model_id, None)) is not None:
            del model
            logger.info("whisper model unloaded", model=model_id, step="MODEL")
            return True
        return False

    async def is_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    async def download(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub (thread-offloaded)."""
        logger.info("downloading model", model=model_id, step="DOWNLOAD")
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=model_id,
            repo_type="model",
            local_dir=str(st.MODELS_PATH / "stt" / model_id.replace("/", "--")),
            allow_patterns=_HF_ALLOW_PATTERNS,
        )
        logger.info("model downloaded", model=model_id, path=path, step="DOWNLOAD")
        return Path(path)

    def _lc_create_model(self, model_id: str) -> WhisperModel:
        """Instantiate WhisperModel (CPU-bound, runs in thread)."""
        return WhisperModel(
            model_id,
            device=self._config.device.value,
            device_index=self._config.device_index,
            compute_type=resolve_compute_type(self._config.device, self._config.compute_type).value,
            cpu_threads=self._config.cpu_threads,
            num_workers=self._config.num_workers,
            download_root=str(st.MODELS_PATH / "stt"),
        )

    def _lc_resolve(self, model_id: str | None) -> WhisperModel:
        """Resolve model_id to a loaded WhisperModel. Raises if not loaded."""
        mid = model_id or st.stt.model
        if mid not in self._models:
            raise RuntimeError(f"Model '{mid}' not loaded. Call load() first.")
        return self._models[mid]

    ##### TRANSCRIPTION #####

    async def transcribe(
        self,
        audio_data: NDArray[np.float32],
        *,
        model_id: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        hotwords: str | None = None,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Run transcription (thread-offloaded). Returns segments and info."""
        model = self._lc_resolve(model_id)
        return await asyncio.to_thread(
            self._tr_run,
            model,
            audio_data,
            "transcribe",
            language,
            prompt,
            temperature,
            word_timestamps,
            vad_filter,
            hotwords,
        )

    async def transcribe_stream(
        self,
        audio_data: NDArray[np.float32],
        *,
        model_id: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        vad_filter: bool = False,
        hotwords: str | None = None,
    ) -> AsyncGenerator[Segment]:
        """Yield segments one at a time for SSE streaming. GPU-locked and thread-offloaded."""
        model = self._lc_resolve(model_id)
        segments, _info = await asyncio.to_thread(
            self._tr_run,
            model,
            audio_data,
            "transcribe",
            language,
            prompt,
            temperature,
            False,
            vad_filter,
            hotwords,
        )
        for segment in segments:
            yield segment

    def _tr_run(
        self,
        model: WhisperModel,
        audio_data: NDArray[np.float32],
        task: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
        vad_filter: bool,
        hotwords: str | None,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Execute transcription and materialize segments (CPU-bound thread). Serialized via lock."""
        with self._gpu_lock:
            segments_gen, info = self._tr_invoke(
                model, audio_data, task, language, prompt, temperature, word_timestamps, vad_filter, hotwords
            )
            return list(segments_gen), info

    def _tr_invoke(
        self,
        model: WhisperModel,
        audio_data: NDArray[np.float32],
        task: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        word_timestamps: bool,
        vad_filter: bool,
        hotwords: str | None,
    ) -> tuple[object, TranscriptionInfo]:
        """Call model.transcribe with unified params. Returns (generator, info)."""
        use_vad = vad_filter or self._vad_config.enabled
        return model.transcribe(
            audio_data,
            task=task,
            language=language or st.stt.default_language,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=use_vad,
            vad_parameters=self._vad_config.to_dict() if use_vad else None,
            hotwords=hotwords,
            condition_on_previous_text=True,
        )

    ##### TRANSLATION #####

    async def translate(
        self,
        audio_data: NDArray[np.float32],
        *,
        model_id: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        vad_filter: bool = False,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Run translation to English (thread-offloaded)."""
        model = self._lc_resolve(model_id)
        return await asyncio.to_thread(
            self._tr_run,
            model,
            audio_data,
            "translate",
            None,
            prompt,
            temperature,
            False,
            vad_filter,
            None,
        )

    ##### RESPONSE FORMATTING #####

    @staticmethod
    def segments_to_text(segments: list[Segment]) -> str:
        """Join segment texts."""
        return "".join(seg.text for seg in segments).strip()

    @staticmethod
    def segment_to_model(seg: Segment, word_timestamps: bool = False) -> TranscriptionSegment:
        """Convert faster-whisper Segment to Pydantic model."""
        words = None
        if word_timestamps and seg.words:
            words = [
                TranscriptionWord(start=w.start, end=w.end, word=w.word, probability=w.probability) for w in seg.words
            ]

        return TranscriptionSegment(
            id=seg.id,
            seek=seg.seek,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            tokens=list(seg.tokens),
            temperature=seg.temperature,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
            words=words,
        )

    @staticmethod
    def build_response(
        segments: list[Segment],
        info: TranscriptionInfo,
        audio_data: NDArray[np.float32],
        response_format: ResponseFormat,
        word_timestamps: bool = False,
        task: str = "transcribe",
    ) -> tuple[str, str]:
        """Build formatted response. Returns (body, content_type)."""
        text = WhisperAdapter.segments_to_text(segments)

        match response_format:
            case ResponseFormat.TEXT:
                return text, "text/plain"

            case ResponseFormat.JSON:
                return TranscriptionResponse(text=text).model_dump_json(), "application/json"

            case ResponseFormat.VERBOSE_JSON:
                seg_models = [WhisperAdapter.segment_to_model(seg, word_timestamps) for seg in segments]
                all_words = [w for seg in seg_models if seg.words for w in seg.words] if word_timestamps else None

                resp = TranscriptionVerboseResponse(
                    task=task,
                    language=info.language,
                    duration=audio_duration(audio_data),
                    text=text,
                    segments=seg_models,
                    words=all_words,
                )
                return resp.model_dump_json(), "application/json"

            case ResponseFormat.SRT:
                return _format_srt(segments), "text/plain"

            case ResponseFormat.VTT:
                return _format_vtt(segments), "text/plain"

    @staticmethod
    def format_segment_for_streaming(
        segment: Segment,
        response_format: ResponseFormat,
        segment_index: int = 0,
    ) -> str:
        """Format a single segment for SSE streaming."""
        match response_format:
            case ResponseFormat.TEXT:
                return segment.text
            case ResponseFormat.JSON:
                return TranscriptionResponse(text=segment.text).model_dump_json()
            case ResponseFormat.VERBOSE_JSON:
                return WhisperAdapter.segment_to_model(segment).model_dump_json()
            case ResponseFormat.SRT:
                return _format_srt_segment(segment, segment_index)
            case ResponseFormat.VTT:
                return _format_vtt_segment(segment)


##### SUBTITLE FORMATTING #####


def _format_srt(segments: list[Segment]) -> str:
    return "\n".join(_format_srt_segment(seg, i) for i, seg in enumerate(segments))


def _format_srt_segment(segment: Segment, index: int) -> str:
    start = format_timestamp(segment.start)
    end = format_timestamp(segment.end)
    return f"{index + 1}\n{start} --> {end}\n{segment.text.strip()}\n"


def _format_vtt(segments: list[Segment]) -> str:
    header = "WEBVTT\n\n"
    return header + "\n".join(_format_vtt_segment(seg) for seg in segments)


def _format_vtt_segment(segment: Segment) -> str:
    start = format_timestamp_vtt(segment.start)
    end = format_timestamp_vtt(segment.end)
    return f"{start} --> {end}\n{segment.text.strip()}\n"
