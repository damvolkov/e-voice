"""Kokoro-ONNX adapter — TTS model lifecycle and speech synthesis."""

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from numpy.typing import NDArray

from e_voice.adapters.base import BaseModelAdapter
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.models.tts import OnnxProvider, SynthesisParams, TTSModelSpec

type AudioChunk = tuple[NDArray[np.float32], int]


# ── Pure Helpers ───────────────────────────────────────────────────────


def _resolve_provider(device: str) -> OnnxProvider:
    """Resolve ONNX provider with fallback to CPU."""
    desired = OnnxProvider.CUDA if device == "cuda" else OnnxProvider.CPU

    if desired in ort.get_available_providers():  # ty: ignore[possibly-missing-attribute]
        return desired

    logger.warning("⚠️ PROVIDER_FALLBACK", extra={"desired": desired, "actual": OnnxProvider.CPU})
    return OnnxProvider.CPU


# ── Adapter ────────────────────────────────────────────────────────────


class KokoroAdapter(BaseModelAdapter[TTSModelSpec]):
    """Manages Kokoro-ONNX TTS model registry and synthesis."""

    __slots__ = ("_models", "_voices")

    def __init__(self) -> None:
        self._models: dict[TTSModelSpec, Kokoro] = {}
        self._voices: list[str] = []

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def voices(self) -> list[str]:
        """Available voice IDs (cached after first model load)."""
        return self._voices

    @property
    def loaded(self) -> list[TTSModelSpec]:
        """Currently loaded model specs."""
        return list(self._models)

    # ── Model Lifecycle ────────────────────────────────────────────────

    async def load(self, spec: TTSModelSpec | None = None) -> None:
        """Download (if needed) and load Kokoro model. Idempotent."""
        target = spec or TTSModelSpec(device=st.tts.device)
        if target in self._models:
            return

        model_dir = st.MODELS_PATH / "tts"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / st.tts.model_filename
        voices_path = model_dir / st.tts.voices_filename

        if not model_path.exists() or not voices_path.exists():
            await self._download(model_path, voices_path)

        provider = _resolve_provider(target.device.value)
        os.environ["ONNX_PROVIDER"] = provider

        logger.info("🔄 MODEL_LOADING", extra={"model": target.model_id, "provider": provider.value})
        kokoro = await asyncio.to_thread(Kokoro, str(model_path), str(voices_path))
        self._models[target] = kokoro
        if not self._voices:
            self._voices = kokoro.get_voices()
        logger.info("✅ MODEL_LOADED", extra={"model": target.model_id, "provider": provider.value})

    async def unload(self, spec: TTSModelSpec | None = None) -> bool:
        """Unload model from memory."""
        target = spec or TTSModelSpec(device=st.tts.device)
        if (model := self._models.pop(target, None)) is not None:
            del model
            logger.info("🗑️ MODEL_UNLOADED", extra={"model": target.model_id})
            return True
        return False

    async def is_loaded(self, spec: TTSModelSpec | None = None) -> bool:
        target = spec or TTSModelSpec(device=st.tts.device)
        return target in self._models

    def loaded_models(self) -> list[TTSModelSpec]:
        return list(self._models)

    async def download(self, model_id: str = "kokoro") -> Path:
        """Download Kokoro model files to disk. Returns model directory."""
        model_dir = st.MODELS_PATH / "tts"
        model_dir.mkdir(parents=True, exist_ok=True)
        await self._download(model_dir / st.tts.model_filename, model_dir / st.tts.voices_filename)
        return model_dir

    # ── Batch Synthesis ────────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        *,
        spec: TTSModelSpec | None = None,
        params: SynthesisParams | None = None,
    ) -> AudioChunk:
        """Full synthesis — returns (samples, sample_rate)."""
        kokoro = self._resolve(spec)
        p = params or SynthesisParams()
        return await asyncio.to_thread(kokoro.create, text, p.voice, p.speed, p.resolved_lang)

    # ── Streaming Synthesis ────────────────────────────────────────────

    async def synthesize_stream(
        self,
        text: str,
        *,
        spec: TTSModelSpec | None = None,
        params: SynthesisParams | None = None,
    ) -> AsyncGenerator[AudioChunk]:
        """Yield audio chunks as generated — true streaming."""
        kokoro = self._resolve(spec)
        p = params or SynthesisParams()
        async for chunk in kokoro.create_stream(text, p.voice, p.speed, p.resolved_lang):
            yield chunk

    # ── Private ────────────────────────────────────────────────────────

    def _resolve(self, spec: TTSModelSpec | None = None) -> Kokoro:
        """Resolve spec → loaded Kokoro instance."""
        target = spec or TTSModelSpec(device=st.tts.device)
        if target not in self._models:
            raise RuntimeError(f"Model {target!r} not loaded. Call load() first.")
        return self._models[target]

    @staticmethod
    async def _download(model_path: Path, voices_path: Path) -> None:
        """Download model + voices from GitHub releases."""
        base_url = st.tts.release_url
        chunk_size = st.tts.download_chunk_size
        logger.info("⬇️ FILES_DOWNLOADING", extra={"source": base_url})

        async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
            for filename, dest in ((st.tts.model_filename, model_path), (st.tts.voices_filename, voices_path)):
                if dest.exists():
                    continue
                url = f"{base_url}/{filename}"
                logger.info("⬇️ FILE_DOWNLOADING", extra={"file": filename})
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with dest.open("wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                            f.write(chunk)

        logger.info("✅ FILES_DOWNLOADED", extra={"path": str(model_path.parent)})
