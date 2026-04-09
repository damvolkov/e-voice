"""Kokoro-ONNX adapter — implements TTSBackend with local ONNX model lifecycle."""

import asyncio
import gc
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import onnxruntime as ort
from kokoro_onnx import Kokoro

from e_voice.adapters.base import TTSBackend
from e_voice.core.logger import logger
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.tts import AudioChunk, OnnxProvider, SynthesisParams, TTSModelSpec, VoiceEntry, resolve_voice_lang

##### HELPERS #####


def _resolve_provider(device: str) -> OnnxProvider:
    """Resolve ONNX provider with fallback to CPU."""
    desired = OnnxProvider.CUDA if device in ("gpu", "cuda") else OnnxProvider.CPU

    if desired in ort.get_available_providers():  # ty: ignore[possibly-missing-attribute]
        return desired

    logger.warning("⚠️ PROVIDER_FALLBACK", extra={"desired": desired, "actual": OnnxProvider.CPU})
    return OnnxProvider.CPU


##### ADAPTER #####


class KokoroAdapter(TTSBackend):
    """Manages Kokoro-ONNX TTS model registry and synthesis."""

    __slots__ = ("_models", "_voice_entries", "_voices")

    def __init__(self) -> None:
        self._models: dict[TTSModelSpec, Kokoro] = {}
        self._voices: list[str] = []
        self._voice_entries: list[VoiceEntry] = []

    # ── Capabilities ──────────────────────────────────────────────────

    @property
    def supported_devices(self) -> frozenset[DeviceType]:
        return frozenset({DeviceType.CPU, DeviceType.GPU})

    # ── Properties ────────────────────────────────────────────────────

    @property
    def voices(self) -> list[str]:
        """Available voice IDs (cached after first model load)."""
        return self._voices

    @property
    def voice_entries(self) -> list[VoiceEntry]:
        """Voice metadata with resolved language codes."""
        return self._voice_entries

    # ── Model Lifecycle ───────────────────────────────────────────────

    async def load(self, spec: TTSModelSpec | None = None) -> None:
        """Download (if needed) and load Kokoro model. Idempotent."""
        target = spec or TTSModelSpec(device=st.tts.device)
        if target in self._models:
            return

        models_dir = st.MODELS_PATH / "tts" / "kokoro" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / st.tts.model_filename
        voices_path = models_dir / st.tts.voices_filename

        if not model_path.exists() or not voices_path.exists():
            await self._ka_download_files(model_path, voices_path)

        provider = _resolve_provider(target.device.value)
        os.environ["ONNX_PROVIDER"] = provider

        logger.info("🔄 MODEL_LOADING", extra={"model": target.model_id, "provider": provider.value})
        kokoro = await asyncio.to_thread(Kokoro, str(model_path), str(voices_path))
        self._models[target] = kokoro
        if not self._voices:
            self._voices = kokoro.get_voices()
            self._voice_entries = self._ka_build_voice_entries(self._voices)
        logger.info("✅ MODEL_LOADED", extra={"model": target.model_id, "voices": len(self._voices)})

    async def unload(self, spec: TTSModelSpec | None = None) -> bool:
        """Unload model and release resources."""
        target = spec or TTSModelSpec(device=st.tts.device)
        if (model := self._models.pop(target, None)) is not None:
            del model
            gc.collect()
            logger.info("🗑️ MODEL_UNLOADED", extra={"model": target.model_id, "device": target.device.value})
            return True
        return False

    async def is_loaded(self, spec: TTSModelSpec | None = None) -> bool:
        target = spec or TTSModelSpec(device=st.tts.device)
        return target in self._models

    def loaded_models(self) -> list[TTSModelSpec]:
        return list(self._models)

    async def download(self, model_id: str = "kokoro") -> Path:
        """Download Kokoro model files to disk. Returns model directory."""
        models_dir = st.MODELS_PATH / "tts" / "kokoro" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        await self._ka_download_files(models_dir / st.tts.model_filename, models_dir / st.tts.voices_filename)
        return models_dir

    # ── Batch Synthesis ───────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        *,
        params: SynthesisParams | None = None,
    ) -> AudioChunk:
        """Full synthesis — returns (samples, sample_rate)."""
        kokoro = self._ka_resolve()
        p = params or SynthesisParams()
        return await asyncio.to_thread(kokoro.create, text, p.voice, p.speed, p.lang)

    # ── Streaming Synthesis ───────────────────────────────────────────

    async def synthesize_stream(  # ty: ignore[invalid-method-override]
        self,
        text: str,
        *,
        params: SynthesisParams | None = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Yield audio chunks as generated — true streaming."""
        kokoro = self._ka_resolve()
        p = params or SynthesisParams()
        async for chunk in kokoro.create_stream(text, p.voice, p.speed, p.lang):
            yield chunk

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _ka_build_voice_entries(voice_ids: list[str]) -> list[VoiceEntry]:
        """Build VoiceEntry list with language resolved from Kokoro prefix convention."""
        entries: list[VoiceEntry] = []
        for vid in voice_ids:
            try:
                lang = resolve_voice_lang(vid)
            except ValueError:
                lang = "unknown"
            entries.append(VoiceEntry(id=vid, language=lang))
        return entries

    def _ka_resolve(self, spec: TTSModelSpec | None = None) -> Kokoro:
        """Resolve spec → loaded Kokoro instance."""
        target = spec or TTSModelSpec(device=st.tts.device)
        if target not in self._models:
            raise RuntimeError(f"Model {target!r} not loaded. Call load() first.")
        return self._models[target]

    @staticmethod
    async def _ka_download_files(model_path: Path, voices_path: Path) -> None:
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
