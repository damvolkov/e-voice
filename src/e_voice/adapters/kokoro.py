"""Kokoro-ONNX adapter — TTS model lifecycle and speech synthesis."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import numpy as np
from kokoro_onnx import Kokoro
from numpy.typing import NDArray

from e_voice.adapters.base import BaseModelAdapter
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st

KOKORO_SAMPLE_RATE = 24_000

_MODEL_FILENAME = "kokoro-v1.0.onnx"
_VOICES_FILENAME = "voices-v1.0.bin"
_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

##### LANGUAGE MAP #####

_VOICE_LANG_MAP: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}


def _resolve_lang(voice: str, lang: str | None) -> str:
    """Infer language from voice prefix if not explicitly provided."""
    if lang:
        return lang
    prefix = voice[0] if voice else "a"
    return _VOICE_LANG_MAP.get(prefix, "en-us")


class KokoroAdapter(BaseModelAdapter):
    """Manages Kokoro-ONNX TTS model lifecycle and synthesis."""

    __slots__ = ("_kokoro", "_model_dir")

    def __init__(self, model_dir: Path | None = None) -> None:
        self._kokoro: Kokoro | None = None
        self._model_dir = model_dir or st.MODELS_PATH / "tts"

    ##### MODEL LIFECYCLE #####

    async def load(self, model_id: str = "kokoro") -> None:
        """Download (if needed) and load Kokoro model."""
        if self._kokoro is not None:
            return

        self._model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_dir / _MODEL_FILENAME
        voices_path = self._model_dir / _VOICES_FILENAME

        if not model_path.exists() or not voices_path.exists():
            await self._lc_download_files(model_path, voices_path)

        logger.info("loading kokoro model", step="MODEL", path=str(self._model_dir))
        self._kokoro = await asyncio.to_thread(Kokoro, str(model_path), str(voices_path))
        logger.info("kokoro model loaded", step="MODEL")

    async def unload(self, model_id: str = "kokoro") -> bool:
        if self._kokoro is not None:
            self._kokoro = None
            logger.info("kokoro model unloaded", step="MODEL")
            return True
        return False

    async def is_loaded(self, model_id: str = "kokoro") -> bool:
        return self._kokoro is not None

    def loaded_models(self) -> list[str]:
        return ["kokoro"] if self._kokoro is not None else []

    async def download(self, model_id: str = "kokoro") -> Path:
        """Download Kokoro model files to disk. Returns model directory."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_dir / _MODEL_FILENAME
        voices_path = self._model_dir / _VOICES_FILENAME
        await self._lc_download_files(model_path, voices_path)
        return self._model_dir

    async def _lc_download_files(self, model_path: Path, voices_path: Path) -> None:
        """Download model + voices from GitHub releases."""
        logger.info("downloading kokoro files", step="DOWNLOAD", source=_RELEASE_BASE)

        async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
            for filename, dest in ((_MODEL_FILENAME, model_path), (_VOICES_FILENAME, voices_path)):
                if dest.exists():
                    continue
                url = f"{_RELEASE_BASE}/{filename}"
                logger.info("downloading", step="DOWNLOAD", file=filename)
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with dest.open("wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)

        logger.info("kokoro files downloaded", step="DOWNLOAD")

    def _lc_resolve(self) -> Kokoro:
        """Return loaded model or raise."""
        if self._kokoro is None:
            raise RuntimeError("Kokoro model not loaded. Call load() first.")
        return self._kokoro

    ##### SYNTHESIS #####

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str | None = None,
    ) -> tuple[NDArray[np.float32], int]:
        """Synthesize full audio. Returns (samples, sample_rate)."""
        kokoro = self._lc_resolve()
        resolved_lang = _resolve_lang(voice, lang)
        return await asyncio.to_thread(kokoro.create, text, voice, speed, resolved_lang)

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str | None = None,
    ) -> AsyncGenerator[tuple[NDArray[np.float32], int]]:
        """Yield audio chunks as they are generated."""
        kokoro = self._lc_resolve()
        resolved_lang = _resolve_lang(voice, lang)
        async for chunk in kokoro.create_stream(text, voice, speed, resolved_lang):
            yield chunk

    ##### VOICES #####

    def get_voices(self) -> list[str]:
        """Return available voice IDs."""
        kokoro = self._lc_resolve()
        return kokoro.get_voices()
