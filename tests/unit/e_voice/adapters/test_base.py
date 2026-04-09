"""Tests for STTBackend/TTSBackend ABC default capabilities."""

from pathlib import Path

import pytest

from e_voice.adapters.base import STTBackend, TTSBackend
from e_voice.adapters.tts.kokoro import KokoroAdapter
from e_voice.models.error import BackendCapabilityError

##### TTS DEFAULTS #####


async def test_tts_default_supports_voice_clone_false() -> None:
    adapter = KokoroAdapter()
    assert adapter.supports_voice_clone is False


async def test_tts_default_clone_voice_raises() -> None:
    adapter = KokoroAdapter()
    with pytest.raises(BackendCapabilityError, match="does not support voice cloning"):
        await adapter.clone_voice("test", Path("/tmp/fake.wav"), "hello")


async def test_tts_default_supported_devices_empty() -> None:
    class MinimalTTS(TTSBackend):
        async def load(self, spec=None) -> None: ...
        async def unload(self, spec=None) -> bool:
            return False

        async def is_loaded(self, spec=None) -> bool:
            return False

        def loaded_models(self):
            return []

        async def synthesize(self, text, *, params=None): ...  # type: ignore[empty-body]
        async def synthesize_stream(self, text, *, params=None):  # type: ignore[empty-body]
            yield  # type: ignore[misc]

        @property
        def voices(self):
            return []

        @property
        def voice_entries(self):
            return []

    tts = MinimalTTS()
    assert tts.supported_devices == frozenset()
    with pytest.raises(BackendCapabilityError, match="does not support local download"):
        await tts.download("test")


##### STT DEFAULTS #####


async def test_stt_default_download_raises() -> None:
    class MinimalSTT(STTBackend):
        async def load(self, spec=None) -> None: ...
        async def unload(self, spec=None) -> bool:
            return False

        async def is_loaded(self, spec=None) -> bool:
            return False

        def loaded_models(self):
            return []

        async def transcribe(self, audio, *, params=None): ...  # type: ignore[empty-body]
        async def transcribe_stream(self, audio, *, params=None):  # type: ignore[empty-body]
            yield  # type: ignore[misc]

    stt = MinimalSTT()
    assert stt.supported_devices == frozenset()

    with pytest.raises(BackendCapabilityError, match="does not support local download"):
        await stt.download("test")

    with pytest.raises(BackendCapabilityError, match="does not support translation"):
        await stt.translate(None)  # type: ignore[arg-type]


async def test_stt_default_translate_stream_raises() -> None:
    class MinimalSTT(STTBackend):
        async def load(self, spec=None) -> None: ...
        async def unload(self, spec=None) -> bool:
            return False

        async def is_loaded(self, spec=None) -> bool:
            return False

        def loaded_models(self):
            return []

        async def transcribe(self, audio, *, params=None): ...  # type: ignore[empty-body]
        async def transcribe_stream(self, audio, *, params=None):  # type: ignore[empty-body]
            yield  # type: ignore[misc]

    stt = MinimalSTT()
    with pytest.raises(BackendCapabilityError, match="does not support streaming translation"):
        async for _ in stt.translate_stream(None):  # type: ignore[arg-type]
            pass
