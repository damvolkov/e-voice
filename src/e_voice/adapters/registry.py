"""Backend registry — maps config backend names to adapter classes."""

from e_voice.adapters.base import STTBackend, TTSBackend
from e_voice.adapters.stt.whisper import WhisperAdapter
from e_voice.adapters.tts.kokoro import KokoroAdapter
from e_voice.core.settings import settings as st

STT_BACKENDS: dict[str, type[STTBackend]] = {
    "whisper": WhisperAdapter,
}

TTS_BACKENDS: dict[str, type[TTSBackend]] = {
    "kokoro": KokoroAdapter,
}


def available_backends() -> dict[str, dict[str, str | list[str]]]:
    """Return active + available backends per service for API/UI discovery."""
    return {
        "stt": {
            "active": st.stt.backend,
            "available": sorted(STT_BACKENDS),
        },
        "tts": {
            "active": st.tts.backend,
            "available": sorted(TTS_BACKENDS),
        },
    }
