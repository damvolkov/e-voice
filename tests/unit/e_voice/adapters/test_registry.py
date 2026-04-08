"""Tests for backend registry and available_backends."""

from e_voice.adapters.registry import STT_BACKENDS, TTS_BACKENDS, available_backends
from e_voice.core.settings import settings as st


async def test_stt_backends_has_whisper() -> None:
    assert "whisper" in STT_BACKENDS


async def test_tts_backends_has_kokoro() -> None:
    assert "kokoro" in TTS_BACKENDS


async def test_available_backends_structure() -> None:
    result = available_backends()
    assert "stt" in result
    assert "tts" in result
    assert "active" in result["stt"]
    assert "available" in result["stt"]
    assert "active" in result["tts"]
    assert "available" in result["tts"]


async def test_available_backends_active_matches_config() -> None:
    result = available_backends()
    assert result["stt"]["active"] == st.stt.backend
    assert result["tts"]["active"] == st.tts.backend


async def test_available_backends_lists_are_sorted() -> None:
    result = available_backends()
    assert result["stt"]["available"] == sorted(result["stt"]["available"])
    assert result["tts"]["available"] == sorted(result["tts"]["available"])
