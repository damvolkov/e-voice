"""Fixtures for API integration tests."""

import httpx
import pytest


@pytest.fixture(scope="session")
def stt_transcription_url(base_url: str) -> str:
    """Full URL for STT transcription endpoint."""
    return f"{base_url}/v1/audio/transcriptions"


@pytest.fixture(scope="session")
def stt_translation_url(base_url: str) -> str:
    """Full URL for STT translation endpoint."""
    return f"{base_url}/v1/audio/translations"


@pytest.fixture(scope="session")
def tts_speech_url(base_url: str) -> str:
    """Full URL for TTS speech endpoint."""
    return f"{base_url}/v1/audio/speech"


@pytest.fixture(scope="session")
def tts_voices_url(base_url: str) -> str:
    """Full URL for TTS voices listing endpoint."""
    return f"{base_url}/v1/audio/voices"


@pytest.fixture(scope="session")
def models_url(base_url: str) -> str:
    """Full URL for models listing endpoint."""
    return f"{base_url}/v1/models"


@pytest.fixture(scope="session")
def api_ps_url(base_url: str) -> str:
    """Full URL for model management endpoint."""
    return f"{base_url}/v1/api/ps"


@pytest.fixture(scope="session")
async def stt_client(base_url: str) -> httpx.AsyncClient:
    """Dedicated httpx client for STT multipart requests."""
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        yield client
