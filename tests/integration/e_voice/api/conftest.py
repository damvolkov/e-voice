from contextlib import suppress

import httpx
import pytest


@pytest.fixture(scope="session")
def stt_transcription_url(base_url: str) -> str:
    return f"{base_url}/v1/audio/transcriptions"


@pytest.fixture(scope="session")
def stt_translation_url(base_url: str) -> str:
    return f"{base_url}/v1/audio/translations"


@pytest.fixture(scope="session")
def tts_speech_url(base_url: str) -> str:
    return f"{base_url}/v1/audio/speech"


@pytest.fixture(scope="session")
def tts_voices_url(base_url: str) -> str:
    return f"{base_url}/v1/audio/voices"


@pytest.fixture(scope="session")
def models_url(base_url: str) -> str:
    return f"{base_url}/v1/models"


@pytest.fixture(scope="session")
def api_ps_url(base_url: str) -> str:
    return f"{base_url}/v1/api/ps"


@pytest.fixture(scope="session")
async def stt_client(base_url: str):
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
    yield client
    with suppress(RuntimeError):
        await client.aclose()
