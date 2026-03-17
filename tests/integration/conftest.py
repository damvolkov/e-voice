"""Session-scoped fixtures for integration tests — auto-starts e-voice server."""

import multiprocessing
import os
import time

import httpx
import pytest
from pytest_audioeval.client import AudioEval

_TEST_PORT = 5599
_TEST_HOST = "127.0.0.1"
_STARTUP_TIMEOUT = 120


def _run_server() -> None:
    """Start e-voice in a subprocess. Imports here to isolate from test process."""
    os.environ["API_HOST"] = _TEST_HOST
    os.environ["API_PORT"] = str(_TEST_PORT)
    os.environ["DEBUG"] = "True"
    os.environ["ENVIRONMENT"] = "DEV"
    from e_voice.main import main

    main()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark all integration tests as slow."""
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


##### SERVER LIFECYCLE #####


@pytest.fixture(scope="session")
def e_voice_server() -> str:
    """Start e-voice server in a subprocess. Yields base URL. Kills on teardown."""
    process = multiprocessing.Process(target=_run_server, daemon=True)
    process.start()

    base_url = f"http://{_TEST_HOST}:{_TEST_PORT}"

    for _ in range(_STARTUP_TIMEOUT):
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError(f"e-voice server failed to start within {_STARTUP_TIMEOUT}s")

    yield base_url

    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()


##### DERIVED FIXTURES #####


@pytest.fixture(scope="session")
def base_url(e_voice_server: str) -> str:
    """Base URL for the running e-voice service."""
    return e_voice_server


@pytest.fixture(scope="session")
async def http_client(base_url: str) -> httpx.AsyncClient:
    """Shared async HTTP client for REST endpoints."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
async def audioeval(e_voice_server: str) -> AudioEval:
    """AudioEval wired to the auto-started server."""
    ws_url = f"ws://{_TEST_HOST}:{_TEST_PORT}/v1/audio/transcriptions"
    tts_url = f"http://{_TEST_HOST}:{_TEST_PORT}/v1/audio/speech"
    client = AudioEval(stt_url=ws_url, tts_url=tts_url)
    yield client
    await client.aclose()
