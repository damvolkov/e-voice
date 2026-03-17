import multiprocessing
import os
import socket
import time

import httpx
import pytest
from pytest_audioeval.client import AudioEval

_HOST = "127.0.0.1"
_STARTUP_TIMEOUT = 120


def _find_free_port() -> int:
    """Bind to port 0, let the OS assign a free port, return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_server(port: int) -> None:
    """Start e-voice on the given port. Runs in a subprocess."""
    os.environ["GRADIO_ENABLED"] = "false"
    from e_voice.core.settings import settings as st

    st.system.port = port
    st.system.host = _HOST
    st.front.enabled = False

    from e_voice.main import main

    main()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


##### SERVER LIFECYCLE #####


@pytest.fixture(scope="session")
def e_voice_server():
    port = _find_free_port()
    process = multiprocessing.Process(target=_run_server, args=(port,), daemon=True)
    process.start()

    base_url = f"http://{_HOST}:{port}"

    for _ in range(_STARTUP_TIMEOUT):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError(f"e-voice failed to start on :{port} within {_STARTUP_TIMEOUT}s")

    yield {"base_url": base_url, "host": _HOST, "port": port}

    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()


##### DERIVED FIXTURES #####


@pytest.fixture(scope="session")
def base_url(e_voice_server: dict) -> str:
    return e_voice_server["base_url"]


@pytest.fixture
async def http_client(base_url: str) -> httpx.AsyncClient:
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
async def audioeval(e_voice_server: dict) -> AudioEval:
    host = e_voice_server["host"]
    port = e_voice_server["port"]
    client = AudioEval(
        stt_url=f"ws://{host}:{port}/v1/audio/transcriptions",
        tts_url=f"http://{host}:{port}/v1/audio/speech",
    )
    yield client
    await client.aclose()
