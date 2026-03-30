import pytest
from pytest_audioeval.client import AudioEval
from pytest_audioeval.stt import STTClient
from pytest_audioeval.tts import TTSClient


@pytest.fixture
def audioeval_thresholds() -> dict[str, float]:
    """Relaxed thresholds for integration tests (Whisper turbo + short samples)."""
    return {"max_wer": 0.5, "max_cer": 0.3, "min_mos": 3.0}


@pytest.fixture(scope="session")
def stt(audioeval: AudioEval) -> STTClient:
    assert audioeval.stt is not None
    return audioeval.stt


@pytest.fixture(scope="session")
def tts(audioeval: AudioEval) -> TTSClient:
    """HTTP TTS client — for POST, SSE, streaming tests."""
    assert audioeval.tts is not None
    return audioeval.tts


@pytest.fixture(scope="session")
def tts_ws(e_voice_server: dict) -> TTSClient:
    """WebSocket TTS client — for WS tests (points to standalone WS server)."""
    host = e_voice_server["host"]
    ws_port = e_voice_server["ws_port"]
    return TTSClient(url=f"ws://{host}:{ws_port}/v1/audio/speech")


@pytest.fixture(scope="session")
def en_sample(audioeval: AudioEval):
    return audioeval.samples.en_hello_world


@pytest.fixture(scope="session")
def es_sample(audioeval: AudioEval):
    return audioeval.samples.es_hola_mundo


@pytest.fixture(scope="session")
def es_long_sample(audioeval: AudioEval):
    return audioeval.samples.es_pangrama


@pytest.fixture(scope="session")
def en_counting_sample(audioeval: AudioEval):
    return audioeval.samples.en_counting
