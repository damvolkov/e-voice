"""Unit tests for models/system.py — system API DTOs."""

from e_voice.models.system import (
    DownloadRequest,
    DownloadResponse,
    ErrorResponse,
    ModelEntry,
    ModelsListResponse,
    ServiceType,
)

##### SERVICE TYPE #####


async def test_service_type_values() -> None:
    assert ServiceType.STT == "stt"
    assert ServiceType.TTS == "tts"


##### DOWNLOAD REQUEST #####


async def test_download_request_stt() -> None:
    req = DownloadRequest(model="Systran/faster-whisper-large-v3", service=ServiceType.STT)
    assert req.model == "Systran/faster-whisper-large-v3"
    assert req.service == ServiceType.STT


async def test_download_request_tts() -> None:
    req = DownloadRequest(model="kokoro", service=ServiceType.TTS)
    assert req.service == ServiceType.TTS


##### DOWNLOAD RESPONSE #####


async def test_download_response_serializes() -> None:
    resp = DownloadResponse(status="downloaded", service="stt", model="test-model", path="/tmp/model")
    data = resp.model_dump()
    assert data["status"] == "downloaded"
    assert data["path"] == "/tmp/model"


##### MODEL ENTRY #####


async def test_model_entry_defaults() -> None:
    entry = ModelEntry(id="kokoro", service="tts", path="/models/tts")
    assert entry.size_mb == 0.0


async def test_model_entry_with_size() -> None:
    entry = ModelEntry(id="whisper-large", service="stt", path="/models/stt", size_mb=3072.5)
    assert entry.size_mb == 3072.5


##### MODELS LIST RESPONSE #####


async def test_models_list_response_empty() -> None:
    resp = ModelsListResponse()
    assert resp.stt == []
    assert resp.tts == []


async def test_models_list_response_with_entries() -> None:
    stt = [ModelEntry(id="whisper", service="stt", path="/m/stt")]
    tts = [ModelEntry(id="kokoro", service="tts", path="/m/tts")]
    resp = ModelsListResponse(stt=stt, tts=tts)
    assert len(resp.stt) == 1
    assert len(resp.tts) == 1


##### ERROR RESPONSE #####


async def test_error_response_minimal() -> None:
    err = ErrorResponse(error="not found")
    assert err.detail is None


async def test_error_response_with_detail() -> None:
    err = ErrorResponse(error="download failed", detail="timeout")
    assert err.detail == "timeout"
    assert "download failed" in err.model_dump_json()
