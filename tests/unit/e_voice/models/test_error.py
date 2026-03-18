"""Unit tests for models/error.py — error response models and factory."""

import orjson
import pytest

from e_voice.models.error import (
    ErrorResponse,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    error_response,
)

##### OPENAI ERROR DETAIL #####


async def test_openai_error_detail_defaults() -> None:
    detail = OpenAIErrorDetail(message="something broke")
    assert detail.type == "invalid_request_error"
    assert detail.param is None
    assert detail.code is None


async def test_openai_error_detail_custom() -> None:
    detail = OpenAIErrorDetail(message="not found", type="not_found_error", code="404")
    assert detail.type == "not_found_error"
    assert detail.code == "404"


##### OPENAI ERROR RESPONSE #####


async def test_openai_error_response_schema() -> None:
    resp = OpenAIErrorResponse(error=OpenAIErrorDetail(message="bad request"))
    parsed = orjson.loads(resp.model_dump_json())
    assert "error" in parsed
    assert parsed["error"]["message"] == "bad request"
    assert parsed["error"]["type"] == "invalid_request_error"
    assert parsed["error"]["param"] is None
    assert parsed["error"]["code"] is None


##### ERROR RESPONSE (SIMPLE) #####


async def test_error_response_minimal() -> None:
    err = ErrorResponse(error="not found")
    assert err.detail is None


async def test_error_response_with_detail() -> None:
    err = ErrorResponse(error="download failed", detail="timeout")
    parsed = orjson.loads(err.model_dump_json())
    assert parsed["error"] == "download failed"
    assert parsed["detail"] == "timeout"


##### ERROR_RESPONSE FACTORY #####

_OPENAI_PATHS = [
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/audio/speech",
    "/v1/audio/voices",
    "/v1/models",
    "/v1/models/whisper-large-v3",
]

_TAXONOMY_PATHS = [
    "/v1/stt/http",
    "/v1/stt/sse",
    "/v1/stt/translate",
    "/v1/tts/http",
    "/v1/tts/sse",
    "/v1/tts/stream",
]


@pytest.mark.parametrize("path", _OPENAI_PATHS, ids=[p.split("/")[-1] for p in _OPENAI_PATHS])
async def test_error_response_openai_format(path: str) -> None:
    resp = error_response(path, 422, "No audio file provided")
    parsed = orjson.loads(resp.description)
    assert "error" in parsed
    assert isinstance(parsed["error"], dict)
    assert parsed["error"]["message"] == "No audio file provided"
    assert parsed["error"]["type"] == "invalid_request_error"
    assert resp.status_code == 422


@pytest.mark.parametrize("path", _TAXONOMY_PATHS, ids=[p.split("/")[-1] for p in _TAXONOMY_PATHS])
async def test_error_response_simple_format(path: str) -> None:
    resp = error_response(path, 422, "No audio file provided")
    parsed = orjson.loads(resp.description)
    assert "error" in parsed
    assert isinstance(parsed["error"], str)
    assert parsed["error"] == "No audio file provided"
    assert resp.status_code == 422


async def test_error_response_custom_error_type() -> None:
    resp = error_response("/v1/audio/transcriptions", 404, "Model not found", error_type="not_found_error")
    parsed = orjson.loads(resp.description)
    assert parsed["error"]["type"] == "not_found_error"


async def test_error_response_simple_with_detail() -> None:
    resp = error_response("/v1/stt/http", 500, "Internal error", detail="stack trace here")
    parsed = orjson.loads(resp.description)
    assert parsed["error"] == "Internal error"
    assert parsed["detail"] == "stack trace here"


async def test_error_response_content_type() -> None:
    resp = error_response("/v1/audio/transcriptions", 422, "test")
    assert resp.headers["content-type"] == "application/json"
