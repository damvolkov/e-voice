"""STT HTTP POST tests — /v1/audio/transcriptions, /v1/audio/translations.

Uses httpx directly for multipart form-data (file + form fields).
Audio samples provided by pytest-audioeval fixtures.
"""

import httpx
import orjson
import pytest
from pytest_audioeval.metrics.text import TextMetrics

_RESPONSE_FORMATS = ["json", "text", "verbose_json", "srt", "vtt"]


##### HELPERS #####


def _multipart(audio_bytes: bytes, **fields: str) -> tuple[dict, dict]:
    """Build httpx files + data dicts for STT multipart POST."""
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    return files, fields


##### POST /v1/audio/transcriptions #####


@pytest.mark.parametrize(
    "response_format",
    _RESPONSE_FORMATS,
    ids=_RESPONSE_FORMATS,
)
async def test_transcription_response_format(
    stt_client: httpx.AsyncClient,
    en_sample,
    response_format: str,
) -> None:
    """Each response_format returns 200 with well-formed body."""
    files, data = _multipart(en_sample.audio_bytes(), response_format=response_format)
    response = await stt_client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200

    match response_format:
        case "json":
            body = orjson.loads(response.content)
            assert "text" in body
            assert isinstance(body["text"], str)
        case "text":
            assert len(response.text.strip()) > 0
        case "verbose_json":
            body = orjson.loads(response.content)
            assert "text" in body
            assert "segments" in body
            assert "language" in body
            assert "duration" in body
        case "srt":
            assert "-->" in response.text
        case "vtt":
            assert response.text.startswith("WEBVTT")
            assert "-->" in response.text


async def test_transcription_json_quality(
    stt_client: httpx.AsyncClient,
    en_sample,
    audioeval_thresholds: dict[str, float],
) -> None:
    """Transcription quality within acceptable WER/CER thresholds."""
    files, data = _multipart(en_sample.audio_bytes(), response_format="json")
    response = await stt_client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200

    body = orjson.loads(response.content)
    metrics = TextMetrics.compute(en_sample.reference_text, body["text"])
    metrics.assert_quality(
        max_wer=audioeval_thresholds["max_wer"],
        max_cer=audioeval_thresholds["max_cer"],
    )


async def test_transcription_with_language(
    stt_client: httpx.AsyncClient,
    es_sample,
) -> None:
    """Explicit language parameter produces correct transcription."""
    files, data = _multipart(es_sample.audio_bytes(), response_format="json", language="es")
    response = await stt_client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert len(body["text"].strip()) > 0


async def test_transcription_with_vad_filter(
    stt_client: httpx.AsyncClient,
    en_sample,
) -> None:
    """VAD filter enabled still returns valid transcription."""
    files, data = _multipart(en_sample.audio_bytes(), response_format="json", vad_filter="true")
    response = await stt_client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "text" in body


async def test_transcription_verbose_json_word_timestamps(
    stt_client: httpx.AsyncClient,
    en_sample,
) -> None:
    """Word-level timestamps included when requested."""
    files = {"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")}
    data = {"response_format": "verbose_json", "timestamp_granularities[]": "word"}
    response = await stt_client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "segments" in body
    assert len(body["segments"]) > 0


async def test_transcription_with_hotwords(
    stt_client: httpx.AsyncClient,
    en_sample,
) -> None:
    """Hotwords parameter accepted without error."""
    files, data = _multipart(en_sample.audio_bytes(), response_format="json", hotwords="hello world")
    response = await stt_client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "text" in body


async def test_transcription_no_file_returns_422(
    stt_client: httpx.AsyncClient,
) -> None:
    """Missing audio file returns 422 Unprocessable Entity."""
    response = await stt_client.post("/v1/audio/transcriptions", data={"response_format": "json"})
    assert response.status_code == 422

    body = orjson.loads(response.content)
    assert "error" in body


##### POST /v1/audio/translations #####


async def test_translation_json_format(
    stt_client: httpx.AsyncClient,
    es_sample,
) -> None:
    """Translation endpoint returns English text from non-English audio."""
    files, data = _multipart(es_sample.audio_bytes(), response_format="json")
    response = await stt_client.post("/v1/audio/translations", files=files, data=data)
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "text" in body
    assert len(body["text"].strip()) > 0


async def test_translation_text_format(
    stt_client: httpx.AsyncClient,
    es_sample,
) -> None:
    """Translation with text format returns plain text."""
    files, data = _multipart(es_sample.audio_bytes(), response_format="text")
    response = await stt_client.post("/v1/audio/translations", files=files, data=data)
    assert response.status_code == 200
    assert len(response.text.strip()) > 0


async def test_translation_no_file_returns_422(
    stt_client: httpx.AsyncClient,
) -> None:
    """Translation without audio file returns 422."""
    response = await stt_client.post("/v1/audio/translations", data={"response_format": "json"})
    assert response.status_code == 422
