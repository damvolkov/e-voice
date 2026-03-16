"""STT SSE streaming tests — /v1/audio/transcriptions with stream=true.

Uses httpx-sse for SSE connection with multipart form-data.
Audio samples provided by pytest-audioeval fixtures.
"""

import httpx
import orjson
from httpx_sse import aconnect_sse
from pytest_audioeval.metrics.text import TextMetrics

##### SSE /v1/audio/transcriptions (stream=true) #####


async def test_transcription_sse_yields_events(
    stt_client: httpx.AsyncClient,
    en_sample,
) -> None:
    """SSE streaming transcription yields at least one event."""
    events: list[str] = []

    async with aconnect_sse(
        stt_client,
        "POST",
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")},
        data={"stream": "true", "response_format": "json"},
    ) as event_source:
        async for event in event_source.aiter_sse():
            events.append(event.data)

    assert len(events) > 0


async def test_transcription_sse_events_contain_text(
    stt_client: httpx.AsyncClient,
    en_sample,
) -> None:
    """Each SSE event contains parseable text content."""
    texts: list[str] = []

    async with aconnect_sse(
        stt_client,
        "POST",
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")},
        data={"stream": "true", "response_format": "json"},
    ) as event_source:
        async for event in event_source.aiter_sse():
            body = orjson.loads(event.data)
            assert "text" in body
            texts.append(body["text"])

    assert len(texts) > 0
    combined = " ".join(texts).strip()
    assert len(combined) > 0


async def test_transcription_sse_text_format(
    stt_client: httpx.AsyncClient,
    en_sample,
) -> None:
    """SSE streaming with text format yields raw text segments."""
    segments: list[str] = []

    async with aconnect_sse(
        stt_client,
        "POST",
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")},
        data={"stream": "true", "response_format": "text"},
    ) as event_source:
        async for event in event_source.aiter_sse():
            segments.append(event.data)

    assert len(segments) > 0


async def test_transcription_sse_quality(
    stt_client: httpx.AsyncClient,
    en_sample,
    audioeval_thresholds: dict[str, float],
) -> None:
    """Concatenated SSE segments meet quality thresholds."""
    texts: list[str] = []

    async with aconnect_sse(
        stt_client,
        "POST",
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")},
        data={"stream": "true", "response_format": "json"},
    ) as event_source:
        async for event in event_source.aiter_sse():
            body = orjson.loads(event.data)
            texts.append(body.get("text", ""))

    combined = " ".join(texts).strip()
    metrics = TextMetrics.compute(en_sample.reference_text, combined)
    metrics.assert_quality(
        max_wer=audioeval_thresholds["max_wer"],
        max_cer=audioeval_thresholds["max_cer"],
    )


async def test_translation_sse_yields_events(
    stt_client: httpx.AsyncClient,
    es_sample,
) -> None:
    """SSE streaming translation yields events with English text."""
    events: list[str] = []

    async with aconnect_sse(
        stt_client,
        "POST",
        "/v1/audio/translations",
        files={"file": ("audio.wav", es_sample.audio_bytes(), "audio/wav")},
        data={"stream": "true", "response_format": "json"},
    ) as event_source:
        async for event in event_source.aiter_sse():
            events.append(event.data)

    assert len(events) > 0
    body = orjson.loads(events[0])
    assert "text" in body
