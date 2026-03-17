import base64

import orjson
from pytest_audioeval.client import AudioEval

##### SSE /v1/audio/speech (stream_format=sse) #####


async def test_speech_sse_yields_delta_events(
    audioeval: AudioEval,
) -> None:
    deltas: list[dict] = []

    async with audioeval.tts.sse(
        json={
            "input": "Hello world.",
            "voice": "af_heart",
            "stream": True,
            "stream_format": "sse",
        },
    ) as event_source:
        async for event in event_source:
            body = orjson.loads(event.data)
            deltas.append(body)

    assert len(deltas) >= 2

    delta_events = [d for d in deltas if d.get("type") == "speech.audio.delta"]
    assert len(delta_events) > 0


async def test_speech_sse_done_event(
    audioeval: AudioEval,
) -> None:
    events: list[dict] = []

    async with audioeval.tts.sse(
        json={
            "input": "Done event test.",
            "voice": "af_heart",
            "stream": True,
            "stream_format": "sse",
        },
    ) as event_source:
        async for event in event_source:
            events.append(orjson.loads(event.data))

    assert len(events) > 0
    last_event = events[-1]
    assert last_event["type"] == "speech.audio.done"


async def test_speech_sse_delta_audio_is_valid_base64(
    audioeval: AudioEval,
) -> None:
    async with audioeval.tts.sse(
        json={
            "input": "Base64 validation.",
            "voice": "af_heart",
            "stream": True,
            "stream_format": "sse",
        },
    ) as event_source:
        async for event in event_source:
            body = orjson.loads(event.data)
            if body.get("type") == "speech.audio.delta":
                audio_bytes = base64.b64decode(body["audio"])
                assert len(audio_bytes) > 0
                assert len(audio_bytes) % 2 == 0
                break


async def test_speech_sse_combined_audio_non_trivial(
    audioeval: AudioEval,
) -> None:
    total_audio_bytes = 0

    async with audioeval.tts.sse(
        json={
            "input": "The quick brown fox jumps over the lazy dog.",
            "voice": "af_heart",
            "stream": True,
            "stream_format": "sse",
        },
    ) as event_source:
        async for event in event_source:
            body = orjson.loads(event.data)
            if body.get("type") == "speech.audio.delta":
                audio_bytes = base64.b64decode(body["audio"])
                total_audio_bytes += len(audio_bytes)

    assert total_audio_bytes > 1000
