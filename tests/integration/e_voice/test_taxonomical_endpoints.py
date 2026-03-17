"""Taxonomical endpoint aliases — verify /v1/stt/* and /v1/tts/* map to the same handlers.

Each alias must produce identical results to the OpenAI-compatible canonical endpoint.
Suffixes follow the transport taxonomy: _http, _sse, _stream, _ws.
"""

import orjson
from httpx_sse import aconnect_sse
from pytest_audioeval.client import AudioEval

##### STT — HTTP #####


async def test_stt_http_alias_returns_200(audioeval: AudioEval, en_sample) -> None:
    response = await audioeval.stt.post(
        url="/v1/stt/http",
        sample=en_sample,
        params={"response_format": "json"},
    )
    assert response.status_code == 200
    assert "text" in response.json()


async def test_stt_http_alias_matches_canonical(audioeval: AudioEval, en_sample) -> None:
    canonical = await audioeval.stt.post(sample=en_sample, params={"response_format": "json"})
    alias = await audioeval.stt.post(url="/v1/stt/http", sample=en_sample, params={"response_format": "json"})
    assert canonical.json()["text"] == alias.json()["text"]


##### STT — SSE #####


async def test_stt_sse_alias_streams_events(stt_client, en_sample) -> None:
    with en_sample.open("rb") as f:
        async with aconnect_sse(
            stt_client,
            "POST",
            "/v1/stt/sse",
            files={"file": ("audio.wav", f)},
            data={"stream": "true", "response_format": "text"},
        ) as event_source:
            events = [event async for event in event_source.aiter_sse()]
    assert len(events) >= 1


##### STT — WS #####


async def test_stt_ws_alias_returns_transcription(audioeval: AudioEval, en_sample) -> None:
    async with audioeval.stt.ws(url="/v1/stt/ws", sample=en_sample) as session:
        await session.send_sample(en_sample, chunk_ms=200)
        import asyncio

        await asyncio.sleep(3)
        result = session.result()
    assert len(result.hypothesis_text.strip()) > 0


##### TTS — HTTP #####


async def test_tts_http_alias_returns_audio(audioeval: AudioEval) -> None:
    response = await audioeval.tts.post(
        url="/v1/tts/http",
        json={"input": "Hello.", "voice": "af_heart", "response_format": "wav", "stream": False},
    )
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


async def test_tts_http_alias_matches_canonical(audioeval: AudioEval) -> None:
    payload = {"input": "Test.", "voice": "af_heart", "response_format": "wav", "stream": False}
    canonical = await audioeval.tts.post(json=payload)
    alias = await audioeval.tts.post(url="/v1/tts/http", json=payload)
    assert canonical.status_code == alias.status_code == 200
    assert abs(len(canonical.content) - len(alias.content)) < 100


##### TTS — SSE #####


async def test_tts_sse_alias_streams_delta_events(http_client) -> None:
    async with aconnect_sse(
        http_client,
        "POST",
        "/v1/tts/sse",
        json={"input": "Hello world.", "voice": "af_heart", "stream": True, "stream_format": "sse"},
    ) as event_source:
        events = []
        async for event in event_source.aiter_sse():
            events.append(orjson.loads(event.data))

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    done_events = [e for e in events if e.get("type") == "speech.audio.done"]
    assert len(delta_events) >= 1
    assert len(done_events) == 1


##### TTS — STREAM (CHUNKED) #####


async def test_tts_stream_alias_returns_chunked_audio(http_client) -> None:
    async with http_client.stream(
        "POST",
        "/v1/tts/stream",
        json={"input": "Hello world.", "voice": "af_heart", "stream": True, "response_format": "pcm"},
    ) as response:
        chunks = [chunk async for chunk in response.aiter_bytes()]

    total = b"".join(chunks)
    assert len(total) > 0
    assert len(total) % 2 == 0


##### TTS — WS #####


async def test_tts_ws_alias_returns_audio_events(audioeval: AudioEval) -> None:
    async with audioeval.tts.ws(url="/v1/tts/ws") as ws:
        await ws.send_text(orjson.dumps({"input": "Hello.", "voice": "af_heart"}).decode())

        events = []
        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
                body = orjson.loads(text)
                events.append(body)
                if body.get("type") == "speech.audio.done":
                    break
            except Exception:
                break

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert len(delta_events) >= 1
