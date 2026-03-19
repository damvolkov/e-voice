"""Taxonomical endpoint aliases — verify /v1/stt/* and /v1/tts/* map to the same handlers.

Each alias must produce identical results to the OpenAI-compatible canonical endpoint.
Suffixes follow the transport taxonomy: _http, _sse, _stream, _ws.
"""

import asyncio

import httpx
import orjson
from httpx_sse import aconnect_sse
from pytest_audioeval.client import AudioEval
from pytest_audioeval.stt import AudioEncoding

##### STT — HTTP #####


async def test_stt_http_alias_returns_200(http_client: httpx.AsyncClient, en_sample) -> None:
    files = {"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")}
    response = await http_client.post("/v1/stt/http", files=files, data={"response_format": "json"})
    assert response.status_code == 200
    assert "text" in orjson.loads(response.content)


async def test_stt_http_alias_matches_canonical(http_client: httpx.AsyncClient, en_sample) -> None:
    files_a = {"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")}
    files_b = {"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")}
    canonical = await http_client.post("/v1/audio/transcriptions", files=files_a, data={"response_format": "json"})
    alias = await http_client.post("/v1/stt/http", files=files_b, data={"response_format": "json"})
    assert orjson.loads(canonical.content)["text"] == orjson.loads(alias.content)["text"]


##### STT — SSE #####


async def test_stt_sse_alias_streams_events(http_client: httpx.AsyncClient, en_sample) -> None:
    async with aconnect_sse(
        http_client,
        "POST",
        "/v1/stt/sse",
        files={"file": ("audio.wav", en_sample.audio_bytes(), "audio/wav")},
        data={"stream": "true", "response_format": "text"},
    ) as event_source:
        events = [event async for event in event_source.aiter_sse()]
    assert len(events) >= 1


##### STT — WS #####


async def test_stt_ws_alias_returns_transcription(
    audioeval: AudioEval,
    en_sample,
    ws_base_url: str,
) -> None:
    from pytest_audioeval.stt import STTClient

    ws_url = ws_base_url + "/v1/stt/ws"
    client = STTClient(url=ws_url)

    async with client.ws(sample=en_sample, params={"response_format": "text"}) as session:
        await session.send_sample(en_sample, chunk_ms=200, encoding=AudioEncoding.PCM16_BASE64)
        await session.send_text("END_OF_AUDIO")
        await asyncio.sleep(1)

        while True:
            try:
                await session.receive_text(timeout=2.0)
            except Exception:
                break

        result = session.result()

    assert len(result.hypothesis_text.strip()) > 0


##### TTS — HTTP #####


async def test_tts_http_alias_returns_audio(http_client: httpx.AsyncClient) -> None:
    response = await http_client.post(
        "/v1/tts/http",
        json={"input": "Hello.", "voice": "af_heart", "response_format": "wav", "stream": False},
    )
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


async def test_tts_http_alias_matches_canonical(http_client: httpx.AsyncClient) -> None:
    payload = {"input": "Test.", "voice": "af_heart", "response_format": "wav", "stream": False}
    canonical = await http_client.post("/v1/audio/speech", json=payload)
    alias = await http_client.post("/v1/tts/http", json=payload)
    assert canonical.status_code == alias.status_code == 200
    assert abs(len(canonical.content) - len(alias.content)) < 100


##### TTS — SSE #####


async def test_tts_sse_alias_streams_delta_events(http_client: httpx.AsyncClient) -> None:
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


async def test_tts_stream_alias_returns_chunked_audio(http_client: httpx.AsyncClient) -> None:
    async with http_client.stream(
        "POST",
        "/v1/tts/stream",
        json={"input": "Hello world.", "voice": "af_heart", "stream": True, "response_format": "pcm"},
    ) as response:
        chunks = [chunk async for chunk in response.aiter_bytes()]

    total = b"".join(chunks)
    assert len(total) > 0


##### TTS — WS #####


async def test_tts_ws_alias_returns_audio_events(
    ws_base_url: str,
) -> None:
    from pytest_audioeval.tts import TTSClient

    ws_url = ws_base_url + "/v1/tts/ws"
    client = TTSClient(url=ws_url)

    async with client.ws() as ws:
        await ws.send_text(orjson.dumps({"input": "Hello.", "voice": "af_heart"}).decode())

        events: list[dict] = []
        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
            except Exception:
                break
            if not text.strip():
                continue
            body = orjson.loads(text)
            events.append(body)
            if body.get("type") == "speech.audio.done":
                break

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert len(delta_events) >= 1
