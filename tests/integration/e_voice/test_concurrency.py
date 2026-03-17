"""Concurrency tests — verify STT and TTS handle parallel requests."""

import asyncio

import httpx
from pytest_audioeval.client import AudioEval

##### CONCURRENT STT #####


async def test_concurrent_stt_transcriptions(
    audioeval: AudioEval,
    en_sample,
    es_sample,
) -> None:
    """Two STT WebSocket sessions running in parallel."""
    async def transcribe(sample, params=None):
        async with audioeval.stt.ws(sample=sample, **({"params": params} if params else {})) as session:
            await session.send_sample(sample, chunk_ms=200)
            await asyncio.sleep(3)
            return session.result()

    results = await asyncio.gather(
        transcribe(en_sample),
        transcribe(es_sample, params={"language": "es"}),
    )

    assert all(len(r.hypothesis_text.strip()) > 0 for r in results)


##### CONCURRENT TTS #####


async def test_concurrent_tts_requests(http_client: httpx.AsyncClient) -> None:
    """Two TTS requests in parallel."""
    async def synthesize(text: str, voice: str):
        resp = await http_client.post(
            "/v1/audio/speech",
            json={"input": text, "model": "kokoro", "voice": voice, "response_format": "wav"},
            timeout=30.0,
        )
        return resp

    responses = await asyncio.gather(
        synthesize("Hello world.", "af_heart"),
        synthesize("Good morning.", "af_bella"),
    )

    assert all(r.status_code == 200 for r in responses)
    assert all(len(r.content) > 0 for r in responses)


##### CONCURRENT STT + TTS #####


async def test_concurrent_stt_and_tts(
    audioeval: AudioEval,
    en_sample,
    http_client: httpx.AsyncClient,
) -> None:
    """STT and TTS running simultaneously."""
    async def transcribe():
        async with audioeval.stt.ws(sample=en_sample) as session:
            await session.send_sample(en_sample, chunk_ms=200)
            await asyncio.sleep(3)
            return session.result()

    async def synthesize():
        return await http_client.post(
            "/v1/audio/speech",
            json={"input": "Testing concurrent access.", "model": "kokoro", "voice": "af_heart", "response_format": "wav"},
            timeout=30.0,
        )

    stt_result, tts_response = await asyncio.gather(transcribe(), synthesize())

    assert len(stt_result.hypothesis_text.strip()) > 0
    assert tts_response.status_code == 200
    assert len(tts_response.content) > 0
