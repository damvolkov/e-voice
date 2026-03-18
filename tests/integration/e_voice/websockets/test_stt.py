import asyncio

import orjson
from pytest_audioeval.stt import AudioEncoding, STTClient

##### WS /v1/audio/transcriptions #####


async def test_ws_transcription_json_format(
    stt: STTClient,
    en_sample,
) -> None:
    async with stt.ws(sample=en_sample, params={"language": "en", "response_format": "json"}) as session:
        await session.send_sample(en_sample, chunk_ms=200, encoding=AudioEncoding.PCM16_BASE64)
        await session.send_text("END_OF_AUDIO")
        await asyncio.sleep(1)

        texts: list[str] = []
        while True:
            try:
                text = await session.receive_text(timeout=2.0)
            except Exception:
                break
            if not text.strip():
                continue
            texts.append(text)

    assert len(texts) > 0
    body = orjson.loads(texts[-1])
    assert "text" in body


async def test_ws_transcription_quality(
    stt: STTClient,
    en_sample,
    audioeval_thresholds: dict[str, float],
) -> None:
    async with stt.ws(
        sample=en_sample,
        params={"language": "en", "response_format": "text"},
    ) as session:
        await session.send_sample(en_sample, chunk_ms=200, encoding=AudioEncoding.PCM16_BASE64)
        await session.send_text("END_OF_AUDIO")
        await asyncio.sleep(1)

        while True:
            try:
                await session.receive_text(timeout=2.0)
            except Exception:
                break

        result = session.result()

    result.compute_metrics(en_sample.reference_text)
    result.assert_quality(
        max_wer=audioeval_thresholds["max_wer"],
        max_cer=audioeval_thresholds["max_cer"],
    )


async def test_ws_transcription_text_format(
    stt: STTClient,
    en_sample,
) -> None:
    async with stt.ws(
        sample=en_sample,
        params={"response_format": "text"},
    ) as session:
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


async def test_ws_transcription_verbose_json_format(
    stt: STTClient,
    en_sample,
) -> None:
    async with stt.ws(
        sample=en_sample,
        params={"response_format": "verbose_json"},
    ) as session:
        await session.send_sample(en_sample, chunk_ms=200, encoding=AudioEncoding.PCM16_BASE64)
        await session.send_text("END_OF_AUDIO")
        await asyncio.sleep(1)

        texts: list[str] = []
        while True:
            try:
                text = await session.receive_text(timeout=2.0)
            except Exception:
                break
            if not text.strip():
                continue
            texts.append(text)

    assert len(texts) > 0
    body = orjson.loads(texts[-1])
    assert "text" in body
    assert "partial" in body


async def test_ws_transcription_with_language(
    stt: STTClient,
    es_sample,
) -> None:
    async with stt.ws(
        sample=es_sample,
        params={"language": "es", "response_format": "text"},
    ) as session:
        await session.send_sample(es_sample, chunk_ms=200, encoding=AudioEncoding.PCM16_BASE64)
        await session.send_text("END_OF_AUDIO")
        await asyncio.sleep(1)

        while True:
            try:
                await session.receive_text(timeout=2.0)
            except Exception:
                break

        result = session.result()

    assert len(result.hypothesis_text.strip()) > 0


async def test_ws_transcription_text_message_ignored(
    stt: STTClient,
    en_sample,
) -> None:
    async with stt.ws(
        sample=en_sample,
        params={"response_format": "text"},
    ) as session:
        await session.send_text("this is not audio")
        await asyncio.sleep(0.5)

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


async def test_ws_transcription_multiple_chunks(
    stt: STTClient,
    en_counting_sample,
) -> None:
    async with stt.ws(
        sample=en_counting_sample,
        params={"response_format": "text"},
    ) as session:
        await session.send_sample(en_counting_sample, chunk_ms=100, encoding=AudioEncoding.PCM16_BASE64)
        await session.send_text("END_OF_AUDIO")
        await asyncio.sleep(2)

        while True:
            try:
                await session.receive_text(timeout=2.0)
            except Exception:
                break

        result = session.result()

    assert result.chunks_received > 0
    assert len(result.hypothesis_text.strip()) > 0
