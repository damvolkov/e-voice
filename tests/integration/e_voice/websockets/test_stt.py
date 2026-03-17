import asyncio

import orjson
from pytest_audioeval.client import AudioEval

##### WS /v1/audio/transcriptions #####


async def test_ws_transcription_json_format(
    audioeval: AudioEval,
    en_sample,
) -> None:
    async with audioeval.stt.ws(sample=en_sample) as session:
        await session.send_sample(en_sample, chunk_ms=200)
        await asyncio.sleep(2)

        result = session.result()

    assert len(result.hypothesis_text.strip()) > 0


async def test_ws_transcription_quality(
    audioeval: AudioEval,
    en_sample,
    audioeval_thresholds: dict[str, float],
) -> None:
    async with audioeval.stt.ws(sample=en_sample) as session:
        await session.send_sample(en_sample, chunk_ms=200)
        await asyncio.sleep(3)

        result = session.result()

    result.compute_metrics(en_sample.reference_text)
    result.assert_quality(
        max_wer=audioeval_thresholds["max_wer"],
        max_cer=audioeval_thresholds["max_cer"],
    )


async def test_ws_transcription_text_format(
    audioeval: AudioEval,
    en_sample,
) -> None:
    async with audioeval.stt.ws(
        sample=en_sample,
        params={"response_format": "text"},
    ) as session:
        await session.send_sample(en_sample, chunk_ms=200)
        await asyncio.sleep(2)

        result = session.result()

    assert len(result.hypothesis_text.strip()) > 0


async def test_ws_transcription_verbose_json_format(
    audioeval: AudioEval,
    en_sample,
) -> None:
    async with audioeval.stt.ws(
        sample=en_sample,
        params={"response_format": "verbose_json"},
    ) as session:
        await session.send_sample(en_sample, chunk_ms=200)
        await asyncio.sleep(2)

        text = await session.receive_text(timeout=5.0)

    body = orjson.loads(text)
    assert "text" in body
    assert "segments" in body
    assert "language" in body


async def test_ws_transcription_with_language(
    audioeval: AudioEval,
    es_sample,
) -> None:
    async with audioeval.stt.ws(
        sample=es_sample,
        params={"language": "es"},
    ) as session:
        await session.send_sample(es_sample, chunk_ms=200)
        await asyncio.sleep(2)

        result = session.result()

    assert len(result.hypothesis_text.strip()) > 0


async def test_ws_transcription_text_message_ignored(
    audioeval: AudioEval,
    en_sample,
) -> None:
    async with audioeval.stt.ws(sample=en_sample) as session:
        await session.send_text("this is not audio")
        await asyncio.sleep(0.5)

        await session.send_sample(en_sample, chunk_ms=200)
        await asyncio.sleep(2)

        result = session.result()

    assert len(result.hypothesis_text.strip()) > 0


async def test_ws_transcription_multiple_chunks(
    audioeval: AudioEval,
    en_counting_sample,
) -> None:
    async with audioeval.stt.ws(sample=en_counting_sample) as session:
        await session.send_sample(en_counting_sample, chunk_ms=100)
        await asyncio.sleep(3)

        result = session.result()

    assert result.chunks_received > 0
    assert len(result.hypothesis_text.strip()) > 0
