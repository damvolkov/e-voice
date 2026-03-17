import pytest
from pytest_audioeval.client import AudioEval

_STREAM_FORMATS = ["pcm", "mp3", "wav"]

_CONTENT_TYPE_MAP: dict[str, str] = {
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}


##### POST /v1/audio/speech (chunked audio streaming) #####


@pytest.mark.parametrize(
    "audio_format",
    _STREAM_FORMATS,
    ids=_STREAM_FORMATS,
)
async def test_speech_stream_audio_chunks(
    audioeval: AudioEval,
    audio_format: str,
) -> None:
    chunks: list[bytes] = []

    async with audioeval.tts.stream(
        json={
            "input": "This is a streaming audio test.",
            "voice": "af_heart",
            "response_format": audio_format,
            "stream": True,
            "stream_format": "audio",
        },
    ) as response:
        assert _CONTENT_TYPE_MAP[audio_format] in response.headers.get("content-type", "")
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)

    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0


async def test_speech_stream_multiple_chunks(
    audioeval: AudioEval,
) -> None:
    chunks: list[bytes] = []

    async with audioeval.tts.stream(
        json={
            "input": "The quick brown fox jumps over the lazy dog. This sentence has multiple words to generate longer audio output.",
            "voice": "af_heart",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        },
    ) as response:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)

    assert len(chunks) >= 1
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


async def test_speech_stream_pcm_raw_bytes(
    audioeval: AudioEval,
) -> None:
    all_bytes = b""

    async with audioeval.tts.stream(
        json={
            "input": "PCM test.",
            "voice": "af_heart",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        },
    ) as response:
        async for chunk in response.aiter_bytes():
            all_bytes += chunk

    assert len(all_bytes) > 0
    assert len(all_bytes) % 2 == 0
