"""OpenAI SDK compatibility tests — TTS endpoints.

Validates that e-voice TTS is a drop-in replacement for OpenAI's Audio API
using the official openai Python SDK.
"""

from openai import AsyncOpenAI

##### POST /v1/audio/speech (SDK) #####


async def test_speech_mp3(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis returns MP3 audio."""
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="mp3",
    )
    audio_bytes = response.content
    assert len(audio_bytes) > 0


async def test_speech_wav(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis returns WAV audio."""
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="wav",
    )
    audio_bytes = response.content
    assert len(audio_bytes) > 0


async def test_speech_pcm(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis returns raw PCM audio."""
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="pcm",
    )
    audio_bytes = response.content
    assert len(audio_bytes) > 0
    assert len(audio_bytes) % 2 == 0


async def test_speech_flac(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis returns FLAC audio."""
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="flac",
    )
    assert len(response.content) > 0


async def test_speech_opus(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis returns Opus audio."""
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="opus",
    )
    assert len(response.content) > 0


async def test_speech_with_speed(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis respects speed parameter."""
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="mp3",
        speed=1.5,
    )
    assert len(response.content) > 0


async def test_speech_different_voices(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis works with different voice IDs."""
    for voice in ("af_heart", "af_bella", "bf_emma"):
        response = await openai_client.audio.speech.create(
            model="kokoro",
            input="Testing voice.",
            voice=voice,
            response_format="mp3",
        )
        assert len(response.content) > 0, f"Voice {voice} produced no audio"


async def test_speech_long_text(openai_client: AsyncOpenAI) -> None:
    """SDK speech handles longer text input."""
    long_text = "This is a longer sentence that tests the ability of the TTS engine to handle multi-word input correctly."
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input=long_text,
        voice="af_heart",
        response_format="mp3",
    )
    assert len(response.content) > 1000


##### STREAMING (SDK) #####


async def test_speech_streaming(openai_client: AsyncOpenAI) -> None:
    """SDK speech synthesis supports streaming response."""
    async with openai_client.audio.speech.with_streaming_response.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="mp3",
    ) as response:
        chunks: list[bytes] = []
        async for chunk in response.iter_bytes(chunk_size=4096):
            chunks.append(chunk)

    total = b"".join(chunks)
    assert len(total) > 0
