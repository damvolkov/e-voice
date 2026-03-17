from openai import AsyncOpenAI

##### POST /v1/audio/speech (SDK) #####


async def test_speech_mp3(openai_client: AsyncOpenAI) -> None:
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="mp3",
    )
    audio_bytes = response.content
    assert len(audio_bytes) > 0


async def test_speech_wav(openai_client: AsyncOpenAI) -> None:
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="wav",
    )
    audio_bytes = response.content
    assert len(audio_bytes) > 0


async def test_speech_pcm(openai_client: AsyncOpenAI) -> None:
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
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="flac",
    )
    assert len(response.content) > 0


async def test_speech_opus(openai_client: AsyncOpenAI) -> None:
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="opus",
    )
    assert len(response.content) > 0


async def test_speech_with_speed(openai_client: AsyncOpenAI) -> None:
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input="Hello world.",
        voice="af_heart",
        response_format="mp3",
        speed=1.5,
    )
    assert len(response.content) > 0


async def test_speech_different_voices(openai_client: AsyncOpenAI) -> None:
    for voice in ("af_heart", "af_bella", "bf_emma"):
        response = await openai_client.audio.speech.create(
            model="kokoro",
            input="Testing voice.",
            voice=voice,
            response_format="mp3",
        )
        assert len(response.content) > 0, f"Voice {voice} produced no audio"


async def test_speech_long_text(openai_client: AsyncOpenAI) -> None:
    long_text = (
        "This is a longer sentence that tests the ability of the TTS engine to handle multi-word input correctly."
    )
    response = await openai_client.audio.speech.create(
        model="kokoro",
        input=long_text,
        voice="af_heart",
        response_format="mp3",
    )
    assert len(response.content) > 1000


##### STREAMING (SDK) #####


async def test_speech_streaming(openai_client: AsyncOpenAI) -> None:
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
