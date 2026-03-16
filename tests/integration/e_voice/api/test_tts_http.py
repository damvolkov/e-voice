"""TTS HTTP POST tests — /v1/audio/speech (non-streaming), /v1/audio/voices."""

import orjson
import pytest
from pytest_audioeval.client import AudioEval

_AUDIO_FORMATS = ["pcm", "mp3", "wav", "flac", "opus", "aac"]

_CONTENT_TYPE_MAP: dict[str, str] = {
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}


##### POST /v1/audio/speech (non-streaming) #####


@pytest.mark.parametrize(
    "audio_format",
    _AUDIO_FORMATS,
    ids=_AUDIO_FORMATS,
)
async def test_speech_format(
    audioeval: AudioEval,
    audio_format: str,
) -> None:
    """Each audio format returns 200 with correct content-type and non-empty body."""
    response = await audioeval.tts.post(
        json={
            "input": "Hello world.",
            "voice": "af_heart",
            "response_format": audio_format,
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert _CONTENT_TYPE_MAP[audio_format] in response.headers.get("content-type", "")
    assert len(response.content) > 0


async def test_speech_custom_voice(
    audioeval: AudioEval,
) -> None:
    """Speech synthesis with a specific voice parameter."""
    response = await audioeval.tts.post(
        json={
            "input": "Testing voice selection.",
            "voice": "af_heart",
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert len(response.content) > 0


async def test_speech_custom_speed(
    audioeval: AudioEval,
) -> None:
    """Speech synthesis respects speed parameter."""
    response = await audioeval.tts.post(
        json={
            "input": "Speed test.",
            "voice": "af_heart",
            "response_format": "pcm",
            "speed": 1.5,
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert len(response.content) > 0


async def test_speech_custom_language(
    audioeval: AudioEval,
) -> None:
    """Speech synthesis with explicit lang parameter."""
    response = await audioeval.tts.post(
        json={
            "input": "Hola mundo.",
            "voice": "ef_dora",
            "response_format": "mp3",
            "lang": "es",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert len(response.content) > 0


async def test_speech_invalid_json_returns_422(
    audioeval: AudioEval,
) -> None:
    """Invalid request body returns 422 Unprocessable Entity."""
    response = await audioeval.tts.post(json={"voice": "af_heart"})
    assert response.status_code == 422

    body = orjson.loads(response.content)
    assert "error" in body


async def test_speech_mp3_is_valid_audio(
    audioeval: AudioEval,
) -> None:
    """MP3 output starts with valid MP3/ID3 header bytes."""
    response = await audioeval.tts.post(
        json={
            "input": "Audio validation test.",
            "voice": "af_heart",
            "response_format": "mp3",
            "stream": False,
        },
    )
    assert response.status_code == 200
    # MP3 files start with ID3 tag or MPEG sync word
    header = response.content[:3]
    assert header in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


async def test_speech_wav_is_valid_audio(
    audioeval: AudioEval,
) -> None:
    """WAV output starts with RIFF header."""
    response = await audioeval.tts.post(
        json={
            "input": "WAV validation test.",
            "voice": "af_heart",
            "response_format": "wav",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


##### GET /v1/audio/voices #####


async def test_voices_returns_list(
    audioeval: AudioEval,
    http_client,
) -> None:
    """Voice listing returns non-empty list with expected schema."""
    response = await http_client.get("/v1/audio/voices")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "voices" in body
    assert isinstance(body["voices"], list)
    assert len(body["voices"]) > 0

    voice = body["voices"][0]
    assert "id" in voice
    assert "name" in voice
    assert "language" in voice
