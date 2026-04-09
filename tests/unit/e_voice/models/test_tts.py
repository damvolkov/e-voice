"""Unit tests for models/tts.py — SpeechRequest validation and voice resolution."""

import pytest
from pydantic import ValidationError

from e_voice.models.tts import SpeechRequest, SpeechResponseFormat, StreamFormat

##### SPEECH REQUEST — DEFAULTS #####


async def test_speech_request_defaults() -> None:
    req = SpeechRequest(input="hello")
    assert req.voice == "af_heart"
    assert req.lang == "en-us"
    assert req.speed == 1.0
    assert req.model == "kokoro"
    assert req.stream is True
    assert req.stream_format == StreamFormat.AUDIO
    assert req.response_format == SpeechResponseFormat.MP3
    assert req.sample_rate is None


##### SPEECH REQUEST — LANG AUTO-RESOLUTION #####


@pytest.mark.parametrize(
    ("voice", "expected_lang"),
    [
        ("af_heart", "en-us"),
        ("bf_emma", "en-gb"),
        ("ef_dora", "es"),
        ("ff_siwis", "fr"),
        ("jf_alpha", "ja"),
    ],
    ids=["en-us", "en-gb", "es", "fr", "ja"],
)
async def test_speech_request_infers_lang_from_voice(voice: str, expected_lang: str) -> None:
    req = SpeechRequest(input="test", voice=voice)
    assert req.lang == expected_lang


##### SPEECH REQUEST — EXPLICIT LANG MATCH #####


async def test_speech_request_explicit_lang_matching_voice() -> None:
    req = SpeechRequest(input="test", voice="ef_dora", lang="es")
    assert req.lang == "es"


##### SPEECH REQUEST — BACKEND-AGNOSTIC VOICE/LANG #####


async def test_speech_request_explicit_lang_overrides() -> None:
    req = SpeechRequest(input="test", voice="af_heart", lang="es")
    assert req.lang == "es"


async def test_speech_request_unknown_voice_keeps_default_lang() -> None:
    req = SpeechRequest(input="test", voice="serena")
    assert req.lang == "en-us"


async def test_speech_request_accepts_any_voice_id() -> None:
    req = SpeechRequest(input="test", voice="tatan", lang="es")
    assert req.voice == "tatan"
    assert req.lang == "es"


##### SPEECH REQUEST — INPUT VALIDATION #####


async def test_speech_request_empty_input() -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(input="")


async def test_speech_request_speed_bounds() -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(input="test", speed=0.1)
    with pytest.raises(ValidationError):
        SpeechRequest(input="test", speed=5.0)
