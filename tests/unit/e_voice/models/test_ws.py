"""Unit tests for models/ws.py — STTParams, TTSParams, WSSpeechRequest."""

import pytest
from pydantic import ValidationError

from e_voice.core.settings import ResponseFormatType
from e_voice.core.settings import settings as st
from e_voice.models.ws import STTParams, TTSParams, WSSpeechRequest

##### STT PARAMS — DEFAULTS #####


async def test_stt_params_defaults() -> None:
    params = STTParams()
    assert params.language == st.stt.default_language
    assert params.response_format == st.stt.default_response_format
    assert params.model == st.stt.model
    assert params.segmentation is False


async def test_stt_params_frozen() -> None:
    params = STTParams()
    with pytest.raises(ValidationError):
        params.language = "fr"


##### STT PARAMS — LANGUAGE NORMALIZATION #####


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("es", "es"),
        ("en", "en"),
        ("auto", st.stt.default_language),
        ("", st.stt.default_language),
        (None, st.stt.default_language),
    ],
    ids=["explicit-es", "explicit-en", "auto-to-default", "empty-to-default", "none-to-default"],
)
async def test_stt_params_language_normalization(raw: str | None, expected: str | None) -> None:
    params = STTParams(language=raw)
    assert params.language == expected


##### STT PARAMS — RESPONSE FORMAT #####


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("json", ResponseFormatType.JSON),
        ("text", ResponseFormatType.TEXT),
        ("verbose_json", ResponseFormatType.VERBOSE_JSON),
        ("", st.stt.default_response_format),
        (None, st.stt.default_response_format),
    ],
    ids=["json", "text", "verbose", "empty-to-default", "none-to-default"],
)
async def test_stt_params_response_format(raw: str | None, expected: ResponseFormatType) -> None:
    data: dict[str, str | None] = {"response_format": raw}
    params = STTParams.model_validate(data)
    assert params.response_format == expected


async def test_stt_params_invalid_response_format() -> None:
    with pytest.raises(ValidationError):
        STTParams(response_format="invalid_format")


##### STT PARAMS — MODEL #####


async def test_stt_params_model_explicit() -> None:
    params = STTParams(model="tiny")
    assert params.model == "tiny"


async def test_stt_params_model_empty_falls_to_default() -> None:
    params = STTParams(model="")
    assert params.model == st.stt.model


##### STT PARAMS — SEGMENTATION COERCION #####


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("True", True), ("FALSE", False), ("", False), (True, True), (False, False)],
    ids=["true-str", "false-str", "True-cap", "FALSE-cap", "empty", "bool-true", "bool-false"],
)
async def test_stt_params_segmentation_coercion(raw: str | bool, expected: bool) -> None:
    params = STTParams.model_validate({"segmentation": raw})
    assert params.segmentation is expected


##### STT PARAMS — FROM RAW DICT (simulates parse_qsl) #####


async def test_stt_params_from_raw_dict() -> None:
    raw = {"language": "en", "response_format": "text", "segmentation": "true"}
    params = STTParams.model_validate(raw)
    assert params.language == "en"
    assert params.response_format == ResponseFormatType.TEXT
    assert params.segmentation is True
    assert params.model == st.stt.model


async def test_stt_params_from_empty_dict() -> None:
    params = STTParams.model_validate({})
    assert params.language == st.stt.default_language
    assert params.response_format == st.stt.default_response_format
    assert params.model == st.stt.model
    assert params.segmentation is False


##### TTS PARAMS #####


async def test_tts_params_empty() -> None:
    params = TTSParams()
    assert params.model_fields == {}


async def test_tts_params_ignores_extra() -> None:
    params = TTSParams.model_validate({"unknown_key": "value"})
    assert not hasattr(params, "unknown_key")


##### WS SPEECH REQUEST — DEFAULTS #####


async def test_ws_speech_request_defaults() -> None:
    request = WSSpeechRequest(input="Hello world")
    assert request.voice == st.tts.default_voice
    assert request.speed == st.tts.default_speed
    assert request.lang


async def test_ws_speech_request_frozen() -> None:
    request = WSSpeechRequest(input="Hello")
    with pytest.raises(ValidationError):
        request.input = "changed"


##### WS SPEECH REQUEST — INPUT VALIDATION #####


async def test_ws_speech_request_empty_input_rejected() -> None:
    with pytest.raises(ValidationError, match="Input text must not be empty"):
        WSSpeechRequest(input="")


async def test_ws_speech_request_whitespace_input_rejected() -> None:
    with pytest.raises(ValidationError, match="Input text must not be empty"):
        WSSpeechRequest(input="   ")


##### WS SPEECH REQUEST — VOICE VALIDATION #####


async def test_ws_speech_request_valid_voice() -> None:
    request = WSSpeechRequest(input="Hello", voice="af_heart")
    assert request.voice == "af_heart"


async def test_ws_speech_request_invalid_voice_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown voice prefix"):
        WSSpeechRequest(input="Hello", voice="xf_unknown")


##### WS SPEECH REQUEST — LANG RESOLUTION #####


async def test_ws_speech_request_lang_inferred_from_voice() -> None:
    request = WSSpeechRequest(input="Hello", voice="af_heart")
    assert request.lang == "en-us"


async def test_ws_speech_request_lang_inferred_spanish() -> None:
    request = WSSpeechRequest(input="Hola", voice="ef_dora")
    assert request.lang == "es"


async def test_ws_speech_request_explicit_lang_matching() -> None:
    request = WSSpeechRequest(input="Hello", voice="af_heart", lang="en-us")
    assert request.lang == "en-us"


async def test_ws_speech_request_lang_conflict_rejected() -> None:
    with pytest.raises(ValidationError, match="conflicts with voice"):
        WSSpeechRequest(input="Hello", voice="af_heart", lang="es")


async def test_ws_speech_request_none_lang_inferred() -> None:
    request = WSSpeechRequest.model_validate({"input": "Hello", "voice": "af_heart", "lang": None})
    assert request.lang == "en-us"


##### WS SPEECH REQUEST — JSON PARSING #####


async def test_ws_speech_request_from_json() -> None:
    raw = '{"input": "Hello world", "voice": "af_heart", "speed": 1.5}'
    request = WSSpeechRequest.model_validate_json(raw)
    assert request.input == "Hello world"
    assert request.voice == "af_heart"
    assert request.speed == 1.5
    assert request.lang == "en-us"


async def test_ws_speech_request_invalid_json() -> None:
    with pytest.raises(ValidationError):
        WSSpeechRequest.model_validate_json("not json")


async def test_ws_speech_request_missing_input() -> None:
    with pytest.raises(ValidationError):
        WSSpeechRequest.model_validate_json('{"voice": "af_heart"}')
