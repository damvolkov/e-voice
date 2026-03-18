import pytest

from e_voice.models.transcription import (
    ResponseFormat,
    TimestampGranularity,
    TranscriptionParams,
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionVerboseResponse,
    TranscriptionWord,
)

##### RESPONSE_FORMAT ENUM #####


@pytest.mark.parametrize(
    ("value", "member"),
    [
        ("text", ResponseFormat.TEXT),
        ("json", ResponseFormat.JSON),
        ("verbose_json", ResponseFormat.VERBOSE_JSON),
        ("srt", ResponseFormat.SRT),
        ("vtt", ResponseFormat.VTT),
    ],
    ids=["text", "json", "verbose_json", "srt", "vtt"],
)
async def test_response_format_from_str(value: str, member: ResponseFormat) -> None:
    assert ResponseFormat(value) == member


async def test_response_format_invalid() -> None:
    with pytest.raises(ValueError):
        ResponseFormat("invalid")


##### TIMESTAMP_GRANULARITY #####


async def test_timestamp_granularity_members() -> None:
    assert TimestampGranularity.SEGMENT == "segment"
    assert TimestampGranularity.WORD == "word"


##### TRANSCRIPTION_PARAMS #####


async def test_transcription_params_defaults() -> None:
    params = TranscriptionParams()
    assert params.response_format == ResponseFormat.JSON
    assert params.temperature == 0.0
    assert params.stream is False
    assert params.vad_filter is False
    assert params.language is None


async def test_transcription_params_custom() -> None:
    params = TranscriptionParams(
        language="es",
        response_format=ResponseFormat.TEXT,
        temperature=0.5,
        stream=True,
    )
    assert params.language == "es"
    assert params.stream is True


async def test_transcription_params_temperature_bounds() -> None:
    with pytest.raises(ValueError):
        TranscriptionParams(temperature=-0.1)
    with pytest.raises(ValueError):
        TranscriptionParams(temperature=1.1)


##### TRANSCRIPTION_WORD #####


async def test_transcription_word() -> None:
    word = TranscriptionWord(start=0.0, end=0.5, word="hello", probability=0.95)
    assert word.word == "hello"
    assert word.start == 0.0
    assert word.probability == 0.95


async def test_transcription_word_defaults() -> None:
    word = TranscriptionWord(start=0.0, end=1.0, word="test")
    assert word.probability == 0.0


##### TRANSCRIPTION_SEGMENT #####


async def test_transcription_segment() -> None:
    seg = TranscriptionSegment(id=0, seek=0, start=0.0, end=1.0, text="hello")
    assert seg.text == "hello"
    assert seg.tokens == []
    assert seg.words is None


async def test_transcription_segment_with_words() -> None:
    words = [TranscriptionWord(start=0.0, end=0.5, word="hello")]
    seg = TranscriptionSegment(id=0, seek=0, start=0.0, end=1.0, text="hello", words=words)
    assert seg.words is not None
    assert len(seg.words) == 1


##### TRANSCRIPTION_RESPONSE #####


async def test_transcription_response() -> None:
    resp = TranscriptionResponse(text="Hello world")
    assert resp.text == "Hello world"
    dumped = resp.model_dump_json()
    assert "Hello world" in dumped


##### TRANSCRIPTION_VERBOSE_RESPONSE #####


async def test_verbose_response() -> None:
    resp = TranscriptionVerboseResponse(
        language="en",
        duration=2.5,
        text="Hello",
        segments=[],
    )
    assert resp.task == "transcribe"
    assert resp.language == "en"
    assert resp.duration == 2.5
    assert resp.words is None


async def test_verbose_response_with_segments() -> None:
    seg = TranscriptionSegment(id=0, seek=0, start=0.0, end=1.0, text="hi")
    resp = TranscriptionVerboseResponse(
        language="en",
        duration=1.0,
        text="hi",
        segments=[seg],
    )
    assert len(resp.segments) == 1
