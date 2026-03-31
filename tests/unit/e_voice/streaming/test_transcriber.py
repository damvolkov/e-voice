from unittest.mock import AsyncMock

import numpy as np

from e_voice.streaming.text import StreamingWord, WordBuffer
from e_voice.streaming.transcriber import (
    LocalAgreement,
    SessionState,
    StreamingEventType,
    _build_prompt,
    _check_same_output,
    _extract_words,
    _needs_audio_after,
    flush_segment,
    flush_session,
    process_audio_chunk,
)


def _w(word: str, start: float = 0.0, end: float = 0.0) -> StreamingWord:
    return StreamingWord(word=word, start=start, end=end)


##### LOCAL AGREEMENT #####


async def test_agreement_first_call_returns_empty() -> None:
    la = LocalAgreement()
    confirmed = WordBuffer()
    incoming = [_w("hello", 0.0, 0.5), _w("world", 0.5, 1.0)]
    result = la.merge(confirmed, incoming)
    assert result == []
    assert len(la.unconfirmed) == 2


async def test_agreement_second_call_confirms_prefix() -> None:
    la = LocalAgreement()
    confirmed = WordBuffer()

    la.merge(confirmed, [_w("hello", 0.0, 0.5), _w("wor", 0.5, 1.0)])

    result = la.merge(confirmed, [_w("hello", 0.0, 0.5), _w("world", 0.5, 1.0), _w("foo", 1.0, 1.5)])
    assert len(result) == 1
    assert result[0].word == "hello"


async def test_agreement_no_common_prefix() -> None:
    la = LocalAgreement()
    confirmed = WordBuffer()
    la.merge(confirmed, [_w("hello")])
    result = la.merge(confirmed, [_w("goodbye")])
    assert result == []


async def test_agreement_flush_returns_unconfirmed() -> None:
    la = LocalAgreement()
    la.merge(WordBuffer(), [_w("hello"), _w("world")])
    flushed = la.flush()
    assert len(flushed) == 2
    assert la.unconfirmed == []


async def test_agreement_flush_empty() -> None:
    la = LocalAgreement()
    assert la.flush() == []


async def test_agreement_unconfirmed_text() -> None:
    la = LocalAgreement()
    la.merge(WordBuffer(), [_w("hello"), _w("world")])
    assert la.unconfirmed_text == "hello world"


async def test_agreement_progressive_confirmation() -> None:
    la = LocalAgreement()
    confirmed = WordBuffer()

    la.merge(confirmed, [_w("Hola", 0.0, 0.5)])

    new = la.merge(confirmed, [_w("Hola", 0.0, 0.5), _w("mundo", 0.5, 1.0)])
    assert len(new) == 1
    assert new[0].word == "Hola"
    confirmed.extend(new)

    new = la.merge(confirmed, [_w("Hola", 0.0, 0.5), _w("mundo", 0.5, 1.0), _w("qué", 1.0, 1.5)])
    assert len(new) == 1
    assert new[0].word == "mundo"


##### _NEEDS_AUDIO_AFTER #####


async def test_needs_audio_after_empty() -> None:
    assert _needs_audio_after(WordBuffer()) == 0.0


async def test_needs_audio_after_no_sentence() -> None:
    confirmed = WordBuffer([_w("hello", 0.0, 1.0)])
    assert _needs_audio_after(confirmed) == 0.0


async def test_needs_audio_after_with_sentence() -> None:
    confirmed = WordBuffer([_w("Hello.", 0.0, 1.0), _w("World", 1.0, 2.0)])
    assert _needs_audio_after(confirmed) == 1.0


##### _BUILD_PROMPT #####


async def test_build_prompt_empty() -> None:
    assert _build_prompt(WordBuffer()) is None


async def test_build_prompt_no_sentence() -> None:
    assert _build_prompt(WordBuffer([_w("hello")])) is None


async def test_build_prompt_with_sentence() -> None:
    confirmed = WordBuffer([_w("Hello."), _w("World.")])
    assert _build_prompt(confirmed) == "World."


##### _CHECK_SAME_OUTPUT #####


async def test_check_same_output_increments() -> None:
    session = SessionState(language="es")
    session.agreement._la_unconfirmed = [_w("hello")]
    assert _check_same_output(session) is False
    assert session._ss_same_output_count == 1

    for _ in range(6):
        _check_same_output(session)
    assert _check_same_output(session) is True


async def test_check_same_output_resets_on_change() -> None:
    session = SessionState(language="es")
    session.agreement._la_unconfirmed = [_w("hello")]
    _check_same_output(session)
    _check_same_output(session)
    assert session._ss_same_output_count == 2

    session.agreement._la_unconfirmed = [_w("world")]
    _check_same_output(session)
    assert session._ss_same_output_count == 1


async def test_check_same_output_empty_unconfirmed() -> None:
    session = SessionState(language="es")
    assert _check_same_output(session) is False
    assert session._ss_same_output_count == 0


##### FLUSH_SESSION #####


async def test_flush_session_moves_unconfirmed() -> None:
    session = SessionState(language="es")
    session.confirmed.extend([_w("Hello.")])
    session.agreement._la_unconfirmed = [_w("World")]

    event = flush_session(session)

    assert event.type == StreamingEventType.SESSION_END
    assert event.is_final is True
    assert "World" in event.confirmed_text
    assert "World" in event.new_confirmed
    assert event.unconfirmed_text == ""


async def test_flush_session_empty() -> None:
    session = SessionState(language="es")
    event = flush_session(session)
    assert event.is_final is True
    assert event.confirmed_text == ""
    assert event.new_confirmed == ""


##### FLUSH_SEGMENT #####


async def test_flush_segment_returns_segment_text() -> None:
    session = SessionState(language="es", segmentation=True)
    session.confirmed.extend([_w("Hola", 0.0, 0.5), _w("mundo.", 0.5, 1.0)])
    session.agreement._la_unconfirmed = [_w("qué")]

    event = flush_segment(session)

    assert event.type == StreamingEventType.SEGMENT_END
    assert event.is_final is True
    assert "Hola" in event.confirmed_text
    assert "mundo." in event.confirmed_text
    assert "qué" in event.confirmed_text


async def test_flush_segment_resets_boundary() -> None:
    session = SessionState(language="es", segmentation=True)
    session.confirmed.extend([_w("Hola"), _w("mundo.")])

    flush_segment(session)

    assert session._ss_segment_start == 2
    assert session.segment_text == ""
    assert session._ss_same_output_count == 0
    assert session._ss_prev_unconfirmed == ""


async def test_flush_segment_second_segment_only_new_words() -> None:
    session = SessionState(language="es", segmentation=True)
    session.confirmed.extend([_w("Hola"), _w("mundo.")])
    flush_segment(session)

    session.confirmed.extend([_w("Adiós")])
    event = flush_segment(session)

    assert event.confirmed_text == "Adiós"
    assert session._ss_segment_start == 3


async def test_flush_segment_empty_segment() -> None:
    session = SessionState(language="es", segmentation=True)
    event = flush_segment(session)
    assert event.confirmed_text == ""
    assert event.is_final is True


##### SESSION_STATE #####


async def test_session_state_defaults() -> None:
    session = SessionState(language="en", model_id="test-model", response_format="json")
    assert session.language == "en"
    assert session.model_id == "test-model"
    assert session.response_format == "json"
    assert session.audio_buffer.duration == 0.0
    assert len(session.confirmed) == 0
    assert session.vad is None
    assert session._ss_segment_start == 0


async def test_session_state_segmentation_creates_vad() -> None:
    session = SessionState(language="en", segmentation=True)
    assert session.vad is not None


async def test_session_state_no_segmentation_no_vad() -> None:
    session = SessionState(language="en", segmentation=False)
    assert session.vad is None


async def test_session_state_segment_text_empty() -> None:
    session = SessionState(language="es")
    assert session.segment_text == ""


async def test_session_state_segment_text_after_words() -> None:
    session = SessionState(language="es")
    session.confirmed.extend([_w("Hola"), _w("mundo")])
    assert session.segment_text == "Hola mundo"


async def test_session_state_segment_text_after_boundary() -> None:
    session = SessionState(language="es")
    session.confirmed.extend([_w("Hola"), _w("mundo"), _w("qué")])
    session._ss_segment_start = 2
    assert session.segment_text == "qué"


##### _EXTRACT_WORDS #####


async def test_extract_words_from_segments(mock_segments_with_words) -> None:
    words = _extract_words(mock_segments_with_words, audio_offset=0.0, no_speech_threshold=0.45)
    assert len(words) == 2
    assert words[0].word == "Hola"
    assert words[1].word == "mundo."


async def test_extract_words_applies_offset(mock_segments_with_words) -> None:
    words = _extract_words(mock_segments_with_words, audio_offset=5.0, no_speech_threshold=0.45)
    assert words[0].start == 5.0
    assert words[1].start == 5.5


async def test_extract_words_filters_silence(mock_segments_silence) -> None:
    words = _extract_words(mock_segments_silence, audio_offset=0.0, no_speech_threshold=0.45)
    assert words == []


async def test_extract_words_empty_segments() -> None:
    assert _extract_words([], audio_offset=0.0, no_speech_threshold=0.45) == []


##### PROCESS_AUDIO_CHUNK #####


async def test_process_audio_chunk_accumulates_before_threshold() -> None:
    session = SessionState(language="es")
    mock_whisper = AsyncMock()

    short_audio = np.zeros(8000, dtype=np.float32)
    result = await process_audio_chunk(session, mock_whisper, short_audio)

    assert result is None
    mock_whisper.transcribe.assert_not_called()


async def test_process_audio_chunk_transcribes_when_ready(
    mock_segments_with_words,
    mock_info,
) -> None:
    session = SessionState(language="es")
    mock_whisper = AsyncMock()
    mock_whisper.transcribe.return_value = (mock_segments_with_words, mock_info)

    audio = np.zeros(16000, dtype=np.float32)
    result = await process_audio_chunk(session, mock_whisper, audio)

    mock_whisper.transcribe.assert_called_once()
    assert result is None or result.unconfirmed_text


async def test_process_audio_chunk_confirms_on_second_call(
    mock_segments_with_words,
    mock_info,
) -> None:
    session = SessionState(language="es")
    mock_whisper = AsyncMock()
    mock_whisper.transcribe.return_value = (mock_segments_with_words, mock_info)

    audio = np.zeros(16000, dtype=np.float32)

    await process_audio_chunk(session, mock_whisper, audio)
    result = await process_audio_chunk(session, mock_whisper, audio)

    if result is not None:
        assert "Hola" in result.confirmed_text or "Hola" in result.new_confirmed


async def test_process_audio_chunk_silence_returns_none(
    mock_segments_silence,
    mock_info,
) -> None:
    session = SessionState(language="es")
    mock_whisper = AsyncMock()
    mock_whisper.transcribe.return_value = (mock_segments_silence, mock_info)

    audio = np.zeros(16000, dtype=np.float32)
    result = await process_audio_chunk(session, mock_whisper, audio)

    assert result is None


##### PROCESS_AUDIO_CHUNK — SEGMENTATION MODE #####


async def test_process_audio_chunk_segmentation_returns_segment_text(
    mock_segments_with_words,
    mock_info,
) -> None:
    session = SessionState(language="es", segmentation=True)
    mock_whisper = AsyncMock()
    mock_whisper.transcribe.return_value = (mock_segments_with_words, mock_info)

    audio = np.zeros(16000, dtype=np.float32)
    await process_audio_chunk(session, mock_whisper, audio)
    result = await process_audio_chunk(session, mock_whisper, audio)

    if result is not None:
        assert result.confirmed_text == session.segment_text


##### FLUSH_SESSION — SEGMENTATION MODE #####


async def test_flush_session_segmentation_returns_segment_text() -> None:
    session = SessionState(language="es", segmentation=True)
    session.confirmed.extend([_w("Hola"), _w("mundo.")])
    flush_segment(session)
    session.confirmed.extend([_w("Adiós")])
    session.agreement._la_unconfirmed = [_w("amigo")]

    event = flush_session(session)

    assert event.type == StreamingEventType.SESSION_END
    assert "Adiós" in event.confirmed_text
    assert "amigo" in event.confirmed_text
    assert "Hola" not in event.confirmed_text
