"""Unit tests for websockets/stt.py — streaming STT handler."""

import base64

import numpy as np
import orjson

from e_voice.streaming.transcriber import StreamingEvent, StreamingEventType
from e_voice.websockets.stt import format_event, handle_stt

from .conftest import MockConnection, MockState

##### FORMAT_EVENT #####


async def test_format_event_json() -> None:
    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hello",
        unconfirmed_text="world",
        new_confirmed="Hello",
    )
    result = format_event(event, "json")
    parsed = orjson.loads(result)
    assert parsed["type"] == "transcript_update"
    assert parsed["text"] == "Hello"
    assert parsed["partial"] == "world"
    assert parsed["is_final"] is False


async def test_format_event_text() -> None:
    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hello world",
        unconfirmed_text="",
        new_confirmed="world",
    )
    assert format_event(event, "text") == "Hello world"


##### HANDLE_STT — BINARY FRAMES #####


async def test_handle_stt_binary_pcm16(mocker) -> None:
    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hi",
        unconfirmed_text="there",
        new_confirmed="Hi",
    )
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=event)

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        query_params={"language": "en", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=[pcm16],
    )

    await handle_stt(conn)

    assert len(conn.sent) == 1
    parsed = orjson.loads(conn.sent[0])
    assert parsed["text"] == "Hi"


##### HANDLE_STT — BASE64 BACKWARD COMPAT #####


async def test_handle_stt_base64_text(mocker) -> None:
    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hi",
        unconfirmed_text="",
        new_confirmed="Hi",
    )
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=event)

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())
    b64 = base64.b64encode(np.zeros(160, dtype=np.int16).tobytes()).decode()

    conn = MockConnection(
        query_params={"language": "en", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=[b64],
    )

    await handle_stt(conn)

    assert len(conn.sent) == 1
    parsed = orjson.loads(conn.sent[0])
    assert parsed["text"] == "Hi"


##### HANDLE_STT — END_OF_AUDIO #####


async def test_handle_stt_end_of_audio_flushes(mocker) -> None:
    event = StreamingEvent(
        type=StreamingEventType.SESSION_END,
        confirmed_text="final",
        unconfirmed_text="",
        new_confirmed="final",
        is_final=True,
    )
    mocker.patch("e_voice.websockets.stt.flush_session", return_value=event)

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())
    conn = MockConnection(
        query_params={"language": "en", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=["END_OF_AUDIO"],
    )

    await handle_stt(conn)

    assert len(conn.sent) == 1
    parsed = orjson.loads(conn.sent[0])
    assert parsed["text"] == "final"
    assert parsed["is_final"] is True


##### HANDLE_STT — NONE EVENT SENDS ACK #####


async def test_handle_stt_none_event_sends_ack(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=None)

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        query_params={"language": "en", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=[pcm16],
    )

    await handle_stt(conn)

    assert len(conn.sent) == 1
    parsed = orjson.loads(conn.sent[0])
    assert parsed["type"] == "transcript_update"
    assert parsed["text"] == ""
    assert parsed["partial"] == ""
    assert parsed["is_final"] is False


##### HANDLE_STT — SESSION CLEANUP #####


async def test_handle_stt_cleans_session_on_disconnect(mocker) -> None:
    event = StreamingEvent(
        type=StreamingEventType.SESSION_END,
        confirmed_text="",
        unconfirmed_text="",
        new_confirmed="",
    )
    mocker.patch("e_voice.websockets.stt.flush_session", return_value=event)

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())
    conn = MockConnection(
        query_params={"language": "en", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=[],
    )

    await handle_stt(conn)

    assert conn.id not in state.stt_sessions


##### HANDLE_STT — ERROR HANDLING #####


async def test_handle_stt_error_sends_json(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", side_effect=RuntimeError("boom"))

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        query_params={"language": "en", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=[pcm16],
    )

    await handle_stt(conn)

    assert len(conn.sent) == 1
    parsed = orjson.loads(conn.sent[0])
    assert "boom" in parsed["error"]


##### HANDLE_STT — AUTO LANGUAGE #####


async def test_handle_stt_auto_language_sets_none(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=None)

    state = MockState(stt_sessions={}, whisper=mocker.MagicMock())

    conn = MockConnection(
        query_params={"language": "auto", "response_format": "json", "model": "tiny"},
        state=state,
        _messages=[np.zeros(160, dtype=np.int16).tobytes()],
    )

    await handle_stt(conn)

    session = state.stt_sessions.get(conn.id)
    assert session is None  # cleaned up in finally
