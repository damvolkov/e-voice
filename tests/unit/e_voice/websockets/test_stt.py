"""Unit tests for websockets/stt.py — streaming STT handler."""

import base64

import numpy as np
import orjson

from e_voice.models.ws import STTParams
from e_voice.streaming.stt.transcriber import StreamingEvent, StreamingEventType
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


async def test_format_event_segment_end() -> None:
    event = StreamingEvent(
        type=StreamingEventType.SEGMENT_END,
        confirmed_text="Hola mundo",
        unconfirmed_text="",
        new_confirmed="",
        is_final=True,
    )
    result = format_event(event, "json")
    parsed = orjson.loads(result)
    assert parsed["type"] == "segment_end"
    assert parsed["text"] == "Hola mundo"
    assert parsed["is_final"] is True


##### HANDLE_STT — BINARY FRAMES #####


async def test_handle_stt_binary_pcm16(mocker) -> None:
    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hi",
        unconfirmed_text="there",
        new_confirmed="Hi",
    )
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=event)

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        params=STTParams(language="en", response_format="json"),
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

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    b64 = base64.b64encode(np.zeros(160, dtype=np.int16).tobytes()).decode()

    conn = MockConnection(
        params=STTParams(language="en", response_format="json"),
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

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    conn = MockConnection(
        params=STTParams(language="en", response_format="json"),
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

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        params=STTParams(language="en", response_format="json"),
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

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    conn = MockConnection(
        params=STTParams(language="en"),
        state=state,
        _messages=[],
    )

    await handle_stt(conn)

    assert conn.id not in state.stt_sessions


##### HANDLE_STT — ERROR HANDLING #####


async def test_handle_stt_error_sends_json(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", side_effect=RuntimeError("boom"))

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        params=STTParams(language="en"),
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

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())

    conn = MockConnection(
        params=STTParams(language="auto"),
        state=state,
        _messages=[np.zeros(160, dtype=np.int16).tobytes()],
    )

    await handle_stt(conn)

    session = state.stt_sessions.get(conn.id)
    assert session is None  # cleaned up in finally


##### HANDLE_STT — SEGMENTATION MODE #####


async def test_handle_stt_segmentation_creates_vad(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=None)

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    conn = MockConnection(
        params=STTParams(language="en", segmentation=True),
        state=state,
        _messages=[np.zeros(160, dtype=np.int16).tobytes()],
    )

    await handle_stt(conn)
    assert len(conn.sent) >= 1


async def test_handle_stt_segmentation_false_no_vad(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=None)

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    conn = MockConnection(
        params=STTParams(language="en", segmentation=False),
        state=state,
        _messages=[np.zeros(160, dtype=np.int16).tobytes()],
    )

    await handle_stt(conn)
    assert len(conn.sent) >= 1


async def test_handle_stt_vad_trigger_sends_segment_end(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=None)

    seg_event = StreamingEvent(
        type=StreamingEventType.SEGMENT_END,
        confirmed_text="Hola mundo",
        unconfirmed_text="",
        new_confirmed="",
        is_final=True,
    )
    mocker.patch("e_voice.websockets.stt.flush_segment", return_value=seg_event)

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    pcm16 = np.zeros(160, dtype=np.int16).tobytes()

    conn = MockConnection(
        params=STTParams(language="en", segmentation=True),
        state=state,
        _messages=[pcm16],
    )

    original_init = mocker.patch("e_voice.websockets.stt.SessionState")
    mock_session = mocker.MagicMock()
    mock_session.vad = mocker.MagicMock()
    mock_session.vad.update.return_value = True
    mock_session.response_format = "json"
    mock_session.confirmed.text = "Hola mundo"
    mock_session.agreement.unconfirmed_text = ""
    mock_session.segment_text = "Hola mundo"
    original_init.return_value = mock_session
    state.stt_sessions = {}

    await handle_stt(conn)

    sent_types = []
    for msg in conn.sent:
        parsed = orjson.loads(msg)
        sent_types.append(parsed.get("type"))
    assert "segment_end" in sent_types


async def test_handle_stt_end_of_audio_with_segmentation(mocker) -> None:
    seg_event = StreamingEvent(
        type=StreamingEventType.SEGMENT_END,
        confirmed_text="pending",
        unconfirmed_text="",
        new_confirmed="",
        is_final=True,
    )
    session_event = StreamingEvent(
        type=StreamingEventType.SESSION_END,
        confirmed_text="",
        unconfirmed_text="",
        new_confirmed="",
        is_final=True,
    )
    mocker.patch("e_voice.websockets.stt.flush_segment", return_value=seg_event)
    mocker.patch("e_voice.websockets.stt.flush_session", return_value=session_event)

    state = MockState(stt_sessions={}, stt=mocker.MagicMock())
    conn = MockConnection(
        params=STTParams(language="en", response_format="json", segmentation=True),
        state=state,
        _messages=["END_OF_AUDIO"],
    )

    await handle_stt(conn)

    assert len(conn.sent) == 2
    first = orjson.loads(conn.sent[0])
    second = orjson.loads(conn.sent[1])
    assert first["type"] == "segment_end"
    assert first["text"] == "pending"
    assert second["type"] == "session_end"
