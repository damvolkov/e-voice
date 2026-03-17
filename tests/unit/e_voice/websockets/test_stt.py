"""Unit tests for websockets/stt.py — streaming STT handlers."""

from dataclasses import dataclass, field

import numpy as np
import orjson

from e_voice.websockets.stt import format_event, on_close, on_connect, on_message

##### FIXTURES #####


@dataclass
class MockWSConnector:
    id: str = "ws-001"
    query_params: dict = field(default_factory=dict)


@dataclass
class MockSTTConfig:
    default_language: str | None = None
    default_response_format: type = field(default=None)
    model: str = "Systran/faster-whisper-small"

    def __post_init__(self) -> None:
        if self.default_response_format is None:

            @dataclass
            class FakeEnum:
                value: str = "json"

            self.default_response_format = FakeEnum()


@dataclass
class MockState:
    stt_sessions: dict = field(default_factory=dict)
    whisper: object = None


class MockGlobalDeps:
    def __init__(self, state: MockState) -> None:
        self._state = state

    def get(self, key: str) -> MockState:
        return self._state


##### FORMAT_EVENT #####


async def test_format_event_json() -> None:
    from e_voice.streaming.transcriber import StreamingEvent, StreamingEventType

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
    from e_voice.streaming.transcriber import StreamingEvent, StreamingEventType

    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hello world",
        unconfirmed_text="",
        new_confirmed="world",
    )
    assert format_event(event, "text") == "Hello world"


##### ON_CONNECT #####


async def test_on_connect_creates_session() -> None:
    ws = MockWSConnector(query_params={"language": "es", "response_format": "text", "model": "tiny"})
    state = MockState()
    deps = MockGlobalDeps(state)

    on_connect(ws, deps)

    assert ws.id in state.stt_sessions
    session = state.stt_sessions[ws.id]
    assert session.language == "es"
    assert session.response_format == "text"
    assert session.model_id == "tiny"


async def test_on_connect_defaults(mocker) -> None:
    mocker.patch("e_voice.websockets.stt.st", mocker.MagicMock(stt=MockSTTConfig()))
    ws = MockWSConnector(query_params={})
    state = MockState()
    deps = MockGlobalDeps(state)

    on_connect(ws, deps)

    session = state.stt_sessions[ws.id]
    assert session.model_id == "Systran/faster-whisper-small"


async def test_on_connect_auto_language_sets_none() -> None:
    ws = MockWSConnector(query_params={"language": "auto", "response_format": "json", "model": "tiny"})
    state = MockState()
    deps = MockGlobalDeps(state)

    on_connect(ws, deps)

    session = state.stt_sessions[ws.id]
    assert session.language is None


##### ON_MESSAGE #####


async def test_on_message_no_session() -> None:
    ws = MockWSConnector(id="unknown")
    state = MockState()
    deps = MockGlobalDeps(state)

    result = await on_message(ws, "", deps)
    parsed = orjson.loads(result)
    assert parsed["error"] == "no session"


async def test_on_message_processes_audio(mocker) -> None:
    from e_voice.streaming.transcriber import SessionState, StreamingEvent, StreamingEventType

    session = SessionState(language="en", model_id="tiny", response_format="json")
    state = MockState(stt_sessions={"ws-001": session}, whisper=mocker.MagicMock())
    deps = MockGlobalDeps(state)

    event = StreamingEvent(
        type=StreamingEventType.TRANSCRIPT_UPDATE,
        confirmed_text="Hi",
        unconfirmed_text="there",
        new_confirmed="Hi",
    )
    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=event)

    pcm_samples = np.zeros(160, dtype=np.int16)
    import base64

    msg = base64.b64encode(pcm_samples.tobytes()).decode()

    result = await on_message(MockWSConnector(), msg, deps)
    parsed = orjson.loads(result)
    assert parsed["text"] == "Hi"


async def test_on_message_returns_empty_on_none_event(mocker) -> None:
    from e_voice.streaming.transcriber import SessionState

    session = SessionState(language="en", model_id="tiny", response_format="json")
    state = MockState(stt_sessions={"ws-001": session}, whisper=mocker.MagicMock())
    deps = MockGlobalDeps(state)

    mocker.patch("e_voice.websockets.stt.process_audio_chunk", return_value=None)

    pcm_samples = np.zeros(160, dtype=np.int16)
    import base64

    msg = base64.b64encode(pcm_samples.tobytes()).decode()
    result = await on_message(MockWSConnector(), msg, deps)
    assert result == ""


async def test_on_message_handles_exception(mocker) -> None:
    from e_voice.streaming.transcriber import SessionState

    session = SessionState(language="en", model_id="tiny", response_format="json")
    state = MockState(stt_sessions={"ws-001": session}, whisper=mocker.MagicMock())
    deps = MockGlobalDeps(state)

    mocker.patch("e_voice.websockets.stt.process_audio_chunk", side_effect=RuntimeError("boom"))

    import base64

    msg = base64.b64encode(b"\x00" * 320).decode()
    result = await on_message(MockWSConnector(), msg, deps)
    parsed = orjson.loads(result)
    assert "boom" in parsed["error"]


##### ON_CLOSE #####


async def test_on_close_flushes_session(mocker) -> None:
    from e_voice.streaming.transcriber import SessionState, StreamingEvent, StreamingEventType

    session = SessionState(language="en", model_id="tiny", response_format="json")
    state = MockState(stt_sessions={"ws-001": session})
    deps = MockGlobalDeps(state)

    event = StreamingEvent(
        type=StreamingEventType.SESSION_END,
        confirmed_text="final text",
        unconfirmed_text="",
        new_confirmed="final text",
    )
    mocker.patch("e_voice.websockets.stt.flush_session", return_value=event)

    result = on_close(MockWSConnector(), deps)
    assert result == ""
    assert "ws-001" not in state.stt_sessions


async def test_on_close_no_session() -> None:
    state = MockState()
    deps = MockGlobalDeps(state)

    result = on_close(MockWSConnector(id="nonexistent"), deps)
    assert result == ""
