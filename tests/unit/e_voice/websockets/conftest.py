"""Shared fixtures for websocket handler unit tests."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from e_voice.core.websocket import BaseWSParams
from e_voice.models.session import ConnectionRegistry


@dataclass
class MockConnection:
    """Mock Connection for testing standalone WS handlers."""

    id: str = "ws-001"
    path: str = "/v1/audio/transcriptions"
    params: Any = field(default_factory=BaseWSParams)
    state: Any = None
    _messages: list[str | bytes] = field(default_factory=list)
    sent: list[str | bytes] = field(default_factory=list)

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass

    async def __aiter__(self):
        for msg in self._messages:
            yield msg


@dataclass
class MockState:
    """Mock lifespan state."""

    stt_sessions: dict = field(default_factory=dict)
    stt_connections: ConnectionRegistry = field(default_factory=ConnectionRegistry)
    tts_connections: ConnectionRegistry = field(default_factory=ConnectionRegistry)
    stt: object = None
    tts: object = None


@pytest.fixture()
def mock_state() -> MockState:
    return MockState()


@pytest.fixture()
def make_connection(mock_state: MockState):
    """Factory fixture for MockConnection with pre-configured state."""

    def _make(
        *,
        id: str = "ws-001",
        path: str = "/v1/audio/transcriptions",
        params: Any = None,
        messages: list[str | bytes] | None = None,
    ) -> MockConnection:
        return MockConnection(
            id=id,
            path=path,
            params=params or BaseWSParams(),
            state=mock_state,
            _messages=messages or [],
        )

    return _make
