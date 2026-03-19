"""Shared fixtures for websocket handler unit tests."""

from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class MockConnection:
    """Mock Connection for testing standalone WS handlers."""

    id: str = "ws-001"
    path: str = "/v1/audio/transcriptions"
    query_params: dict[str, str] = field(default_factory=dict)
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
    whisper: object = None
    kokoro: object = None


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
        query_params: dict[str, str] | None = None,
        messages: list[str | bytes] | None = None,
    ) -> MockConnection:
        return MockConnection(
            id=id,
            path=path,
            query_params=query_params or {},
            state=mock_state,
            _messages=messages or [],
        )

    return _make
