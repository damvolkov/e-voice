"""Test fixtures for robyn-ml-api unit tests."""

import json
from collections.abc import Generator
from dataclasses import dataclass, field

import pytest

from app.core.lifespan import State


# -----------------------------------------------------------------------------
# Mock classes for Robyn Request
# -----------------------------------------------------------------------------


@dataclass
class MockHeaders:
    """Mock Headers object for Robyn Request."""

    _data: dict = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._data[key] = value


@dataclass
class MockRequest:
    """Mock Request object for Robyn."""

    _body: dict | str = field(default_factory=dict)
    headers: MockHeaders = field(default_factory=MockHeaders)
    method: str = "GET"
    path: str = "/"

    def json(self) -> dict:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


# -----------------------------------------------------------------------------
# State fixture
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_state() -> State:
    """Create a test state container."""
    return State()


@pytest.fixture
def global_dependencies(test_state: State) -> Generator[dict, None, None]:
    """Setup global dependencies for tests."""
    yield {"state": test_state}
    test_state.clear()


@pytest.fixture
def make_mock_request(global_dependencies):
    """Factory fixture to create mock requests."""

    def _make(body: dict | None = None) -> MockRequest:
        return MockRequest(_body=body or {})

    return _make
