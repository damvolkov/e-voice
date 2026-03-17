import json
from collections.abc import Generator
from dataclasses import dataclass, field

import pytest

from e_voice.core.lifespan import State


@dataclass
class MockHeaders:
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
    _body: dict | str = field(default_factory=dict)
    headers: MockHeaders = field(default_factory=MockHeaders)
    method: str = "GET"
    path: str = "/"

    def json(self) -> dict:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


@pytest.fixture(scope="session")
def test_state() -> State:
    return State()


@pytest.fixture
def global_dependencies(test_state: State) -> Generator[dict, None, None]:
    yield {"state": test_state}
    test_state.clear()


@pytest.fixture
def make_mock_request(global_dependencies):
    def _make(body: dict | None = None) -> MockRequest:
        return MockRequest(_body=body or {})

    return _make
