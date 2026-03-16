"""Session-scoped fixtures for integration tests."""

import httpx
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI options for integration tests."""
    parser.addoption("--base-url", default="http://localhost:5500", help="Base URL for the e-voice API")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark all integration tests as slow."""
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def base_url(request: pytest.FixtureRequest) -> str:
    """Base URL for the running e-voice service."""
    return request.config.getoption("--base-url")


@pytest.fixture(scope="session")
async def http_client(base_url: str) -> httpx.AsyncClient:
    """Shared async HTTP client for REST endpoints."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client
