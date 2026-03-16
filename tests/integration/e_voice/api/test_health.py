"""Health endpoint tests — GET /health."""

import httpx
import orjson

##### GET /health #####


async def test_health_returns_200(http_client: httpx.AsyncClient) -> None:
    """Health check returns 200 with healthy status."""
    response = await http_client.get("/health")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert body["status"] == "healthy"


async def test_health_response_schema(http_client: httpx.AsyncClient) -> None:
    """Health response contains service name and version."""
    response = await http_client.get("/health")
    assert response.status_code == 200

    body = orjson.loads(response.content)
    assert "service" in body
    assert "version" in body
    assert isinstance(body["service"], str)
    assert isinstance(body["version"], str)
