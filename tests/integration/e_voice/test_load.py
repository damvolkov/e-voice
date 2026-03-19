"""Load tests — sustained WS TTS connections, concurrent HTTP, mixed workloads."""

import asyncio
import base64
from collections import Counter

import httpx
import orjson
import pytest
import websockets

##### CONSTANTS #####

_SHORT_TEXT = "Load test."
_MEDIUM_TEXT = "The quick brown fox jumps over the lazy dog."

_WS_TTS_CASES: list[tuple[int, str]] = [
    (50, "sequential-50"),
    (100, "sequential-100"),
    (150, "sequential-150"),
]

_WS_TTS_CONCURRENT_CASES: list[tuple[int, int, str]] = [
    (5, 10, "5x10-burst"),
    (10, 5, "10x5-burst"),
    (3, 20, "3x20-burst"),
]

_HTTP_TTS_CASES: list[tuple[int, str]] = [
    (50, "http-50"),
    (100, "http-100"),
]

_MIXED_CASES: list[tuple[int, int, str]] = [
    (25, 25, "mixed-25ws-25http"),
    (50, 50, "mixed-50ws-50http"),
]


##### HELPERS #####


async def _ws_tts_single(ws_url: str, text: str, timeout: float = 30.0) -> str:
    """One WS TTS round-trip. Returns 'ok' or error description."""
    try:
        async with websockets.connect(ws_url, open_timeout=5, close_timeout=5) as ws:
            await ws.send(orjson.dumps({"input": text, "voice": "af_heart"}).decode())
            audio_bytes = 0
            async for msg in ws:
                if not msg or not msg.strip():
                    continue
                body = orjson.loads(msg)
                if body.get("type") == "speech.audio.delta":
                    audio_bytes += len(base64.b64decode(body["audio"]))
                elif body.get("type") == "speech.audio.done":
                    break
                elif "error" in body:
                    return f"server_error:{body['error']}"
            return "ok" if audio_bytes > 0 else "empty_audio"
    except websockets.exceptions.ConnectionClosedError as e:
        return f"ws_closed:{e.code}"
    except TimeoutError:
        return "timeout"
    except OSError as e:
        return f"os_error:{e}"
    except Exception as e:
        return f"{type(e).__name__}:{e}"


async def _http_tts_single(client: httpx.AsyncClient, text: str) -> str:
    """One HTTP TTS request. Returns 'ok' or error description."""
    try:
        resp = await client.post(
            "/v1/audio/speech",
            json={"input": text, "voice": "af_heart", "response_format": "pcm", "stream": False},
            timeout=30.0,
        )
        if resp.status_code != 200:
            return f"http_{resp.status_code}"
        return "ok" if len(resp.content) > 0 else "empty"
    except Exception as e:
        return f"{type(e).__name__}:{e}"


async def _health_ok(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.get(f"{base_url}/health", timeout=5.0)
            return resp.status_code == 200
    except Exception:
        return False


def _assert_results(results: list[str], min_pass_ratio: float = 1.0) -> None:
    counts = Counter(results)
    total = len(results)
    passed = counts.get("ok", 0)
    ratio = passed / total if total else 0
    summary = ", ".join(f"{k}={v}" for k, v in counts.most_common())
    assert ratio >= min_pass_ratio, f"{passed}/{total} passed ({ratio:.0%}), need {min_pass_ratio:.0%} — {summary}"


##### WS TTS SEQUENTIAL #####


@pytest.mark.parametrize(
    ("count", "label"),
    _WS_TTS_CASES,
    ids=[c[1] for c in _WS_TTS_CASES],
)
async def test_ws_tts_sequential(base_url: str, count: int, label: str) -> None:
    """Sequential WS TTS connections — detect resource leaks and connection exhaustion."""
    ws_url = base_url.replace("http://", "ws://") + "/v1/audio/speech"
    results: list[str] = []

    for i in range(count):
        result = await _ws_tts_single(ws_url, _SHORT_TEXT)
        results.append(result)
        if result != "ok" and not await _health_ok(base_url):
            results.extend(["server_down"] * (count - i - 1))
            break

    _assert_results(results)


##### WS TTS CONCURRENT BURSTS #####


@pytest.mark.parametrize(
    ("bursts", "per_burst", "label"),
    _WS_TTS_CONCURRENT_CASES,
    ids=[c[2] for c in _WS_TTS_CONCURRENT_CASES],
)
async def test_ws_tts_concurrent_bursts(base_url: str, bursts: int, per_burst: int, label: str) -> None:
    """Bursts of concurrent WS TTS connections — detect thread/connection pool exhaustion."""
    ws_url = base_url.replace("http://", "ws://") + "/v1/audio/speech"
    results: list[str] = []

    for _ in range(bursts):
        batch = await asyncio.gather(
            *[_ws_tts_single(ws_url, _SHORT_TEXT) for _ in range(per_burst)],
            return_exceptions=True,
        )
        for r in batch:
            results.append(str(r) if isinstance(r, Exception) else r)
        if not await _health_ok(base_url):
            results.append("server_down")
            break
        await asyncio.sleep(0.5)

    _assert_results(results)


##### HTTP TTS SEQUENTIAL #####


@pytest.mark.parametrize(
    ("count", "label"),
    _HTTP_TTS_CASES,
    ids=[c[1] for c in _HTTP_TTS_CASES],
)
async def test_http_tts_sequential(http_client: httpx.AsyncClient, base_url: str, count: int, label: str) -> None:
    """Sequential HTTP TTS requests — baseline comparison for WS load."""
    results: list[str] = []

    for i in range(count):
        result = await _http_tts_single(http_client, _SHORT_TEXT)
        results.append(result)
        if result != "ok" and not await _health_ok(base_url):
            results.extend(["server_down"] * (count - i - 1))
            break

    _assert_results(results)


##### MIXED WS + HTTP #####


@pytest.mark.parametrize(
    ("ws_count", "http_count", "label"),
    _MIXED_CASES,
    ids=[c[2] for c in _MIXED_CASES],
)
async def test_mixed_ws_and_http(
    http_client: httpx.AsyncClient,
    base_url: str,
    ws_count: int,
    http_count: int,
    label: str,
) -> None:
    """Interleaved WS and HTTP TTS — detect cross-transport resource contention."""
    ws_url = base_url.replace("http://", "ws://") + "/v1/audio/speech"
    results: list[str] = []

    for i in range(max(ws_count, http_count)):
        if i < ws_count:
            results.append(await _ws_tts_single(ws_url, _SHORT_TEXT))
        if i < http_count:
            results.append(await _http_tts_single(http_client, _SHORT_TEXT))
        if results[-1] != "ok" and not await _health_ok(base_url):
            results.append("server_down")
            break

    _assert_results(results)


##### RAPID CONNECT/DISCONNECT #####


async def test_ws_rapid_connect_disconnect(base_url: str) -> None:
    """100 rapid WS open+close without sending — detect connection leak."""
    ws_url = base_url.replace("http://", "ws://") + "/v1/audio/speech"
    results: list[str] = []

    for _ in range(100):
        try:
            async with websockets.connect(ws_url, open_timeout=5, close_timeout=2):
                pass
            results.append("ok")
        except Exception as e:
            results.append(f"{type(e).__name__}")

    assert await _health_ok(base_url), "Server down after rapid connect/disconnect"
    _assert_results(results)


##### SUSTAINED LONG TEXT #####


async def test_ws_tts_sustained_long_text(base_url: str) -> None:
    """30 sequential WS TTS with longer text — detect memory/buffer leaks."""
    ws_url = base_url.replace("http://", "ws://") + "/v1/audio/speech"
    results: list[str] = []

    for _ in range(30):
        result = await _ws_tts_single(ws_url, _MEDIUM_TEXT)
        results.append(result)

    assert await _health_ok(base_url), "Server down after sustained long text"
    _assert_results(results)
