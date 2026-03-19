import base64

import orjson
from pytest_audioeval.tts import TTSClient

##### WS /v1/audio/speech #####


async def test_ws_speech_returns_audio_deltas(
    tts_ws: TTSClient,
) -> None:
    events: list[dict] = []

    async with tts_ws.ws() as ws:
        await ws.send_text(orjson.dumps({"input": "Hello world.", "voice": "af_heart"}).decode())

        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
            except Exception:
                break
            if not text.strip():
                continue
            body = orjson.loads(text)
            events.append(body)
            if body.get("type") == "speech.audio.done":
                break

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert len(delta_events) > 0

    audio_bytes = base64.b64decode(delta_events[0]["audio"])
    assert len(audio_bytes) > 0
    assert len(audio_bytes) % 2 == 0


async def test_ws_speech_done_event(
    tts_ws: TTSClient,
) -> None:
    events: list[dict] = []

    async with tts_ws.ws() as ws:
        await ws.send_text(orjson.dumps({"input": "Done event test.", "voice": "af_heart"}).decode())

        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
            except Exception:
                break
            if not text.strip():
                continue
            events.append(orjson.loads(text))
            if events[-1].get("type") == "speech.audio.done":
                break

    assert len(events) >= 2
    assert events[-1]["type"] == "speech.audio.done"


async def test_ws_speech_custom_voice_and_speed(
    tts_ws: TTSClient,
) -> None:
    events: list[dict] = []

    async with tts_ws.ws() as ws:
        await ws.send_text(
            orjson.dumps(
                {
                    "input": "Speed test.",
                    "voice": "af_heart",
                    "speed": 1.5,
                }
            ).decode()
        )

        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
            except Exception:
                break
            if not text.strip():
                continue
            body = orjson.loads(text)
            events.append(body)
            if body.get("type") == "speech.audio.done":
                break

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert len(delta_events) > 0


async def test_ws_speech_empty_input_returns_error(
    tts_ws: TTSClient,
) -> None:
    async with tts_ws.ws() as ws:
        await ws.send_text(orjson.dumps({"input": "", "voice": "af_heart"}).decode())

        while not (text := await ws.receive_text(timeout=5.0)).strip():
            pass
        body = orjson.loads(text)

    assert "error" in body
    assert body["error"] == "Empty input"


async def test_ws_speech_invalid_json_returns_error(
    tts_ws: TTSClient,
) -> None:
    async with tts_ws.ws() as ws:
        await ws.send_text("this is not json {{{")

        while not (text := await ws.receive_text(timeout=5.0)).strip():
            pass
        body = orjson.loads(text)

    assert "error" in body
    assert "Invalid JSON" in body["error"]


async def test_ws_speech_binary_message_ignored(
    tts_ws: TTSClient,
) -> None:
    events: list[dict] = []

    async with tts_ws.ws() as ws:
        await ws.send_bytes(b"\x00\x01\x02\x03")

        await ws.send_text(orjson.dumps({"input": "After binary.", "voice": "af_heart"}).decode())

        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
            except Exception:
                break
            if not text.strip():
                continue
            body = orjson.loads(text)
            events.append(body)
            if body.get("type") == "speech.audio.done":
                break

    delta_events = [e for e in events if e.get("type") == "speech.audio.delta"]
    assert len(delta_events) > 0


async def test_ws_speech_combined_audio_non_trivial(
    tts_ws: TTSClient,
) -> None:
    total_audio_bytes = 0

    async with tts_ws.ws() as ws:
        await ws.send_text(
            orjson.dumps(
                {
                    "input": "The quick brown fox jumps over the lazy dog.",
                    "voice": "af_heart",
                }
            ).decode()
        )

        while True:
            try:
                text = await ws.receive_text(timeout=10.0)
            except Exception:
                break
            if not text.strip():
                continue
            body = orjson.loads(text)
            if body.get("type") == "speech.audio.delta":
                total_audio_bytes += len(base64.b64decode(body["audio"]))
            elif body.get("type") == "speech.audio.done":
                break

    assert total_audio_bytes > 1000
