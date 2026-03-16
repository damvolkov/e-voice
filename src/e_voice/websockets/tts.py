"""WebSocket /v1/audio/speech — real-time streaming TTS."""

import orjson
from robyn.robyn import WebSocketConnector

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.core.helpers import float32_to_base64_pcm16
from e_voice.core.logger import logger
from e_voice.core.websocket import BaseWebSocket

ws_tts = BaseWebSocket("/v1/audio/speech")


@ws_tts.on("connect")
def on_connect(ws: WebSocketConnector) -> str:
    logger.info("tts connected", step="WS", client=ws.id)
    return ""


@ws_tts.on("message")
async def on_message(ws: WebSocketConnector, msg: str, global_dependencies) -> str:
    """Receive JSON text request, stream back base64 PCM16 audio chunks."""
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro

    payload = orjson.loads(msg)
    text = payload.get("input", "")
    voice = payload.get("voice", "af_heart")
    speed = float(payload.get("speed", 1.0))
    lang = payload.get("lang")

    if not text:
        return orjson.dumps({"error": "Empty input"}).decode()

    logger.info("speech request", step="WS", client=ws.id, voice=voice, text_len=len(text))

    async for samples, _sr in kokoro.synthesize_stream(text, voice=voice, speed=speed, lang=lang):
        chunk = orjson.dumps(
            {
                "type": "speech.audio.delta",
                "audio": float32_to_base64_pcm16(samples),
            }
        ).decode()
        await ws.async_send_to(ws.id, chunk)

    done = orjson.dumps({"type": "speech.audio.done"}).decode()
    await ws.async_send_to(ws.id, done)

    return ""


@ws_tts.on("close")
def on_close(ws: WebSocketConnector) -> str:
    logger.info("tts disconnected", step="WS", client=ws.id)
    return ""
