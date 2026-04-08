"""WebSocket /v1/audio/speech — streaming TTS."""

import orjson
from pydantic import ValidationError

from e_voice.adapters.base import TTSBackend
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.websocket import Connection, WebSocketRouter
from e_voice.models.tts import SynthesisParams
from e_voice.models.ws import TTSParams, WSSpeechRequest

router = WebSocketRouter()


@router("/v1/audio/speech", "/v1/tts/ws", params=TTSParams)
async def handle_tts(conn: Connection[TTSParams]) -> None:
    """Receive JSON request, stream back base64 PCM16 audio chunks."""
    await conn.state.tts_connections.add(conn.id)
    try:
        await _handle_tts_messages(conn)
    finally:
        await conn.state.tts_connections.remove(conn.id)


async def _handle_tts_messages(conn: Connection[TTSParams]) -> None:
    """Process TTS messages for a connection."""
    async for msg in conn:
        if not isinstance(msg, str) or not msg.strip():
            continue

        try:
            request = WSSpeechRequest.model_validate_json(msg)
        except ValidationError as exc:
            error_msg = exc.errors()[0].get("msg", "Invalid request")
            await conn.send(orjson.dumps({"error": error_msg}).decode())
            continue

        tts: TTSBackend = conn.state.tts
        params = SynthesisParams(voice=request.voice, speed=request.speed, lang=request.lang)
        logger.info(
            "speech request",
            step="WS",
            client=conn.id,
            voice=params.voice,
            lang=params.lang,
            text_len=len(request.input),
        )

        async for samples, _sr in tts.synthesize_stream(request.input, params=params):
            await conn.send(
                orjson.dumps(
                    {
                        "type": "speech.audio.delta",
                        "audio": Audio.float32_to_base64_pcm16(samples),
                    }
                ).decode()
            )

        await conn.send(orjson.dumps({"type": "speech.audio.done"}).decode())
