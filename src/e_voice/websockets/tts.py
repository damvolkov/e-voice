"""WebSocket /v1/audio/speech — streaming TTS."""

import orjson

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.websocket import Connection, WebSocketRouter
from e_voice.models.tts import SynthesisParams, resolve_voice_lang

router = WebSocketRouter()


@router("/v1/audio/speech", "/v1/tts/ws")
async def handle_tts(conn: Connection) -> None:
    """Receive JSON request, stream back base64 PCM16 audio chunks."""
    async for msg in conn:
        if not isinstance(msg, str) or not msg.strip():
            continue

        try:
            payload = orjson.loads(msg)
        except orjson.JSONDecodeError:
            await conn.send(orjson.dumps({"error": "Invalid JSON"}).decode())
            continue

        text = payload.get("input", "")
        if not text:
            await conn.send(orjson.dumps({"error": "Empty input"}).decode())
            continue

        voice = payload.get("voice", "af_heart")
        try:
            voice_lang = resolve_voice_lang(voice)
        except ValueError as e:
            await conn.send(orjson.dumps({"error": str(e)}).decode())
            continue

        explicit_lang = payload.get("lang")
        if explicit_lang and explicit_lang != voice_lang:
            await conn.send(
                orjson.dumps(
                    {
                        "error": f"Language '{explicit_lang}' conflicts with voice '{voice}' (expected '{voice_lang}').",
                    }
                ).decode()
            )
            continue

        kokoro: KokoroAdapter = conn.state.kokoro
        params = SynthesisParams(voice=voice, speed=float(payload.get("speed", 1.0)), lang=explicit_lang or voice_lang)
        logger.info(
            "speech request", step="WS", client=conn.id, voice=params.voice, lang=params.lang, text_len=len(text)
        )

        async for samples, _sr in kokoro.synthesize_stream(text, params=params):
            await conn.send(
                orjson.dumps(
                    {
                        "type": "speech.audio.delta",
                        "audio": Audio.float32_to_base64_pcm16(samples),
                    }
                ).decode()
            )

        await conn.send(orjson.dumps({"type": "speech.audio.done"}).decode())
