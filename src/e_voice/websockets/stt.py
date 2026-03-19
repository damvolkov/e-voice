"""WebSocket /v1/audio/transcriptions — streaming STT with binary frame support."""

import base64

import orjson

from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.core.websocket import Connection, WebSocketRouter
from e_voice.streaming.transcriber import SessionState, StreamingEvent, flush_session, process_audio_chunk

router = WebSocketRouter()


def format_event(event: StreamingEvent, response_format: str) -> str:
    """Format streaming event based on response_format."""
    match response_format:
        case "text":
            return event.confirmed_text
        case _:
            return orjson.dumps(
                {
                    "type": event.type.value,
                    "text": event.confirmed_text,
                    "partial": event.unconfirmed_text,
                    "is_final": event.is_final,
                }
            ).decode()


@router("/v1/audio/transcriptions", "/v1/stt/ws")
async def handle_stt(conn: Connection) -> None:
    """Stream audio → transcription. Accepts binary PCM16-LE or base64 text frames."""
    lang = conn.query_params.get("language") or st.stt.default_language
    fmt = conn.query_params.get("response_format") or st.stt.default_response_format.value
    model = conn.query_params.get("model") or st.stt.model

    session = SessionState(
        language=lang if lang != "auto" else None,
        model_id=model,
        response_format=fmt,
    )
    conn.state.stt_sessions[conn.id] = session
    logger.info("stt stream started", step="WS", client=conn.id, lang=lang or "auto", fmt=fmt)

    try:
        async for msg in conn:
            match msg:
                case str() if msg.strip() == "END_OF_AUDIO":
                    event = flush_session(session)
                    if event.new_confirmed:
                        logger.info(event.new_confirmed, step="STT", final=True)
                    await conn.send(format_event(event, session.response_format))
                    continue
                case bytes():
                    samples = Audio.pcm16_to_float32(msg)
                case str():
                    samples = Audio.pcm16_to_float32(base64.b64decode(msg))

            if (event := await process_audio_chunk(session, conn.state.whisper, samples)) is None:
                continue

            if event.new_confirmed:
                logger.info(event.new_confirmed, step="STT", lang=session.language or "auto")

            await conn.send(format_event(event, session.response_format))

    except Exception as exc:
        logger.error("streaming transcription failed", step="WS", error=str(exc))
        await conn.send(orjson.dumps({"error": str(exc)}).decode())
    finally:
        if (session := conn.state.stt_sessions.pop(conn.id, None)) is not None:
            event = flush_session(session)
            if event.new_confirmed:
                logger.info(event.new_confirmed, step="STT", final=True)
