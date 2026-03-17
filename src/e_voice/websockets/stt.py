"""WebSocket /v1/audio/transcriptions — streaming STT with LocalAgreement."""

import base64

import orjson
from robyn.robyn import WebSocketConnector

from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.core.websocket import BaseWebSocket
from e_voice.streaming.transcriber import (
    SessionState,
    StreamingEvent,
    flush_session,
    process_audio_chunk,
)

ws_stt = BaseWebSocket("/v1/audio/transcriptions")
ws_stt_alias = BaseWebSocket("/v1/stt/ws")


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


@ws_stt.on("connect")
def on_connect(ws: WebSocketConnector, global_dependencies) -> str:
    lang = ws.query_params.get("language", None) or st.stt.default_language
    fmt = ws.query_params.get("response_format", None) or st.stt.default_response_format.value
    model = ws.query_params.get("model", None) or st.stt.model

    sessions: dict[str, SessionState] = global_dependencies.get("state").stt_sessions
    sessions[ws.id] = SessionState(
        language=lang if lang != "auto" else None,
        model_id=model,
        response_format=fmt,
    )

    logger.info("stt stream started", step="WS", client=ws.id, lang=lang or "auto", fmt=fmt)
    return ""


@ws_stt.on("message")
async def on_message(ws: WebSocketConnector, msg: str, global_dependencies) -> str:
    """Receive base64 PCM16 chunk, process through streaming pipeline."""
    try:
        state = global_dependencies.get("state")
        if (session := state.stt_sessions.get(ws.id)) is None:
            return orjson.dumps({"error": "no session"}).decode()

        whisper = state.whisper

        raw_bytes = base64.b64decode(msg)
        audio_samples = Audio.pcm16_to_float32(raw_bytes)

        if (event := await process_audio_chunk(session, whisper, audio_samples)) is None:
            return ""

        if event.new_confirmed:
            logger.info(event.new_confirmed, step="STT", lang=session.language or "auto")

        return format_event(event, session.response_format)

    except Exception as exc:
        logger.error("streaming transcription failed", step="WS", error=str(exc))
        return orjson.dumps({"error": str(exc)}).decode()


@ws_stt.on("close")
def on_close(ws: WebSocketConnector, global_dependencies) -> str:
    sessions: dict[str, SessionState] = global_dependencies.get("state").stt_sessions
    if (session := sessions.pop(ws.id, None)) is not None:
        event = flush_session(session)
        if event.new_confirmed:
            logger.info(event.new_confirmed, step="STT", final=True)
    logger.info("stt stream ended", step="WS", client=ws.id)
    return ""


ws_stt_alias._handlers = ws_stt._handlers
