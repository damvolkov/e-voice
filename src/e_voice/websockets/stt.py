"""WebSocket /v1/audio/transcriptions — streaming STT with binary frame support."""

import base64

import orjson

from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.websocket import Connection, WebSocketRouter
from e_voice.models.ws import STTParams
from e_voice.streaming.stt.transcriber import (
    SessionState,
    StreamingEvent,
    StreamingEventType,
    flush_segment,
    flush_session,
    process_audio_chunk,
)

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


@router("/v1/audio/transcriptions", "/v1/stt/ws", params=STTParams)
async def handle_stt(conn: Connection[STTParams]) -> None:
    """Stream audio -> transcription. Accepts binary PCM16-LE or base64 text frames."""
    session = SessionState(
        language=conn.params.language,
        model_id=conn.params.model,
        response_format=conn.params.response_format,
        segmentation=conn.params.segmentation,
    )
    conn.state.stt_sessions[conn.id] = session
    await conn.state.stt_connections.add(conn.id)
    logger.info(
        "stt stream started",
        step="WS",
        client=conn.id,
        lang=conn.params.language or "auto",
        fmt=conn.params.response_format,
        seg=conn.params.segmentation,
    )

    try:
        async for msg in conn:
            match msg:
                case str() if msg.strip() == "END_OF_AUDIO":
                    if session.vad is not None:
                        seg_event = flush_segment(session)
                        if seg_event.confirmed_text:
                            logger.info(seg_event.confirmed_text, step="STT", segment=True)
                            await conn.send(format_event(seg_event, session.response_format))
                    event = flush_session(session)
                    if event.new_confirmed:
                        logger.info(event.new_confirmed, step="STT", final=True)
                    await conn.send(format_event(event, session.response_format))
                    continue
                case bytes():
                    samples = Audio.pcm16_to_float32(msg)
                case str():
                    samples = Audio.pcm16_to_float32(base64.b64decode(msg))

            event = await process_audio_chunk(session, conn.state.stt, samples)

            if event is not None:
                if event.new_confirmed:
                    logger.info(event.new_confirmed, step="STT", lang=session.language or "auto")
                await conn.send(format_event(event, session.response_format))

            if session.vad is not None and session.vad.update(samples):
                seg_event = flush_segment(session)
                if seg_event.confirmed_text:
                    logger.info(seg_event.confirmed_text, step="STT", segment=True)
                    await conn.send(format_event(seg_event, session.response_format))
                continue

            if event is None:
                ack = StreamingEvent(
                    type=StreamingEventType.TRANSCRIPT_UPDATE,
                    confirmed_text=session.segment_text if session.vad is not None else session.confirmed.text,
                    unconfirmed_text=session.agreement.unconfirmed_text,
                    new_confirmed="",
                )
                await conn.send(format_event(ack, session.response_format))

    except Exception as exc:
        logger.error("streaming transcription failed", step="WS", error=str(exc))
        await conn.send(orjson.dumps({"error": str(exc)}).decode())
    finally:
        await conn.state.stt_connections.remove(conn.id)
        if (session := conn.state.stt_sessions.pop(conn.id, None)) is not None:
            event = flush_session(session)
            if event.new_confirmed:
                logger.info(event.new_confirmed, step="STT", final=True)
