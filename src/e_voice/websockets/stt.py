"""WebSocket /v1/audio/transcriptions — real-time streaming STT."""

import orjson
from robyn import Robyn, WebSocket
from robyn.robyn import WebSocketConnector

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.helpers import audio_samples_from_file
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.models.transcription import ResponseFormat, TranscriptionResponse


def create_ws_stt(app: Robyn) -> WebSocket:
    """Create WebSocket endpoint for real-time transcription."""
    ws = WebSocket(app, "/v1/audio/transcriptions")

    @ws.on("connect")
    def on_connect(ws: WebSocketConnector) -> str:
        logger.info("connected", step="WS", client=ws.id)
        return ""

    @ws.on("message")
    async def on_message(ws: WebSocketConnector, msg: str, global_dependencies) -> str:
        """Receive audio bytes, transcribe, send back result."""
        whisper: WhisperAdapter = global_dependencies.get("state").whisper

        model_id = st.WHISPER_MODEL
        language = st.DEFAULT_LANGUAGE
        response_format = ResponseFormat(st.DEFAULT_RESPONSE_FORMAT)
        temperature = 0.0
        vad_filter = True

        try:
            audio_data = audio_samples_from_file(msg.encode("latin-1") if isinstance(msg, str) else msg)
        except Exception as exc:
            return orjson.dumps({"error": f"Audio decode failed: {exc}"}).decode()

        logger.info("transcription request", step="WS", client=ws.id, audio_len=len(audio_data))

        try:
            segments, info = await whisper.transcribe(
                audio_data,
                model_id=model_id,
                language=language,
                temperature=temperature,
                vad_filter=vad_filter,
            )

            match response_format:
                case ResponseFormat.TEXT:
                    return WhisperAdapter.segments_to_text(segments)
                case ResponseFormat.JSON:
                    return TranscriptionResponse(text=WhisperAdapter.segments_to_text(segments)).model_dump_json()
                case ResponseFormat.VERBOSE_JSON:
                    seg_models = [WhisperAdapter.segment_to_model(seg) for seg in segments]
                    return orjson.dumps(
                        {
                            "task": "transcribe",
                            "language": info.language,
                            "duration": info.duration,
                            "text": WhisperAdapter.segments_to_text(segments),
                            "segments": [s.model_dump() for s in seg_models],
                        }
                    ).decode()
                case _:
                    return WhisperAdapter.segments_to_text(segments)

        except Exception as exc:
            logger.error("transcription error", error=str(exc), client=ws.id)
            return orjson.dumps({"error": str(exc)}).decode()

    @ws.on("close")
    def on_close(ws: WebSocketConnector) -> str:
        logger.info("disconnected", step="WS", client=ws.id)
        return ""

    return ws
