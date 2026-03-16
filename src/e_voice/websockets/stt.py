"""WebSocket /v1/audio/transcriptions — real-time streaming STT (raw PCM16)."""

import orjson
import numpy as np
import soundfile as sf
from io import BytesIO
from robyn.robyn import WebSocketConnector

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.core.websocket import BaseWebSocket
from e_voice.models.transcription import ResponseFormat, TranscriptionResponse

SAMPLES_PER_SECOND = 16_000

ws_stt = BaseWebSocket("/v1/audio/transcriptions")


def _pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Decode raw PCM16-LE bytes to float32 samples at 16kHz."""
    audio, _ = sf.read(
        BytesIO(raw),
        format="RAW",
        channels=1,
        samplerate=SAMPLES_PER_SECOND,
        subtype="PCM_16",
        dtype="float32",
        endian="LITTLE",
    )
    return audio


@ws_stt.on("connect")
def on_connect(ws: WebSocketConnector) -> str:
    logger.info("stt connected", step="WS", client=ws.id)
    return ""


@ws_stt.on("message")
async def on_message(ws: WebSocketConnector, msg: str, global_dependencies) -> str:
    """Receive raw PCM16 audio, transcribe, return result."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper

    raw_bytes = msg.encode("latin-1") if isinstance(msg, str) else msg
    audio_data = _pcm16_to_float32(raw_bytes)

    duration = len(audio_data) / SAMPLES_PER_SECOND
    logger.info("transcription request", step="WS", client=ws.id, duration=f"{duration:.1f}s")

    model_id = st.WHISPER_MODEL
    language = st.DEFAULT_LANGUAGE
    response_format = ResponseFormat(st.DEFAULT_RESPONSE_FORMAT)

    segments, info = await whisper.transcribe(
        audio_data,
        model_id=model_id,
        language=language,
        temperature=0.0,
        vad_filter=True,
    )

    logger.info("transcription done", step="WS", client=ws.id, segments=len(segments))

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


@ws_stt.on("close")
def on_close(ws: WebSocketConnector) -> str:
    logger.info("stt disconnected", step="WS", client=ws.id)
    return ""
