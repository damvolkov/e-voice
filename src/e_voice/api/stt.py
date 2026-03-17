"""OpenAI-compatible Speech-to-Text API — transcriptions, translations, models."""

import orjson
from robyn import Request, Response, SSEMessage, SSEResponse, status_codes
from robyn.types import Files, FormData

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.core.settings import settings as st
from e_voice.models.transcription import (
    ListModelsResponse,
    ModelObject,
    TimestampGranularity,
    TranscriptionParams,
    TranslationParams,
)

router = Router(__file__, prefix="/v1")

_TRANSCRIPTION_FIELDS = (
    "model",
    "language",
    "prompt",
    "response_format",
    "temperature",
    "stream",
    "hotwords",
    "vad_filter",
)
_TRANSLATION_FIELDS = ("model", "prompt", "response_format", "temperature", "stream", "vad_filter")


##### POST /v1/audio/transcriptions #####


@router.post("/audio/transcriptions")
async def transcriptions(request: Request, form_data: FormData, files: Files, global_dependencies):
    """Transcribe audio to text (OpenAI-compatible). Supports SSE streaming."""
    if not (file_bytes := next(iter(files.values()), None) if files else None):
        return Response(
            status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": "No audio file provided"}).decode(),
        )

    whisper: WhisperAdapter = global_dependencies.get("state").whisper

    raw = {k: v for k in _TRANSCRIPTION_FIELDS if (v := form_data.get(k)) is not None}
    if (tg := form_data.get("timestamp_granularities[]")) is not None:
        raw["timestamp_granularities"] = [tg] if isinstance(tg, str) else tg
    params = TranscriptionParams(**raw)

    model_id = params.model or st.stt.model
    word_timestamps = TimestampGranularity.WORD in params.timestamp_granularities

    logger.info("transcription request", step="STT", model=model_id, language=params.language, stream=params.stream)

    audio_data = Audio.samples_from_file(file_bytes)

    segments, info = await whisper.transcribe(
        audio_data,
        model_id=model_id,
        language=params.language,
        prompt=params.prompt,
        temperature=params.temperature,
        word_timestamps=word_timestamps,
        vad_filter=params.vad_filter,
        hotwords=params.hotwords,
    )

    if params.stream:

        def sse_generator():
            for seg in segments:
                yield SSEMessage(data=WhisperAdapter.format_segment_for_streaming(seg, params.response_format))
            yield SSEMessage(data="[DONE]")

        return SSEResponse(sse_generator())

    body, content_type = WhisperAdapter.build_response(
        segments, info, audio_data, params.response_format, word_timestamps
    )
    logger.info("transcription complete", step="STT", segments=len(segments))
    return Response(status_code=status_codes.HTTP_200_OK, headers={"content-type": content_type}, description=body)


##### POST /v1/audio/translations #####


@router.post("/audio/translations")
async def translations(request: Request, form_data: FormData, files: Files, global_dependencies):
    """Translate audio to English text (OpenAI-compatible). Supports SSE streaming."""
    if not (file_bytes := next(iter(files.values()), None) if files else None):
        return Response(
            status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": "No audio file provided"}).decode(),
        )

    whisper: WhisperAdapter = global_dependencies.get("state").whisper

    raw = {k: v for k in _TRANSLATION_FIELDS if (v := form_data.get(k)) is not None}
    params = TranslationParams(**raw)

    model_id = params.model or st.stt.model

    logger.info("translation request", step="STT", model=model_id, stream=params.stream)

    audio_data = Audio.samples_from_file(file_bytes)

    segments, info = await whisper.translate(
        audio_data,
        model_id=model_id,
        prompt=params.prompt,
        temperature=params.temperature,
        vad_filter=params.vad_filter,
    )

    if params.stream:

        def sse_generator():
            for seg in segments:
                yield SSEMessage(data=WhisperAdapter.format_segment_for_streaming(seg, params.response_format))
            yield SSEMessage(data="[DONE]")

        return SSEResponse(sse_generator())

    body, content_type = WhisperAdapter.build_response(
        segments, info, audio_data, params.response_format, task="translate"
    )
    logger.info("translation complete", step="STT", segments=len(segments))
    return Response(status_code=status_codes.HTTP_200_OK, headers={"content-type": content_type}, description=body)


##### GET /v1/models #####


@router.get("/models")
async def list_models(global_dependencies) -> ListModelsResponse:
    """List all loaded models (OpenAI-compatible)."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    return ListModelsResponse(data=[ModelObject(id=mid, owned_by="whisper") for mid in whisper.loaded_models()])


@router.get("/models/:model_id")
async def get_model(model_id: str, global_dependencies) -> Response:
    """Get a specific model info."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    if not await whisper.is_loaded(model_id):
        return Response(
            status_code=status_codes.HTTP_404_NOT_FOUND,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": f"Model '{model_id}' not loaded"}).decode(),
        )
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=ModelObject(id=model_id, owned_by="whisper").model_dump_json(),
    )


##### TAXONOMICAL ALIASES #####

router.alias("/audio/transcriptions", "/stt/http", "/stt/sse")
router.alias("/audio/translations", "/stt/translate")
