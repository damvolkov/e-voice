"""OpenAI-compatible Speech-to-Text API — transcriptions, translations, models."""

from urllib.parse import unquote  # noqa: F401

from robyn import Request, Response, SSEMessage, SSEResponse, status_codes
from robyn.types import Files, FormData

from e_voice.adapters.base import STTBackend
from e_voice.core.audio import Audio
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.core.settings import settings as st
from e_voice.models.error import BackendCapabilityError, error_response
from e_voice.models.stt import InferenceParams
from e_voice.models.transcription import (
    ListModelsResponse,
    ModelObject,
    TimestampGranularity,
    TranscriptionParams,
    TranslationParams,
    build_transcript_response,
    format_span,
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
    """Transcribe audio to text (OpenAI-compatible). Supports true SSE streaming."""
    if not (file_bytes := next(iter(files.values()), None) if files else None):
        return error_response(request.url.path, 422, "No audio file provided")

    stt: STTBackend = global_dependencies.get("state").stt

    raw = {k: v for k in _TRANSCRIPTION_FIELDS if (v := form_data.get(k)) is not None}
    if (tg := form_data.get("timestamp_granularities[]")) is not None:
        raw["timestamp_granularities"] = [tg] if isinstance(tg, str) else tg
    params = TranscriptionParams.model_validate(raw)

    logger.info("transcription request", step="STT", model=params.model or st.stt.model, stream=params.stream)

    audio_data = Audio.samples_from_file(file_bytes)
    inference = InferenceParams(
        language=params.language,
        prompt=params.prompt,
        temperature=params.temperature,
        word_timestamps=TimestampGranularity.WORD in params.timestamp_granularities,
        vad_filter=params.vad_filter,
        hotwords=params.hotwords,
    )
    fmt = params.response_format.value

    if params.stream:

        async def sse_generator():
            idx = 0
            async for span in stt.transcribe_stream(audio_data, params=inference):
                yield SSEMessage(data=format_span(span, fmt, idx))
                idx += 1
            yield SSEMessage(data="[DONE]")

        return SSEResponse(sse_generator())

    transcript = await stt.transcribe(audio_data, params=inference)
    body, content_type = build_transcript_response(transcript, fmt, word_timestamps=inference.word_timestamps)

    logger.info("transcription complete", step="STT")
    return Response(status_code=status_codes.HTTP_200_OK, headers={"content-type": content_type}, description=body)


##### POST /v1/audio/translations #####


@router.post("/audio/translations")
async def translations(request: Request, form_data: FormData, files: Files, global_dependencies):
    """Translate audio to English text (OpenAI-compatible). Supports true SSE streaming."""
    if not (file_bytes := next(iter(files.values()), None) if files else None):
        return error_response(request.url.path, 422, "No audio file provided")

    stt: STTBackend = global_dependencies.get("state").stt

    raw = {k: v for k in _TRANSLATION_FIELDS if (v := form_data.get(k)) is not None}
    params = TranslationParams.model_validate(raw)

    logger.info("translation request", step="STT", model=params.model or st.stt.model, stream=params.stream)

    audio_data = Audio.samples_from_file(file_bytes)
    inference = InferenceParams(
        prompt=params.prompt,
        temperature=params.temperature,
        vad_filter=params.vad_filter,
    )
    fmt = params.response_format.value

    try:
        if params.stream:

            async def sse_generator():
                idx = 0
                async for span in stt.translate_stream(audio_data, params=inference):
                    yield SSEMessage(data=format_span(span, fmt, idx))
                    idx += 1
                yield SSEMessage(data="[DONE]")

            return SSEResponse(sse_generator())

        transcript = await stt.translate(audio_data, params=inference)
        body, content_type = build_transcript_response(
            transcript, fmt, word_timestamps=inference.word_timestamps, task="translate"
        )

        logger.info("translation complete", step="STT")
        return Response(status_code=status_codes.HTTP_200_OK, headers={"content-type": content_type}, description=body)

    except BackendCapabilityError:
        return error_response(request.url.path, 501, "Current STT backend does not support translation")


##### TAXONOMICAL ALIASES #####

router.alias("/audio/transcriptions", "/stt/http", "/stt/sse")
router.alias("/audio/translations", "/stt/translate")


##### GET /v1/models #####


@router.get("/models")
async def list_models(global_dependencies) -> ListModelsResponse:
    """List all loaded models (OpenAI-compatible)."""
    stt: STTBackend = global_dependencies.get("state").stt
    return ListModelsResponse(data=[ModelObject(id=spec.model_id) for spec in stt.loaded_models()])


@router.get("/models/:model_id")
async def get_model(request: Request, model_id: str, global_dependencies) -> Response:
    """Get a specific model info."""
    model_id = unquote(model_id)
    stt: STTBackend = global_dependencies.get("state").stt
    if not any(spec.model_id == model_id for spec in stt.loaded_models()):
        return error_response(request.url.path, 404, f"Model '{model_id}' not loaded", error_type="not_found_error")
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=ModelObject(id=model_id).model_dump_json(),
    )
