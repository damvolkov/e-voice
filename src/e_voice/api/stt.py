"""OpenAI-compatible Speech-to-Text API — transcriptions, translations, models."""

import orjson
from robyn import Request, Response, SSEMessage, SSEResponse, status_codes
from robyn.types import Files, FormData

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.helpers import audio_samples_from_file
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


##### FORM PARSING #####


def _parse_transcription_params(form_data: FormData) -> TranscriptionParams:
    """Parse form fields into TranscriptionParams."""
    raw: dict = {}

    if (v := form_data.get("model")) is not None:
        raw["model"] = v
    if (v := form_data.get("language")) is not None:
        raw["language"] = v
    if (v := form_data.get("prompt")) is not None:
        raw["prompt"] = v
    if (v := form_data.get("response_format")) is not None:
        raw["response_format"] = v
    if (v := form_data.get("temperature")) is not None:
        raw["temperature"] = float(v)
    if (v := form_data.get("stream")) is not None:
        raw["stream"] = v.lower() in ("true", "1", "yes")
    if (v := form_data.get("hotwords")) is not None:
        raw["hotwords"] = v
    if (v := form_data.get("vad_filter")) is not None:
        raw["vad_filter"] = v.lower() in ("true", "1", "yes")
    if (v := form_data.get("timestamp_granularities[]")) is not None:
        raw["timestamp_granularities"] = [v] if isinstance(v, str) else v

    return TranscriptionParams(**raw)


def _parse_translation_params(form_data: FormData) -> TranslationParams:
    """Parse form fields into TranslationParams."""
    raw: dict = {}

    if (v := form_data.get("model")) is not None:
        raw["model"] = v
    if (v := form_data.get("prompt")) is not None:
        raw["prompt"] = v
    if (v := form_data.get("response_format")) is not None:
        raw["response_format"] = v
    if (v := form_data.get("temperature")) is not None:
        raw["temperature"] = float(v)
    if (v := form_data.get("stream")) is not None:
        raw["stream"] = v.lower() in ("true", "1", "yes")
    if (v := form_data.get("vad_filter")) is not None:
        raw["vad_filter"] = v.lower() in ("true", "1", "yes")

    return TranslationParams(**raw)


def _require_audio(files: Files) -> bytes | None:
    """Extract first uploaded file or None."""
    return next(iter(files.values()), None) if files else None


##### POST /v1/audio/transcriptions #####


@router.post("/audio/transcriptions")
async def transcriptions(request: Request, form_data: FormData, files: Files, global_dependencies):
    """Transcribe audio to text (OpenAI-compatible). Supports SSE streaming."""
    if not (file_bytes := _require_audio(files)):
        return Response(
            status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": "No audio file provided"}).decode(),
        )

    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    params = _parse_transcription_params(form_data)
    model_id = params.model or st.WHISPER_MODEL
    word_timestamps = TimestampGranularity.WORD in params.timestamp_granularities

    logger.info("transcription request", step="STT", model=model_id, language=params.language, stream=params.stream)

    audio_data = audio_samples_from_file(file_bytes)

    if params.stream:

        async def sse_generator():
            async for segment in whisper.transcribe_stream(
                audio_data,
                model_id=model_id,
                language=params.language,
                prompt=params.prompt,
                temperature=params.temperature,
                vad_filter=params.vad_filter,
                hotwords=params.hotwords,
            ):
                yield SSEMessage(data=WhisperAdapter.format_segment_for_streaming(segment, params.response_format))

        return SSEResponse(sse_generator())

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

    body, content_type = WhisperAdapter.build_response(
        segments, info, audio_data, params.response_format, word_timestamps
    )
    logger.info("transcription complete", step="STT", segments=len(segments))
    return Response(status_code=status_codes.HTTP_200_OK, headers={"content-type": content_type}, description=body)


##### POST /v1/audio/translations #####


@router.post("/audio/translations")
async def translations(request: Request, form_data: FormData, files: Files, global_dependencies):
    """Translate audio to English text (OpenAI-compatible). Supports SSE streaming."""
    if not (file_bytes := _require_audio(files)):
        return Response(
            status_code=status_codes.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": "No audio file provided"}).decode(),
        )

    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    params = _parse_translation_params(form_data)
    model_id = params.model or st.WHISPER_MODEL

    logger.info("translation request", step="STT", model=model_id, stream=params.stream)

    audio_data = audio_samples_from_file(file_bytes)

    segments, info = await whisper.translate(
        audio_data,
        model_id=model_id,
        prompt=params.prompt,
        temperature=params.temperature,
        vad_filter=params.vad_filter,
    )

    if params.stream:

        async def sse_generator():
            for seg in segments:
                yield SSEMessage(data=WhisperAdapter.format_segment_for_streaming(seg, params.response_format))

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


##### EXPERIMENTAL — MODEL MANAGEMENT #####


@router.get("/api/ps")
async def list_loaded(global_dependencies) -> Response:
    """List currently loaded models."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=orjson.dumps({"models": whisper.loaded_models()}).decode(),
    )


@router.post("/api/ps/:model_id")
async def load_model_endpoint(model_id: str, global_dependencies) -> Response:
    """Load a model into memory."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    if await whisper.is_loaded(model_id):
        return Response(
            status_code=status_codes.HTTP_409_CONFLICT,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"error": "Model already loaded"}).decode(),
        )

    logger.info("loading model via API", step="MODEL", model=model_id)
    await whisper.load(model_id)
    return Response(
        status_code=status_codes.HTTP_201_CREATED,
        headers={"content-type": "application/json"},
        description=orjson.dumps({"status": "loaded", "model": model_id}).decode(),
    )


@router.delete("/api/ps/:model_id")
async def unload_model_endpoint(model_id: str, global_dependencies) -> Response:
    """Unload a model from memory."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    if await whisper.unload(model_id):
        return Response(
            status_code=status_codes.HTTP_200_OK,
            headers={"content-type": "application/json"},
            description=orjson.dumps({"status": "unloaded", "model": model_id}).decode(),
        )
    return Response(
        status_code=status_codes.HTTP_404_NOT_FOUND,
        headers={"content-type": "application/json"},
        description=orjson.dumps({"error": "Model not loaded"}).decode(),
    )


@router.post("/api/pull/:model_id")
async def pull_model(model_id: str, global_dependencies) -> Response:
    """Download a model from HuggingFace Hub."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    logger.info("pulling model via API", step="DOWNLOAD", model=model_id)
    path = await whisper.download(model_id)
    return Response(
        status_code=status_codes.HTTP_201_CREATED,
        headers={"content-type": "application/json"},
        description=orjson.dumps({"status": "downloaded", "model": model_id, "path": str(path)}).decode(),
    )
