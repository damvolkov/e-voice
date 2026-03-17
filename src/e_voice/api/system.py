"""System API — model download and management endpoints."""

from pathlib import Path

from robyn import Response, status_codes

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.core.settings import settings as st
from e_voice.models.system import (
    DownloadRequest,
    DownloadResponse,
    ErrorResponse,
    ModelEntry,
    ModelsListResponse,
    ServiceType,
)

router = Router(__file__, prefix="/v1")


##### ENDPOINTS #####


@router.post("/models/download")
async def download_model(body: DownloadRequest, global_dependencies) -> Response:
    """Download a model to disk for the specified service."""
    try:
        match body.service:
            case ServiceType.STT:
                whisper: WhisperAdapter = global_dependencies.get("state").whisper
                logger.info("downloading STT model", step="DOWNLOAD", model=body.model)
                path = await whisper.download(body.model)

            case ServiceType.TTS:
                kokoro: KokoroAdapter = global_dependencies.get("state").kokoro
                logger.info("downloading TTS model", step="DOWNLOAD", model=body.model)
                path = await kokoro.download(body.model)

        logger.info("model downloaded", step="DOWNLOAD", service=body.service.value, model=body.model)

        return Response(
            status_code=status_codes.HTTP_201_CREATED,
            headers={"content-type": "application/json"},
            description=DownloadResponse(
                status="downloaded",
                service=body.service.value,
                model=body.model,
                path=str(path),
            ).model_dump_json(),
        )

    except (OSError, RuntimeError) as exc:
        logger.error("download failed", step="DOWNLOAD", service=body.service.value, model=body.model, error=str(exc))
        return Response(
            status_code=status_codes.HTTP_502_BAD_GATEWAY,
            headers={"content-type": "application/json"},
            description=ErrorResponse(
                error=f"Download failed for {body.service.value} model '{body.model}'",
                detail=str(exc),
            ).model_dump_json(),
        )
    except Exception as exc:
        logger.error("download error", step="DOWNLOAD", error=str(exc))
        return Response(
            status_code=status_codes.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"content-type": "application/json"},
            description=ErrorResponse(
                error="Internal error during download",
                detail=str(exc),
            ).model_dump_json(),
        )


def _dir_size_mb(path: Path) -> float:
    """Total size of directory contents in MB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


@router.get("/models/list")
async def list_downloaded_models() -> Response:
    """List all downloaded models on disk, grouped by service."""
    stt_dir = st.MODELS_PATH / "stt"
    stt_models = (
        [
            ModelEntry(
                id=d.name.removeprefix("models--").replace("--", "/"),
                service="stt",
                path=str(d),
                size_mb=round(_dir_size_mb(d), 1),
            )
            for d in sorted(stt_dir.iterdir())
            if d.is_dir() and d.name.startswith("models--")
        ]
        if stt_dir.exists()
        else []
    )

    tts_dir = st.MODELS_PATH / "tts"
    tts_models = (
        [
            ModelEntry(
                id="kokoro",
                service="tts",
                path=str(tts_dir),
                size_mb=round(_dir_size_mb(tts_dir), 1),
            )
        ]
        if tts_dir.exists() and any(tts_dir.glob("*.onnx"))
        else []
    )

    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=ModelsListResponse(stt=stt_models, tts=tts_models).model_dump_json(),
    )
