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


##### HELPERS #####


def _dir_size_mb(path: Path) -> float:
    """Total size of directory contents in MB."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _scan_stt_models() -> list[ModelEntry]:
    """Scan data/models/stt/ for downloaded STT models (HuggingFace cache layout)."""
    stt_dir = st.MODELS_PATH / "stt"
    if not stt_dir.exists():
        return []
    return [
        ModelEntry(
            id=d.name.removeprefix("models--").replace("--", "/"),
            service="stt",
            path=str(d),
            size_mb=round(_dir_size_mb(d), 1),
        )
        for d in sorted(stt_dir.iterdir())
        if d.is_dir() and d.name.startswith("models--")
    ]


def _scan_tts_models() -> list[ModelEntry]:
    """Scan data/models/tts/ for downloaded TTS models."""
    tts_dir = st.MODELS_PATH / "tts"
    if not tts_dir.exists():
        return []
    onnx_files = list(tts_dir.glob("*.onnx"))
    if not onnx_files:
        return []
    return [
        ModelEntry(
            id="kokoro",
            service="tts",
            path=str(tts_dir),
            size_mb=round(_dir_size_mb(tts_dir), 1),
        )
    ]


def _json_response(status_code: int, body: str) -> Response:
    return Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        description=body,
    )


def _json_error(status_code: int, error: str, detail: str | None = None) -> Response:
    return _json_response(
        status_code,
        ErrorResponse(error=error, detail=detail).model_dump_json(),
    )


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

        return _json_response(
            status_codes.HTTP_201_CREATED,
            DownloadResponse(
                status="downloaded",
                service=body.service.value,
                model=body.model,
                path=str(path),
            ).model_dump_json(),
        )

    except (OSError, RuntimeError) as exc:
        logger.error("download failed", step="DOWNLOAD", service=body.service.value, model=body.model, error=str(exc))
        return _json_error(
            status_codes.HTTP_502_BAD_GATEWAY,
            f"Download failed for {body.service.value} model '{body.model}'",
            str(exc),
        )
    except Exception as exc:
        logger.error("download error", step="DOWNLOAD", error=str(exc))
        return _json_error(
            status_codes.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal error during download",
            str(exc),
        )


@router.get("/models/list")
async def list_downloaded_models() -> Response:
    """List all downloaded models on disk, grouped by service."""
    resp = ModelsListResponse(
        stt=_scan_stt_models(),
        tts=_scan_tts_models(),
    )
    return _json_response(status_codes.HTTP_200_OK, resp.model_dump_json())
