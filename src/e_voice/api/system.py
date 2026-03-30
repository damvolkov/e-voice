"""System API — model download, management, device control, and loaded-model endpoints."""

from pathlib import Path
from urllib.parse import unquote

import orjson
from robyn import Request, Response, status_codes

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.error import error_response
from e_voice.models.system import (
    DownloadRequest,
    DownloadResponse,
    LoadedModelsResponse,
    ModelEntry,
    ModelsListResponse,
    ServiceType,
)
from e_voice.operational.controller import DeviceController
from e_voice.operational.monitor import SystemMonitor

router = Router(__file__, prefix="/v1")


##### ENDPOINTS #####


@router.post("/models/download")
async def download_model(request: Request, body: DownloadRequest, global_dependencies) -> Response:
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
        return error_response(
            request.url.path, 502, f"Download failed for {body.service.value} model '{body.model}': {exc}"
        )
    except Exception as exc:
        logger.error("download error", step="DOWNLOAD", error=str(exc))
        return error_response(request.url.path, 500, f"Internal error during download: {exc}")


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


##### LOADED MODELS (PROCESS STATUS) #####


@router.get("/api/ps")
async def list_loaded_models(global_dependencies) -> LoadedModelsResponse:
    """List all loaded/running models across adapters."""
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro
    models = [s.model_id for s in whisper.loaded_models()] + [s.model_id for s in kokoro.loaded_models()]
    return LoadedModelsResponse(models=models)


@router.post("/api/ps/:model_id")
async def load_model(request: Request, model_id: str, global_dependencies) -> Response:
    """Load a model into memory. Returns 409 if already loaded."""
    model_id = unquote(model_id)
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro
    loaded = {s.model_id for s in whisper.loaded_models()} | {s.model_id for s in kokoro.loaded_models()}

    if model_id in loaded:
        return error_response(request.url.path, 409, f"Model '{model_id}' is already loaded")

    return error_response(request.url.path, 404, f"Unknown model '{model_id}'", error_type="not_found_error")


@router.delete("/api/ps/:model_id")
async def unload_model(request: Request, model_id: str, global_dependencies) -> Response:
    """Unload a model from memory. Returns 404 if not loaded."""
    model_id = unquote(model_id)
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro

    for spec in whisper.loaded_models():
        if spec.model_id == model_id:
            await whisper.unload(spec)
            return Response(
                status_code=status_codes.HTTP_200_OK,
                headers={"content-type": "application/json"},
                description=orjson.dumps({"status": "unloaded", "model": model_id}).decode(),
            )

    for spec in kokoro.loaded_models():
        if spec.model_id == model_id:
            await kokoro.unload(spec)
            return Response(
                status_code=status_codes.HTTP_200_OK,
                headers={"content-type": "application/json"},
                description=orjson.dumps({"status": "unloaded", "model": model_id}).decode(),
            )

    return error_response(request.url.path, 404, f"Model '{model_id}' is not loaded", error_type="not_found_error")


##### SYSTEM MONITOR #####


@router.get("/system/monitor")
async def get_monitor(global_dependencies) -> Response:
    """Poll and return system metrics snapshot with sparkline history."""
    monitor: SystemMonitor = global_dependencies.get("state").monitor
    snap = monitor.poll()
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=orjson.dumps(
            {
                "cpu_pct": snap.cpu_pct,
                "ram_used_gb": snap.ram_used_gb,
                "ram_total_gb": snap.ram_total_gb,
                "ram_pct": snap.ram_pct,
                "gpu_util_pct": snap.gpu_util_pct,
                "vram_used_mb": snap.vram_used_mb,
                "vram_total_mb": snap.vram_total_mb,
                "vram_pct": snap.vram_pct,
                "gpu_available": snap.gpu_available,
                "history": {
                    "cpu": list(monitor.cpu_history),
                    "ram": list(monitor.ram_history),
                    "gpu_util": list(monitor.gpu_util_history),
                    "vram": list(monitor.vram_history),
                },
            }
        ).decode(),
    )


##### DEVICE CONTROL #####


@router.get("/system/device")
async def get_device(global_dependencies) -> Response:
    """Return current device mode and transition state."""
    ctrl: DeviceController = global_dependencies.get("state").device_controller
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=orjson.dumps(
            {
                "device": ctrl.active_device.value,
                "state": ctrl.state.value,
                "transitioning": ctrl.transitioning,
            }
        ).decode(),
    )


@router.post("/system/device")
async def switch_device(request: Request, global_dependencies) -> Response:
    """Switch STT+TTS to a different device (gpu/cpu)."""
    body = orjson.loads(request.body)
    target_str = body.get("device", "").lower()

    if target_str not in ("gpu", "cpu"):
        return error_response(request.url.path, 400, f"Invalid device: '{target_str}'. Use 'gpu' or 'cpu'.")

    target = DeviceType.GPU if target_str == "gpu" else DeviceType.CPU
    ctrl: DeviceController = global_dependencies.get("state").device_controller
    whisper: WhisperAdapter = global_dependencies.get("state").whisper
    kokoro: KokoroAdapter = global_dependencies.get("state").kokoro

    result = await ctrl.switch(target, whisper, kokoro)

    return Response(
        status_code=status_codes.HTTP_200_OK if result.success else status_codes.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"content-type": "application/json"},
        description=orjson.dumps(
            {
                "success": result.success,
                "device": result.device.value,
                "message": result.message,
            }
        ).decode(),
    )
