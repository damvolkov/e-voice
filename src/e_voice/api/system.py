"""System API — model download, management, device control, and loaded-model endpoints."""

from pathlib import Path
from urllib.parse import unquote

import orjson
from robyn import Request, Response, status_codes

from e_voice.adapters.base import STTBackend, TTSBackend
from e_voice.adapters.registry import available_backends
from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.error import BackendCapabilityError, error_response
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
                stt: STTBackend = global_dependencies.get("state").stt
                logger.info("downloading STT model", step="DOWNLOAD", model=body.model)
                path = await stt.download(body.model)

            case ServiceType.TTS:
                tts: TTSBackend = global_dependencies.get("state").tts
                logger.info("downloading TTS model", step="DOWNLOAD", model=body.model)
                path = await tts.download(body.model)

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

    except BackendCapabilityError as exc:
        return error_response(request.url.path, 501, str(exc))
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


def _scan_stt_models() -> list[ModelEntry]:
    """Scan STT backend model directory for downloaded models."""
    stt_dir = st.MODELS_PATH / "stt" / st.stt.backend
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
    """Scan TTS backend model directory for downloaded models."""
    tts_dir = st.MODELS_PATH / "tts" / st.tts.backend
    if not tts_dir.exists():
        return []
    has_models = any(tts_dir.glob("*.onnx")) or any(tts_dir.glob("*.bin")) or any(tts_dir.glob("*.pt"))
    if not has_models:
        return []
    return [
        ModelEntry(
            id=st.tts.backend,
            service="tts",
            path=str(tts_dir),
            size_mb=round(_dir_size_mb(tts_dir), 1),
        )
    ]


@router.get("/system/backends")
async def get_backends() -> Response:
    """Return active and available backends per service."""
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=orjson.dumps(available_backends()).decode(),
    )


@router.get("/models/list")
async def list_downloaded_models() -> Response:
    """List all downloaded models on disk for the active backends."""
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=ModelsListResponse(stt=_scan_stt_models(), tts=_scan_tts_models()).model_dump_json(),
    )


##### LOADED MODELS (PROCESS STATUS) #####


@router.get("/api/ps")
async def list_loaded_models(global_dependencies) -> LoadedModelsResponse:
    """List all loaded/running models across adapters."""
    stt: STTBackend = global_dependencies.get("state").stt
    tts: TTSBackend = global_dependencies.get("state").tts
    models = [s.model_id for s in stt.loaded_models()] + [s.model_id for s in tts.loaded_models()]
    return LoadedModelsResponse(models=models)


@router.post("/api/ps/:model_id")
async def load_model(request: Request, model_id: str, global_dependencies) -> Response:
    """Load a model into memory. Returns 409 if already loaded."""
    model_id = unquote(model_id)
    stt: STTBackend = global_dependencies.get("state").stt
    tts: TTSBackend = global_dependencies.get("state").tts
    loaded = {s.model_id for s in stt.loaded_models()} | {s.model_id for s in tts.loaded_models()}

    if model_id in loaded:
        return error_response(request.url.path, 409, f"Model '{model_id}' is already loaded")

    return error_response(request.url.path, 404, f"Unknown model '{model_id}'", error_type="not_found_error")


@router.delete("/api/ps/:model_id")
async def unload_model(request: Request, model_id: str, global_dependencies) -> Response:
    """Unload a model from memory. Returns 404 if not loaded."""
    model_id = unquote(model_id)
    stt: STTBackend = global_dependencies.get("state").stt
    tts: TTSBackend = global_dependencies.get("state").tts

    for spec in stt.loaded_models():
        if spec.model_id == model_id:
            await stt.unload(spec)
            return Response(
                status_code=status_codes.HTTP_200_OK,
                headers={"content-type": "application/json"},
                description=orjson.dumps({"status": "unloaded", "model": model_id}).decode(),
            )

    for spec in tts.loaded_models():
        if spec.model_id == model_id:
            await tts.unload(spec)
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
    """Return per-service device state for STT and TTS."""
    ctrl: DeviceController = global_dependencies.get("state").device_controller
    return Response(
        status_code=status_codes.HTTP_200_OK,
        headers={"content-type": "application/json"},
        description=orjson.dumps(
            {
                "stt": {
                    "device": ctrl.active_device(ServiceType.STT).value,
                    "state": ctrl.state(ServiceType.STT).value,
                    "transitioning": ctrl.transitioning(ServiceType.STT),
                },
                "tts": {
                    "device": ctrl.active_device(ServiceType.TTS).value,
                    "state": ctrl.state(ServiceType.TTS).value,
                    "transitioning": ctrl.transitioning(ServiceType.TTS),
                },
            }
        ).decode(),
    )


@router.post("/system/device")
async def switch_device(request: Request, global_dependencies) -> Response:
    """Switch a service to a different device (gpu/cpu). Requires 'service' and 'device' in body."""
    body = orjson.loads(request.body)
    target_str = body.get("device", "").lower()
    service_str = body.get("service", "").lower()

    if target_str not in ("gpu", "cpu"):
        return error_response(request.url.path, 400, f"Invalid device: '{target_str}'. Use 'gpu' or 'cpu'.")

    if service_str not in ("stt", "tts"):
        return error_response(request.url.path, 400, f"Invalid service: '{service_str}'. Use 'stt' or 'tts'.")

    target = DeviceType.GPU if target_str == "gpu" else DeviceType.CPU
    service = ServiceType.STT if service_str == "stt" else ServiceType.TTS
    ctrl: DeviceController = global_dependencies.get("state").device_controller
    stt: STTBackend = global_dependencies.get("state").stt
    tts: TTSBackend = global_dependencies.get("state").tts

    result = await ctrl.switch(service, target, stt, tts)

    return Response(
        status_code=status_codes.HTTP_200_OK if result.success else status_codes.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"content-type": "application/json"},
        description=orjson.dumps(
            {
                "success": result.success,
                "service": result.service.value,
                "device": result.device.value,
                "message": result.message,
            }
        ).decode(),
    )
