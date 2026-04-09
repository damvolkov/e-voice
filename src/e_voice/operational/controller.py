"""Device and backend lifecycle controller — GPU↔CPU switching + backend hot-swap."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import yaml

from e_voice.adapters.registry import STT_BACKENDS, TTS_BACKENDS
from e_voice.core.logger import logger
from e_voice.core.settings import ComputeType, DeviceType, resolve_compute_type
from e_voice.core.settings import settings as st
from e_voice.models.error import BackendCapabilityError
from e_voice.models.session import ConnectionRegistry
from e_voice.models.stt import ModelSpec
from e_voice.models.system import ServiceType
from e_voice.models.tts import TTSModelSpec

if TYPE_CHECKING:
    from e_voice.adapters.base import STTBackend, TTSBackend


class DeviceState(StrEnum):
    """Visual state for the device semaphore."""

    GPU = auto()
    CPU = auto()
    TRANSITIONING = auto()


@dataclass(frozen=True, slots=True)
class SwitchResult:
    """Outcome of a device or backend switch attempt."""

    success: bool
    service: ServiceType
    device: DeviceType
    message: str


class DeviceController:
    """Manages GPU↔CPU switching and backend hot-swap for STT and TTS."""

    __slots__ = ("_lock", "_transitioning")

    def __init__(self) -> None:
        self._transitioning: dict[ServiceType, bool] = {
            ServiceType.STT: False,
            ServiceType.TTS: False,
        }
        self._lock = asyncio.Lock()

    def active_device(self, service: ServiceType) -> DeviceType:
        """Current device for a given service."""
        match service:
            case ServiceType.STT:
                return st.stt.device
            case ServiceType.TTS:
                return st.tts.device

    def transitioning(self, service: ServiceType) -> bool:
        return self._transitioning[service]

    def state(self, service: ServiceType) -> DeviceState:
        if self._transitioning[service]:
            return DeviceState.TRANSITIONING
        return DeviceState.GPU if self.active_device(service) == DeviceType.GPU else DeviceState.CPU

    ##### DEVICE SWITCH #####

    async def switch(
        self,
        service: ServiceType,
        target: DeviceType,
        stt: STTBackend,
        tts: TTSBackend,
    ) -> SwitchResult:
        """Switch a single service to the target device."""
        if target == self.active_device(service) and not self._transitioning[service]:
            return SwitchResult(
                success=True, service=service, device=target, message=f"{service.value} already on {target.value}"
            )

        self._transitioning[service] = True
        try:
            match service:
                case ServiceType.STT:
                    await self._dc_switch_stt(target, stt)
                case ServiceType.TTS:
                    await self._dc_switch_tts(target, tts)

            await self._dc_persist_config(service, device=target)

            logger.info("device switched", step="OK", service=service.value, device=target.value)
            return SwitchResult(
                success=True, service=service, device=target, message=f"{service.value} switched to {target.value}"
            )

        except BackendCapabilityError as exc:
            logger.warning("device switch not supported", step="WARN", service=service.value, error=str(exc))
            return SwitchResult(success=False, service=service, device=self.active_device(service), message=str(exc))

        except Exception as exc:
            logger.error("device switch failed", step="ERROR", service=service.value, error=str(exc))
            return SwitchResult(success=False, service=service, device=self.active_device(service), message=str(exc))
        finally:
            self._transitioning[service] = False

    ##### BACKEND SWITCH #####

    async def switch_backend(
        self,
        service: ServiceType,
        target_backend: str,
        state: Any,
    ) -> SwitchResult:
        """Hot-swap a service backend. Drains connections, shuts down old, loads new."""
        async with self._lock:
            return await self._dc_swap_backend(service, target_backend, state)

    async def _dc_swap_backend(
        self,
        service: ServiceType,
        target_backend: str,
        state: Any,
    ) -> SwitchResult:
        """Internal backend swap — must be called under self._lock."""
        registry = TTS_BACKENDS if service == ServiceType.TTS else STT_BACKENDS
        current_backend = st.tts.backend if service == ServiceType.TTS else st.stt.backend
        attr = service.value

        if target_backend == current_backend:
            return SwitchResult(
                success=True,
                service=service,
                device=self.active_device(service),
                message=f"{service.value} already using {target_backend}",
            )

        if target_backend not in registry:
            available = sorted(registry)
            return SwitchResult(
                success=False,
                service=service,
                device=self.active_device(service),
                message=f"Unknown backend '{target_backend}'. Available: {available}",
            )

        self._transitioning[service] = True
        try:
            conn_registry: ConnectionRegistry | None = getattr(state, f"{attr}_connections", None)
            if conn_registry is not None:
                logger.info("draining connections", step="SWAP", service=attr, active=conn_registry.count)
                await conn_registry.wait_empty(timeout=10.0)

            old_adapter = getattr(state, attr, None)
            if old_adapter is not None:
                logger.info("shutting down old backend", step="SWAP", service=attr, backend=current_backend)
                for spec in list(old_adapter.loaded_models()):
                    await old_adapter.unload(spec)

            adapter_cls = registry[target_backend]
            new_adapter = adapter_cls()  # ty: ignore[too-many-positional-arguments]

            device = self.active_device(service)
            if device not in new_adapter.supported_devices and new_adapter.supported_devices:
                device = next(iter(new_adapter.supported_devices))
                match service:
                    case ServiceType.TTS:
                        st.tts.device = device
                    case ServiceType.STT:
                        st.stt.device = device

            (st.MODELS_PATH / attr / target_backend).mkdir(parents=True, exist_ok=True)

            match service:
                case ServiceType.TTS:
                    tts_adapter: TTSBackend = new_adapter  # ty: ignore[invalid-assignment]
                    await tts_adapter.load(TTSModelSpec(model_id=target_backend, device=device))
                case ServiceType.STT:
                    stt_adapter: STTBackend = new_adapter  # ty: ignore[invalid-assignment]
                    await stt_adapter.load(
                        ModelSpec(model_id=st.stt.model, device=device.value, compute_type=st.stt.compute_type.value)
                    )

            setattr(state, attr, new_adapter)

            match service:
                case ServiceType.TTS:
                    st.tts.backend = target_backend
                case ServiceType.STT:
                    st.stt.backend = target_backend

            await self._dc_persist_config(service, backend=target_backend, device=device)

            logger.info("backend switched", step="OK", service=attr, backend=target_backend, device=device.value)
            return SwitchResult(
                success=True,
                service=service,
                device=device,
                message=f"{service.value} switched to {target_backend} on {device.value}",
            )

        except BackendCapabilityError as exc:
            logger.warning("backend switch failed", step="WARN", service=attr, error=str(exc))
            return SwitchResult(success=False, service=service, device=self.active_device(service), message=str(exc))
        except Exception as exc:
            logger.error("backend switch failed", step="ERROR", service=attr, error=str(exc))
            return SwitchResult(success=False, service=service, device=self.active_device(service), message=str(exc))
        finally:
            self._transitioning[service] = False

    ##### DEVICE SWITCH INTERNALS #####

    async def _dc_switch_stt(self, target: DeviceType, stt: STTBackend) -> None:
        """Load STT on target device, update settings, unload previous."""
        if target not in stt.supported_devices:
            raise BackendCapabilityError(
                f"STT backend does not support {target.value}. Supported: {stt.supported_devices}"
            )

        previous = st.stt.device
        resolved_compute = resolve_compute_type(target, ComputeType.DEFAULT)

        stt_spec = ModelSpec(
            model_id=st.stt.model,
            device=target.value,
            compute_type=resolved_compute.value,
        )
        if not await stt.is_loaded(stt_spec):
            logger.info("loading STT on target device", step="MODEL", device=target.value)
            await stt.load(stt_spec)

        st.stt.device = target
        st.stt.compute_type = resolved_compute

        if previous != target:
            prev_val = previous.value
            for spec in [s for s in stt.loaded_models() if s.device == prev_val]:
                if await stt.unload(spec):
                    logger.info("unloaded STT from previous device", step="MODEL", device=prev_val)

    async def _dc_switch_tts(self, target: DeviceType, tts: TTSBackend) -> None:
        """Load TTS on target device, update settings, unload previous."""
        if target not in tts.supported_devices:
            raise BackendCapabilityError(
                f"TTS backend does not support {target.value}. Supported: {tts.supported_devices}"
            )

        previous = st.tts.device

        tts_spec = TTSModelSpec(device=target)
        if not await tts.is_loaded(tts_spec):
            logger.info("loading TTS on target device", step="MODEL", device=target.value)
            await tts.load(tts_spec)

        st.tts.device = target

        if previous != target:
            for spec in [s for s in tts.loaded_models() if s.device == previous]:
                if await tts.unload(spec):
                    logger.info("unloaded TTS from previous device", step="MODEL", device=previous.value)

    ##### CONFIG PERSISTENCE #####

    async def _dc_persist_config(
        self,
        service: ServiceType,
        *,
        device: DeviceType | None = None,
        backend: str | None = None,
        config_dir: Path | None = None,
    ) -> None:
        """Write updated device and/or backend for a service to config.yaml."""
        config_path = (config_dir or st.CONFIG_PATH) / "config.yaml"
        if not config_path.exists():
            return

        try:
            async with aiofiles.open(config_path) as fh:
                raw = await fh.read()
            data = yaml.safe_load(raw) or {}

            section = data.setdefault(service.value, {})
            if device is not None:
                section["device"] = device.value
            if backend is not None:
                section["backend"] = backend

            async with aiofiles.open(config_path, "w") as fh:
                await fh.write(yaml.dump(data, default_flow_style=False, sort_keys=False))

            logger.info("config persisted", step="OK", service=service.value, device=device, backend=backend)
        except OSError:
            logger.info("config read-only — skipping persist", step="WARN")
