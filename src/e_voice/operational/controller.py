"""Device lifecycle controller — GPU↔CPU switching with config persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import yaml

from e_voice.core.logger import logger
from e_voice.core.settings import ComputeType, DeviceType, resolve_compute_type
from e_voice.core.settings import settings as st
from e_voice.models.error import BackendCapabilityError
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
    """Outcome of a device switch attempt."""

    success: bool
    service: ServiceType
    device: DeviceType
    message: str


class DeviceController:
    """Manages GPU↔CPU switching for STT and TTS backends independently."""

    __slots__ = ("_transitioning",)

    def __init__(self) -> None:
        self._transitioning: dict[ServiceType, bool] = {
            ServiceType.STT: False,
            ServiceType.TTS: False,
        }

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

    async def switch(
        self,
        service: ServiceType,
        target: DeviceType,
        stt: STTBackend,
        tts: TTSBackend,
    ) -> SwitchResult:
        """Switch a single service to the target device. Loads model, persists config."""
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

            await self._dc_persist_config(service, target)

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

    async def _dc_persist_config(
        self, service: ServiceType, device: DeviceType, config_dir: Path | None = None
    ) -> None:
        """Write updated device for a single service to config.yaml."""
        config_path = (config_dir or st.CONFIG_PATH) / "config.yaml"
        if not config_path.exists():
            return

        try:
            async with aiofiles.open(config_path) as fh:
                raw = await fh.read()
            data = yaml.safe_load(raw) or {}

            data.setdefault(service.value, {})["device"] = device.value

            async with aiofiles.open(config_path, "w") as fh:
                await fh.write(yaml.dump(data, default_flow_style=False, sort_keys=False))

            logger.info("config persisted", step="OK", service=service.value, device=device.value)
        except OSError:
            logger.info("config read-only — skipping persist", step="WARN")
