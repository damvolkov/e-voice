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
from e_voice.models.stt import ModelSpec
from e_voice.models.tts import TTSModelSpec

if TYPE_CHECKING:
    from e_voice.adapters.kokoro import KokoroAdapter
    from e_voice.adapters.whisper import WhisperAdapter


class DeviceState(StrEnum):
    """Visual state for the device semaphore."""

    GPU = auto()
    CPU = auto()
    TRANSITIONING = auto()


@dataclass(frozen=True, slots=True)
class SwitchResult:
    """Outcome of a device switch attempt."""

    success: bool
    device: DeviceType
    message: str


class DeviceController:
    """Manages GPU↔CPU switching for STT and TTS adapters."""

    __slots__ = ("_transitioning",)

    def __init__(self) -> None:
        self._transitioning = False

    @property
    def active_device(self) -> DeviceType:
        return st.stt.device

    @property
    def transitioning(self) -> bool:
        return self._transitioning

    @property
    def state(self) -> DeviceState:
        if self._transitioning:
            return DeviceState.TRANSITIONING
        return DeviceState.GPU if st.stt.device == DeviceType.GPU else DeviceState.CPU

    async def switch(
        self,
        target: DeviceType,
        whisper: WhisperAdapter,
        kokoro: KokoroAdapter,
    ) -> SwitchResult:
        """Switch both adapters to target device. Loads model if needed, persists config."""
        if target == self.active_device and not self._transitioning:
            return SwitchResult(success=True, device=target, message=f"Already on {target.value}")

        self._transitioning = True
        try:
            resolved_compute = resolve_compute_type(target, ComputeType.DEFAULT)
            stt_spec = ModelSpec(
                model_id=st.stt.model,
                device=target.value,
                compute_type=resolved_compute.value,
            )
            if not await whisper.is_loaded(stt_spec):
                logger.info("loading STT on target device", step="MODEL", device=target.value)
                await whisper.load(stt_spec)

            tts_spec = TTSModelSpec(device=target)
            if not await kokoro.is_loaded(tts_spec):
                logger.info("loading TTS on target device", step="MODEL", device=target.value)
                await kokoro.load(tts_spec)

            st.stt.device = target
            st.tts.device = target

            await self._dc_persist_config(target)

            logger.info("device switched", step="OK", device=target.value)
            return SwitchResult(success=True, device=target, message=f"Switched to {target.value}")

        except Exception as exc:
            logger.error("device switch failed", step="ERROR", error=str(exc))
            return SwitchResult(success=False, device=self.active_device, message=str(exc))
        finally:
            self._transitioning = False

    async def _dc_persist_config(self, device: DeviceType, config_dir: Path | None = None) -> None:
        """Write updated device to config.yaml. No-op if file is read-only or missing."""
        config_path = (config_dir or st.CONFIG_PATH) / "config.yaml"
        if not config_path.exists():
            return

        try:
            async with aiofiles.open(config_path) as fh:
                raw = await fh.read()
            data = yaml.safe_load(raw) or {}

            data.setdefault("stt", {})["device"] = device.value
            data.setdefault("tts", {})["device"] = device.value

            async with aiofiles.open(config_path, "w") as fh:
                await fh.write(yaml.dump(data, default_flow_style=False, sort_keys=False))

            logger.info("config persisted", step="OK", device=device.value)
        except OSError:
            logger.info("config read-only — skipping persist", step="WARN")
