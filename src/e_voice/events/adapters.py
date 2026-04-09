"""Adapter lifespan events — backend-agnostic STT/TTS startup and shutdown."""

from e_voice.adapters.base import STTBackend, TTSBackend
from e_voice.adapters.registry import STT_BACKENDS, TTS_BACKENDS
from e_voice.core.lifespan import BaseEvent
from e_voice.core.logger import logger
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.stt import ModelSpec
from e_voice.models.tts import TTSModelSpec


class STTAdapterEvent(BaseEvent[STTBackend]):
    """Manages STT backend lifecycle — creates adapter from config, loads model, shuts down."""

    name = "stt"

    async def startup(self) -> STTBackend:
        """Create STT adapter from registry and pre-load model on configured device."""
        backend_name = st.stt.backend
        if backend_name not in STT_BACKENDS:
            raise RuntimeError(f"Unknown STT backend: '{backend_name}'. Available: {sorted(STT_BACKENDS)}")

        adapter_cls = STT_BACKENDS[backend_name]
        logger.info("creating STT backend", step="START", backend=backend_name)

        adapter = adapter_cls(st.stt, st.vad)  # ty: ignore[too-many-positional-arguments]

        (st.MODELS_PATH / "stt" / backend_name).mkdir(parents=True, exist_ok=True)
        await adapter.load(
            ModelSpec(
                model_id=st.stt.model,
                device=st.stt.device.value,
                compute_type=st.stt.compute_type.value,
            )
        )

        if st.stt.cpu_fallback and st.stt.device != DeviceType.CPU:
            logger.info("loading CPU fallback model", step="START", model=st.stt.model)
            await adapter.load(ModelSpec(model_id=st.stt.model, device="cpu"))

        return adapter

    async def shutdown(self, instance: STTBackend) -> None:
        """Unload all models."""
        for spec in list(instance.loaded_models()):
            await instance.unload(spec)


class TTSAdapterEvent(BaseEvent[TTSBackend]):
    """Manages TTS backend lifecycle — creates adapter from config, loads model, shuts down."""

    name = "tts"

    async def startup(self) -> TTSBackend:
        """Create TTS adapter from registry and load default model."""
        backend_name = st.tts.backend
        if backend_name not in TTS_BACKENDS:
            available = sorted(TTS_BACKENDS)
            hint = ""
            if backend_name == "qwen" and "qwen" not in TTS_BACKENDS:
                hint = " Install deps with: uv add --group qwen qwen-tts faster-qwen3-tts torch torchaudio"
            raise RuntimeError(f"Unknown TTS backend: '{backend_name}'. Available: {available}.{hint}")

        adapter_cls = TTS_BACKENDS[backend_name]
        logger.info("creating TTS backend", step="START", backend=backend_name)

        (st.MODELS_PATH / "tts" / backend_name).mkdir(parents=True, exist_ok=True)
        adapter = adapter_cls()  # ty: ignore[too-many-positional-arguments]
        await adapter.load(TTSModelSpec(model_id=backend_name, device=st.tts.device))
        return adapter

    async def shutdown(self, instance: TTSBackend) -> None:
        """Unload all models."""
        for spec in list(instance.loaded_models()):
            await instance.unload(spec)
