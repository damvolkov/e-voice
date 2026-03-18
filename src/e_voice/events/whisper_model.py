"""Whisper model lifespan event — preload default model on startup."""

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.lifespan import BaseEvent
from e_voice.core.logger import logger
from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.stt import ModelSpec


class WhisperModelEvent(BaseEvent[WhisperAdapter]):
    """Manages WhisperAdapter lifecycle: load on startup, unload on shutdown."""

    name = "whisper"

    async def startup(self) -> WhisperAdapter:
        """Create adapter and pre-load model on configured device (+ CPU fallback)."""
        (st.MODELS_PATH / "stt").mkdir(parents=True, exist_ok=True)
        adapter = WhisperAdapter(st.stt, st.vad)

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

    async def shutdown(self, instance: WhisperAdapter) -> None:
        """Unload all models."""
        for spec in list(instance.loaded_models()):
            await instance.unload(spec)
