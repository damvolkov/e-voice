"""Whisper model lifespan event — preload default model on startup."""

from e_voice.adapters.whisper import WhisperAdapter
from e_voice.core.lifespan import BaseEvent
from e_voice.core.settings import settings as st


class WhisperModelEvent(BaseEvent[WhisperAdapter]):
    """Manages WhisperAdapter lifecycle: load on startup, unload on shutdown."""

    name = "whisper"

    async def startup(self) -> WhisperAdapter:
        """Create adapter and load default model."""
        (st.MODELS_PATH / "stt").mkdir(parents=True, exist_ok=True)
        adapter = WhisperAdapter(st.whisper_config, st.vad_config)
        await adapter.load(st.WHISPER_MODEL)
        return adapter

    async def shutdown(self, instance: WhisperAdapter) -> None:
        """Unload all models."""
        for model_id in list(instance.loaded_models()):
            await instance.unload(model_id)
