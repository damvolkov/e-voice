"""Kokoro TTS model lifespan event — preload on startup."""

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.core.lifespan import BaseEvent
from e_voice.core.settings import settings as st


class KokoroModelEvent(BaseEvent[KokoroAdapter]):
    """Manages KokoroAdapter lifecycle."""

    name = "kokoro"

    async def startup(self) -> KokoroAdapter:
        """Create adapter and load default model."""
        adapter = KokoroAdapter(st.MODELS_PATH / "tts")
        await adapter.load()
        return adapter

    async def shutdown(self, instance: KokoroAdapter) -> None:
        """Unload model."""
        await instance.unload()
