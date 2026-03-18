"""Kokoro TTS model lifespan event — preload on startup."""

from e_voice.adapters.kokoro import KokoroAdapter
from e_voice.core.lifespan import BaseEvent


class KokoroModelEvent(BaseEvent[KokoroAdapter]):
    """Manages KokoroAdapter lifecycle."""

    name = "kokoro"

    async def startup(self) -> KokoroAdapter:
        """Create adapter and load default model."""
        adapter = KokoroAdapter()
        await adapter.load()
        return adapter

    async def shutdown(self, instance: KokoroAdapter) -> None:
        """Unload all models."""
        for spec in list(instance.loaded):
            await instance.unload(spec)
