"""Base adapter for ML model services."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelAdapter[S](ABC):
    """Base for adapters that manage local ML model lifecycle.

    S: spec type for model identity (e.g., ModelSpec for Whisper, str for Kokoro).
    """

    @abstractmethod
    async def load(self, spec: S) -> None:
        """Load a model into memory."""
        ...

    @abstractmethod
    async def unload(self, spec: S) -> bool:
        """Unload a model from memory."""
        ...

    @abstractmethod
    async def is_loaded(self, spec: S) -> bool:
        """Check if a model is loaded."""
        ...

    @abstractmethod
    def loaded_models(self) -> list[S]:
        """Return specs of currently loaded models."""
        ...

    @abstractmethod
    async def download(self, model_id: str) -> Path:
        """Download model files to disk. Returns the model directory."""
        ...
