"""Base adapter for ML model services."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelAdapter(ABC):
    """Base for adapters that manage local ML model lifecycle."""

    @abstractmethod
    async def load(self, model_id: str) -> None:
        """Load a model into memory."""
        ...

    @abstractmethod
    async def unload(self, model_id: str) -> bool:
        """Unload a model from memory."""
        ...

    @abstractmethod
    async def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        ...

    @abstractmethod
    def loaded_models(self) -> list[str]:
        """Return IDs of currently loaded models."""
        ...

    @abstractmethod
    async def download(self, model_id: str) -> Path:
        """Download model files to disk. Returns the model directory."""
        ...
