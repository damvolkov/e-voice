"""Core models for request/response handling."""

from enum import StrEnum


class BodyType(StrEnum):
    """Body content type classification for request parsing."""

    PYDANTIC = "pydantic"
    JSONABLE = "jsonable"
    RAW = "raw"
    FILE = "file"


class UploadFile:
    """Container for uploaded files from multipart/form-data requests."""

    __slots__ = ("files",)

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}

    def __bool__(self) -> bool:
        return bool(self.files)

    def __iter__(self):
        return iter(self.files.items())

    def get(self, name: str) -> bytes | None:
        """Get file bytes by field name."""
        return self.files.get(name)

    def keys(self) -> list[str]:
        """Get all file field names."""
        return list(self.files.keys())
