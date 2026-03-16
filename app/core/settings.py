"""Unified settings for robyn-ml-api."""

import tomllib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def read_pyproject(pyproject_path: Path) -> dict:
    """Read pyproject.toml into a dict."""
    with pyproject_path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def get_version(base_dir: Path) -> str:
    """Get version from git tags or fallback to package metadata."""
    try:
        import git

        repo = git.Repo(base_dir, search_parent_directories=True)
        latest_tag = max(repo.tags, key=lambda t: t.commit.committed_datetime, default=None)
        return str(latest_tag) if latest_tag else "0.0.0"
    except Exception:
        try:
            import importlib.metadata

            return importlib.metadata.version("robyn-ml-api")
        except Exception:
            return "0.0.0"


class Settings(BaseSettings):
    """Unified settings for robyn-ml-api service."""

    DEBUG: bool = True
    ENVIRONMENT: Literal["DEV", "PROD"] = "DEV"

    # ClassVar to prevent Pydantic from trying to load from env
    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent.parent
    PROJECT: ClassVar[dict] = read_pyproject(BASE_DIR / "pyproject.toml")
    API_NAME: ClassVar[str] = PROJECT.get("project", {}).get("name", "robyn-ml-api")
    API_DESCRIPTION: ClassVar[str] = PROJECT.get("project", {}).get("description", "ML API Template")
    API_VERSION: ClassVar[str] = get_version(BASE_DIR)

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Paths
    DATA_PATH: ClassVar[Path] = BASE_DIR / "data"
    MODELS_PATH: ClassVar[Path] = DATA_PATH / "models"

    # Workers
    MAX_WORKERS: int = 4

    @property
    def api_url(self) -> str:
        return f"http://{self.API_HOST}:{self.API_PORT}"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()  # type: ignore
