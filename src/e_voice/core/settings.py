"""Unified settings for e-voice."""

import tomllib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel
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

            return importlib.metadata.version("e-voice")
        except Exception:
            return "0.0.0"


type Device = Literal["auto", "cpu", "cuda"]
type Quantization = Literal[
    "default",
    "auto",
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]

_GPU_COMPUTE_TYPE: Quantization = "float16"
_CPU_COMPUTE_TYPE: Quantization = "int8"


def resolve_compute_type(device: Device, compute_type: Quantization) -> Quantization:
    """Pick optimal compute_type when 'default' based on resolved device."""
    if compute_type != "default":
        return compute_type
    if device == "cuda":
        return _GPU_COMPUTE_TYPE
    if device == "cpu":
        return _CPU_COMPUTE_TYPE
    # auto — probe at runtime
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return _GPU_COMPUTE_TYPE
    except Exception:
        pass
    return _CPU_COMPUTE_TYPE


class VadConfig(BaseModel):
    """Silero VAD filter parameters for faster-whisper."""

    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400

    def to_dict(self) -> dict:
        """Convert to dict for faster-whisper vad_parameters kwarg."""
        return {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }


class WhisperConfig(BaseModel):
    """Faster-whisper model configuration."""

    model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    inference_device: Device = "cuda"
    device_index: int = 0
    compute_type: Quantization = "default"
    cpu_threads: int = 0
    num_workers: int = 1


class Settings(BaseSettings):
    """Unified settings for e-voice service."""

    DEBUG: bool = True
    ENVIRONMENT: Literal["DEV", "PROD"] = "DEV"

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    PROJECT: ClassVar[dict] = read_pyproject(BASE_DIR / "pyproject.toml")
    API_NAME: ClassVar[str] = PROJECT.get("project", {}).get("name", "e-voice")
    API_DESCRIPTION: ClassVar[str] = PROJECT.get("project", {}).get("description", "Speech-to-text API")
    API_VERSION: ClassVar[str] = get_version(BASE_DIR)

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5500

    # Paths
    MODELS_PATH: Path = BASE_DIR / "models"

    # Workers
    MAX_WORKERS: int = 4

    # Whisper
    WHISPER_MODEL: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    WHISPER_DEVICE: Device = "cuda"
    WHISPER_DEVICE_INDEX: int = 0
    WHISPER_COMPUTE_TYPE: Quantization = "default"
    WHISPER_CPU_THREADS: int = 0
    WHISPER_NUM_WORKERS: int = 1

    # VAD (Silero Voice Activity Detection)
    VAD_ENABLED: bool = True
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_DURATION_MS: int = 250
    VAD_MAX_SPEECH_DURATION_S: float = float("inf")
    VAD_MIN_SILENCE_DURATION_MS: int = 2000
    VAD_SPEECH_PAD_MS: int = 400

    # Transcription defaults
    DEFAULT_LANGUAGE: str | None = None
    DEFAULT_RESPONSE_FORMAT: Literal["text", "json", "verbose_json", "srt", "vtt"] = "json"

    # WebSocket
    WS_MAX_INACTIVITY_SECONDS: float = 3.0

    @property
    def api_url(self) -> str:
        return f"http://{self.API_HOST}:{self.API_PORT}"

    @property
    def vad_config(self) -> VadConfig:
        return VadConfig(
            enabled=self.VAD_ENABLED,
            threshold=self.VAD_THRESHOLD,
            min_speech_duration_ms=self.VAD_MIN_SPEECH_DURATION_MS,
            max_speech_duration_s=self.VAD_MAX_SPEECH_DURATION_S,
            min_silence_duration_ms=self.VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=self.VAD_SPEECH_PAD_MS,
        )

    @property
    def whisper_config(self) -> WhisperConfig:
        return WhisperConfig(
            model=self.WHISPER_MODEL,
            inference_device=self.WHISPER_DEVICE,
            device_index=self.WHISPER_DEVICE_INDEX,
            compute_type=resolve_compute_type(self.WHISPER_DEVICE, self.WHISPER_COMPUTE_TYPE),
            cpu_threads=self.WHISPER_CPU_THREADS,
            num_workers=self.WHISPER_NUM_WORKERS,
        )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
