"""Unified settings for e-voice — YAML-based configuration with typed sub-models."""

import importlib.metadata
import tomllib
from contextlib import suppress
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

##### OPTIONAL DEPENDENCIES #####

_HAS_GIT = False
with suppress(ImportError):
    import git

    _HAS_GIT = True

_HAS_CT2 = False
with suppress(ImportError):
    import ctranslate2

    _HAS_CT2 = True

##### HELPERS #####


def read_pyproject(pyproject_path: Path) -> dict:
    """Read pyproject.toml into a dict."""
    with pyproject_path.open("rb") as file_handle:
        return tomllib.load(file_handle)


def get_version(base_dir: Path) -> str:
    """Get version from git tags or fallback to package metadata."""
    if _HAS_GIT:
        with suppress(Exception):
            repo = git.Repo(base_dir, search_parent_directories=True)
            if latest_tag := max(repo.tags, key=lambda t: t.commit.committed_datetime, default=None):
                return str(latest_tag)
    with suppress(Exception):
        return importlib.metadata.version("e-voice")
    return "0.0.0"


##### TYPE ALIASES #####


class DeviceType(StrEnum):
    AUTO = auto()
    CPU = auto()
    CUDA = auto()


class ComputeType(StrEnum):
    DEFAULT = auto()
    AUTO = auto()
    INT8 = auto()
    INT8_FLOAT16 = auto()
    INT8_FLOAT32 = auto()
    INT8_BFLOAT16 = auto()
    INT16 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()
    FLOAT32 = auto()


class ResponseFormatType(StrEnum):
    TEXT = auto()
    JSON = auto()
    VERBOSE_JSON = auto()
    SRT = auto()
    VTT = auto()


class EnvironmentType(StrEnum):
    DEV = auto()
    PROD = auto()


_GPU_COMPUTE: ComputeType = ComputeType.FLOAT16
_CPU_COMPUTE: ComputeType = ComputeType.INT8


def resolve_compute_type(device: DeviceType, compute_type: ComputeType) -> ComputeType:
    """Pick optimal compute_type when 'default' based on resolved device."""
    if compute_type != ComputeType.DEFAULT:
        return compute_type
    if device == DeviceType.CUDA:
        return _GPU_COMPUTE
    if device == DeviceType.CPU:
        return _CPU_COMPUTE
    if _HAS_CT2:
        with suppress(Exception):
            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):  # ty: ignore[possibly-missing-attribute]
                return _GPU_COMPUTE
    return _CPU_COMPUTE


##### CONFIG SUB-MODELS #####


class SystemConfig(BaseModel):
    """Server and application settings."""

    debug: bool = True
    environment: EnvironmentType = EnvironmentType.DEV
    host: str = "0.0.0.0"
    port: int = Field(default=5500, ge=1, le=65535)
    max_workers: int = Field(default=4, ge=1)


class STTConfig(BaseModel):
    """Speech-to-Text (faster-whisper) settings."""

    model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    device: DeviceType = DeviceType.CUDA
    device_index: int = Field(default=0, ge=0)
    compute_type: ComputeType = ComputeType.FLOAT16
    cpu_threads: int = Field(default=0, ge=0)
    num_workers: int = Field(default=1, ge=1)
    default_language: str | None = None
    default_response_format: ResponseFormatType = ResponseFormatType.JSON
    sample_rate: int = Field(default=16_000, ge=8000, le=48000)
    cpu_fallback: bool = True
    hf_allow_patterns: list[str] = Field(
        default=["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
    )


class TTSConfig(BaseModel):
    """Text-to-Speech (Kokoro-ONNX) settings."""

    device: DeviceType = DeviceType.CUDA
    sample_rate: int = Field(default=24_000, ge=8000, le=48000)
    default_voice: str = "af_heart"
    default_speed: float = Field(default=1.0, ge=0.1, le=5.0)
    model_filename: str = "kokoro-v1.0.onnx"
    voices_filename: str = "voices-v1.0.bin"
    release_url: str = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    download_chunk_size: int = Field(default=65_536, ge=4096)


class VADConfig(BaseModel):
    """Silero VAD (Voice Activity Detection) settings."""

    enabled: bool = True
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=300, ge=0)
    max_speech_duration_s: float = Field(default=float("inf"), ge=0.0)
    min_silence_duration_ms: int = Field(default=2000, ge=0)
    speech_pad_ms: int = Field(default=300, ge=0)

    def to_dict(self) -> dict:
        """Convert to dict for faster-whisper vad_parameters kwarg."""
        return {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }


class StreamingConfig(BaseModel):
    """Streaming STT (LocalAgreement) settings."""

    min_duration: float = Field(default=0.75, ge=0.1)
    max_buffer_seconds: float = Field(default=45.0, ge=5.0)
    trim_seconds: float = Field(default=30.0, ge=1.0)
    same_output_threshold: int = Field(default=7, ge=1)
    no_speech_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    inactivity_flush_seconds: float = Field(default=3.0, ge=0.5)


class FrontConfig(BaseModel):
    """Gradio UI settings."""

    enabled: bool = True
    port: int = Field(default=5600, ge=1, le=65535)
    share: bool = False


##### MAIN SETTINGS #####


class Settings(BaseSettings):
    """Root settings — loaded from data/config/config.yaml."""

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    PROJECT: ClassVar[dict] = read_pyproject(BASE_DIR / "pyproject.toml")
    API_NAME: ClassVar[str] = PROJECT.get("project", {}).get("name", "e-voice")
    API_DESCRIPTION: ClassVar[str] = PROJECT.get("project", {}).get("description", "Speech API")
    API_VERSION: ClassVar[str] = get_version(BASE_DIR)
    DATA_PATH: ClassVar[Path] = BASE_DIR / "data"
    MODELS_PATH: ClassVar[Path] = DATA_PATH / "models"
    CONFIG_PATH: ClassVar[Path] = DATA_PATH / "config"

    system: SystemConfig = SystemConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()
    streaming: StreamingConfig = StreamingConfig()
    front: FrontConfig = FrontConfig()

    model_config = SettingsConfigDict(
        yaml_file=str(BASE_DIR / "data" / "config" / "config.yaml"),
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @property
    def api_url(self) -> str:
        return f"http://{self.system.host}:{self.system.port}"


settings = Settings()
