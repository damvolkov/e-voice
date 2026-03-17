"""Unit tests for core/settings.py — config resolution and model construction."""

from pathlib import Path

import pytest

from e_voice.core.settings import (
    ComputeType,
    DeviceType,
    STTConfig,
    Settings,
    TTSConfig,
    VADConfig,
    read_pyproject,
    resolve_compute_type,
)

##### RESOLVE_COMPUTE_TYPE #####


@pytest.mark.parametrize(
    ("device", "compute_type", "expected"),
    [
        (DeviceType.CUDA, ComputeType.DEFAULT, ComputeType.FLOAT16),
        (DeviceType.CPU, ComputeType.DEFAULT, ComputeType.INT8),
        (DeviceType.CUDA, ComputeType.INT8, ComputeType.INT8),
        (DeviceType.CPU, ComputeType.FLOAT32, ComputeType.FLOAT32),
        (DeviceType.CUDA, ComputeType.BFLOAT16, ComputeType.BFLOAT16),
    ],
    ids=["cuda-default", "cpu-default", "cuda-explicit", "cpu-explicit", "cuda-bfloat16"],
)
async def test_resolve_compute_type(device: DeviceType, compute_type: ComputeType, expected: ComputeType) -> None:
    assert resolve_compute_type(device, compute_type) == expected


##### VAD_CONFIG #####


async def test_vad_config_defaults() -> None:
    cfg = VADConfig()
    assert cfg.enabled is True
    assert cfg.threshold == 0.65
    assert cfg.min_speech_duration_ms == 300


async def test_vad_config_to_dict() -> None:
    cfg = VADConfig(threshold=0.7, min_speech_duration_ms=300)
    d = cfg.to_dict()
    assert d["threshold"] == 0.7
    assert d["min_speech_duration_ms"] == 300
    assert "enabled" not in d


async def test_vad_config_to_dict_keys() -> None:
    d = VADConfig().to_dict()
    expected_keys = {
        "threshold",
        "min_speech_duration_ms",
        "max_speech_duration_s",
        "min_silence_duration_ms",
        "speech_pad_ms",
    }
    assert set(d.keys()) == expected_keys


##### STT_CONFIG #####


async def test_stt_config_defaults() -> None:
    cfg = STTConfig()
    assert cfg.device == DeviceType.CUDA
    assert cfg.compute_type == ComputeType.FLOAT16
    assert cfg.num_workers == 1


async def test_stt_config_custom() -> None:
    cfg = STTConfig(model="tiny", device=DeviceType.CPU, compute_type=ComputeType.INT8)
    assert cfg.model == "tiny"
    assert cfg.device == DeviceType.CPU
    assert cfg.compute_type == ComputeType.INT8


##### TTS_CONFIG #####


async def test_tts_config_defaults() -> None:
    cfg = TTSConfig()
    assert cfg.device == DeviceType.CUDA
    assert cfg.default_voice == "af_heart"
    assert cfg.default_speed == 1.0


##### READ_PYPROJECT #####


async def test_read_pyproject() -> None:
    base = Path(__file__).resolve().parents[4]
    data = read_pyproject(base / "pyproject.toml")
    assert "project" in data
    assert "name" in data["project"]


##### SETTINGS SINGLETON #####


async def test_settings_class_vars() -> None:
    assert Settings.BASE_DIR.is_dir()
    assert "name" in Settings.PROJECT.get("project", {})


async def test_settings_vad_section() -> None:
    s = Settings()
    assert isinstance(s.vad, VADConfig)
    assert s.vad.enabled is True


async def test_settings_stt_section() -> None:
    s = Settings()
    assert isinstance(s.stt, STTConfig)


async def test_settings_api_url() -> None:
    s = Settings()
    assert s.api_url.startswith("http://")
