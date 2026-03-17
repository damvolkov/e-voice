"""Unit tests for core/settings.py — config resolution and model construction."""

from pathlib import Path

import pytest

from e_voice.core.settings import Settings, VadConfig, WhisperConfig, read_pyproject, resolve_compute_type

##### RESOLVE_COMPUTE_TYPE #####


@pytest.mark.parametrize(
    ("device", "compute_type", "expected"),
    [
        ("cuda", "default", "float16"),
        ("cpu", "default", "int8"),
        ("cuda", "int8", "int8"),
        ("cpu", "float32", "float32"),
        ("cuda", "bfloat16", "bfloat16"),
    ],
    ids=["cuda-default", "cpu-default", "cuda-explicit", "cpu-explicit", "cuda-bfloat16"],
)
async def test_resolve_compute_type(device: str, compute_type: str, expected: str) -> None:
    assert resolve_compute_type(device, compute_type) == expected


##### VAD_CONFIG #####


async def test_vad_config_defaults() -> None:
    cfg = VadConfig()
    assert cfg.enabled is True
    assert cfg.threshold == 0.5
    assert cfg.min_speech_duration_ms == 250


async def test_vad_config_to_dict() -> None:
    cfg = VadConfig(threshold=0.7, min_speech_duration_ms=300)
    d = cfg.to_dict()
    assert d["threshold"] == 0.7
    assert d["min_speech_duration_ms"] == 300
    assert "enabled" not in d


async def test_vad_config_to_dict_keys() -> None:
    d = VadConfig().to_dict()
    expected_keys = {
        "threshold",
        "min_speech_duration_ms",
        "max_speech_duration_s",
        "min_silence_duration_ms",
        "speech_pad_ms",
    }
    assert set(d.keys()) == expected_keys


##### WHISPER_CONFIG #####


async def test_whisper_config_defaults() -> None:
    cfg = WhisperConfig()
    assert cfg.inference_device == "cuda"
    assert cfg.compute_type == "default"
    assert cfg.num_workers == 1


async def test_whisper_config_custom() -> None:
    cfg = WhisperConfig(model="tiny", inference_device="cpu", compute_type="int8")
    assert cfg.model == "tiny"
    assert cfg.inference_device == "cpu"
    assert cfg.compute_type == "int8"


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


async def test_settings_vad_config_property() -> None:
    st = Settings()
    cfg = st.vad_config
    assert isinstance(cfg, VadConfig)
    assert cfg.enabled is True


async def test_settings_whisper_config_property() -> None:
    st = Settings()
    cfg = st.whisper_config
    assert isinstance(cfg, WhisperConfig)
    assert cfg.model == st.WHISPER_MODEL


async def test_settings_api_url() -> None:
    st = Settings()
    assert st.api_url.startswith("http://")
