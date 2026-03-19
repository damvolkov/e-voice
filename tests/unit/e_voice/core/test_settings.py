from pathlib import Path

import pytest

from e_voice.core.settings import (
    ComputeType,
    DeviceType,
    FrontConfig,
    Settings,
    StreamingConfig,
    STTConfig,
    SystemConfig,
    TTSConfig,
    VADConfig,
    WebSocketConfig,
    get_version,
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


##### GET_VERSION #####


async def test_get_version_returns_string() -> None:
    version = get_version(Settings.BASE_DIR)
    assert isinstance(version, str)
    assert len(version) > 0


async def test_get_version_fallback_on_bad_path() -> None:
    version = get_version(Path("/nonexistent"))
    assert isinstance(version, str)


##### SYSTEM CONFIG #####


async def test_system_config_defaults() -> None:
    cfg = SystemConfig()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 5500
    assert cfg.max_workers == 4
    assert cfg.debug is True


##### STREAMING CONFIG #####


async def test_streaming_config_defaults() -> None:
    cfg = StreamingConfig()
    assert cfg.min_duration == 0.75
    assert cfg.max_buffer_seconds == 45.0
    assert cfg.same_output_threshold == 7


##### WEBSOCKET CONFIG #####


async def test_websocket_config_defaults() -> None:
    cfg = WebSocketConfig()
    assert cfg.port == 5700


##### FRONT CONFIG #####


async def test_front_config_defaults() -> None:
    cfg = FrontConfig()
    assert cfg.enabled is True
    assert cfg.port == 5600
    assert cfg.share is False


##### RESOLVE_COMPUTE_TYPE — AUTO DEVICE #####


async def test_resolve_compute_type_auto_without_ct2(mocker) -> None:
    mocker.patch("e_voice.core.settings._HAS_CT2", False)
    result = resolve_compute_type(DeviceType.AUTO, ComputeType.DEFAULT)
    assert result == ComputeType.INT8
