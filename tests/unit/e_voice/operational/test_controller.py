"""Tests for DeviceController — GPU↔CPU switching with config persistence."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from pytest_mock import MockerFixture

from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.stt import ModelSpec
from e_voice.models.tts import TTSModelSpec
from e_voice.operational.controller import DeviceController, DeviceState, SwitchResult

##### FIXTURES #####


@pytest.fixture
def controller() -> DeviceController:
    return DeviceController()


@pytest.fixture
def mock_whisper() -> AsyncMock:
    adapter = AsyncMock()
    adapter.is_loaded = AsyncMock(return_value=True)
    adapter.load = AsyncMock()
    adapter.unload = AsyncMock(return_value=True)
    adapter.loaded_models = lambda: []
    return adapter


@pytest.fixture
def mock_kokoro() -> AsyncMock:
    adapter = AsyncMock()
    adapter.is_loaded = AsyncMock(return_value=True)
    adapter.load = AsyncMock()
    adapter.unload = AsyncMock(return_value=True)
    adapter.loaded_models = lambda: []
    return adapter


##### PROPERTIES #####


async def test_controller_reads_device_from_settings(controller: DeviceController) -> None:
    assert controller.active_device == st.stt.device


async def test_controller_not_transitioning_by_default(controller: DeviceController) -> None:
    assert controller.transitioning is False


@pytest.mark.parametrize(
    ("device", "expected_state"),
    [(DeviceType.GPU, DeviceState.GPU), (DeviceType.CPU, DeviceState.CPU)],
    ids=["gpu", "cpu"],
)
async def test_controller_state_matches_device(
    controller: DeviceController, device: DeviceType, expected_state: DeviceState
) -> None:
    original = st.stt.device
    try:
        st.stt.device = device
        assert controller.state == expected_state
    finally:
        st.stt.device = original


async def test_controller_state_transitioning(controller: DeviceController) -> None:
    controller._transitioning = True
    assert controller.state == DeviceState.TRANSITIONING


##### SWITCH — ALREADY ON TARGET #####


async def test_switch_noop_when_already_on_target(
    controller: DeviceController, mock_whisper: AsyncMock, mock_kokoro: AsyncMock
) -> None:
    current = controller.active_device
    result = await controller.switch(current, mock_whisper, mock_kokoro)
    assert result.success is True
    assert "Already" in result.message
    mock_whisper.load.assert_not_called()


##### SWITCH — SUCCESS #####


async def test_switch_changes_device(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
    mocker: MockerFixture,
) -> None:
    original_stt = st.stt.device
    original_tts = st.tts.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    target = DeviceType.CPU if original_stt == DeviceType.GPU else DeviceType.GPU
    try:
        result = await controller.switch(target, mock_whisper, mock_kokoro)
        assert result.success is True
        assert result.device == target
        assert st.stt.device == target
        assert st.tts.device == target
    finally:
        st.stt.device = original_stt
        st.tts.device = original_tts


##### SWITCH — LOADS MISSING MODEL #####


async def test_switch_loads_model_if_not_loaded(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
    mocker: MockerFixture,
) -> None:
    original_stt = st.stt.device
    original_tts = st.tts.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)
    mock_whisper.is_loaded = AsyncMock(return_value=False)
    mock_kokoro.is_loaded = AsyncMock(return_value=False)

    target = DeviceType.CPU if original_stt == DeviceType.GPU else DeviceType.GPU
    try:
        result = await controller.switch(target, mock_whisper, mock_kokoro)
        assert result.success is True
        mock_whisper.load.assert_called_once()
        mock_kokoro.load.assert_called_once()
    finally:
        st.stt.device = original_stt
        st.tts.device = original_tts


##### SWITCH — UNLOADS PREVIOUS DEVICE #####


async def test_switch_unloads_previous_device_models(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
    mocker: MockerFixture,
) -> None:
    """Switching GPU→CPU must unload GPU models from both adapters."""
    original_stt = st.stt.device
    original_tts = st.tts.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    gpu_stt_spec = ModelSpec(model_id=st.stt.model, device="gpu", compute_type="float16")
    gpu_tts_spec = TTSModelSpec(device=DeviceType.GPU)
    mock_whisper.loaded_models = lambda: [gpu_stt_spec]
    mock_kokoro.loaded_models = lambda: [gpu_tts_spec]

    st.stt.device = DeviceType.GPU
    try:
        result = await controller.switch(DeviceType.CPU, mock_whisper, mock_kokoro)
        assert result.success is True
        mock_whisper.unload.assert_called_once_with(gpu_stt_spec)
        mock_kokoro.unload.assert_called_once_with(gpu_tts_spec)
    finally:
        st.stt.device = original_stt
        st.tts.device = original_tts


async def test_switch_skips_unload_when_same_device(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
) -> None:
    """_dc_unload_previous is a no-op when previous == target."""
    await controller._dc_unload_previous(DeviceType.GPU, DeviceType.GPU, mock_whisper, mock_kokoro)
    mock_whisper.unload.assert_not_called()
    mock_kokoro.unload.assert_not_called()


async def test_unload_previous_only_targets_previous_device(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
) -> None:
    """Only models on the previous device are unloaded, not the target device."""
    gpu_spec = ModelSpec(model_id="test", device="gpu", compute_type="float16")
    cpu_spec = ModelSpec(model_id="test", device="cpu", compute_type="int8")
    mock_whisper.loaded_models = lambda: [gpu_spec, cpu_spec]
    mock_kokoro.loaded_models = lambda: []

    await controller._dc_unload_previous(DeviceType.GPU, DeviceType.CPU, mock_whisper, mock_kokoro)
    mock_whisper.unload.assert_called_once_with(gpu_spec)


##### SWITCH — FAILURE #####


async def test_switch_handles_error(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
) -> None:
    original = st.stt.device
    mock_whisper.is_loaded = AsyncMock(return_value=False)
    mock_whisper.load = AsyncMock(side_effect=RuntimeError("CUDA OOM"))

    target = DeviceType.CPU if original == DeviceType.GPU else DeviceType.GPU
    result = await controller.switch(target, mock_whisper, mock_kokoro)
    assert result.success is False
    assert "CUDA OOM" in result.message
    assert controller.transitioning is False


##### PERSIST CONFIG #####


async def test_persist_config_writes_yaml(controller: DeviceController, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("stt:\n  device: gpu\ntts:\n  device: gpu\n")

    await controller._dc_persist_config(DeviceType.CPU, config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["stt"]["device"] == "cpu"
    assert data["tts"]["device"] == "cpu"


async def test_persist_config_noop_when_missing(controller: DeviceController, tmp_path: Path) -> None:
    await controller._dc_persist_config(DeviceType.CPU, config_dir=tmp_path / "nonexistent")


async def test_persist_config_preserves_other_fields(controller: DeviceController, tmp_path: Path) -> None:
    """Config YAML must keep all non-device fields intact after persist."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "system:\n  debug: true\n  port: 5500\n"
        "stt:\n  device: gpu\n  model: whisper-large\n  compute_type: int8\n"
        "tts:\n  device: gpu\n  default_voice: af_heart\n"
    )

    await controller._dc_persist_config(DeviceType.CPU, config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["stt"]["device"] == "cpu"
    assert data["stt"]["model"] == "whisper-large"
    assert data["stt"]["compute_type"] == "int8"
    assert data["tts"]["device"] == "cpu"
    assert data["tts"]["default_voice"] == "af_heart"
    assert data["system"]["debug"] is True
    assert data["system"]["port"] == 5500


async def test_persist_config_readonly_graceful(controller: DeviceController, tmp_path: Path) -> None:
    """Read-only config file should not raise — just skip."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("stt:\n  device: gpu\n")
    config_file.chmod(0o444)

    await controller._dc_persist_config(DeviceType.CPU, config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["stt"]["device"] == "gpu"


async def test_switch_end_to_end_persists_yaml(
    controller: DeviceController,
    mock_whisper: AsyncMock,
    mock_kokoro: AsyncMock,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """Full switch calls persist with the target device."""
    mock_persist = mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    original_stt = st.stt.device
    original_tts = st.tts.device
    target = DeviceType.CPU if original_stt == DeviceType.GPU else DeviceType.GPU
    try:
        result = await controller.switch(target, mock_whisper, mock_kokoro)
        assert result.success is True
        assert st.stt.device == target
        assert st.tts.device == target
        mock_persist.assert_called_once_with(target)
    finally:
        st.stt.device = original_stt
        st.tts.device = original_tts


##### SWITCH RESULT DATACLASS #####


async def test_switch_result_frozen() -> None:
    result = SwitchResult(success=True, device=DeviceType.GPU, message="ok")
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]
