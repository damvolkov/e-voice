"""Tests for DeviceController — per-service GPU↔CPU switching with config persistence."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from pytest_mock import MockerFixture

from e_voice.core.settings import DeviceType
from e_voice.core.settings import settings as st
from e_voice.models.stt import ModelSpec
from e_voice.models.system import ServiceType
from e_voice.operational.controller import DeviceController, DeviceState, SwitchResult

##### FIXTURES #####


@pytest.fixture
def controller() -> DeviceController:
    return DeviceController()


@pytest.fixture
def mock_stt() -> AsyncMock:
    adapter = AsyncMock()
    adapter.is_loaded = AsyncMock(return_value=True)
    adapter.load = AsyncMock()
    adapter.unload = AsyncMock(return_value=True)
    adapter.loaded_models = lambda: []
    adapter.supported_devices = frozenset({DeviceType.CPU, DeviceType.GPU})
    return adapter


@pytest.fixture
def mock_tts() -> AsyncMock:
    adapter = AsyncMock()
    adapter.is_loaded = AsyncMock(return_value=True)
    adapter.load = AsyncMock()
    adapter.unload = AsyncMock(return_value=True)
    adapter.loaded_models = lambda: []
    adapter.supported_devices = frozenset({DeviceType.CPU, DeviceType.GPU})
    return adapter


##### PROPERTIES #####


async def test_controller_reads_stt_device(controller: DeviceController) -> None:
    assert controller.active_device(ServiceType.STT) == st.stt.device


async def test_controller_reads_tts_device(controller: DeviceController) -> None:
    assert controller.active_device(ServiceType.TTS) == st.tts.device


async def test_controller_not_transitioning_by_default(controller: DeviceController) -> None:
    assert controller.transitioning(ServiceType.STT) is False
    assert controller.transitioning(ServiceType.TTS) is False


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
        assert controller.state(ServiceType.STT) == expected_state
    finally:
        st.stt.device = original


async def test_controller_state_transitioning(controller: DeviceController) -> None:
    controller._transitioning[ServiceType.STT] = True
    assert controller.state(ServiceType.STT) == DeviceState.TRANSITIONING


##### SWITCH — ALREADY ON TARGET #####


async def test_switch_noop_when_already_on_target(
    controller: DeviceController, mock_stt: AsyncMock, mock_tts: AsyncMock
) -> None:
    current = controller.active_device(ServiceType.STT)
    result = await controller.switch(ServiceType.STT, current, mock_stt, mock_tts)
    assert result.success is True
    assert "already" in result.message
    mock_stt.load.assert_not_called()


##### SWITCH — SUCCESS #####


async def test_switch_stt_changes_device(
    controller: DeviceController,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
    mocker: MockerFixture,
) -> None:
    original = st.stt.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    target = DeviceType.CPU if original == DeviceType.GPU else DeviceType.GPU
    try:
        result = await controller.switch(ServiceType.STT, target, mock_stt, mock_tts)
        assert result.success is True
        assert result.device == target
        assert result.service == ServiceType.STT
        assert st.stt.device == target
    finally:
        st.stt.device = original


async def test_switch_tts_changes_device(
    controller: DeviceController,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
    mocker: MockerFixture,
) -> None:
    original = st.tts.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    target = DeviceType.CPU if original == DeviceType.GPU else DeviceType.GPU
    try:
        result = await controller.switch(ServiceType.TTS, target, mock_stt, mock_tts)
        assert result.success is True
        assert result.service == ServiceType.TTS
        assert st.tts.device == target
    finally:
        st.tts.device = original


##### SWITCH — LOADS MISSING MODEL #####


async def test_switch_stt_loads_model_if_not_loaded(
    controller: DeviceController,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
    mocker: MockerFixture,
) -> None:
    original = st.stt.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)
    mock_stt.is_loaded = AsyncMock(return_value=False)

    target = DeviceType.CPU if original == DeviceType.GPU else DeviceType.GPU
    try:
        result = await controller.switch(ServiceType.STT, target, mock_stt, mock_tts)
        assert result.success is True
        mock_stt.load.assert_called_once()
        mock_tts.load.assert_not_called()
    finally:
        st.stt.device = original


##### SWITCH — UNLOADS PREVIOUS DEVICE #####


async def test_switch_stt_unloads_previous_device(
    controller: DeviceController,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
    mocker: MockerFixture,
) -> None:
    original = st.stt.device
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    gpu_spec = ModelSpec(model_id=st.stt.model, device="gpu", compute_type="float16")
    mock_stt.loaded_models = lambda: [gpu_spec]

    st.stt.device = DeviceType.GPU
    try:
        result = await controller.switch(ServiceType.STT, DeviceType.CPU, mock_stt, mock_tts)
        assert result.success is True
        mock_stt.unload.assert_called_once_with(gpu_spec)
        mock_tts.unload.assert_not_called()
    finally:
        st.stt.device = original


##### SWITCH — CAPABILITY ERROR #####


async def test_switch_fails_when_device_not_supported(
    controller: DeviceController,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
) -> None:
    mock_stt.supported_devices = frozenset({DeviceType.CPU})
    original = st.stt.device
    st.stt.device = DeviceType.CPU
    try:
        result = await controller.switch(ServiceType.STT, DeviceType.GPU, mock_stt, mock_tts)
        assert result.success is False
        assert "does not support" in result.message
    finally:
        st.stt.device = original


##### SWITCH — FAILURE #####


async def test_switch_handles_error(
    controller: DeviceController,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
) -> None:
    original = st.stt.device
    mock_stt.is_loaded = AsyncMock(return_value=False)
    mock_stt.load = AsyncMock(side_effect=RuntimeError("CUDA OOM"))

    target = DeviceType.CPU if original == DeviceType.GPU else DeviceType.GPU
    result = await controller.switch(ServiceType.STT, target, mock_stt, mock_tts)
    assert result.success is False
    assert "CUDA OOM" in result.message
    assert controller.transitioning(ServiceType.STT) is False


##### PERSIST CONFIG #####


async def test_persist_config_writes_yaml(controller: DeviceController, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("stt:\n  device: gpu\ntts:\n  device: gpu\n")

    await controller._dc_persist_config(ServiceType.STT, device=DeviceType.CPU, config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["stt"]["device"] == "cpu"
    assert data["tts"]["device"] == "gpu"


async def test_persist_config_noop_when_missing(controller: DeviceController, tmp_path: Path) -> None:
    await controller._dc_persist_config(ServiceType.STT, device=DeviceType.CPU, config_dir=tmp_path / "nonexistent")


async def test_persist_config_preserves_other_fields(controller: DeviceController, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "system:\n  debug: true\n  port: 5500\n"
        "stt:\n  device: gpu\n  model: whisper-large\n  compute_type: int8\n"
        "tts:\n  device: gpu\n  default_voice: af_heart\n"
    )

    await controller._dc_persist_config(ServiceType.STT, device=DeviceType.CPU, config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["stt"]["device"] == "cpu"
    assert data["stt"]["model"] == "whisper-large"
    assert data["tts"]["device"] == "gpu"
    assert data["tts"]["default_voice"] == "af_heart"
    assert data["system"]["debug"] is True


async def test_persist_config_readonly_graceful(controller: DeviceController, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("stt:\n  device: gpu\n")
    config_file.chmod(0o444)

    await controller._dc_persist_config(ServiceType.STT, device=DeviceType.CPU, config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["stt"]["device"] == "gpu"


##### SWITCH BACKEND #####


async def test_switch_backend_same_backend_noop(controller: DeviceController) -> None:
    from e_voice.core.lifespan import State

    state = State()
    state.tts = AsyncMock()
    original = st.tts.backend
    result = await controller.switch_backend(ServiceType.TTS, original, state)
    assert result.success is True
    assert "already using" in result.message


async def test_switch_backend_unknown_backend(controller: DeviceController) -> None:
    from e_voice.core.lifespan import State

    state = State()
    result = await controller.switch_backend(ServiceType.TTS, "nonexistent", state)
    assert result.success is False
    assert "Unknown backend" in result.message


async def test_switch_backend_success(controller: DeviceController, mocker: MockerFixture) -> None:
    from e_voice.core.lifespan import State
    from e_voice.models.session import ConnectionRegistry

    mock_old_adapter = AsyncMock()
    mock_old_adapter.loaded_models = lambda: []

    mock_new_adapter = AsyncMock()
    mock_new_adapter.supported_devices = frozenset({DeviceType.CPU, DeviceType.GPU})
    mock_new_adapter.load = AsyncMock()

    state = State()
    state.tts = mock_old_adapter
    state.tts_connections = ConnectionRegistry()

    original_backend = st.tts.backend
    mocker.patch.dict(
        "e_voice.operational.controller.TTS_BACKENDS",
        {"test_backend": lambda: mock_new_adapter},
    )
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    try:
        result = await controller.switch_backend(ServiceType.TTS, "test_backend", state)
        assert result.success is True
        assert "test_backend" in result.message
        assert state.tts is mock_new_adapter
        mock_new_adapter.load.assert_called_once()
    finally:
        st.tts.backend = original_backend


async def test_switch_backend_adjusts_device_if_unsupported(
    controller: DeviceController, mocker: MockerFixture
) -> None:
    from e_voice.core.lifespan import State
    from e_voice.models.session import ConnectionRegistry

    mock_old = AsyncMock()
    mock_old.loaded_models = lambda: []

    mock_new = AsyncMock()
    mock_new.supported_devices = frozenset({DeviceType.GPU})
    mock_new.load = AsyncMock()

    state = State()
    state.tts = mock_old
    state.tts_connections = ConnectionRegistry()

    original_backend = st.tts.backend
    original_device = st.tts.device
    st.tts.device = DeviceType.CPU

    mocker.patch.dict(
        "e_voice.operational.controller.TTS_BACKENDS",
        {"gpu_only": lambda: mock_new},
    )
    mocker.patch.object(DeviceController, "_dc_persist_config", new_callable=AsyncMock)

    try:
        result = await controller.switch_backend(ServiceType.TTS, "gpu_only", state)
        assert result.success is True
        assert result.device == DeviceType.GPU
    finally:
        st.tts.backend = original_backend
        st.tts.device = original_device


##### PERSIST CONFIG — BACKEND #####


async def test_persist_config_writes_backend(controller: DeviceController, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("tts:\n  backend: kokoro\n  device: gpu\n")

    await controller._dc_persist_config(ServiceType.TTS, backend="qwen", config_dir=tmp_path)

    data = yaml.safe_load(config_file.read_text())
    assert data["tts"]["backend"] == "qwen"
    assert data["tts"]["device"] == "gpu"


##### SWITCH RESULT DATACLASS #####


async def test_switch_result_frozen() -> None:
    result = SwitchResult(success=True, service=ServiceType.STT, device=DeviceType.GPU, message="ok")
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]
