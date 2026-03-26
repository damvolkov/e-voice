"""Tests for SystemMonitor — CPU, RAM, GPU metrics polling."""

from types import SimpleNamespace

import pynvml
from pytest_mock import MockerFixture

from e_voice.operational.monitor import MetricsSnapshot, SystemMonitor

##### METRICS SNAPSHOT #####


async def test_metrics_snapshot_defaults() -> None:
    snap = MetricsSnapshot()
    assert snap.cpu_pct == 0.0
    assert snap.gpu_available is False
    assert snap.vram_used_mb == 0


##### POLL — NO GPU #####


async def test_poll_without_gpu(mocker: MockerFixture) -> None:
    mocker.patch(
        "e_voice.operational.monitor.pynvml.nvmlInit", side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)
    )
    mon = SystemMonitor()
    assert mon._nvml_ok is False

    mocker.patch("e_voice.operational.monitor.psutil.cpu_percent", return_value=42.0)
    mocker.patch(
        "e_voice.operational.monitor.psutil.virtual_memory",
        return_value=SimpleNamespace(used=8 * 1024**3, total=32 * 1024**3, percent=25.0),
    )
    snap = mon.poll()
    assert snap.cpu_pct == 42.0
    assert snap.ram_pct == 25.0
    assert snap.gpu_available is False


##### POLL — WITH GPU #####


async def test_poll_with_gpu(mocker: MockerFixture) -> None:
    mocker.patch("e_voice.operational.monitor.pynvml.nvmlInit")
    mocker.patch("e_voice.operational.monitor.pynvml.nvmlShutdown")
    mon = SystemMonitor()
    assert mon._nvml_ok is True

    mocker.patch("e_voice.operational.monitor.psutil.cpu_percent", return_value=10.0)
    mocker.patch(
        "e_voice.operational.monitor.psutil.virtual_memory",
        return_value=SimpleNamespace(used=4 * 1024**3, total=16 * 1024**3, percent=25.0),
    )
    mocker.patch("e_voice.operational.monitor.pynvml.nvmlDeviceGetHandleByIndex", return_value="h")
    mocker.patch(
        "e_voice.operational.monitor.pynvml.nvmlDeviceGetMemoryInfo",
        return_value=SimpleNamespace(used=2 * 1024**3, total=8 * 1024**3),
    )
    mocker.patch(
        "e_voice.operational.monitor.pynvml.nvmlDeviceGetUtilizationRates",
        return_value=SimpleNamespace(gpu=55),
    )
    snap = mon.poll()
    assert snap.gpu_available is True
    assert snap.gpu_util_pct == 55.0
    assert snap.vram_used_mb == 2048
    assert snap.vram_total_mb == 8192


##### HISTORY #####


async def test_poll_appends_history(mocker: MockerFixture) -> None:
    mocker.patch(
        "e_voice.operational.monitor.pynvml.nvmlInit", side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)
    )
    mon = SystemMonitor()
    mocker.patch("e_voice.operational.monitor.psutil.cpu_percent", return_value=50.0)
    mocker.patch(
        "e_voice.operational.monitor.psutil.virtual_memory",
        return_value=SimpleNamespace(used=1, total=2, percent=50.0),
    )
    mon.poll()
    mon.poll()
    assert len(mon.cpu_history) == 2
    assert len(mon.ram_history) == 2


##### SHUTDOWN #####


async def test_shutdown_cleans_nvml(mocker: MockerFixture) -> None:
    mocker.patch("e_voice.operational.monitor.pynvml.nvmlInit")
    mock_shutdown = mocker.patch("e_voice.operational.monitor.pynvml.nvmlShutdown")
    mon = SystemMonitor()
    mon.shutdown()
    mock_shutdown.assert_called_once()
    assert mon._nvml_ok is False
