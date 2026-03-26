"""Live system metrics polling — CPU, RAM, GPU utilization and VRAM."""

import contextlib
from collections import deque
from dataclasses import dataclass, field

import psutil
import pynvml

_HISTORY_SIZE = 30


@dataclass(slots=True)
class MetricsSnapshot:
    """Single point-in-time system metrics reading."""

    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_pct: float = 0.0
    gpu_util_pct: float = 0.0
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    vram_pct: float = 0.0
    gpu_available: bool = False


@dataclass(slots=True)
class SystemMonitor:
    """Polls system metrics and maintains sparkline history buffers."""

    current: MetricsSnapshot = field(default_factory=MetricsSnapshot)
    cpu_history: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY_SIZE))
    ram_history: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY_SIZE))
    gpu_util_history: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY_SIZE))
    vram_history: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY_SIZE))
    _nvml_ok: bool = False

    def __post_init__(self) -> None:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlInit()
            self._nvml_ok = True

    def poll(self) -> MetricsSnapshot:
        """Read all metrics synchronously. Call from a thread."""
        mem = psutil.virtual_memory()
        snap = MetricsSnapshot(
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_used_gb=round(mem.used / (1024**3), 1),
            ram_total_gb=round(mem.total / (1024**3), 1),
            ram_pct=mem.percent,
        )

        if self._nvml_ok:
            with contextlib.suppress(pynvml.NVMLError):
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snap.gpu_available = True
                snap.gpu_util_pct = float(util.gpu)
                snap.vram_used_mb = int(mem_info.used / (1024**2))
                snap.vram_total_mb = int(mem_info.total / (1024**2))
                snap.vram_pct = round(mem_info.used / mem_info.total * 100, 1) if mem_info.total else 0.0

        self.current = snap
        self.cpu_history.append(snap.cpu_pct)
        self.ram_history.append(snap.ram_pct)
        self.gpu_util_history.append(snap.gpu_util_pct)
        self.vram_history.append(snap.vram_pct)
        return snap

    def shutdown(self) -> None:
        """Clean up NVML resources."""
        if self._nvml_ok:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
            self._nvml_ok = False
