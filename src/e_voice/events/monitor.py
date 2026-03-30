"""SystemMonitor lifespan event — poll system metrics as a backend singleton."""

from e_voice.core.lifespan import BaseEvent
from e_voice.operational.monitor import SystemMonitor


class MonitorEvent(BaseEvent[SystemMonitor]):
    """Manages SystemMonitor lifecycle: create on startup, shutdown NVML on stop."""

    name = "monitor"

    async def startup(self) -> SystemMonitor:
        return SystemMonitor()

    async def shutdown(self, instance: SystemMonitor) -> None:
        instance.shutdown()
