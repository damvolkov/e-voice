"""Session tracking — typed connection registries shared across HTTP and WS servers."""

import asyncio
from dataclasses import dataclass, field


@dataclass
class ConnectionRegistry:
    """Thread-safe registry of active WebSocket connection IDs per service.

    Shared via lifespan State between Robyn HTTP and standalone WS server.
    Used by backend switch to drain active connections before swapping adapters.
    """

    _ids: set[str] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _drain_event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self._drain_event.set()

    @property
    def count(self) -> int:
        return len(self._ids)

    @property
    def active(self) -> frozenset[str]:
        return frozenset(self._ids)

    async def add(self, conn_id: str) -> None:
        async with self._lock:
            self._ids.add(conn_id)
            self._drain_event.clear()

    async def remove(self, conn_id: str) -> None:
        async with self._lock:
            self._ids.discard(conn_id)
            if not self._ids:
                self._drain_event.set()

    async def wait_empty(self, timeout: float = 10.0) -> bool:
        """Wait until all connections are drained. Returns True if drained, False if timeout."""
        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False
