"""Process pool lifespan event example."""

import multiprocessing as mp
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

from app.core.lifespan import BaseEvent
from app.core.settings import settings as st


def create_process_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    """Create ProcessPoolExecutor with spawn context for asyncio compatibility."""
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=max_workers or mp.cpu_count(),
        mp_context=ctx,
    )


@contextmanager
def process_pool_context(max_workers: int | None = None) -> Generator[ProcessPoolExecutor, None, None]:
    """Context manager for temporary process pool usage."""
    pool = create_process_pool(max_workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


class ProcessPoolEvent(BaseEvent[ProcessPoolExecutor]):
    """Manages ProcessPoolExecutor lifecycle."""

    name = "process_pool"

    async def startup(self) -> ProcessPoolExecutor:
        """Create and return the process pool."""
        max_workers = st.MAX_WORKERS or mp.cpu_count()
        return create_process_pool(max_workers=max_workers)

    async def shutdown(self, instance: ProcessPoolExecutor) -> None:
        """Shutdown the process pool."""
        instance.shutdown(wait=True)
