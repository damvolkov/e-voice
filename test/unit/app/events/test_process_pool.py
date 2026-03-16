"""Tests for process pool event."""

import asyncio
from concurrent.futures import ProcessPoolExecutor

import pytest

from app.events.process_pool import create_process_pool, process_pool_context


def _cpu_bound_task(x: int) -> int:
    """Simple CPU-bound task for testing."""
    return x * x


def test_create_process_pool_returns_executor() -> None:
    """Verify create_process_pool returns ProcessPoolExecutor."""
    pool = create_process_pool(max_workers=2)
    try:
        assert isinstance(pool, ProcessPoolExecutor)
    finally:
        pool.shutdown(wait=True)


def test_create_process_pool_executes_tasks() -> None:
    """Verify pool can execute tasks."""
    pool = create_process_pool(max_workers=2)
    try:
        results = list(pool.map(_cpu_bound_task, [1, 2, 3, 4]))
        assert results == [1, 4, 9, 16]
    finally:
        pool.shutdown(wait=True)


def test_process_pool_context_manager() -> None:
    """Verify context manager creates and shuts down pool."""
    with process_pool_context(max_workers=2) as pool:
        assert isinstance(pool, ProcessPoolExecutor)
        result = pool.submit(_cpu_bound_task, 5).result()
        assert result == 25


async def test_process_pool_asyncio_compatibility() -> None:
    """Verify ProcessPoolExecutor works with asyncio.to_thread."""
    pool = create_process_pool(max_workers=2)

    def run_in_pool() -> list[int]:
        return list(pool.map(_cpu_bound_task, range(5)))

    try:
        result = await asyncio.to_thread(run_in_pool)
        assert result == [0, 1, 4, 9, 16]
    finally:
        pool.shutdown(wait=True)


async def test_process_pool_run_in_executor() -> None:
    """Verify ProcessPoolExecutor works with loop.run_in_executor."""
    pool = create_process_pool(max_workers=2)
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(pool, _cpu_bound_task, 7)
        assert result == 49
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_create_process_pool_respects_max_workers(workers: int) -> None:
    """Verify max_workers parameter is respected."""
    pool = create_process_pool(max_workers=workers)
    try:
        assert pool._max_workers == workers  # type: ignore[unresolved-attribute]
    finally:
        pool.shutdown(wait=True)
