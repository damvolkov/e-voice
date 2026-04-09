"""Tests for ConnectionRegistry — typed session tracking."""

import asyncio

from e_voice.models.session import ConnectionRegistry


async def test_registry_starts_empty() -> None:
    reg = ConnectionRegistry()
    assert reg.count == 0
    assert reg.active == frozenset()


async def test_registry_add_remove() -> None:
    reg = ConnectionRegistry()
    await reg.add("ws-001")
    assert reg.count == 1
    assert "ws-001" in reg.active

    await reg.remove("ws-001")
    assert reg.count == 0


async def test_registry_remove_nonexistent() -> None:
    reg = ConnectionRegistry()
    await reg.remove("ghost")
    assert reg.count == 0


async def test_registry_wait_empty_already_empty() -> None:
    reg = ConnectionRegistry()
    result = await reg.wait_empty(timeout=0.1)
    assert result is True


async def test_registry_wait_empty_drains() -> None:
    reg = ConnectionRegistry()
    await reg.add("ws-001")

    async def _remove_later() -> None:
        await asyncio.sleep(0.05)
        await reg.remove("ws-001")

    asyncio.create_task(_remove_later())
    result = await reg.wait_empty(timeout=1.0)
    assert result is True


async def test_registry_wait_empty_timeout() -> None:
    reg = ConnectionRegistry()
    await reg.add("ws-001")
    result = await reg.wait_empty(timeout=0.05)
    assert result is False
