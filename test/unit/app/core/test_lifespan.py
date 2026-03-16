"""Tests for lifespan management."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.lifespan import BaseEvent, Lifespan, State, create_lifespan


# -----------------------------------------------------------------------------
# State Tests
# -----------------------------------------------------------------------------


class TestState:
    """Tests for the State class."""

    def test_state_set_and_get_attribute(self) -> None:
        """Verify attribute set/get operations."""
        state = State()
        state.foo = "bar"
        assert state.foo == "bar"

    def test_state_get_nonexistent_attribute_raises(self) -> None:
        """Verify AttributeError for missing attributes."""
        state = State()
        with pytest.raises(AttributeError, match="State has no attribute 'missing'"):
            _ = state.missing

    def test_state_delete_attribute(self) -> None:
        """Verify attribute deletion."""
        state = State()
        state.foo = "bar"
        del state.foo
        with pytest.raises(AttributeError):
            _ = state.foo

    def test_state_delete_nonexistent_raises(self) -> None:
        """Verify AttributeError when deleting missing attribute."""
        state = State()
        with pytest.raises(AttributeError):
            del state.missing

    def test_state_contains(self) -> None:
        """Verify __contains__ check."""
        state = State()
        state.foo = "bar"
        assert "foo" in state
        assert "missing" not in state

    def test_state_iter(self) -> None:
        """Verify iteration over keys."""
        state = State()
        state.a = 1
        state.b = 2
        assert set(state) == {"a", "b"}

    def test_state_repr(self) -> None:
        """Verify string representation."""
        state = State()
        state.foo = "bar"
        assert "foo" in repr(state)
        assert "bar" in repr(state)

    def test_state_get_with_default(self) -> None:
        """Verify get method with default value."""
        state = State()
        assert state.get("missing") is None
        assert state.get("missing", "default") == "default"
        state.foo = "bar"
        assert state.get("foo") == "bar"

    def test_state_clear(self) -> None:
        """Verify clear removes all data."""
        state = State()
        state.a = 1
        state.b = 2
        state.clear()
        assert "a" not in state
        assert "b" not in state


# -----------------------------------------------------------------------------
# BaseEvent Tests
# -----------------------------------------------------------------------------


class TestBaseEvent:
    """Tests for BaseEvent abstract class."""

    def test_has_shutdown_returns_false_when_not_overridden(self) -> None:
        """Verify has_shutdown returns False for default implementation."""

        class NoShutdownEvent(BaseEvent[str]):
            name = "no_shutdown"

            async def startup(self) -> str:
                return "started"

        assert not NoShutdownEvent.has_shutdown()

    def test_has_shutdown_returns_true_when_overridden(self) -> None:
        """Verify has_shutdown returns True when shutdown is overridden."""

        class WithShutdownEvent(BaseEvent[str]):
            name = "with_shutdown"

            async def startup(self) -> str:
                return "started"

            async def shutdown(self, instance: str) -> None:
                pass

        assert WithShutdownEvent.has_shutdown()


# -----------------------------------------------------------------------------
# Lifespan Tests
# -----------------------------------------------------------------------------


class TestLifespan:
    """Tests for Lifespan class."""

    def test_create_lifespan_returns_lifespan(self) -> None:
        """Verify create_lifespan returns Lifespan instance."""
        mock_app = MagicMock()
        lifespan = create_lifespan(mock_app)
        assert isinstance(lifespan, Lifespan)

    def test_register_returns_self_for_chaining(self) -> None:
        """Verify register returns self for method chaining."""
        mock_app = MagicMock()
        lifespan = Lifespan(mock_app)

        class DummyEvent(BaseEvent[str]):
            name = "dummy"

            async def startup(self) -> str:
                return "dummy"

        result = lifespan.register(DummyEvent)
        assert result is lifespan

    def test_state_is_none_before_startup(self) -> None:
        """Verify state is None before startup runs."""
        mock_app = MagicMock()
        lifespan = Lifespan(mock_app)
        assert lifespan.state is None

    def test_events_empty_before_startup(self) -> None:
        """Verify events list is empty before startup."""
        mock_app = MagicMock()
        lifespan = Lifespan(mock_app)
        assert lifespan.events == []

    async def test_startup_initializes_state(self) -> None:
        """Verify startup creates state and runs events."""
        mock_app = MagicMock()
        mock_app.inject_global = MagicMock()
        lifespan = Lifespan(mock_app)

        class TestEvent(BaseEvent[str]):
            name = "test_event"

            async def startup(self) -> str:
                return "test_value"

        lifespan.register(TestEvent)
        await lifespan.startup()

        assert lifespan.state is not None
        assert "test_event" in lifespan.state
        assert lifespan.state.test_event == "test_value"
        mock_app.inject_global.assert_called_once()

    async def test_shutdown_calls_event_shutdown(self) -> None:
        """Verify shutdown calls event shutdown methods in reverse order."""
        mock_app = MagicMock()
        mock_app.inject_global = MagicMock()
        shutdown_called = []

        class EventA(BaseEvent[str]):
            name = "event_a"

            async def startup(self) -> str:
                return "a"

            async def shutdown(self, instance: str) -> None:
                shutdown_called.append("a")

        class EventB(BaseEvent[str]):
            name = "event_b"

            async def startup(self) -> str:
                return "b"

            async def shutdown(self, instance: str) -> None:
                shutdown_called.append("b")

        lifespan = Lifespan(mock_app)
        lifespan.register(EventA).register(EventB)

        await lifespan.startup()
        await lifespan.shutdown()

        # Shutdown in reverse order
        assert shutdown_called == ["b", "a"]

    async def test_shutdown_clears_state(self) -> None:
        """Verify shutdown clears state."""
        mock_app = MagicMock()
        mock_app.inject_global = MagicMock()
        lifespan = Lifespan(mock_app)

        class TestEvent(BaseEvent[str]):
            name = "test"

            async def startup(self) -> str:
                return "value"

        lifespan.register(TestEvent)
        await lifespan.startup()
        await lifespan.shutdown()

        assert lifespan.state is not None and "test" not in lifespan.state

    async def test_shutdown_handles_no_state(self) -> None:
        """Verify shutdown handles case when state is None."""
        mock_app = MagicMock()
        lifespan = Lifespan(mock_app)
        # Should not raise
        await lifespan.shutdown()
