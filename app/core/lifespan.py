"""Lifespan management with event-based architecture for robyn-ml-api."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from robyn import Robyn

from app.core.logger import LogIcon, logger
from app.core.settings import settings as st

AsyncHandler = Callable[[], Coroutine[Any, Any, None]]


class State:
    """Mutable application state container with attribute access."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        object.__setattr__(self, "_data", {})

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"State has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self._data[name]
        except KeyError:
            raise AttributeError(f"State has no attribute '{name}'") from None

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __iter__(self):
        return iter(self._data.keys())

    def __repr__(self) -> str:
        return f"State({self._data})"

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def clear(self) -> None:
        self._data.clear()


class BaseEvent[T](ABC):
    """Abstract base class for lifespan events."""

    name: str
    state: State

    @abstractmethod
    async def startup(self) -> T:
        """Initialize and return the event instance."""
        ...

    async def shutdown(self, instance: T) -> None:  # noqa: B027
        """Optional cleanup. Override if cleanup is needed."""

    @classmethod
    def has_shutdown(cls) -> bool:
        """Check if shutdown was overridden."""
        return cls.shutdown is not BaseEvent.shutdown


class Lifespan:
    """Manages application lifespan with event registration."""

    def __init__(self, app: Robyn) -> None:
        self._app = app
        self._event_classes: list[type[BaseEvent[Any]]] = []
        self._events: list[BaseEvent[Any]] = []
        self._state: State | None = None

    def register(self, event_cls: type[BaseEvent[Any]]) -> "Lifespan":
        """Register an event class. Returns self for chaining."""
        self._event_classes.append(event_cls)
        return self

    @property
    def state(self) -> State | None:
        """Access to state after startup execution."""
        return self._state

    @property
    def events(self) -> list[BaseEvent[Any]]:
        """Access to instantiated events after startup."""
        return self._events

    @property
    def startup(self) -> AsyncHandler:
        """Return async startup handler function."""

        async def _startup() -> None:
            logger.info("Starting application lifespan", icon=LogIcon.START, version=st.API_VERSION)
            self._state = State()

            for event_cls in self._event_classes:
                event = event_cls()
                event.state = self._state

                logger.info(f"Starting event: {event.name}", icon=LogIcon.PROCESSING)
                instance = await event.startup()
                setattr(self._state, event.name, instance)
                logger.info(f"Event ready: {event.name}", icon=LogIcon.SUCCESS)

                self._events.append(event)

            self._app.inject_global(state=self._state)
            logger.info("App state ready", icon=LogIcon.COMPLETE)

        return _startup

    @property
    def shutdown(self) -> AsyncHandler:
        """Return async shutdown handler function."""

        async def _shutdown() -> None:
            logger.info("Cleaning up app state", icon=LogIcon.TOOL)

            if not self._state:
                logger.info("No state to cleanup", icon=LogIcon.WARNING)
                return

            for event in reversed(self._events):
                if event.has_shutdown() and event.name in self._state:
                    logger.info(f"Shutting down: {event.name}", icon=LogIcon.PROCESSING)
                    instance = getattr(self._state, event.name)
                    await event.shutdown(instance)
                    logger.info(f"Shutdown complete: {event.name}", icon=LogIcon.SUCCESS)

            self._state.clear()
            logger.info("Cleanup complete", icon=LogIcon.COMPLETE)

        return _shutdown


def create_lifespan(app: Robyn) -> Lifespan:
    """Create lifespan manager for event registration."""
    return Lifespan(app)
