"""Declarative WebSocket architecture for Robyn applications."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Literal, Self

from robyn import Robyn, WebSocket
from robyn.robyn import FunctionInfo

from e_voice.core.logger import logger

type WSEventType = Literal["connect", "message", "close"]


class BaseWebSocket:
    """Declarative WebSocket definition without app coupling."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._handlers: dict[WSEventType, Callable] = {}

    def on(self, event: WSEventType) -> Callable:
        """Decorator to register event handlers."""

        def decorator(handler: Callable) -> Callable:
            if event not in ("connect", "message", "close"):
                raise ValueError(f"Invalid event type: {event}")
            self._handlers[event] = handler
            return handler

        return decorator

    @property
    def handlers(self) -> dict[WSEventType, Callable]:
        return self._handlers


class WebSocketHandler:
    """Manages WebSocket registration and dependency injection for a Robyn app."""

    def __init__(self, app: Robyn, prefix: str = "") -> None:
        self._app = app
        self._prefix = prefix
        self._registered: list[WebSocket] = []

    def register(self, base_ws: BaseWebSocket) -> Self:
        """Register a BaseWebSocket with its handlers (dependencies injected later)."""
        if "message" not in base_ws.handlers:
            raise ValueError(f"WebSocket {base_ws.endpoint} must have a 'message' handler")

        endpoint = f"{self._prefix}{base_ws.endpoint}"
        websocket = WebSocket(self._app, endpoint)

        for event, handler in base_ws.handlers.items():
            self._wsh_register_handler(websocket, event, handler)

        self._app.add_web_socket(endpoint, websocket)
        self._registered.append(websocket)
        logger.info("registered websocket", step="START", endpoint=endpoint)
        return self

    def inject_dependencies(self) -> None:
        """Copy app global dependencies into every registered WebSocket. Call after startup."""
        global_deps = self._app.dependencies.get_global_dependencies()

        for websocket in self._registered:
            for key, value in global_deps.items():
                websocket.dependencies.add_global_dependency(**{key: value})

            for _event, func_info in websocket.methods.items():
                handler_args = func_info.args
                injected = websocket.dependencies.get_dependency_map(websocket)
                filtered = {k: v for k, v in injected.items() if k in handler_args}
                func_info.kwargs.update(filtered)

        logger.info("injected dependencies into websockets", step="OK", count=len(self._registered))

    def _wsh_register_handler(self, websocket: WebSocket, event: WSEventType, handler: Callable) -> None:
        """Register a single handler on the Robyn WebSocket."""
        params = dict(inspect.signature(handler).parameters)
        is_async = asyncio.iscoroutinefunction(handler)

        websocket.methods[event] = FunctionInfo(handler, is_async, len(params), params, kwargs={})
