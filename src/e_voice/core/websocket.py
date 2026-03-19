"""WebSocket architecture — Robyn text-frame handlers and standalone binary-frame server."""

import asyncio
import inspect
import threading
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, urlparse

import websockets
from robyn import Robyn, WebSocket
from robyn.robyn import FunctionInfo
from websockets.exceptions import ConnectionClosed

from e_voice.core.logger import logger

##### TYPES #####

type WSEventType = Literal["connect", "message", "close"]

##### ROBYN TEXT-FRAME WEBSOCKETS #####


class BaseWebSocket:
    """Robyn WebSocket definition — text frames only (str/base64)."""

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
    """Robyn WebSocket registration and dependency injection."""

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


##### STANDALONE BINARY-FRAME WEBSOCKETS #####


class Connection:
    """Binary-aware WebSocket connection — native str and bytes frames."""

    __slots__ = ("id", "path", "query_params", "state", "_ws")

    def __init__(self, *, id: str, path: str, query_params: dict[str, str], state: Any, ws: Any) -> None:
        self.id = id
        self.path = path
        self.query_params: dict[str, str] = query_params
        self.state = state
        self._ws = ws

    async def send(self, data: str | bytes) -> None:
        """Send text frame (str) or binary frame (bytes)."""
        await self._ws.send(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code, reason)

    async def __aiter__(self) -> AsyncGenerator[str | bytes, None]:
        """Yield messages until connection closes. str = text frame, bytes = binary frame."""
        try:
            async for msg in self._ws:
                yield msg
        except ConnectionClosed:
            return


class WebSocketRouter:
    """Declarative route registration for the standalone WebSocket server."""

    __slots__ = ("_routes",)

    def __init__(self) -> None:
        self._routes: dict[str, Callable] = {}

    @property
    def routes(self) -> dict[str, Callable]:
        return self._routes

    def __call__(self, *paths: str) -> Callable:
        """Register handler for one or more paths: @router("/path1", "/path2")."""

        def decorator(handler: Callable) -> Callable:
            for p in paths:
                self._routes[p] = handler
            return handler

        return decorator


class WebSocketServer:
    """Standalone websockets server — native binary frame support, runs in a daemon thread."""

    __slots__ = ("_port", "_routes", "_state")

    def __init__(self, port: int) -> None:
        self._port = port
        self._routes: dict[str, Callable] = {}
        self._state: Any = None

    @property
    def port(self) -> int:
        return self._port

    def include(self, router: WebSocketRouter) -> None:
        self._routes.update(router.routes)

    def launch_background(self, state: Any) -> None:
        """Start server in a daemon thread with its own event loop."""
        self._state = state
        thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        thread.start()
        logger.info("🔌 WS_SERVER_STARTED", extra={"port": self._port, "routes": sorted(self._routes)})

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        async with websockets.serve(self._dispatch, "0.0.0.0", self._port):
            await asyncio.Future()

    async def _dispatch(self, ws) -> None:
        """Route incoming connection to the matching handler."""
        parsed = urlparse(ws.request.path)
        path = parsed.path

        if (handler := self._routes.get(path)) is None:
            await ws.close(4004, f"No handler for {path}")
            return

        conn = Connection(
            id=ws.id.hex,
            path=path,
            query_params=dict(parse_qsl(parsed.query)),
            state=self._state,
            ws=ws,
        )

        logger.info("ws connected", step="WS", path=path, client=conn.id)
        try:
            await handler(conn)
        except Exception as exc:
            logger.error("ws handler error", step="WS", path=path, error=str(exc))
        finally:
            logger.info("ws disconnected", step="WS", path=path, client=conn.id)
