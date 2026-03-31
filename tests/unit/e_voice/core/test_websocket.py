"""Unit tests for core/websocket.py — BaseWebSocket, WebSocketHandler, Connection, WebSocketRouter, WSRoute."""

import pytest

from e_voice.core.websocket import BaseWebSocket, BaseWSParams, Connection, WebSocketHandler, WebSocketRouter, WSRoute

##### BASE WEBSOCKET #####


async def test_base_websocket_init() -> None:
    ws = BaseWebSocket("/v1/test")
    assert ws.endpoint == "/v1/test"
    assert ws.handlers == {}


async def test_base_websocket_on_registers_handler() -> None:
    ws = BaseWebSocket("/v1/test")

    @ws.on("connect")
    def on_connect():
        return ""

    assert "connect" in ws.handlers
    assert ws.handlers["connect"] is on_connect


async def test_base_websocket_on_message() -> None:
    ws = BaseWebSocket("/v1/test")

    @ws.on("message")
    async def on_message(msg):
        return msg

    assert "message" in ws.handlers


async def test_base_websocket_on_close() -> None:
    ws = BaseWebSocket("/v1/test")

    @ws.on("close")
    def on_close():
        return ""

    assert "close" in ws.handlers


async def test_base_websocket_on_invalid_event() -> None:
    ws = BaseWebSocket("/v1/test")
    with pytest.raises(ValueError, match="Invalid event type"):
        ws.on("invalid")(lambda: "")  # ty: ignore[invalid-argument-type]


async def test_base_websocket_handlers_property() -> None:
    ws = BaseWebSocket("/v1/test")

    @ws.on("connect")
    def h1():
        return ""

    @ws.on("message")
    async def h2(msg):
        return ""

    assert len(ws.handlers) == 2
    assert ws.handlers is ws._handlers


async def test_base_websocket_handler_shared_via_assignment() -> None:
    ws1 = BaseWebSocket("/v1/main")
    ws2 = BaseWebSocket("/v1/alias")

    @ws1.on("connect")
    def on_connect():
        return ""

    @ws1.on("message")
    async def on_message(msg):
        return msg

    ws2._handlers = ws1._handlers
    assert ws2.handlers["connect"] is ws1.handlers["connect"]
    assert ws2.handlers["message"] is ws1.handlers["message"]


##### WEBSOCKET HANDLER #####


async def test_websocket_handler_register_requires_message(mocker) -> None:
    mock_app = mocker.MagicMock()
    handler = WebSocketHandler(mock_app)

    ws = BaseWebSocket("/v1/test")
    ws.on("connect")(lambda: "")

    with pytest.raises(ValueError, match="must have a 'message' handler"):
        handler.register(ws)


async def test_websocket_handler_register_success(mocker) -> None:
    mock_app = mocker.MagicMock()
    mock_ws_instance = mocker.MagicMock()
    mocker.patch("e_voice.core.websocket.WebSocket", return_value=mock_ws_instance)

    handler = WebSocketHandler(mock_app, prefix="/api")
    ws = BaseWebSocket("/ws")

    @ws.on("message")
    async def on_msg(msg):
        return msg

    handler.register(ws)

    mock_app.add_web_socket.assert_called_once_with("/api/ws", mock_ws_instance)
    assert len(handler._registered) == 1


async def test_websocket_handler_inject_dependencies(mocker) -> None:
    mock_app = mocker.MagicMock()
    mock_app.dependencies.get_global_dependencies.return_value = {"state": "mock_state"}

    mock_ws = mocker.MagicMock()
    mock_ws.dependencies.get_dependency_map.return_value = {"state": "mock_state"}

    mock_func_info = mocker.MagicMock()
    mock_func_info.args = {"ws": None, "state": None}
    mock_func_info.kwargs = {}
    mock_ws.methods = {"message": mock_func_info}

    handler = WebSocketHandler(mock_app)
    handler._registered = [mock_ws]

    handler.inject_dependencies()

    mock_ws.dependencies.add_global_dependency.assert_called_once()


async def test_websocket_handler_wsh_register_handler_sync(mocker) -> None:
    mock_app = mocker.MagicMock()
    mock_ws = mocker.MagicMock()
    mock_ws.methods = {}

    handler = WebSocketHandler(mock_app)

    def sync_handler(ws):
        return ""

    handler._wsh_register_handler(mock_ws, "connect", sync_handler)
    fi = mock_ws.methods["connect"]
    assert fi.handler is sync_handler
    assert fi.is_async is False


async def test_websocket_handler_wsh_register_handler_async(mocker) -> None:
    mock_app = mocker.MagicMock()
    mock_ws = mocker.MagicMock()
    mock_ws.methods = {}

    handler = WebSocketHandler(mock_app)

    async def async_handler(ws, msg):
        return msg

    handler._wsh_register_handler(mock_ws, "message", async_handler)
    fi = mock_ws.methods["message"]
    assert fi.handler is async_handler
    assert fi.is_async is True


##### BASE WS PARAMS #####


async def test_base_ws_params_empty() -> None:
    params = BaseWSParams()
    assert params.model_fields == {}


async def test_base_ws_params_frozen() -> None:
    assert BaseWSParams.model_config.get("frozen") is True


async def test_base_ws_params_ignores_extra() -> None:
    params = BaseWSParams.model_validate({"unknown": "value"})
    assert not hasattr(params, "unknown")


##### WS ROUTE #####


async def test_ws_route_frozen() -> None:
    async def handler(conn):
        pass

    route = WSRoute(handler=handler, params_cls=BaseWSParams)
    assert route.handler is handler
    assert route.params_cls is BaseWSParams


##### CONNECTION #####


async def test_connection_send_delegates(mocker) -> None:
    mock_ws = mocker.AsyncMock()
    conn = Connection(id="abc", path="/v1/test", params=BaseWSParams(), state=None, ws=mock_ws)

    await conn.send("hello")
    mock_ws.send.assert_awaited_once_with("hello")


async def test_connection_send_bytes(mocker) -> None:
    mock_ws = mocker.AsyncMock()
    conn = Connection(id="abc", path="/v1/test", params=BaseWSParams(), state=None, ws=mock_ws)

    await conn.send(b"\x00\x01")
    mock_ws.send.assert_awaited_once_with(b"\x00\x01")


async def test_connection_close(mocker) -> None:
    mock_ws = mocker.AsyncMock()
    conn = Connection(id="abc", path="/v1/test", params=BaseWSParams(), state=None, ws=mock_ws)

    await conn.close(4000, "test")
    mock_ws.close.assert_awaited_once_with(4000, "test")


async def test_connection_aiter_yields_messages() -> None:
    class FakeWS:
        def __init__(self, msgs):
            self._msgs = msgs

        async def __aiter__(self):
            for m in self._msgs:
                yield m

    conn = Connection(id="abc", path="/v1/test", params=BaseWSParams(), state=None, ws=FakeWS(["text", b"binary"]))

    collected = [msg async for msg in conn]
    assert collected == ["text", b"binary"]


async def test_connection_params_typed() -> None:
    class CustomParams(BaseWSParams):
        language: str = "es"

    params = CustomParams()
    mock_ws = type("FakeWS", (), {"send": lambda *a: None, "close": lambda *a: None})()
    conn = Connection(id="abc", path="/test", params=params, state=None, ws=mock_ws)
    assert conn.params.language == "es"


##### WEBSOCKET ROUTER #####


async def test_router_registers_handler() -> None:
    router = WebSocketRouter()

    @router("/v1/test")
    async def handler(conn):
        pass

    assert "/v1/test" in router.routes
    assert router.routes["/v1/test"].handler is handler
    assert router.routes["/v1/test"].params_cls is BaseWSParams


async def test_router_registers_multiple_paths() -> None:
    router = WebSocketRouter()

    @router("/v1/a", "/v1/b", "/v1/c")
    async def handler(conn):
        pass

    assert len(router.routes) == 3
    assert all(router.routes[p].handler is handler for p in ("/v1/a", "/v1/b", "/v1/c"))


async def test_router_registers_with_params_cls() -> None:
    class CustomParams(BaseWSParams):
        lang: str = "en"

    router = WebSocketRouter()

    @router("/v1/test", params=CustomParams)
    async def handler(conn):
        pass

    assert router.routes["/v1/test"].params_cls is CustomParams


async def test_router_routes_property() -> None:
    router = WebSocketRouter()
    assert router.routes == {}


##### WEBSOCKET SERVER #####


async def test_server_include_merges_routes() -> None:
    from e_voice.core.websocket import WebSocketServer

    server = WebSocketServer(port=9999)
    r1 = WebSocketRouter()
    r2 = WebSocketRouter()

    @r1("/a")
    async def h1(conn):
        pass

    @r2("/b")
    async def h2(conn):
        pass

    server.include(r1)
    server.include(r2)

    assert "/a" in server._routes
    assert "/b" in server._routes


async def test_server_port_property() -> None:
    from e_voice.core.websocket import WebSocketServer

    server = WebSocketServer(port=5700)
    assert server.port == 5700


async def test_server_dispatch_calls_handler(mocker) -> None:
    from uuid import UUID

    from e_voice.core.websocket import WebSocketServer

    handler = mocker.AsyncMock()
    server = WebSocketServer(port=9999)

    class TestParams(BaseWSParams):
        lang: str = "en"

    server._routes = {"/v1/test": WSRoute(handler=handler, params_cls=TestParams)}
    server._state = mocker.MagicMock()

    mock_ws = mocker.AsyncMock()
    mock_ws.request.path = "/v1/test?lang=es"
    mock_ws.id = UUID("12345678123456781234567812345678")

    await server._dispatch(mock_ws)

    handler.assert_awaited_once()
    conn = handler.call_args[0][0]
    assert conn.path == "/v1/test"
    assert conn.params.lang == "es"


async def test_server_dispatch_validates_params(mocker) -> None:
    from uuid import UUID

    from e_voice.core.websocket import WebSocketServer

    handler = mocker.AsyncMock()
    server = WebSocketServer(port=9999)
    server._routes = {"/v1/test": WSRoute(handler=handler, params_cls=BaseWSParams)}
    server._state = mocker.MagicMock()

    mock_ws = mocker.AsyncMock()
    mock_ws.request.path = "/v1/test"
    mock_ws.id = UUID("12345678123456781234567812345678")

    await server._dispatch(mock_ws)

    conn = handler.call_args[0][0]
    assert isinstance(conn.params, BaseWSParams)


async def test_server_dispatch_unknown_path_closes(mocker) -> None:
    from uuid import UUID

    from e_voice.core.websocket import WebSocketServer

    server = WebSocketServer(port=9999)
    server._routes = {}
    server._state = None

    mock_ws = mocker.AsyncMock()
    mock_ws.request.path = "/unknown"
    mock_ws.id = UUID("12345678123456781234567812345678")

    await server._dispatch(mock_ws)

    mock_ws.close.assert_awaited_once()
    assert mock_ws.close.call_args[0][0] == 4004


async def test_server_dispatch_handler_exception_logged(mocker) -> None:
    from uuid import UUID

    from e_voice.core.websocket import WebSocketServer

    async def failing_handler(conn):
        raise RuntimeError("handler exploded")

    server = WebSocketServer(port=9999)
    server._routes = {"/v1/boom": WSRoute(handler=failing_handler, params_cls=BaseWSParams)}
    server._state = mocker.MagicMock()

    mock_ws = mocker.AsyncMock()
    mock_ws.request.path = "/v1/boom"
    mock_ws.id = UUID("12345678123456781234567812345678")

    await server._dispatch(mock_ws)
