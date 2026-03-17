"""Unit tests for core/websocket.py — BaseWebSocket and WebSocketHandler."""

import pytest

from e_voice.core.websocket import BaseWebSocket, WebSocketHandler

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
        ws.on("invalid")(lambda: "")


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
