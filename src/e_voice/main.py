"""e-voice — Speech API powered by Robyn with pluggable STT/TTS backends."""

from robyn import Robyn

from e_voice.api.health import router as health_router
from e_voice.api.stt import router as stt_router
from e_voice.api.system import router as system_router
from e_voice.api.tts import router as tts_router
from e_voice.core.lifespan import create_lifespan
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.core.websocket import WebSocketServer
from e_voice.events.adapters import STTAdapterEvent, TTSAdapterEvent
from e_voice.events.monitor import MonitorEvent
from e_voice.events.process_pool import ProcessPoolEvent
from e_voice.front import launch_background as launch_gradio
from e_voice.middlewares.base import MiddlewareHandler
from e_voice.middlewares.files import FileUploadOpenAPIMiddleware
from e_voice.middlewares.swagger import SwaggerBrandingMiddleware
from e_voice.models.session import ConnectionRegistry
from e_voice.operational.controller import DeviceController
from e_voice.websockets.stt import router as ws_stt_router
from e_voice.websockets.tts import router as ws_tts_router

app = Robyn(__file__)

lifespan = create_lifespan(app)
lifespan.register(ProcessPoolEvent)
lifespan.register(STTAdapterEvent)
lifespan.register(TTSAdapterEvent)
lifespan.register(MonitorEvent)

ws_server = WebSocketServer(port=st.ws.port)
ws_server.include(ws_stt_router)
ws_server.include(ws_tts_router)


async def startup() -> None:
    """Startup: lifespan first, then WS server, then Gradio UI."""
    await lifespan.startup()
    assert lifespan.state is not None
    lifespan.state.stt_connections = ConnectionRegistry()
    lifespan.state.tts_connections = ConnectionRegistry()
    lifespan.state.stt_sessions = {}
    lifespan.state.device_controller = DeviceController()
    ws_server.launch_background(state=lifespan.state)
    launch_gradio()


app.startup_handler(startup)
app.shutdown_handler(lifespan.shutdown)

middlewares = MiddlewareHandler(app)
middlewares.register(FileUploadOpenAPIMiddleware)
middlewares.register(SwaggerBrandingMiddleware)

app.include_router(health_router)
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(system_router)


def main() -> None:
    logger.info(
        "starting server",
        name=st.API_NAME,
        host=st.system.host,
        port=st.system.port,
        ws_port=st.ws.port,
        step="START",
    )
    app.start(host=st.system.host, port=st.system.port)


if __name__ == "__main__":
    main()
