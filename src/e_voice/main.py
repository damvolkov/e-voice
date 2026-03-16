"""e-voice — Speech API powered by Robyn, faster-whisper, and Kokoro."""

from robyn import Robyn

from e_voice.api.health import router as health_router
from e_voice.api.stt import router as stt_router
from e_voice.api.tts import router as tts_router
from e_voice.core.lifespan import create_lifespan
from e_voice.core.logger import logger
from e_voice.core.settings import settings as st
from e_voice.core.websocket import WebSocketHandler
from e_voice.events.kokoro_model import KokoroModelEvent
from e_voice.events.process_pool import ProcessPoolEvent
from e_voice.events.whisper_model import WhisperModelEvent
from e_voice.middlewares.base import MiddlewareHandler
from e_voice.middlewares.files import FileUploadOpenAPIMiddleware
from e_voice.websockets.stt import ws_stt
from e_voice.websockets.tts import ws_tts

app = Robyn(__file__)

# Lifespan events
lifespan = create_lifespan(app)
lifespan.register(ProcessPoolEvent)
lifespan.register(WhisperModelEvent)
lifespan.register(KokoroModelEvent)

# WebSockets (register now, inject dependencies during startup)
websockets = WebSocketHandler(app)
websockets.register(ws_stt)
websockets.register(ws_tts)


async def startup() -> None:
    """Startup: lifespan first, then inject WS dependencies."""
    await lifespan.startup()
    websockets.inject_dependencies()


app.startup_handler(startup)
app.shutdown_handler(lifespan.shutdown)

# Middlewares
middlewares = MiddlewareHandler(app)
middlewares.register(FileUploadOpenAPIMiddleware)

# HTTP Routers
app.include_router(health_router)
app.include_router(stt_router)
app.include_router(tts_router)


def main() -> None:
    logger.info("starting server", name=st.API_NAME, host=st.API_HOST, port=st.API_PORT, step="START")
    app.start(host=st.API_HOST, port=st.API_PORT)


if __name__ == "__main__":
    main()
