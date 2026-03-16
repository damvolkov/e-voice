"""robyn-ml-api - ML API Template powered by Robyn."""

from robyn import Robyn

from app.api.health import router as health_router
from app.core.lifespan import create_lifespan
from app.core.logger import logger
from app.core.router import Router
from app.core.settings import settings as st
from app.events.process_pool import ProcessPoolEvent
from app.middlewares.base import MiddlewareHandler
from app.middlewares.files import FileUploadOpenAPIMiddleware
from app.models.core import UploadFile

app = Robyn(__file__)

# Lifespan events
lifespan = create_lifespan(app)
lifespan.register(ProcessPoolEvent)

app.startup_handler(lifespan.startup)
app.shutdown_handler(lifespan.shutdown)

# Middlewares
middlewares = MiddlewareHandler(app)
middlewares.register(FileUploadOpenAPIMiddleware)

# Routers
app.include_router(health_router)


# --- TEMPORAL: Test file upload ---
upload_router = Router(__file__, prefix="/files")


@upload_router.post("/upload")
async def test_upload(files: UploadFile):
    """Upload a file and get its detected type."""
    return {"files": [{"name": name, "size": len(data)} for name, data in files]}


app.include_router(upload_router)


def main() -> None:
    logger.info("🚀 STARTING %s | HOST=%s | PORT=%s", st.API_NAME, st.API_HOST, st.API_PORT)
    app.start(host=st.API_HOST, port=st.API_PORT)


if __name__ == "__main__":
    main()
