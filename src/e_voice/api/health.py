"""Health check endpoint."""

from pydantic import BaseModel

from e_voice.core.logger import logger
from e_voice.core.router import Router
from e_voice.core.settings import settings as st

router = Router(__file__)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str


@router.get("/health")
async def health_check() -> HealthResponse:
    logger.info("health check requested", step="OK")
    return HealthResponse(status="healthy", service=st.API_NAME, version=st.API_VERSION)
