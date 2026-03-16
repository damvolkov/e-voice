"""Health check endpoint."""

from pydantic import BaseModel

from app.core.logger import LogIcon, logger
from app.core.router import Router
from app.core.settings import settings as st

router = Router(__file__)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str


@router.get("/health")
async def health_check() -> HealthResponse:
    logger.info("Health check requested", icon=LogIcon.HEALTHCHECK)
    return HealthResponse(status="healthy", service=st.API_NAME, version=st.API_VERSION)
