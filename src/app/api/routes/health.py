from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse, ReadyResponse
from app.dependencies import get_redis_index, get_settings
from app.integrations.readiness import ReadinessChecker
from app.integrations.redis import RedisIndex
from app.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    """Reports basic liveness of the service."""
    return HealthResponse()


@router.get("/ready")
async def ready(
    settings: Annotated[Settings, Depends(get_settings)],
    redis_index: Annotated[RedisIndex, Depends(get_redis_index)],
) -> ReadyResponse:
    """Reports whether the LLM backend and Redis dependencies are ready to serve requests."""
    return ReadinessChecker().check(settings, redis_index)
