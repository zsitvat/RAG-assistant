from typing import Annotated

import redis
from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse, ReadinessCheck, ReadyResponse
from app.core.config import Settings
from app.dependencies import get_redis_client, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


def _check_redis(redis_client: redis.Redis | None) -> ReadinessCheck:
    if redis_client is None:
        return ReadinessCheck(
            name="redis", status="unavailable", detail="Redis was unreachable at startup."
        )
    try:
        redis_client.ping()
    except redis.RedisError:
        return ReadinessCheck(name="redis", status="unavailable", detail="Redis ping failed.")
    return ReadinessCheck(name="redis", status="ok", detail="Redis is reachable.")


@router.get("/ready")
async def ready(
    settings: Annotated[Settings, Depends(get_settings)],
    redis_client: Annotated[redis.Redis | None, Depends(get_redis_client)],
) -> ReadyResponse:
    is_dummy = settings.llm_backend == "dummy"
    llm_check = ReadinessCheck(
        name="llm",
        status="ok" if is_dummy else "not_configured",
        detail=(
            "Dummy backend requires no external model."
            if is_dummy
            else f"Ollama backend at {settings.ollama_base_url} is not health-checked yet."
        ),
    )
    redis_check = _check_redis(redis_client)
    return ReadyResponse(ready=redis_check.status != "unavailable", checks=[llm_check, redis_check])
