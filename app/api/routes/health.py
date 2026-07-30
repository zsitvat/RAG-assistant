from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse, ReadinessCheck, ReadyResponse
from app.core.config import Settings
from app.dependencies import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready")
async def ready(settings: Annotated[Settings, Depends(get_settings)]) -> ReadyResponse:
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
    redis_check = ReadinessCheck(
        name="redis",
        status="not_configured",
        detail="Redis integration is not part of this milestone yet.",
    )
    return ReadyResponse(ready=True, checks=[llm_check, redis_check])
