from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.agent.service import AgentService
from app.api.schemas import EvaluationRequest, EvaluationResponse
from app.dependencies import get_agent_service

router = APIRouter(prefix="/admin", tags=["evaluation"])


@router.post("/eval")
async def evaluate(
    request: EvaluationRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
) -> EvaluationResponse:
    """Runs one evaluation turn and returns the internal typed outputs the eval harness needs."""
    return await run_in_threadpool(
        agent_service.evaluate,
        request.thread_id,
        request.message,
        request.reference_date,
        request.dataset_item_id,
        request.experiment_name,
    )
