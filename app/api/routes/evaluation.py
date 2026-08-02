from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.agent.service import AgentService
from app.api.schemas import EvaluationRequest, EvaluationResponse, LoadTestRequest, LoadTestResult
from app.dependencies import get_agent_service, get_observability
from app.evaluation.load import LoadTestRunner, LoadTestValidationError
from app.integrations.langfuse import Observability

router = APIRouter(prefix="/admin", tags=["evaluation"])

LANGFUSE_UNAVAILABLE_DETAIL = (
    "Langfuse must be enabled and configured (LANGFUSE_ENABLED=true plus credentials) to run "
    "the evaluation or load-test harness."
)


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


@router.post("/load-test")
async def load_test(
    request: LoadTestRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
    observability: Annotated[Observability, Depends(get_observability)],
) -> LoadTestResult:
    """Replays a Langfuse dataset synchronously under bounded concurrency."""
    if not observability.enabled:
        raise HTTPException(status_code=503, detail=LANGFUSE_UNAVAILABLE_DETAIL)

    runner = LoadTestRunner(agent_service, observability)
    try:
        return await run_in_threadpool(
            runner.run, request.dataset_name, request.repetitions, request.max_concurrency
        )
    except LoadTestValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
