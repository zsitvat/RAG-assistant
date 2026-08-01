from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.agent.service import AgentService
from app.api.schemas import ChatRequest, ChatResponse
from app.dependencies import get_agent_service

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
) -> ChatResponse:
    """Runs a chat message through the agent and returns the resulting reply."""
    return await run_in_threadpool(agent_service.respond, request.thread_id, request.message)
