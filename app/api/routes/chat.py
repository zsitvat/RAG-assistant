from typing import Annotated

from fastapi import APIRouter, Depends
from langgraph.checkpoint.base import BaseCheckpointSaver
from starlette.concurrency import run_in_threadpool

from app.agent.service import AgentService
from app.api.schemas import ChatRequest, ChatResponse, ThreadResetResponse
from app.dependencies import get_agent_service, get_checkpointer

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
) -> ChatResponse:
    """Runs a chat message through the agent and returns the resulting reply."""
    return await run_in_threadpool(agent_service.respond, request.thread_id, request.message)


@router.delete("/threads/{thread_id}")
async def reset_thread(
    thread_id: str,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
) -> ThreadResetResponse:
    """Deletes a conversation's stored state so the next message starts a new conversation."""
    await run_in_threadpool(checkpointer.delete_thread, thread_id)
    return ThreadResetResponse(thread_id=thread_id)
