from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
) -> StreamingResponse:
    """Streams the agent's public step, source and token events, then the complete reply."""

    async def events() -> AsyncIterator[str]:
        """Renders each public stream event in the server-sent events wire format."""
        async for event in agent_service.stream(request.thread_id, request.message):
            yield event.to_sse()

    return StreamingResponse(events(), media_type="text/event-stream")


@router.delete("/threads/{thread_id}")
async def reset_thread(
    thread_id: str,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
) -> ThreadResetResponse:
    """Deletes a conversation's stored state so the next message starts a new conversation."""
    await run_in_threadpool(checkpointer.delete_thread, thread_id)
    return ThreadResetResponse(thread_id=thread_id)
