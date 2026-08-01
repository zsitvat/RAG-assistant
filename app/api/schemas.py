from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Reports that the service process is alive."""

    status: Literal["ok"] = "ok"


class ReadinessCheck(BaseModel):
    """Reports the readiness status of a single dependency."""

    name: str
    status: Literal["ok", "not_configured", "unavailable"]
    detail: str


class ReadyResponse(BaseModel):
    """Aggregates the readiness checks for all dependencies."""

    ready: bool
    checks: list[ReadinessCheck]


class ChatRequest(BaseModel):
    """Carries an incoming chat message for a conversation thread."""

    thread_id: str
    message: str


class ChatSource(BaseModel):
    """Identifies a policy document citation surfaced during a chat reply."""

    source_id: str
    doc_id: str
    title: str
    section: str


class ChatResponse(BaseModel):
    """Carries the agent's answer for a chat message along with its sources and steps."""

    thread_id: str
    answer: str
    generated_at: datetime
    response_time_ms: int
    sources: list[ChatSource]
    steps: list[str]
