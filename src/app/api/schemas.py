import json
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.model import CalculationResult, Decision, ExpenseClaim, Finding, Intent
from app.rules.model import Category

THREAD_ID_PATTERN = r"^[A-Za-z0-9_.:-]+$"
THREAD_ID_MAX_LENGTH = 128
MESSAGE_MAX_CHARS = 500


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

    thread_id: str = Field(min_length=1, max_length=THREAD_ID_MAX_LENGTH, pattern=THREAD_ID_PATTERN)
    message: str = Field(min_length=1, max_length=MESSAGE_MAX_CHARS)


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
    decision: Decision | None
    sources: list[ChatSource]
    steps: list[str]
    degraded: bool


class ThreadResetResponse(BaseModel):
    """Confirms that a conversation thread's stored state was deleted."""

    thread_id: str
    reset: bool = True


class StreamEvent(BaseModel):
    """One public server-sent event of a streamed chat turn."""

    event: Literal["step", "source", "token", "result"]
    data: str | ChatSource | ChatResponse

    def to_sse(self) -> str:
        """Renders the event in the server-sent events wire format."""
        payload = self.model_dump_json(include={"data"})
        return f"event: {self.event}\ndata: {payload}\n\n"


def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, object]]:
    """Parses 'event:'/'data:' lines into (event, data) pairs, the inverse of StreamEvent.to_sse."""
    event_name = None
    for line in lines:
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ").strip()
        elif line.startswith("data: ") and event_name:
            yield event_name, json.loads(line.removeprefix("data: "))["data"]


class EvaluationRequest(BaseModel):
    """Carries one evaluation turn's input, pinned for deterministic scoring."""

    thread_id: str
    message: str
    reference_date: date
    dataset_item_id: str | None = None
    experiment_name: str | None = None


class EvaluationResponse(BaseModel):
    """Carries the internal, typed graph outputs needed to score one evaluation turn."""

    thread_id: str
    intent: Intent
    category: Category | None
    decision: Decision | None
    claim: ExpenseClaim
    missing_slots: list[str]
    tool_calls: list[str]
    calculation: CalculationResult | None
    findings: list[Finding]
    retrieved_doc_ids: list[str]
    cited_doc_ids: list[str]
    degraded: bool
    answer: str
