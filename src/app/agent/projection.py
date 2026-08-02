import time
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.message_history import MessageHistory
from app.agent.model import CalculationResult, ExpenseClaim, Finding
from app.agent.slots import RequiredSlotTable
from app.api.schemas import ChatResponse, ChatSource, EvaluationResponse
from app.rag.model import RagResult

STEP_LABELS = {
    "search_policies": "Policies searched",
    "check_rules": "Rules checked",
    "calculate": "Amount calculated",
}
UNDERSTOOD_STEP = "Request understood"
EXTRACTED_STEP = "Information extracted"
FINAL_STEP = "Answer prepared"
ALWAYS_FIRST_STEPS = [UNDERSTOOD_STEP, EXTRACTED_STEP]


def collect_cited_sources(request_messages: list[BaseMessage]) -> list[ChatSource]:
    """Collects deduplicated cited sources from policy-search tool messages."""
    sources: list[ChatSource] = []
    seen: set[tuple[str, str]] = set()
    for message in request_messages:
        if not (isinstance(message, ToolMessage) and message.name == "search_policies"):
            continue
        artifact = RagResult.from_artifact(message.artifact)
        for citation in artifact.citations:
            key = (citation.doc_id, citation.section or "")
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                ChatSource(
                    source_id=citation.marker,
                    doc_id=citation.doc_id,
                    title=citation.doc_title,
                    section=citation.section or "",
                )
            )
    return sources


class TurnProjector:
    """Builds the public chat and evaluation contracts from a finished graph turn."""

    def __init__(self) -> None:
        """Creates the slot table used to derive missing slots for the evaluation contract."""
        self._slots = RequiredSlotTable()

    def project_chat(self, thread_id: str, state: dict, start: float) -> ChatResponse:
        """Builds the public reply from final graph state, shared by both chat endpoints."""
        request_messages = MessageHistory(state["messages"]).messages()
        return ChatResponse(
            thread_id=thread_id,
            answer=state["messages"][-1].content,
            generated_at=datetime.now(UTC),
            response_time_ms=round((time.monotonic() - start) * 1000),
            decision=state.get("decision"),
            sources=collect_cited_sources(request_messages),
            steps=self._collect_step_labels(request_messages),
        )

    def project_evaluation(self, thread_id: str, state: dict) -> EvaluationResponse:
        """Builds the internal evaluation contract from final graph state."""
        request_messages = MessageHistory(state["messages"]).messages()
        claim = ExpenseClaim.from_state(state.get("claim"))
        intent = state["intent"]
        category = state.get("category")
        return EvaluationResponse(
            thread_id=thread_id,
            intent=intent,
            category=category,
            decision=state.get("decision"),
            claim=claim,
            missing_slots=self._slots.missing(intent, category, claim),
            tool_calls=self._collect_tool_calls(request_messages),
            calculation=self._collect_calculation(request_messages),
            findings=self._collect_findings(request_messages),
            retrieved_doc_ids=self._collect_retrieved_doc_ids(request_messages),
            cited_doc_ids=self._collect_cited_doc_ids(request_messages),
            degraded=state.get("degraded", False),
            answer=state["messages"][-1].content,
        )

    @staticmethod
    def _collect_step_labels(request_messages: list[BaseMessage]) -> list[str]:
        """Collects stable public step labels from the completed request."""
        steps = list(ALWAYS_FIRST_STEPS)
        for message in request_messages:
            if not isinstance(message, ToolMessage):
                continue
            label = STEP_LABELS.get(message.name)
            if label and label not in steps:
                steps.append(label)
        if isinstance(request_messages[-1], AIMessage):
            steps.append(FINAL_STEP)
        return steps

    @staticmethod
    def _collect_tool_calls(request_messages: list[BaseMessage]) -> list[str]:
        """Collects the ordered tool-call names issued during the current request."""
        return [
            call["name"]
            for message in request_messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]

    @staticmethod
    def _collect_calculation(request_messages: list[BaseMessage]) -> CalculationResult | None:
        """Returns the last calculation result from the current request, if the tool ran."""
        return next(
            (
                message.artifact
                for message in reversed(request_messages)
                if isinstance(message, ToolMessage)
                and message.name == "calculate"
                and isinstance(message.artifact, CalculationResult)
            ),
            None,
        )

    @staticmethod
    def _collect_findings(request_messages: list[BaseMessage]) -> list[Finding]:
        """Returns the last rule-check findings from the current request, if the tool ran."""
        return next(
            (
                message.artifact
                for message in reversed(request_messages)
                if isinstance(message, ToolMessage)
                and message.name == "check_rules"
                and isinstance(message.artifact, list)
            ),
            [],
        )

    @staticmethod
    def _collect_retrieved_doc_ids(request_messages: list[BaseMessage]) -> list[str]:
        """Collects deduplicated doc ids retrieved by policy-search tool calls, before budgeting."""
        doc_ids: list[str] = []
        for message in request_messages:
            if not (isinstance(message, ToolMessage) and message.name == "search_policies"):
                continue
            for result in RagResult.from_artifact(message.artifact).results:
                if result.doc_id not in doc_ids:
                    doc_ids.append(result.doc_id)
        return doc_ids

    @staticmethod
    def _collect_cited_doc_ids(request_messages: list[BaseMessage]) -> list[str]:
        """Collects deduplicated doc ids actually placed in the answer's citation context."""
        doc_ids: list[str] = []
        for message in request_messages:
            if not (isinstance(message, ToolMessage) and message.name == "search_policies"):
                continue
            for citation in RagResult.from_artifact(message.artifact).citations:
                if citation.doc_id not in doc_ids:
                    doc_ids.append(citation.doc_id)
        return doc_ids
