import time
from collections import Counter
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.agent.message_history import MessageHistory
from app.agent.model import CalculationResult, ExpenseClaim, Finding
from app.agent.slots import RequiredSlotTable
from app.api.schemas import ChatResponse, ChatSource, EvaluationResponse
from app.rag.model import RagResult

CLASSIFY_INTENT_STEP = "Intent classified"
EXTRACT_INFORMATION_STEP = "Details extracted"
GENERATE_RESPONSE_STEP = "Answer generated"
ASK_CLARIFICATION_STEP = "Clarification asked"
OUT_OF_SCOPE_STEP = "Marked out of scope"
STATE_DETAIL_NODES = ["classify_intent", "extract_information"]

STEP_LABELS = {
    # tool names, keyed by the ToolMessage produced when execute_tools finishes
    "search_policies": "Policies searched",
    "check_rules": "Rules checked",
    "calculate": "Amount calculated",
    # graph node names, keyed by the node itself for nodes that emit no ToolMessage
    "classify_intent": CLASSIFY_INTENT_STEP,
    "extract_information": EXTRACT_INFORMATION_STEP,
    "generate_response": GENERATE_RESPONSE_STEP,
    "ask_clarification": ASK_CLARIFICATION_STEP,
    "out_of_scope": OUT_OF_SCOPE_STEP,
}

# The decision alone tells us, from message history, which terminal node produced the last
# AIMessage (extract_information always resets it, so only one terminal node can have set it).
DECISION_FINAL_STEPS = {
    "needs_info": ASK_CLARIFICATION_STEP,
    "out_of_scope": OUT_OF_SCOPE_STEP,
}


def step_label(tool_name: str) -> str:
    """Returns the curated public label for a tool, or a derived one for an unlisted tool."""
    return STEP_LABELS.get(tool_name, tool_name.replace("_", " ").capitalize())


def _final_step_label(decision: str | None) -> str:
    """Returns the public label for the terminal node, inferred from the turn's decision."""
    return DECISION_FINAL_STEPS.get(decision, GENERATE_RESPONSE_STEP)


def labelled_step(label: str, detail: str) -> str:
    """Joins a public step label with its short result summary, when there is one."""
    return f"{label} — {detail}" if detail else label


def tool_step_detail(message: ToolMessage) -> str:
    """Summarizes what one finished tool call produced, for the public step list."""
    if message.status == "error":
        return "failed"
    if message.name == "search_policies":
        return _search_detail(message.artifact)
    if message.name == "check_rules":
        return _findings_detail(message.artifact)
    if message.name == "calculate":
        return _calculation_detail(message.artifact)
    return ""


def node_step_detail(node: str, state: dict) -> str:
    """Summarizes the state a non-tool node produced, for the public step list."""
    if node == "classify_intent":
        return ", ".join(str(v) for v in (state.get("intent"), state.get("category")) if v)
    if node == "extract_information":
        return _claim_detail(state.get("claim"))
    return ""


def _search_detail(artifact: object) -> str:
    """Summarizes a policy search as the cited passage count and the documents they came from."""
    citations = RagResult.from_artifact(artifact).citations
    if not citations:
        return "no matching passage"
    titles = list(dict.fromkeys(citation.doc_title for citation in citations))
    passages = f"{len(citations)} passage{'' if len(citations) == 1 else 's'}"
    return f"{passages} from {', '.join(titles)}"


def _findings_detail(artifact: object) -> str:
    """Summarizes a rule check as its per-status finding counts."""
    if not isinstance(artifact, list):
        return "no applicable rule"
    findings = [f for f in artifact if isinstance(f, Finding)]
    if not findings:
        return "no applicable rule"
    counts = Counter(finding.status for finding in findings)
    return ", ".join(f"{count} {status}" for status, count in counts.items())


def _calculation_detail(artifact: object) -> str:
    """Summarizes a calculation using its own compact summary."""
    return artifact.compact_summary() if isinstance(artifact, CalculationResult) else ""


def _claim_detail(claim: object) -> str:
    """Summarizes the extracted claim as its populated fields."""
    fields = ExpenseClaim.from_state(claim).model_dump(exclude_none=True)
    return ", ".join(f"{name}={_field_value(value)}" for name, value in fields.items())


def _field_value(value: object) -> str:
    """Renders one claim field value without list punctuation or trailing float zeros."""
    if isinstance(value, list):
        return " / ".join(str(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


class ResponseBuilder:
    """Builds the public chat and evaluation contracts from a finished graph turn."""

    def __init__(self) -> None:
        """Creates the slot table used to derive missing slots for the evaluation contract."""
        self._slots = RequiredSlotTable()

    def build_chat(self, thread_id: str, state: dict, start: float) -> ChatResponse:
        """Builds the public reply from final graph state, shared by both chat endpoints."""
        request_messages = MessageHistory(state["messages"]).messages()
        decision = state.get("decision")
        return ChatResponse(
            thread_id=thread_id,
            answer=state["messages"][-1].content,
            generated_at=datetime.now(UTC),
            response_time_ms=round((time.monotonic() - start) * 1000),
            decision=decision,
            sources=collect_cited_sources(request_messages),
            steps=self._collect_step_labels(request_messages, state, decision),
            degraded=state.get("degraded", False),
        )

    def build_evaluation(self, thread_id: str, state: dict) -> EvaluationResponse:
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
    def _collect_step_labels(
        request_messages: list[BaseMessage], state: dict, decision: str | None
    ) -> list[str]:
        """Collects stable public step labels, each with its result summary, from the request."""
        steps = [
            labelled_step(STEP_LABELS[node], node_step_detail(node, state))
            for node in STATE_DETAIL_NODES
        ]
        for message in request_messages:
            if not isinstance(message, ToolMessage):
                continue
            label = labelled_step(step_label(message.name), tool_step_detail(message))
            if label not in steps:
                steps.append(label)
        if isinstance(request_messages[-1], AIMessage):
            steps.append(_final_step_label(decision))
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
