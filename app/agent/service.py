import time
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent.current_request import CurrentRequest
from app.agent.state import RECURSION_LIMIT
from app.api.schemas import ChatResponse, ChatSource
from app.rag.model import RagResult

STEP_LABELS = {
    "search_policies": "Policies searched",
    "check_rules": "Rules checked",
    "calculate": "Amount calculated",
}
ALWAYS_FIRST_STEPS = ["Request understood", "Information extracted"]
FINAL_STEP = "Answer prepared"


class AgentService:
    """The one place graph messages/artifacts become the public chat contract."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        """Stores the compiled agent graph used to process requests."""
        self._graph = graph

    def respond(self, thread_id: str, message: str) -> ChatResponse:
        """Runs the user's message through the agent graph and projects the result into a reply."""
        start = time.monotonic()
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT},
        )
        response_time_ms = round((time.monotonic() - start) * 1000)

        request_messages = CurrentRequest(result["messages"]).messages()
        return ChatResponse(
            thread_id=thread_id,
            answer=result["messages"][-1].content,
            generated_at=datetime.now(UTC),
            response_time_ms=response_time_ms,
            decision=result.get("decision"),
            sources=self._collect_cited_sources(request_messages),
            steps=self._collect_step_labels(request_messages),
        )

    @staticmethod
    def _collect_cited_sources(request_messages: list[BaseMessage]) -> list[ChatSource]:
        sources: list[ChatSource] = []
        seen: set[tuple[str, str]] = set()
        for message in request_messages:
            if not (isinstance(message, ToolMessage) and message.name == "search_policies"):
                continue
            artifact: RagResult = message.artifact
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

    @staticmethod
    def _collect_step_labels(request_messages: list[BaseMessage]) -> list[str]:
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
