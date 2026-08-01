import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent.current_request import CurrentRequest
from app.agent.state import RECURSION_LIMIT
from app.api.schemas import ChatResponse, ChatSource, StreamEvent
from app.integrations.langfuse import Observability
from app.rag.model import RagResult

STEP_LABELS = {
    "search_policies": "Policies searched",
    "check_rules": "Rules checked",
    "calculate": "Amount calculated",
}
UNDERSTOOD_STEP = "Request understood"
EXTRACTED_STEP = "Information extracted"
FINAL_STEP = "Answer prepared"
NODE_STEP_LABELS = {
    "classify_intent": UNDERSTOOD_STEP,
    "extract_information": EXTRACTED_STEP,
    "generate_response": FINAL_STEP,
    "ask_clarification": FINAL_STEP,
    "out_of_scope": FINAL_STEP,
}
ANSWER_NODE = "generate_response"
TOOL_NODE = "execute_tools"
ALWAYS_FIRST_STEPS = [UNDERSTOOD_STEP, EXTRACTED_STEP]


class AgentService:
    """The one place graph messages/artifacts become the public chat contract."""

    def __init__(
        self, graph: CompiledStateGraph, observability: Observability | None = None
    ) -> None:
        """Stores the compiled agent graph and the observability adapter used to trace turns."""
        self._graph = graph
        self._observability = observability or Observability(None)

    def respond(self, thread_id: str, message: str) -> ChatResponse:
        """Runs the user's message through the agent graph and projects the result into a reply."""
        start = time.monotonic()
        result = self._graph.invoke(self._input(message), config=self._config(thread_id))
        return self._project(thread_id, result, start)

    async def stream(self, thread_id: str, message: str) -> AsyncIterator[StreamEvent]:
        """Streams public step, source and answer-token events, then the complete reply."""
        start = time.monotonic()
        config = self._config(thread_id)
        emitted_steps: set[str] = set()
        emitted_sources: set[tuple[str, str]] = set()

        async for mode, payload in self._graph.astream(
            self._input(message), config=config, stream_mode=["updates", "messages"]
        ):
            if mode == "updates":
                for node, update in payload.items():
                    for event in self._node_events(node, update, emitted_steps, emitted_sources):
                        yield event
            elif mode == "messages":
                token = self._answer_token(payload)
                if token is not None:
                    yield token

        final_state = self._graph.get_state(config).values
        yield StreamEvent(event="result", data=self._project(thread_id, final_state, start))

    @staticmethod
    def _input(message: str) -> dict:
        return {"messages": [HumanMessage(content=message)]}

    def _config(self, thread_id: str) -> dict:
        """Builds the graph config, attaching one Langfuse trace per turn when enabled."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        return config | self._observability.trace_config(thread_id)

    def _project(self, thread_id: str, state: dict, start: float) -> ChatResponse:
        """Builds the public reply from final graph state, shared by both endpoints."""
        request_messages = CurrentRequest(state["messages"]).messages()
        self._observability.update_trace(
            thread_id=thread_id,
            intent=state.get("intent"),
            category=state.get("category"),
            decision=state.get("decision"),
        )
        return ChatResponse(
            thread_id=thread_id,
            answer=state["messages"][-1].content,
            generated_at=datetime.now(UTC),
            response_time_ms=round((time.monotonic() - start) * 1000),
            decision=state.get("decision"),
            sources=self._collect_cited_sources(request_messages),
            steps=self._collect_step_labels(request_messages),
        )

    def _node_events(
        self,
        node: str,
        update: dict,
        emitted_steps: set[str],
        emitted_sources: set[tuple[str, str]],
    ) -> list[StreamEvent]:
        """Maps one finished node update to its deduplicated public step and source events."""
        messages = (update or {}).get("messages") or []
        events: list[StreamEvent] = []

        for source in self._collect_cited_sources(messages):
            key = (source.doc_id, source.section)
            if key not in emitted_sources:
                emitted_sources.add(key)
                events.append(StreamEvent(event="source", data=source))

        for label in self._node_step_labels(node, messages):
            if label not in emitted_steps:
                emitted_steps.add(label)
                events.append(StreamEvent(event="step", data=label))
        return events

    @staticmethod
    def _node_step_labels(node: str, messages: list[BaseMessage]) -> list[str]:
        """Returns the allow-listed public labels a finished node may announce."""
        if node == TOOL_NODE:
            return [
                STEP_LABELS[message.name]
                for message in messages
                if isinstance(message, ToolMessage) and message.name in STEP_LABELS
            ]
        label = NODE_STEP_LABELS.get(node)
        return [label] if label else []

    @staticmethod
    def _answer_token(payload: tuple) -> StreamEvent | None:
        """Returns a token event only for chunks produced by the final-answer node."""
        chunk, metadata = payload
        if metadata.get("langgraph_node") != ANSWER_NODE:
            return None
        return StreamEvent(event="token", data=chunk.content) if chunk.content else None

    @staticmethod
    def _collect_cited_sources(request_messages: list[BaseMessage]) -> list[ChatSource]:
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
