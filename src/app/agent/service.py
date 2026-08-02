import time
from collections.abc import AsyncIterator
from datetime import date

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent.projection import TurnProjector
from app.agent.state import RECURSION_LIMIT
from app.agent.streaming import StreamEventMapper
from app.api.schemas import ChatResponse, EvaluationResponse, StreamEvent
from app.integrations.langfuse import Observability


class AgentService:
    """The one place graph messages/artifacts become the public chat and evaluation contracts."""

    def __init__(
        self, graph: CompiledStateGraph, observability: Observability | None = None
    ) -> None:
        """Stores the compiled agent graph and the observability adapter used to trace turns."""
        self._graph = graph
        self._observability = observability or Observability(None)
        self._projector = TurnProjector()
        self._streaming = StreamEventMapper()

    def invoke_graph(self, thread_id: str, message: str) -> ChatResponse:
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
                    events = self._streaming.node_events(
                        node, update, emitted_steps, emitted_sources
                    )
                    for event in events:
                        yield event
            elif mode == "messages":
                token = self._streaming.answer_token(payload)
                if token is not None:
                    yield token

        final_state = (await self._graph.aget_state(config)).values
        yield StreamEvent(event="result", data=self._project(thread_id, final_state, start))

    def evaluate(
        self,
        thread_id: str,
        message: str,
        reference_date: date,
        dataset_item_id: str | None = None,
        experiment_name: str | None = None,
    ) -> EvaluationResponse:
        """Runs one evaluation turn and projects the graph state into the internal eval contract."""
        config = self._config(
            thread_id,
            tags=("eval",),
            dataset_item_id=dataset_item_id,
            experiment_name=experiment_name,
        )
        result = self._graph.invoke(self._input(message, reference_date), config=config)
        return self._projector.project_evaluation(thread_id, result)

    @staticmethod
    def _input(message: str, reference_date: date | None = None) -> dict:
        """Builds graph input containing the current user message and optional pinned date."""
        payload: dict = {"messages": [HumanMessage(content=message)]}
        if reference_date is not None:
            payload["reference_date"] = reference_date
        return payload

    def _config(self, thread_id: str, **trace_kwargs: object) -> dict:
        """Builds the graph config, attaching one Langfuse trace per turn when enabled."""
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        return config | self._observability.trace_config(thread_id, **trace_kwargs)

    def _project(self, thread_id: str, state: dict, start: float) -> ChatResponse:
        """Updates the trace with this turn's outcome and projects the final reply."""
        self._observability.update_trace(
            thread_id=thread_id,
            intent=state.get("intent"),
            category=state.get("category"),
            decision=state.get("decision"),
            degraded=state.get("degraded", False),
        )
        return self._projector.project_chat(thread_id, state, start)
