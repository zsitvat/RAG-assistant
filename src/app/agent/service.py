import logging
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, date, datetime

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.agent.graph_state import RECURSION_LIMIT
from app.agent.responses import ResponseBuilder
from app.agent.static_texts import STREAM_FAILED_MESSAGE
from app.agent.streaming import StreamEventMapper
from app.api.schemas import ChatResponse, EvaluationResponse, StreamEvent
from app.integrations.langfuse import Observability

logger = logging.getLogger(__name__)


class AgentService:
    """The one place graph messages/artifacts become the public chat and evaluation contracts."""

    def __init__(
        self, graph: CompiledStateGraph, observability: Observability | None = None
    ) -> None:
        """Stores the compiled agent graph and the observability adapter used to trace turns."""
        self._graph = graph
        self._observability = observability or Observability(None)
        self._responses = ResponseBuilder()
        self._streaming = StreamEventMapper()

    async def ainvoke_graph(self, thread_id: str, message: str) -> ChatResponse:
        """Runs the user's message through the agent graph and builds the resulting reply."""
        start = time.monotonic()
        with self._observability.traced_turn(thread_id, message) as (trace_config, update_trace):
            config = self._config(thread_id) | trace_config
            result = await self._graph.ainvoke(self._input(message), config=config)
            return self._build_response(thread_id, result, start, update_trace)

    async def astream(self, thread_id: str, message: str) -> AsyncIterator[StreamEvent]:
        """Streams public step, source and answer-token events, then the complete reply."""
        start = time.monotonic()
        emitted_steps: set[str] = set()
        emitted_sources: set[tuple[str, str]] = set()

        with self._observability.traced_turn(thread_id, message) as (trace_config, update_trace):
            config = self._config(thread_id) | trace_config

            try:
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
                response = self._build_response(thread_id, final_state, start, update_trace)
            except Exception as e:
                logger.exception(f"agent turn failed mid-stream: {type(e).__name__}: {e}")
                response = self._failed_response(thread_id, start)
                update_trace(degraded=True, output=response.answer)

            yield StreamEvent(event="result", data=response)

    async def evaluate(
        self,
        thread_id: str,
        message: str,
        reference_date: date,
        dataset_item_id: str | None = None,
        experiment_name: str | None = None,
    ) -> EvaluationResponse:
        """Runs one evaluation turn and builds the internal eval contract from the graph state."""
        with self._observability.traced_turn(
            thread_id,
            message,
            tags=("eval",),
            dataset_item_id=dataset_item_id,
            experiment_name=experiment_name,
        ) as (trace_config, update_trace):
            config = self._config(thread_id) | trace_config
            result = await self._graph.ainvoke(self._input(message, reference_date), config=config)
            response = self._responses.build_evaluation(thread_id, result)
            update_trace(output=response.answer)
            return response

    @staticmethod
    def _input(message: str, reference_date: date | None = None) -> dict:
        """Builds graph input containing the current user message and optional pinned date."""
        payload: dict = {"messages": [HumanMessage(content=message)]}
        if reference_date is not None:
            payload["reference_date"] = reference_date
        return payload

    @staticmethod
    def _config(thread_id: str) -> dict:
        """Builds the base graph config shared by every turn."""
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    @staticmethod
    def _failed_response(thread_id: str, start: float) -> ChatResponse:
        """Builds a degraded reply for when the agent turn raises mid-stream, so the SSE
        connection still ends cleanly instead of breaking after the 200 has already been sent."""
        return ChatResponse(
            thread_id=thread_id,
            answer=STREAM_FAILED_MESSAGE,
            generated_at=datetime.now(UTC),
            response_time_ms=round((time.monotonic() - start) * 1000),
            decision=None,
            sources=[],
            steps=[],
            degraded=True,
        )

    def _build_response(
        self, thread_id: str, state: dict, start: float, update_trace: Callable[..., None]
    ) -> ChatResponse:
        """Updates the trace with this turn's outcome and builds the final reply."""
        response = self._responses.build_chat(thread_id, state, start)
        update_trace(
            thread_id=thread_id,
            intent=state.get("intent"),
            category=state.get("category"),
            decision=state.get("decision"),
            degraded=state.get("degraded", False),
            output=response.answer,
        )
        return response
