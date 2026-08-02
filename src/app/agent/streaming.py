from langchain_core.messages import BaseMessage, ToolMessage

from app.agent.projection import (
    EXTRACTED_STEP,
    FINAL_STEP,
    STEP_LABELS,
    UNDERSTOOD_STEP,
    collect_cited_sources,
)
from app.api.schemas import StreamEvent

ANSWER_NODE = "generate_response"
TOOL_NODE = "execute_tools"
NODE_STEP_LABELS = {
    "classify_intent": UNDERSTOOD_STEP,
    "extract_information": EXTRACTED_STEP,
    "generate_response": FINAL_STEP,
    "ask_clarification": FINAL_STEP,
    "out_of_scope": FINAL_STEP,
}


class StreamEventMapper:
    """Maps finished graph node updates and answer-model chunks to public stream events."""

    def node_events(
        self,
        node: str,
        update: dict,
        emitted_steps: set[str],
        emitted_sources: set[tuple[str, str]],
    ) -> list[StreamEvent]:
        """Maps one finished node update to its deduplicated public step and source events."""
        messages = (update or {}).get("messages") or []
        events: list[StreamEvent] = []

        for source in collect_cited_sources(messages):
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
    def answer_token(payload: tuple) -> StreamEvent | None:
        """Returns a token event only for chunks produced by the final-answer node."""
        chunk, metadata = payload
        if metadata.get("langgraph_node") != ANSWER_NODE:
            return None
        return StreamEvent(event="token", data=chunk.content) if chunk.content else None
