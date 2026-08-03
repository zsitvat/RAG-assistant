from langchain_core.messages import ToolMessage

from app.agent.responses import (
    STEP_LABELS,
    collect_cited_sources,
    labelled_step,
    node_step_detail,
    step_label,
    tool_step_detail,
)
from app.api.schemas import StreamEvent


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

        for label in self._node_step_labels(node, update or {}):
            if label not in emitted_steps:
                emitted_steps.add(label)
                events.append(StreamEvent(event="step", data=label))
        return events

    @staticmethod
    def _node_step_labels(node: str, update: dict) -> list[str]:
        """Returns the allow-listed public labels, with result summaries, a finished node
        announces."""
        if node == "execute_tools":
            return [
                labelled_step(step_label(message.name), tool_step_detail(message))
                for message in update.get("messages") or []
                if isinstance(message, ToolMessage)
            ]
        label = STEP_LABELS.get(node)
        return [labelled_step(label, node_step_detail(node, update))] if label else []

    @staticmethod
    def answer_token(payload: tuple) -> StreamEvent | None:
        """Returns a token event only for chunks produced by the final-answer node."""
        chunk, metadata = payload
        if metadata.get("langgraph_node") != "generate_response":
            return None
        return StreamEvent(event="token", data=chunk.content) if chunk.content else None
