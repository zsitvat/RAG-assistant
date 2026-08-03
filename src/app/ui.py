import logging
import uuid
from collections.abc import Iterator
from datetime import datetime

import httpx2
import streamlit as st

from app.api.schemas import MESSAGE_MAX_CHARS, parse_sse_lines
from app.settings import get_settings

STREAM_TIMEOUT_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 5.0
CLARIFICATION_DECISION = "needs_info"

logger = logging.getLogger(__name__)


class ChatApiClient:
    """Talks to the assistant API on behalf of the fully vibecoded chat UI."""

    def __init__(self, base_url: str) -> None:
        """Stores the API base URL used for every call."""
        self._base_url = base_url

    def index_stats(self) -> dict:
        """Returns the read-only policy index statistics."""
        return self._get("/admin/stats")

    def reset_thread(self, thread_id: str) -> None:
        """Deletes the server-side conversation state for a thread."""
        response = httpx2.delete(
            f"{self._base_url}/threads/{thread_id}", timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()

    def stream_turn(self, thread_id: str, message: str) -> Iterator[tuple[str, object]]:
        """Yields (event, data) pairs from the streaming chat endpoint."""
        payload = {"thread_id": thread_id, "message": message}
        with httpx2.stream(
            "POST",
            f"{self._base_url}/chat/stream",
            json=payload,
            timeout=STREAM_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            yield from parse_sse_lines(response.iter_lines())

    def _get(self, path: str) -> dict:
        """Returns decoded JSON from a read-only API endpoint."""
        response = httpx2.get(f"{self._base_url}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()


def _render_source(source: dict) -> str:
    """Renders one citation as a single markdown line."""
    section = f" › {source['section']}" if source["section"] else ""
    return f"**[{source['source_id']}]** {source['title']}{section}"


def _render_details(entry: dict) -> None:
    """Renders the completion caption and the collapsed steps/sources expander."""
    generated_at = entry.get("generated_at")
    if generated_at:
        local_time = datetime.fromisoformat(generated_at).astimezone().strftime("%H:%M:%S")
        st.caption(f"{local_time} · {entry.get('response_time_ms', 0)} ms")
    if entry.get("degraded"):
        st.caption("⚠️ This answer may be incomplete or less reliable.")
    steps, sources = entry.get("steps") or [], entry.get("sources") or []
    if not steps and not sources:
        return
    with st.expander("Steps and sources", expanded=False):
        for step in steps:
            st.write(f"- {step}")
        for source in sources:
            st.write(_render_source(source))


def _render_history(history: list[dict]) -> None:
    """Renders every stored chat message with its steps and sources."""
    for entry in history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant" and entry.get("decision") == CLARIFICATION_DECISION:
                st.info(entry["content"], icon="❓")
            else:
                st.markdown(entry["content"])
            if entry["role"] == "assistant":
                _render_details(entry)


def _consume_stream(client: ChatApiClient, thread_id: str, prompt: str) -> dict:
    """Streams one turn into the live status area and returns the final reply."""
    with st.chat_message("assistant"):
        status = st.status("Working on it…", expanded=True)
        answer_area = st.empty()
        streamed, result = "", None
        try:
            for event, data in client.stream_turn(thread_id, prompt):
                if event == "step":
                    status.write(f"- {data}")
                elif event == "source":
                    status.write(_render_source(data))
                elif event == "token":
                    streamed += data
                    answer_area.markdown(streamed)
                elif event == "result":
                    result = data
        except httpx2.HTTPError:
            status.update(label="Failed", state="error", expanded=True)
            raise
        status.update(label="Done", state="complete", expanded=False)
    return result or {}


def _call_or_warn(action, error_message: str, *, level=st.warning):
    """Runs action(), reporting an HTTP failure with the given message and returning None."""
    try:
        return action()
    except httpx2.HTTPError as e:
        logger.warning(f"{error_message}: {type(e).__name__}: {e}")
        level(error_message)
        return None


def _render_sidebar(client: ChatApiClient) -> None:
    """Renders the conversation controls and policy index stats in the sidebar."""
    with st.sidebar:
        st.subheader("Conversation")
        st.caption(f"Thread `{st.session_state.thread_id}`")
        if st.button("Reset conversation"):
            _call_or_warn(
                lambda: client.reset_thread(st.session_state.thread_id),
                "Could not reset the server-side thread",
            )
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()

        st.subheader("Policy index")
        stats = _call_or_warn(client.index_stats, "Index stats unavailable")
        if stats is not None:
            st.metric("Indexed chunks", stats["total_chunks"])
            for category, count in sorted(stats["category_counts"].items()):
                st.write(f"- {category}: {count}")


def main() -> None:
    """Renders the chat page and handles one turn per Streamlit rerun."""
    settings = get_settings()
    client = ChatApiClient(settings.api_base_url)

    st.set_page_config(page_title="RAG Assistant")
    st.title("Corporate Expense & Benefits Assistant")
    st.caption("Company policies. Not tax or legal advice.")
    st.info(
        "You are interacting with an AI system, not a human. Responses are generated "
        "automatically and may contain mistakes.",
        icon="🤖",
    )

    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("history", [])

    _render_sidebar(client)
    _render_history(st.session_state.history)

    if prompt := st.chat_input(
        "Ask about expenses, benefits, deadlines or documents",
        max_chars=MESSAGE_MAX_CHARS,
    ):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = _call_or_warn(
            lambda: _consume_stream(client, st.session_state.thread_id, prompt),
            "Could not reach the assistant API",
            level=st.error,
        )
        if reply is not None:
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": reply.get("answer", ""),
                    "decision": reply.get("decision"),
                    "steps": reply.get("steps", []),
                    "sources": reply.get("sources", []),
                    "generated_at": reply.get("generated_at"),
                    "response_time_ms": reply.get("response_time_ms", 0),
                    "degraded": reply.get("degraded", False),
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
