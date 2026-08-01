import json
import uuid
from collections.abc import Iterator
from datetime import datetime

import httpx2
import streamlit as st

from app.settings import get_settings

STREAM_TIMEOUT_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 5.0
CLARIFICATION_DECISION = "needs_info"


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
            event_name = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_name = line.removeprefix("event: ").strip()
                elif line.startswith("data: ") and event_name:
                    yield event_name, json.loads(line.removeprefix("data: "))["data"]

    def _get(self, path: str) -> dict:
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
        status.update(label="Done", state="complete", expanded=False)
    return result or {}


settings = get_settings()
client = ChatApiClient(settings.api_base_url)

st.set_page_config(page_title="RAG Assistant")
st.title("Corporate Expense & Benefits Assistant")
st.caption("Demo policies describe a fictional company. Not tax or legal advice.")

st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("history", [])

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Thread `{st.session_state.thread_id}`")
    if st.button("Reset conversation"):
        try:
            client.reset_thread(st.session_state.thread_id)
        except httpx2.HTTPError as exc:
            st.warning(f"Could not reset the server-side thread: {exc}")
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.subheader("Policy index")
    try:
        stats = client.index_stats()
    except httpx2.HTTPError as exc:
        st.warning(f"Index stats unavailable: {exc}")
    else:
        st.metric("Indexed chunks", stats["total_chunks"])
        for category, count in sorted(stats["category_counts"].items()):
            st.write(f"- {category}: {count}")

_render_history(st.session_state.history)

if prompt := st.chat_input("Ask about expenses, benefits, deadlines or documents"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        reply = _consume_stream(client, st.session_state.thread_id, prompt)
    except httpx2.HTTPError as exc:
        st.error(f"Could not reach the assistant API: {exc}")
    else:
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": reply.get("answer", ""),
                "decision": reply.get("decision"),
                "steps": reply.get("steps", []),
                "sources": reply.get("sources", []),
                "generated_at": reply.get("generated_at"),
                "response_time_ms": reply.get("response_time_ms", 0),
            }
        )
        st.rerun()
