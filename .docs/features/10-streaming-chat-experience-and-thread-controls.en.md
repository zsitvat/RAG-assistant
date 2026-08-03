# Feature: Streaming chat experience and thread controls

Implements task
[`09-streaming-chat-experience-and-thread-controls.md`](../tasks/09-streaming-chat-experience-and-thread-controls.md).

## What it does

Adds `POST /chat/stream`, which turns the graph's internal event flow into a four-word public SSE
vocabulary (`step`, `source`, `token`, `result`), and replaces the placeholder Streamlit page with a
real chat that shows live progress, streams the answer, and can reset the server-side thread.

## How it works

### The public event vocabulary (`src/app/agent/streaming.py`)

`AgentService.astream()` consumes `graph.astream(..., stream_mode=["updates", "messages"])` and
delegates translation to `StreamEventMapper`; no LangGraph object reaches the client.

| Source | Event | Rule |
| --- | --- | --- |
| node update | `step` | `STEP_LABELS` allow-list, deduplicated per turn |
| `search_policies` tool message | `source` | one `ChatSource` per citation, deduplicated by `(doc_id, section)` |
| message chunk | `token` | only when `metadata["langgraph_node"] == "generate_response"` |
| stream end | `result` | the complete `ChatResponse` |

**Token filtering is load-bearing.** `agent_step` and `execute_tools` also emit message chunks — the
tool-calling `AIMessage`s and the raw `ToolMessage` payloads. Without the node filter, retrieved
policy text and tool JSON would stream straight into the chat window.

Step labels come from an allow-list keyed by node name, never from node output: `classify_intent` →
"Intent classified", `extract_information` → "Details extracted", `execute_tools` → the tool's
label, and each terminal node has its own distinct label (`generate_response` → "Answer generated",
`ask_clarification` → "Clarification asked", `out_of_scope` → "Marked out of scope"). Deterministic
clarification and refusal turns therefore produce no `token` events at all; their text arrives in the
final `result`.

### Parity with the blocking endpoint

Both endpoints end in the same `AgentService._build_response(thread_id, state, start)` method, which
updates the trace and delegates to `ResponseBuilder.build_chat()`, so `/chat` and the `result` event
of `/chat/stream` cannot drift. `ainvoke_graph()` builds it from `ainvoke()`'s return value;
`astream()` builds it from `graph.aget_state(config).values` once the stream is exhausted.

**A checkpoint round-trip flattens tool artifacts.** `get_state()` returns *serialized* channel
values, so a `search_policies` `ToolMessage.artifact` is a plain `dict`, not a `RagResult` — the
same class of problem `ExpenseClaim.from_state` already solves for the claim. `RagResult`
now has a matching `from_artifact()` coercion, and source collection goes through it.

### Streamlit chat (`src/app/ui.py`)

`ChatApiClient` owns every HTTP call; the page keeps only presentation state
(`thread_id`, `history`) — conversation truth stays in the LangGraph checkpoint. During a turn,
`st.status` shows steps and sources live while tokens accumulate into a placeholder; afterwards the
status collapses and each stored assistant message renders its answer, a local-time/duration
caption, and a collapsed "Steps and sources" expander. A `needs_info` reply is rendered with
`st.info` and a distinct icon. The sidebar holds thread reset (`DELETE /threads/{id}`, then a fresh
local `thread_id`) and read-only index stats — no model or retrieval tuning controls. An HTTP
failure surfaces the backend detail via `st.error` and leaves the visible conversation and thread id
intact, so the next message retries on the same thread.

## Key files

| File | Responsibility |
| --- | --- |
| `src/app/agent/service.py` | `AgentService.astream()`, shared `_build_response` |
| `src/app/agent/streaming.py` | `StreamEventMapper` — the event allow-lists |
| `src/app/agent/responses.py` | `ResponseBuilder` — shared `build_chat()` |
| `src/app/api/schemas.py` | `StreamEvent` and its `to_sse()` wire rendering |
| `src/app/api/routes/chat.py` | `POST /chat/stream` as a `StreamingResponse` |
| `src/app/rag/model.py` | `RagResult.from_artifact` |
| `src/app/ui.py` | `ChatApiClient` and the chat page |
| `src/app/tests/journeys/test_chat_stream.py` | event vocabulary, allow-listing, dedup, token filtering, blocking parity, no-token turns, SSE wire format, HTTP contract |

## Related restructuring

`app/core/` (back when `app/` itself lived at the repo root, before the `src/` layout) was split in
the same change: `app/core/config.py` → `app/settings.py`, `app/core/logging.py` →
`app/logging/config.py`.
