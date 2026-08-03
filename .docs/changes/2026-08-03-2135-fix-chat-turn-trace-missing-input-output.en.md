# 2026-08-03 21:35 — Fix `chat_turn` trace missing input/output in Langfuse

## What changed

- `Observability.traced_turn(thread_id, **metadata)` now takes the turn's user message as a required
  second positional argument and passes it as `input=message` to
  `client.start_as_current_observation(name="chat_turn", ...)`.
- The `update_trace(**attrs)` callback yielded by `traced_turn` now accepts an `output=` keyword,
  separate from the outcome metadata, and calls `span.update(metadata=attrs, output=output)`.
- `AgentService.ainvoke_graph`, `astream` and `evaluate` pass the user's message into `traced_turn`,
  and their `update_trace(...)` calls now also pass `output=<final answer text>`.

## Why

- After the turn's Langfuse span was renamed to `chat_turn` and promoted to the trace's root span
  (see the 21:14 change), the Langfuse UI's trace list stopped showing anything in the Input/Output
  columns for `chat_turn` traces, even though the older `LangGraph`-named traces had shown both. A
  root span's `input`/`output` are what the Langfuse UI displays as the trace's input/output; the
  `chat_turn` span was opened and closed without either being set, so both stayed empty regardless of
  what the nested LangChain callback span captured for the graph call itself.

## Quality gates run

`ruff check .` (clean); full `pytest` suite (311 passed).
