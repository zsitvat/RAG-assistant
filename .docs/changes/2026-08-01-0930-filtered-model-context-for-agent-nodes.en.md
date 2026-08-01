# 2026-08-01 09:30 UTC — Filtered model context for agent nodes

## What changed

- Added `CurrentRequest.model_context()` (`app/agent/current_request.py`): for every completed
  previous request it keeps only that request's `HumanMessage` and final (non-tool-calling)
  `AIMessage`, dropping that request's `ToolMessage`s and intermediate tool-calling `AIMessage`s;
  the current request (from the latest `HumanMessage` onward) is kept unchanged. Backed by a new
  private `_split_into_requests()`/`_condense()` pair.
- `AgentNodes.classify_intent`, `extract_information`, `agent_step`, and `generate_response`
  (`app/agent/nodes.py`) now send the chat model `CurrentRequest(...).model_context()` instead of
  the raw `state["messages"]`. Guardrails, duplicate-call detection, source collection, and decision
  derivation still read `CurrentRequest.messages()` (unfiltered current-request slice) — unchanged.
- LangGraph state/checkpoints are untouched: filtering happens only when building model input, not
  when writing to `AgentState`.
- `tests/fakes.py::ScriptedChatModel` now records every `_generate()`/`with_structured_output()`
  call's input (`captured_chat_messages`, `captured_structured_inputs`) so tests can assert on what
  a node actually sent the model.
- Tests: extended `tests/test_current_request.py` with `model_context()` coverage (single and
  multiple previous requests, current-request preservation, no-`HumanMessage` fallback); added
  `tests/test_agent_nodes_model_context.py` asserting each of the four model-calling nodes receives
  the filtered context.
- Updated `.docs/plan/02-technical-design.en.md` §5 and
  `.docs/features/05-first-policy-question-agent-journey.en.md`.

## Why

Every model call previously received the full `state["messages"]`, including old tool calls and
`ToolMessage`s from completed prior requests. That let the model treat stale tool evidence as if it
were current, and grew the prompt unboundedly with old tool payloads. `CurrentRequest.messages()`
already scoped loop guardrails to the current request, but did not limit what was actually sent to
the model.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` — all clean
(137 passed), no behavior change to guardrails, sources, or decision derivation.
