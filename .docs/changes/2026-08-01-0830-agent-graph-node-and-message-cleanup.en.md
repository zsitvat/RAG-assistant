# 2026-08-01 08:30 UTC — Agent graph node and message cleanup

## What changed

- Removed the `check_request` node from the main agent graph
  (`app/agent/graph.py`, `app/agent/nodes.py`). It was a pure no-op sitting between
  `extract_information` and its conditional routing; `route_after_extraction` is now wired
  directly as a conditional edge off `extract_information`. The graph has seven nodes
  instead of eight.
- Extracted the user-facing message constants (`CLARIFICATION_QUESTIONS`,
  `DEFAULT_CLARIFICATION_QUESTION`, `OUT_OF_SCOPE_MESSAGE`, `NO_TOOL_ARTIFACT_MESSAGE`,
  `LLM_UNAVAILABLE_MESSAGE`, `INCOMPLETE_EVIDENCE_NOTE`) out of `app/agent/nodes.py` into
  a new `app/agent/messages.py`.
- Updated `.docs/features/05-first-policy-question-agent-journey.en.md`'s node table and
  wiring description accordingly.

## Why

`check_request` had no distinct behavior and no step label depends on its node name
(`AgentService`'s step projection keys off tool-message names, not graph node names), so
keeping it only added an unnecessary hop. Splitting the message constants out of
`nodes.py` keeps that module focused on node behavior rather than mixing it with
user-facing copy.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` —
all clean, no behavior change.
