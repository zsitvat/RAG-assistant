# 2026-08-02 11:00 UTC — Streaming chat, Langfuse observability, and app/core split

## What changed

### Task 9 — Streaming and thread controls
- `app/agent/service.py`: added `AgentService.stream()`, projecting `graph.astream(...,
  stream_mode=["updates","messages"])` into `step`/`source`/`token` events plus a final `result`,
  sharing `_project()` with the blocking `respond()` so both endpoints stay in parity.
- `app/api/schemas.py`: added `StreamEvent` (`to_sse()`), `ThreadResetResponse`.
- `app/api/routes/chat.py`: added `POST /chat/stream` (SSE) and `DELETE /threads/{thread_id}`.
- `app/rag/model.py`: added `RagResult.from_artifact` — `graph.get_state()` returns serialized
  channel values, so a checkpointed `search_policies` artifact is a plain dict, the same class of
  problem `ExpenseClaim.from_state` (task 7) already solved for the claim.
- `app/ui.py`: replaced the placeholder page with a real streaming chat (`ChatApiClient`, live
  `st.status` progress, thread reset, read-only index stats).
- `app/main.py`, `app/settings.py`: `ui_origin`-restricted CORS.
- `app/core/` split: `config.py` → `app/settings.py`, `logging.py`/`observability.py` →
  `app/logging/{config,middleware}.py`. `RequestContextMiddleware` was later removed entirely
  (below), not just moved.

### Task 10 — Langfuse observability and prompt resolution
- Added `app/integrations/langfuse.py` (`Observability`) and `app/agent/prompt_library.py`
  (`PromptLibrary`, `PromptSpec`, shared validation for every prompt, remote-or-embedded
  resolution with caching).
- `app/agent/nodes.py`, `app/dependencies.py`, `app/agent/service.py`: wired the trace config and
  resolved prompts through the graph.
- Added `langfuse==4.14.2` (and its OpenTelemetry dependencies).
- Removed `app/logging/middleware.py` (`RequestContextMiddleware`) and the `request_id_var`/
  `thread_id_var` contextvars entirely — request-id correlation was judged unnecessary for this
  project's scale; log lines and Langfuse traces now correlate by `thread_id` alone.
- Refactored `RuleCatalogue._validate_references` (Sonar: cognitive complexity 19 → within the
  15 limit) into `_document_id_errors`/`_declared_references`/`_reference_errors`.
- Fixed a real test-isolation gap: `ApplicationDependencies.build()` constructs `Observability`
  before checking Redis, so every full-app test that didn't explicitly disable Langfuse was
  building a **real** client from the developer's `.env` credentials — and, since `AgentService`
  now attaches a live `CallbackHandler()` per turn, may have sent test traces to a real Langfuse
  Cloud project. Fixed by setting `LANGFUSE_ENABLED=false` in every fixture/test that boots the
  real app or constructs `Settings()` en route to `ApplicationDependencies.build()`.

## Why

Tasks 9 and 10. Streaming needed a public event vocabulary narrow enough that internal reasoning,
tool JSON, and retrieval scores never reach the browser, while staying provably identical to the
blocking response. Langfuse tracing needed to be genuinely optional (never a chat-blocking
dependency) while still giving one linked trace per turn, and prompt resolution needed the same
guardrail validation regardless of whether the text came from Langfuse or the embedded fallback.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` — all clean
(260 passed).
