# Feature: Langfuse observability, prompt resolution, and operational logging

Implements task
[`10-langfuse-observability-prompt-resolution-and-operational-logging.md`](../tasks/10-langfuse-observability-prompt-resolution-and-operational-logging.md).

## What it does

Attaches one Langfuse trace per chat turn (nested through every LangGraph node, tool, and model
call automatically), resolves each of the four agent prompts from a production-labelled Langfuse
version with an embedded fallback, and keeps this entirely optional — chat, the agent
graph, and application logs work identically whether or not Langfuse is configured.

## How it works

### Tracing (`src/app/integrations/langfuse.py`)

`Observability.build(settings)` degrades to a disabled adapter (`client=None`) on any setup problem
— `langfuse_enabled=False`, missing `langfuse_public_key`/`langfuse_secret_key`, or the `Langfuse()`
constructor itself raising — logging a warning but never blocking chat. `AgentService._config()`
merges `Observability.trace_config(thread_id)` into the graph's runnable config: when enabled, this
attaches one `langfuse.langchain.CallbackHandler()` and `langfuse_session_id=thread_id`, so every
node, tool, retriever call and model generation nested under that one `graph.invoke()`/`astream()`
becomes a linked observation in one trace — no tracing calls live inside graph nodes.
`AgentService._project()` calls `Observability.update_trace(thread_id=..., intent=..., category=...,
decision=...)` once per turn so the trace also carries the turn's classified/derived outcome.

### Prompt resolution (`src/app/agent/prompt_library.py`)

`PromptLibrary.get(name)` tries the `production`-labelled Langfuse prompt first, and falls back to
the embedded `ChatPromptTemplate` (`src/app/agent/prompts.py`) on any failure — missing prompt or
Langfuse unreachable — logging one warning, never the prompt text itself. Resolution is cached per
`PromptLibrary` instance (one per process), so a disabled or unreachable Langfuse causes at most one
resolution attempt per prompt name, not a retry per turn. There is no content validation of the
resolved prompt (structural correctness or guardrail wording): whichever prompt Langfuse returns for
the `production` label is trusted as-is, matching the embedded fallback's own contract.

### Structured logging (`src/app/logging/config.py`, unchanged this task)

Already JSON, UTC, stdout + `TimedRotatingFileHandler` (midnight UTC, 7-day `backupCount`) from
task 1. No application log call passes prompt text, claim data, retrieved context, or answers —
verified by an AST-based test scanning every `logger.*()` call site for forbidden argument names,
so a future call that accidentally logs `claim`/`prompt`/`answer`/`context` fails the test instead
of shipping.

## Deliberate deviations

- **Retention and rollover testing already existed from task 1** (`TimedRotatingFileHandler`'s
  own `backupCount`); this task did not need to add separate retention-cleanup code, since the
  stdlib handler already deletes rotated files beyond the count on each rollover.

## Key files

| File | Responsibility |
| --- | --- |
| `src/app/integrations/langfuse.py` | `Observability` — client lifecycle, trace config, trace updates |
| `src/app/agent/prompt_library.py` | `PromptLibrary`, `ResolvedPrompt` |
| `src/app/agent/nodes.py` | resolves each of the four prompts through `PromptLibrary` instead of the static import |
| `src/app/agent/service.py` | attaches the trace config per turn, updates trace outcome attributes |
| `src/app/dependencies.py` | builds `Observability` and `PromptLibrary` once at startup |
| `src/app/integrations/tests/test_observability.py` | disabled/enabled paths, degrade-on-failure, trace config shape |
| `src/app/agent/tests/test_prompt_library.py` | every embedded prompt resolves, production-labelled remote resolution, remote-unreachable fallback, caching |
| `src/app/logging/tests/test_config.py` | payload-exclusion AST guard |

## A test-isolation issue this task surfaced

Building `Observability` inside `ApplicationDependencies.build()` (before the Redis check) meant
every full-app integration test (`test_api.py`, `test_admin.py`, `test_offline_api.py`) was
constructing a **real** `Langfuse` client from the developer's `.env` credentials, since none of
those fixtures overrode `LANGFUSE_ENABLED`. Since `AgentService` now attaches a real
`CallbackHandler()` per turn when enabled, tests that hit `/chat`/`/chat/stream` may have sent test
traces to the developer's live Langfuse Cloud project. Fixed by setting `LANGFUSE_ENABLED=false` in
every fixture/test that boots the real app (`monkeypatch.setenv`) or constructs `Settings()` en
route to `ApplicationDependencies.build()`.
