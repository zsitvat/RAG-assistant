# Feature: Langfuse observability, prompt resolution, and operational logging

Implements task
[`10-langfuse-observability-prompt-resolution-and-operational-logging.md`](../tasks/10-langfuse-observability-prompt-resolution-and-operational-logging.md).

## What it does

Attaches one Langfuse trace per chat turn (nested through every LangGraph node, tool, and model
call automatically), resolves each of the four agent prompts from a production-labelled Langfuse
version with a validated embedded fallback, and keeps this entirely optional — chat, the agent
graph, and application logs work identically whether or not Langfuse is configured.

## How it works

### Tracing (`app/integrations/langfuse.py`)

`Observability.build(settings)` degrades to a disabled adapter (`client=None`) on any setup problem
— `langfuse_enabled=False`, missing `langfuse_public_key`/`langfuse_secret_key`, or the `Langfuse()`
constructor itself raising — logging a warning but never blocking chat. `AgentService._config()`
merges `Observability.trace_config(thread_id)` into the graph's runnable config: when enabled, this
attaches one `langfuse.langchain.CallbackHandler()` and `langfuse_session_id=thread_id`, so every
node, tool, retriever call and model generation nested under that one `graph.invoke()`/`astream()`
becomes a linked observation in one trace — no tracing calls live inside graph nodes.
`AgentService._project()` calls `Observability.update_trace(thread_id=..., intent=..., category=...,
decision=...)` once per turn so the trace also carries the turn's classified/derived outcome.

### Prompt resolution (`app/agent/prompt_library.py`)

`PromptLibrary.get(name)` tries the `production`-labelled Langfuse prompt first, validates it, and
falls back to the embedded `ChatPromptTemplate` (`app/agent/prompts.py`) on any failure — missing
prompt, invalid content, or Langfuse unreachable — logging one warning, never the prompt text
itself. Resolution is cached per `PromptLibrary` instance (one per process), so a disabled or
unreachable Langfuse causes at most one resolution attempt per prompt name, not a retry per turn.

**Shared validation, not duplicated per prompt.** `PromptSpec` declares what a name requires
(`requires_no_fabrication` — every prompt; `requires_citation_markers` — `generate_response` only;
`requires_schema` — `extract_information` only), and `PromptLibrary.validate()` checks both the
remote and the embedded version against the same rules: must accept the `messages` placeholder,
must forbid inventing values ("never invent"/"never fabricate"/"never guess" — the three phrasings
already in use), and the two prompt-specific requirements. Writing this validator surfaced that
`extract_information`'s guardrail is phrased "never guess" rather than "never invent"/"never
fabricate" like the other three — a real wording inconsistency the check now tolerates explicitly
rather than silently passing by accident.

### Structured logging (`app/logging/config.py`, unchanged this task)

Already JSON, UTC, stdout + `TimedRotatingFileHandler` (midnight UTC, 7-day `backupCount`) from
task 1. No application log call passes prompt text, claim data, retrieved context, or answers —
verified by an AST-based test scanning every `logger.*()` call site for forbidden argument names,
so a future call that accidentally logs `claim`/`prompt`/`answer`/`context` fails the test instead
of shipping.

## Deliberate deviations

- **No per-request correlation id.** An earlier `RequestContextMiddleware` (binding `X-Request-ID`
  into log lines and traces) was removed as unneeded for this project's scale — see
  `.docs/features/09-streaming-chat-experience-and-thread-controls.en.md`. Log lines and Langfuse
  traces correlate by `thread_id` instead.
- **Retention and rollover testing already existed from task 1** (`TimedRotatingFileHandler`'s
  own `backupCount`); this task did not need to add separate retention-cleanup code, since the
  stdlib handler already deletes rotated files beyond the count on each rollover.

## Key files

| File | Responsibility |
| --- | --- |
| `app/integrations/langfuse.py` | `Observability` — client lifecycle, trace config, trace updates |
| `app/agent/prompt_library.py` | `PromptLibrary`, `PromptSpec`, shared validation, `ResolvedPrompt` |
| `app/agent/nodes.py` | resolves each of the four prompts through `PromptLibrary` instead of the static import |
| `app/agent/service.py` | attaches the trace config per turn, updates trace outcome attributes |
| `app/dependencies.py` | builds `Observability` and `PromptLibrary` once at startup |
| `tests/test_observability.py` | disabled/enabled paths, degrade-on-failure, trace config shape |
| `tests/test_prompt_library.py` | every embedded prompt validates, remote resolution, remote-invalid and remote-unreachable fallback, caching |
| `tests/test_logging.py` | payload-exclusion AST guard |

## A test-isolation issue this task surfaced

Building `Observability` inside `ApplicationDependencies.build()` (before the Redis check) meant
every full-app integration test (`test_api.py`, `test_admin.py`, `test_offline_api.py`) was
constructing a **real** `Langfuse` client from the developer's `.env` credentials, since none of
those fixtures overrode `LANGFUSE_ENABLED`. Since `AgentService` now attaches a real
`CallbackHandler()` per turn when enabled, tests that hit `/chat`/`/chat/stream` may have sent test
traces to the developer's live Langfuse Cloud project. Fixed by setting `LANGFUSE_ENABLED=false` in
every fixture/test that boots the real app (`monkeypatch.setenv`) or constructs `Settings()` en
route to `ApplicationDependencies.build()`.
