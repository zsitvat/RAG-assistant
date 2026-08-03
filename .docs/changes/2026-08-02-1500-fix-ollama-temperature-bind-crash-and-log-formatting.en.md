# 2026-08-02 15:00 — Fix Ollama chat-model crash on every real request

## What changed

- `app/integrations/llm.py::build_chat_model` now constructs `ChatOllama(..., temperature=0)`
  directly instead of leaving `temperature` unset.
- `app/dependencies.py` no longer calls `chat_model.bind(temperature=0)` for the two node-bound
  model instances; both `AgentNodes` model params now receive the same `chat_model` built with
  `temperature=0` already set.
- `_invoke_with_retry` (`app/agent/nodes.py`) and `StructuredOutputRunner` (`app/agent/structured.py`)
  now log the caught exception's type and message instead of swallowing it silently.
- Converted the remaining `%s`-style `logger.*` calls across `app/` to f-strings.

## Why

`temperature` is a `ChatOllama` constructor field, not a per-call keyword argument. `.bind(temperature=0)`
attached it as an extra invocation-time kwarg instead, which `ChatOllama._chat_params` forwards
straight through to the underlying `ollama.Client.chat(...)` call — raising
`TypeError: Client.chat() got an unexpected keyword argument 'temperature'` on every single model
call. Wrapped in `Runnable.with_retry()`, this error was retried with exponential backoff before the
node gave up and returned the generic `LLM_UNAVAILABLE_MESSAGE`, which is why every real chat request
against the 7B Ollama model failed end-to-end after 60-90+ seconds despite every model capability
(tool binding, structured output, plain generation) working correctly in isolation — those isolated
tests never went through the `.bind(temperature=0)` code path.

The exception was invisible in logs because both `_invoke_with_retry` and `StructuredOutputRunner`
caught `Exception` and logged a fixed message with no exception detail, which is what made this take
so long to diagnose. Verified live: after the fix, `POST /chat` against the real
`qwen2.5:7b-instruct-q4_K_M` model returns an actual generated answer instead of the fallback
message (remaining end-to-end latency, ~80s across five sequential model calls on CPU inference, is
expected hardware behavior, not a bug).
