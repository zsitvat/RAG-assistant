# Feature: Runnable application shell

Implements task [`01-runnable-application-shell.md`](../tasks/01-runnable-application-shell.md).

## What it does

Provides the FastAPI + Streamlit runtime seam the rest of the project builds on: validated
settings, structured logging, request correlation, and a LangChain-compatible LLM backend switch
(`ollama` / `dummy`) — all runnable without Redis, Ollama or the policy corpus.

## How it works

- **Settings** (`src/app/core/config.py`): a `pydantic-settings` `Settings` model validated once at
  startup, cached via `get_settings()`. Fields match the technical design's configuration table
  (`LLM_BACKEND`, `OLLAMA_BASE_URL`, `LLM_MODEL`, `API_BASE_URL`, `REDIS_URL`, `LANGFUSE_*`,
  `LOG_LEVEL`); Langfuse credentials are optional so offline/dummy development needs no secrets.
  `Settings` reads `.env`; `.env.example` is the committed Compose-oriented template, while the
  ignored local `.env` selects dummy mode, loopback service URLs and disabled Langfuse.
- **Logging** (`src/app/logging/config.py`): `configure_logging()` attaches one JSON formatter to a
  stdout handler and a stdlib `TimedRotatingFileHandler` (UTC midnight, `backupCount=7`) writing
  under `./logs` — the handler's own count-based retention is enough for a daily-rotating file, so
  no custom retention helper was needed. Uvicorn, FastAPI and Streamlit loggers are redirected
  through the same handlers so framework logs share the format. There is no per-request
  correlation id — an earlier `RequestContextMiddleware` binding an `X-Request-ID` into log lines
  was removed as unneeded for this project's scale.
- **Error handling**: no custom exception handler. FastAPI/Starlette's own default error shapes
  (404s, validation errors, unhandled exceptions) are used as-is, and unhandled exceptions are
  logged by Uvicorn's default error logging. `debug=False` (FastAPI's default) already guarantees
  no traceback reaches the client — a custom problem-details wrapper was considered and dropped as
  unnecessary complexity for what the design actually needs (see deviation note below).
- **LLM backend factory** (`src/app/integrations/llm.py`): `build_chat_model(settings)` returns a
  LangChain `BaseChatModel` — `ChatOllama` for `LLM_BACKEND=ollama`, or LangChain's own
  `FakeListChatModel` (cycles canned responses, never raises) for `LLM_BACKEND=dummy`. No custom
  chat-model class was written; LangChain's framework-native fakes
  (`FakeListChatModel`, `GenericFakeChatModel`) cover both the runtime dummy backend and scripted
  tool-call tests, consistent with the design's "no second model client" boundary.
- **FastAPI shell** (`src/app/main.py`, `src/app/dependencies.py`, `src/app/api/`): the lifespan loads settings,
  configures logging and builds the chat model once, stored on `app.state`. The lifespan is an
  `asynccontextmanager` annotated as `AsyncGenerator[None, None]`; route modules only
  depend on `src/app/dependencies.py` providers. `GET /health` reports liveness; `GET /ready` reports
  one status per dependency (`llm`, `redis`) — `redis` is always `not_configured` until task 2 wires
  it in.
- **Streamlit shell** (`src/app/ui.py`): a thin HTTP client — no graph or domain imports — that calls
  `GET /ready` and renders the per-dependency status, or a connection error if the API is
  unreachable.

## How to use

```bash
uv sync --dev
cp .env.example .env
# Set LLM_BACKEND=dummy, LANGFUSE_ENABLED=false and loopback URLs in .env.
uv run uvicorn app.main:app --port 8000
uv run streamlit run src/app/ui.py
```

`make check` runs the full local quality gate (ruff lint + format check, bandit, pytest + coverage).
With `SONAR_TOKEN` exported, `make sonar` regenerates `coverage.xml`, runs the locked `pysonar`
scanner and waits for the SonarQube Cloud quality gate.
`make clean` removes local caches (`.ruff_cache`, `.pytest_cache`, `.sonar`, coverage files,
`__pycache__`).

## Key files

| File | Responsibility |
| --- | --- |
| `.env.example` | committed application-environment template |
| `pyproject.toml`, `uv.lock` | dependency declarations and reproducible lock |
| `Makefile` | local quality, Sonar and run commands |
| `sonar-project.properties` | SonarQube Cloud project, scope, coverage and gate settings |
| `src/app/settings.py` | `Settings`, `get_settings()` |
| `src/app/logging/config.py` | JSON logging, retention |
| `src/app/integrations/llm.py` | chat-model backend factory |
| `src/app/dependencies.py` | FastAPI dependency providers |
| `src/app/api/schemas.py` | `HealthResponse`, `ReadyResponse` |
| `src/app/api/routes/health.py` | `/health`, `/ready` |
| `src/app/main.py` | app assembly, lifespan |
| `src/app/ui.py` | Streamlit shell |
| `tests/api/test_api.py`, `src/app/integrations/tests/test_llm.py`, `src/app/logging/tests/test_config.py` | focused tests |

## Deliberate deviation from the technical design

`pyproject.toml` + `uv.lock` (via `uv sync`/`uv add`) replace the originally planned
`requirements.in`/`requirements.txt` (+ `-dev`) pip-tools pattern — see
[`02-technical-design.en.md`](../plan/02-technical-design.en.md) §3. `uv` already compiles and locks
dependencies, so a second lock mechanism would be redundant. Runtime dependencies are pinned to
their latest resolvable versions at the time of writing, including `langchain-core>=1.0`.

The design's originally planned RFC-7807 problem-details error shape (§10.1) was dropped after
review: FastAPI's own default error shapes and Uvicorn's default error logging already satisfy the
only real requirement (no stack trace reaches the client), so the extra normalisation layer and its
`ProblemDetail` schema were removed as unneeded complexity.
