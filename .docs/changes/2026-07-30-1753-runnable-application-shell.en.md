# 2026-07-30 17:53 UTC — Runnable application shell (task 1)

## What changed

- Added `pyproject.toml` (uv-managed, Python `>=3.12,<3.13`), `uv.lock`, `.venv`, and a `Makefile`
  with `install`/`lint`/`format`/`format-check`/`security`/`test`/`check`/`sonar`/`run-api`/
  `run-ui`/`clean` targets.
- Added `.env.example` as the committed application-setting template. The ignored local `.env`
  uses dummy mode, loopback service URLs and disabled Langfuse for an offline shell.
- Implemented the FastAPI shell: `app/core/config.py` (settings), `app/core/logging.py` (JSON
  logging with 7-day retention), `app/core/observability.py` (request correlation middleware),
  `app/integrations/llm.py` (LangChain chat-model backend factory), `app/dependencies.py`,
  `app/api/schemas.py`, `app/api/routes/health.py`, `app/api/router.py`, `app/main.py`.
- Implemented the Streamlit shell: `app/ui.py` (HTTP-only client, reports `/ready` status).
- Added focused tests: `tests/test_api.py`, `tests/test_llm.py`, `tests/test_logging.py` (14 tests,
  99% coverage of `app/`, `app/ui.py` excluded from coverage — it is a Streamlit script exercised by
  manual smoke run, not pytest).
- Set up SonarQube Cloud integration for project `zsitvat_RAG-assistant`: locked `pysonar`,
  `sonar-project.properties`, a token-guarded `make sonar` target that waits for the quality gate,
  and the separately installed `sonar` management CLI.
- Updated [`02-technical-design.en.md`](../plan/02-technical-design.en.md) §3 to reflect the
  `pyproject.toml`/`uv.lock` dependency management decision (see below).
- Added [`.docs/features/01-runnable-application-shell.en.md`](../features/01-runnable-application-shell.en.md).

## Why

Executing task 1 of the ordered development plan
([`00-development-flow.en.md`](../plan/00-development-flow.en.md) §7): establish the runtime seam,
configuration, logging/correlation and error-response conventions every later endpoint and trace
depends on, before any Redis/Ollama/graph code exists.

Two decisions diverged from the original technical design and were corrected in the design document
directly, per the development flow's "do not silently diverge" rule:

- **`pyproject.toml` + `uv.lock` instead of `requirements.in`/`.txt` (+ `-dev`).** The design
  originally planned a pip-tools-style split. `uv` already provides compilation and locking through
  `pyproject.toml`/`uv.lock`, so keeping the second pattern would duplicate the same responsibility.
- **LangChain's own `FakeListChatModel`/`GenericFakeChatModel` instead of a hand-written `DummyLLM`.**
  The design's repository layout implied a custom test chat model; LangChain 1.0 already ships
  framework-native fakes that satisfy the same acceptance criterion (scripted messages and tool
  calls, no Ollama) without introducing a second model-client implementation, which the design's own
  framework-boundary note (§1.1) rules out.

Dependencies were pinned to their latest resolvable versions via `uv add` at the user's request,
including `langchain-core>=1.0` (resolved to 1.5.3), `langchain-ollama>=1.1.0` and
`pysonar>=1.7.0.5143`.

The FastAPI lifespan uses `@asynccontextmanager` with an `AsyncGenerator[None, None]` return
annotation. File-level and module-level docstrings were removed from the current Python source and
test files.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`,
`pytest --cov=app --cov-report=term-missing --cov-report=xml` — all clean (14 passed, 99% coverage).
Live smoke test: `uvicorn app.main:app` (`LLM_BACKEND=dummy`) + `streamlit run app/ui.py` against it,
both start and respond correctly. The scanner configuration dry-run recognises the project, source,
test and coverage settings. A real SonarQube Cloud submission remains pending a `SONAR_TOKEN`; the
optional management CLI browser connection remains pending `sonar auth login -o zsitvat`.
