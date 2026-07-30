# 01 — Runnable application shell

**What to build:** A reviewer can start the project in dummy-model mode, reach a FastAPI application, and open a thin Streamlit shell that reports whether the backend is available. This slice establishes the LangChain/LangGraph-compatible runtime seam, configuration loading, transport conventions, and test harness without requiring Redis, Ollama, or the policy corpus to work yet.

**Blocked by:** None — can start immediately.

**Status:** done — see [`.docs/features/01-runnable-application-shell.en.md`](../features/01-runnable-application-shell.en.md)

**Design references:** Technical design §1.1, §1.2, §3, §9, §10.1, §11, and the relevant shell failure modes in §15.

**Technical notes:**

- Use LangChain chat-model and message interfaces from the first slice so the dummy backend and later ChatOllama backend remain interchangeable without a second provider protocol.
- Assemble FastAPI resources in the lifespan and expose them through dependency injection; module imports must remain side-effect free.
- Keep Streamlit as an HTTP-only client. Shared Python types may describe transport payloads, but the UI must not import graph construction or domain implementations.
- Centralise validated settings and request correlation now, and use FastAPI's default error-response convention consistently because every later endpoint and trace depends on those boundaries.
- Keep deployment settings separate from stable implementation constants. URLs, credentials, model selection, and log level are configurable; graph budgets, retrieval size, key prefixes, timeouts, and retention stay named near their owning implementation.

- [x] The project installs on Python 3.12 from declared runtime and development dependencies without relying on undeclared global packages.
- [x] `.env.example` documents the complete application environment contract, while the ignored local `.env` supports dummy/offline development without committing credentials.
- [x] Deployment-dependent settings are validated at startup and have safe local defaults for dummy-model development.
- [x] The initial settings contract recognises the model backend/tag, Ollama URL, API URL, Redis URL, Langfuse enablement/credentials/host, and log level without requiring optional cloud credentials in offline mode.
- [x] Defaults match the design: Ollama backend, local Compose service URLs for Ollama/API/Redis, the selected Qwen model tag, Langfuse enabled with its cloud host, and `INFO` application logging.
- [x] The application is assembled through a lifespan and dependency providers; importing application modules does not open network connections or create runtime resources.
- [x] A LangChain-compatible deterministic test chat model can emit scripted assistant messages and tool calls without Ollama.
- [x] The liveness endpoint reports that the process is running, while readiness can clearly report that later dependencies are not configured or unavailable.
- [x] Errors use FastAPI's default response shapes and never expose a stack trace to the client (custom problem-details normalisation was tried and dropped as unneeded — see the feature doc's deviation note).
- [x] The Streamlit shell calls the backend over HTTP, displays a useful connection/readiness error, and imports no graph or business logic.
- [x] The generated OpenAPI document includes the shell endpoints and their typed response contracts.
- [x] The process binds through the Uvicorn command rather than introducing redundant host/port environment settings.
- [x] Focused tests prove that the API and UI-facing client can be exercised with the dummy model and no external services.
- [x] The repository remains runnable after this ticket, with a documented minimal local smoke command.
- [x] SonarQube Cloud analysis is reproducible through the locked Python scanner, `sonar-project.properties`, generated XML coverage, an environment-supplied token, and a quality-gate-waiting `make sonar` target.
