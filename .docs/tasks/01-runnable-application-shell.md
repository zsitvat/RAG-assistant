# 01 — Runnable application shell

**What to build:** A reviewer can start the project in dummy-model mode, reach a FastAPI application, and open a thin Streamlit shell that reports whether the backend is available. This slice establishes the LangChain/LangGraph-compatible runtime seam, configuration loading, transport conventions, and test harness without requiring Redis, Ollama, or the policy corpus to work yet.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

**Design references:** Technical design §1.1, §1.2, §3, §9, §10.1, §11, and the relevant shell failure modes in §15.

**Technical notes:**

- Use LangChain chat-model and message interfaces from the first slice so the dummy backend and later ChatOllama backend remain interchangeable without a second provider protocol.
- Assemble FastAPI resources in the lifespan and expose them through dependency injection; module imports must remain side-effect free.
- Keep Streamlit as an HTTP-only client. Shared Python types may describe transport payloads, but the UI must not import graph construction or domain implementations.
- Centralise validated settings, request correlation, and problem-details error mapping now because every later endpoint and trace depends on those conventions.
- Keep deployment settings separate from stable implementation constants. URLs, credentials, model selection, and log level are configurable; graph budgets, retrieval size, key prefixes, timeouts, and retention stay named near their owning implementation.

- [ ] The project installs on Python 3.12 from declared runtime and development dependencies without relying on undeclared global packages.
- [ ] Deployment-dependent settings are validated at startup and have safe local defaults for dummy-model development.
- [ ] The initial settings contract recognises the model backend/tag, Ollama URL, API URL, Redis URL, Langfuse enablement/credentials/host, and log level without requiring optional cloud credentials in offline mode.
- [ ] Defaults match the design: Ollama backend, local Compose service URLs for Ollama/API/Redis, the selected Qwen model tag, Langfuse enabled with its cloud host, and `INFO` application logging.
- [ ] The application is assembled through a lifespan and dependency providers; importing application modules does not open network connections or create runtime resources.
- [ ] A LangChain-compatible deterministic test chat model can emit scripted assistant messages and tool calls without Ollama.
- [ ] The liveness endpoint reports that the process is running, while readiness can clearly report that later dependencies are not configured or unavailable.
- [ ] Errors use one stable problem-details response shape containing a request identifier and never expose a stack trace to the client.
- [ ] The Streamlit shell calls the backend over HTTP, displays a useful connection/readiness error, and imports no graph or business logic.
- [ ] The generated OpenAPI document includes the shell endpoints and their typed response contracts.
- [ ] The process binds through the Uvicorn command rather than introducing redundant host/port environment settings.
- [ ] Focused tests prove that the API and UI-facing client can be exercised with the dummy model and no external services.
- [ ] The repository remains runnable after this ticket, with a documented minimal local smoke command.
