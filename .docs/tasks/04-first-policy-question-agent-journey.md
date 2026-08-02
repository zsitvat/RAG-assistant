# 04 — First policy-question agent journey

**What to build:** An employee can ask a general policy question through the HTTP chat endpoint and receive a grounded answer produced by the compiled LangGraph agent, including the policies used. Unsupported tax, legal, or unrelated questions receive a deterministic refusal instead of entering the tool loop.

**Blocked by:** 03 — Grounded RAG subgraph.

**Status:** ready-for-agent

**Design references:** Technical design §5, §6.1–§6.4, §9, §10.1, and graph/model failure modes in §15.

**Technical notes:**

- Use an `AgentState` built around messages, intent, category, claim, and decision as LangChain-message-backed working memory; calculation, findings, retrieval results, counters, and timings belong in tool artifacts or traces rather than duplicate state fields. `reference_date` (evaluation date pinning) and `degraded` (observability) were added later as two narrowly-scoped, non-domain fields.
- Compose classification and extraction as LangChain structured-output runnables with Pydantic validation and one repair retry.
- Build the autonomous loop with a tool-bound chat model, conditional edges, and LangGraph `ToolNode`; do not encode an intent-to-tool dispatch table.
- Derive loop budgets and duplicate-call detection from messages since the latest human turn, with the graph recursion limit as a final safety backstop.
- Project public responses from graph messages and typed artifacts through the agent module so HTTP routes remain transport-only.
- Keep the model-facing tool result concise and the full Pydantic artifact outside the prompt; state and artifacts must have one authoritative home.

- [x] The agent state's core domain fields are messages, intent, category, claim, and decision, with LangChain message reduction and typed domain models; `reference_date` and `degraded` were added afterwards as non-domain fields for evaluation pinning and observability, not additional business state.
- [x] `ExpenseClaim` supports only fields consumed by routing or deterministic tools: category/subtype, HUF amount, date, headcount, distance/direction/days, excluded amount, receipt/approval, and used annual benefit budget.
- [x] The LangChain model factory supplies ChatOllama in normal operation and the scripted test model in dummy mode through the same chat-model interface.
- [x] Production uses `qwen2.5:7b-instruct-q4_K_M` for every call at temperature 0. Structured classification, extraction, tool selection, and final response generation all share one chat-model instance and one temperature; the design's originally suggested 0/0.2 split was deliberately dropped in favour of fully deterministic generation across every node.
- [x] Model selection and prompts permit best-effort Hungarian interaction over the English corpus while the canonical prompts, rule identifiers, enum values, and official evaluation remain English.
- [x] Intent classification and information extraction use structured output with typed validation, one repair attempt, and an observable degraded fallback.
- [x] The compiled main graph contains seven focused nodes and conditional routes for complete, incomplete, unsupported, tool-calling, and response-generating paths.
- [x] The four prompt contracts are classification, information extraction, agent step, and final response generation, all represented as composable LangChain runnables.
- [x] The agent step uses `bind_tools()` and autonomously selects the policy-search tool instead of following a hard-coded intent-to-tool lookup.
- [x] Tool execution uses LangGraph `ToolNode`, returns compact `ToolMessage` content, and retains the typed artifact for downstream response construction and evaluation.
- [x] A general policy question completes through classification, extraction, deterministic routing, tool selection, retrieval, and grounded response generation.
- [x] The blocking chat endpoint returns a typed response with answer, UTC completion time, duration, deduplicated sources, and stable public step labels.
- [x] A policy-dependent answer without a supporting tool artifact is refused or clearly marked as lacking evidence.
- [x] Unsupported requests never call policy, calculation, or rule tools and return the fictional-policy and no-legal-advice disclaimer.
- [x] Loop budgets, invalid tool arguments, and repeated identical calls terminate deterministically and never expose chain-of-thought.
- [x] The loop permits at most four tool-calling agent steps (`MAX_AGENT_STEPS = 4`) and uses a graph recursion limit as a hard backstop. The limit is 15, not the design's originally suggested 20: the worst-case four-tool-call path reaches `generate_response` in 12 node executions, empirically proven (via `GraphRecursionError`) to require at least 13 as the recursion limit; 15 keeps a small margin above that proven minimum.
- [x] The same tool may return invalid arguments at most twice before being disabled for the turn, and an identical repeated call reuses the prior artifact while recording a warning.
- [x] Exhausting the tool budget routes to final response generation with available evidence and explicitly states when that evidence is incomplete.
- [x] Ollama calls retry with bounded backoff twice and then return a clear failure rather than a fabricated answer.
- [x] Scripted integration tests verify the expected graph path, tool sequence, source projection, unsupported path, and recursion backstop.
