# 04 — First policy-question agent journey

**What to build:** An employee can ask a general policy question through the HTTP chat endpoint and receive a grounded answer produced by the compiled LangGraph agent, including the policies used. Unsupported tax, legal, or unrelated questions receive a deterministic refusal instead of entering the tool loop.

**Blocked by:** 03 — Grounded RAG subgraph.

**Status:** ready-for-agent

**Design references:** Technical design §5, §6.1–§6.4, §9, §10.1, and graph/model failure modes in §15.

**Technical notes:**

- Use a five-key `AgentState` with LangChain messages as the working memory; calculation, findings, retrieval results, counters, and timings belong in tool artifacts or traces rather than duplicate state fields.
- Compose classification and extraction as LangChain structured-output runnables with Pydantic validation and one repair retry.
- Build the autonomous loop with a tool-bound chat model, conditional edges, and LangGraph `ToolNode`; do not encode an intent-to-tool dispatch table.
- Derive loop budgets and duplicate-call detection from messages since the latest human turn, with the graph recursion limit as a final safety backstop.
- Project public responses from graph messages and typed artifacts through the agent module so HTTP routes remain transport-only.
- Keep the model-facing tool result concise and the full Pydantic artifact outside the prompt; state and artifacts must have one authoritative home.

- [ ] The agent state contains only messages, intent, category, claim, and decision, with LangChain message reduction and typed domain models.
- [ ] `ExpenseClaim` supports the documented incremental fields for category/subtype, submitted and original amounts/currency, dates, headcount, distances/direction/days, transport, excluded amount, receipt/approval, destination/business purpose, item identity, and used annual benefit budget.
- [ ] The LangChain model factory supplies ChatOllama in normal operation and the scripted test model in dummy mode through the same chat-model interface.
- [ ] Production uses `qwen2.5:7b-instruct-q4_K_M`; structured classification, extraction, and tool selection run at temperature 0, while final response generation runs at temperature 0.2.
- [ ] Model selection and prompts permit best-effort Hungarian interaction over the English corpus while the canonical prompts, rule identifiers, enum values, and official evaluation remain English.
- [ ] Intent classification and information extraction use structured output with typed validation, one repair attempt, and an observable degraded fallback.
- [ ] The compiled main graph contains the eight designed nodes and conditional routes for complete, incomplete, unsupported, tool-calling, and response-generating paths.
- [ ] The four prompt contracts are classification, information extraction, agent step, and final response generation, all represented as composable LangChain runnables.
- [ ] The agent step uses `bind_tools()` and autonomously selects the policy-search tool instead of following a hard-coded intent-to-tool lookup.
- [ ] Tool execution uses LangGraph `ToolNode`, returns compact `ToolMessage` content, and retains the typed artifact for downstream response construction and evaluation.
- [ ] A general policy question completes through classification, extraction, request checking, tool selection, retrieval, and grounded response generation.
- [ ] The blocking chat endpoint returns a typed response with answer, UTC completion time, duration, deduplicated sources, and stable public step labels.
- [ ] A policy-dependent answer without a supporting tool artifact is refused or clearly marked as lacking evidence.
- [ ] Unsupported requests never call policy, calculation, or rule tools and return the fictional-policy and no-legal-advice disclaimer.
- [ ] Loop budgets, invalid tool arguments, and repeated identical calls terminate deterministically and never expose chain-of-thought.
- [ ] The loop permits at most four tool-calling agent steps and uses a graph recursion limit of 20 as a hard backstop.
- [ ] The same tool may return invalid arguments at most twice before being disabled for the turn, and an identical repeated call reuses the prior artifact while recording a warning.
- [ ] Exhausting the tool budget routes to final response generation with available evidence and explicitly states when that evidence is incomplete.
- [ ] Ollama calls retry with bounded backoff twice and then return a clear failure rather than a fabricated answer.
- [ ] Scripted integration tests verify the expected graph path, tool sequence, source projection, unsupported path, and recursion backstop.
