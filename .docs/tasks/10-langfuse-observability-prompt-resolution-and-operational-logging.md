# 10 — Langfuse observability, prompt resolution, and operational logging

**What to build:** Developers can inspect each agent turn, graph node, tool choice, retrieval, and model generation in Langfuse while operators receive bounded, privacy-conscious application logs. Prompt versions are remotely manageable without making Langfuse availability a requirement for normal chat.

**Blocked by:** 04 — First policy-question agent journey.

**Status:** ready-for-agent

**Design references:** Technical design §9 prompt operations, §11 observability/configuration, §13 score integration, and Langfuse/logging failures in §15.

**Technical notes:**

- Attach the Langfuse callback at graph invocation so nested LangChain runnables, retrievers, tools, and model calls inherit one trace hierarchy without trace calls inside graph nodes.
- Resolve prompts through one prompt module that returns validated LangChain `ChatPromptTemplate` objects from either a production-labelled remote version or an embedded fallback.
- Separate diagnostic payloads from operational logs: Langfuse may hold explicitly configured trace content, while application-owned JSON logs contain metadata only.
- Make retention age-based and UTC-calendar-aware, testing startup cleanup and rollover cleanup independently.
- Use Langfuse Cloud as an external dependency for official evaluation/load runs; do not add a self-hosted observability container to the constrained local runtime.

- [ ] One Langfuse trace represents each turn and carries request, thread/session, intent, category, decision, degradation, and model metadata.
- [x] Graph nodes, tool executions, retriever activity, and LLM generations appear as linked observations with useful durations and error status.
- [x] Generation observations record model, token usage, and latency so performance bottlenecks can be attributed to individual calls (prompt identity/version linkage is tracked separately below).
- [ ] Prompt name, resolved version, and prompt source are linked to each generation without copying observability data into graph state.
- [x] Every prompt has a valid embedded development fallback represented as a LangChain chat prompt.
- [x] When enabled, prompt resolution requests the production-labelled Langfuse version and falls back on missing, invalid, or unavailable remote content.
- [x] Remote and embedded prompts are validated for the same required variables, structured-output expectations, policy-number guardrail, and citation markers.
- [x] Disabling Langfuse leaves normal chat operational and uses embedded prompts without repeated network attempts.
- [x] Official functional and load runs refuse to claim complete observability when Langfuse tracing is disabled.
- [x] Prompts, answers, retrieved text, tool artifacts, credentials, and sensitive claim fields never enter application-owned logs.
- [x] Logs are written to stdout and separate service files, rotate at UTC midnight, and remove archives outside the seven-calendar-day retention window at startup and rollover.
- [x] A stopped service may leave expired archives, but startup cleanup removes them before new requests are served.
- [x] Container stdout uses bounded delivery settings so disabling file rotation cannot create unbounded local logs.
- [ ] Tests with mocked Langfuse and controlled time verify tracing metadata, fallback behaviour, prompt validation, rollover, retention, and payload exclusion.
- [x] Langfuse prompt failure emits one warning event without logging prompt text, then uses the validated embedded version.
