# 09 — Streaming chat experience and thread controls

**What to build:** A user can watch the assistant progress through understandable public steps, see sources as they are found, receive streamed answer text, and keep or reset a checkpointed conversation from the Streamlit interface. Internal prompts, tool arguments, state, and reasoning remain hidden.

**Blocked by:** 07 — Commuting and mileage clarification journey.

**Status:** ready-for-agent

**Design references:** Technical design §10.1–§10.2, request/log correlation in §11, and UI/API failure modes in §15.

**Technical notes:**

- Consume LangGraph stream events in the agent module and translate them into a small public SSE vocabulary; do not forward framework event objects directly.
- Filter message chunks by node metadata so only final-answer generation becomes user-visible tokens.
- Maintain per-request deduplication sets for public step identifiers and citation identities, then build the final response from the same accumulated projection.
- Store only presentation data in Streamlit session state. Conversation truth and resumability stay in LangGraph checkpoints addressed by the thread identifier.
- Keep the blocking and streaming endpoints behaviourally equivalent by routing both through the same agent module and response projection.
- Use timezone-aware UTC in the transport contract and localise timestamps only in the UI presentation layer.

- [x] The streaming chat endpoint emits only the documented `step`, `source`, `token`, and final `result` event types.
- [x] Step events use an allow-listed presentation mapping and are deduplicated even when graph nodes or tool retries emit repeated updates.
- [x] Source events contain only citation-safe document metadata and are deduplicated by their stable identity.
- [x] Token events include only chunks from final response generation; classifier, extractor, and tool JSON never appears in the chat stream.
- [x] The final result contains the same complete response that the blocking chat endpoint would return, including accumulated steps and sources.
- [x] Public responses contain only thread identity, answer, generated time, response duration, sources, and stable steps; claims, findings, calculations, retrieval scores, prompts, and traces remain internal.
- [x] Deterministic clarification and out-of-scope messages complete correctly even when they produce no token stream.
- [x] The Streamlit chat displays a live status area during execution and moves completed steps and sources into a collapsed expander after the answer.
- [x] Public step labels cover request understanding, information extraction, policy search, rule checking, and answer preparation without revealing model reasoning.
- [x] Assistant messages show locally formatted completion time and response duration, and clarification messages receive a distinct visual treatment.
- [x] Connection failures display the stable backend error detail while preserving the local thread and visible conversation for retry.
- [x] The sidebar exposes thread reset and read-only index statistics without model or retrieval tuning controls.
- [ ] Request identifiers propagate through streaming and blocking requests, and CORS allows only the configured UI origin.
- [ ] Contract tests verify event order, token filtering, deduplication, final-result parity, reset behaviour, and client recovery from an interrupted stream.
- [x] API unavailability preserves the server-owned conversation semantics: the UI keeps its visible thread and retries with the same thread identifier rather than creating divergent local state.
