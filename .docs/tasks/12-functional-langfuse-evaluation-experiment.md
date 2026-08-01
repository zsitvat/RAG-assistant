# 12 — Functional Langfuse evaluation experiment

**What to build:** A reviewer can synchronise the repository-owned functional dataset to Langfuse, execute the deployed agent over HTTP, inspect every case as a linked experiment trace, and obtain deterministic local and Langfuse scores for classification, extraction, retrieval, tool choice, outcome, and citations.

**Blocked by:** 08 — Benefits, deadlines, and document requirements (which chains through 06 and 07); 10 — Langfuse observability, prompt resolution, and operational logging.

**Status:** ready-for-agent

**Design references:** Technical design §10.1 evaluation contract, §11 Langfuse scores, and §13.1–§13.4.

**Technical notes:**

- Keep the repository JSON dataset authoritative and treat Langfuse as a synchronised execution copy, using stable item identifiers for idempotent updates and trace linkage.
- Expose an internal typed evaluation response that projects state and artifacts already produced by the graph; do not add expected values to prompts or public chat responses.
- Implement metrics as deterministic functions over expected values and typed actual values, then submit their outputs as Langfuse scores.
- Run cases over HTTP to exercise the deployed application seam, while pinning reference date and creating a fresh thread for isolation.
- Generate local human-readable and machine-readable summaries from the same per-item result records used to score Langfuse.
- Keep evaluation data fictional and free of personal/confidential information because traced request content is sent to the configured Langfuse host.

- [ ] The version-controlled dataset contains 20 stable cases spanning general policy, meals, travel, commuting, mileage, equipment, benefits, deadlines, documents, clarification, prohibited expenses, missing receipts, and unsupported requests.
- [ ] Dataset validation rejects duplicate identifiers, malformed expected fields, impossible decisions, and references to unknown categories or documents before a run starts.
- [ ] Synchronisation is idempotent: stable local identifiers map to Langfuse dataset items, questions map to item input, and expected values map to expected output or metadata.
- [ ] The internal evaluation endpoint returns typed projected state and tool artifacts without adding diagnostics to the public chat contract or requiring answer-prose parsing.
- [ ] The evaluation projection includes decision, intent, claim, missing slots, ordered tool calls, calculation, findings, retrieval, and degradation state.
- [ ] Each evaluation request uses a fresh thread, pinned reference date, dataset item identity, experiment name, and trace metadata.
- [ ] The runner executes the running application over HTTP and links each response trace to its Langfuse dataset item and experiment run.
- [ ] Classification accuracy, slot accuracy, retrieval hit@4, tool-selection accuracy, outcome accuracy, and citation accuracy are computed from typed values and pushed as Langfuse scores.
- [ ] Outcome scoring compares calculation `amount_huf` with the expected amount when present, and citation scoring requires an expected retrieved document actually used by the answer.
- [ ] Clarification and unsupported cases are scored according to their explicit decisions without requiring irrelevant calculation or retrieval expectations.
- [ ] One failing item does not abort the remaining experiment and is reported with enough context to open the corresponding trace.
- [ ] A timestamped Markdown summary and machine-readable result contain aggregate percentages, per-case outcomes, failure notes, prompt/model metadata, and the Langfuse experiment link.
- [ ] An intent-only mode reuses the same dataset and experiment conventions while evaluating only the classifier.
- [ ] Tests cover dataset validation, idempotent synchronisation, metric edge cases, endpoint projection, scoring, partial failures, and report generation with mocked Langfuse calls.
- [ ] Contract tests snapshot the public response, internal evaluation response, load-test result, and OpenAPI schemas so evaluation-only changes cannot silently leak into chat.
