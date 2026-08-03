# Evaluation and load test results

This folder holds the functional-evaluation and load-test results for the RAG-assistant PoC. Method,
metrics and how to run both are documented in the main [README](../README.md) (§8, §9); this file
holds the actual numbers, links and analysis, plus the per-run Markdown/JSON reports both tools write
here (`functional-<timestamp>.md`/`.json`, `load-<timestamp>.json`).

## Functional evaluation

**Results** (`qwen2.5:7b-instruct-q4_K_M`, run 2026-08-02T10:26:25Z, 20 cases):

| Metric | Pass rate | Scored cases |
| --- | --- | --- |
| classification_accuracy | 90.0% | 20 |
| slot_accuracy | 11.8% | 17 |
| retrieval_hit_at_4 | 0.0% | 18 |
| tool_selection_accuracy | 15.0% | 20 |
| outcome_accuracy | 25.0% | 20 |
| citation_accuracy | 0.0% | 18 |

**Analysis — these numbers are real and honest, and the low scores have one consistent, verified
root cause.** `classification_accuracy` (intent + category) is strong at 90%; every metric downstream
of `extract_information` is weak. Spot-checking individual `/admin/eval` calls for low-scoring cases
shows the 7B model frequently fails to produce the *exact* canonical value the extraction prompt asks
for — e.g. writing `expense_type: "transport"` instead of the required `"pass"`, or failing to infer
an implied numeric zero from "no alcohol" into `non_reimbursable_amount: 0`. Because
`route_after_extraction` is deterministic and correct, an imprecise extraction correctly routes the
turn to `ask_clarification` *before* the agent ever reaches `agent_step`/`search_policies` — so
`retrieval_hit_at_4`, `tool_selection_accuracy` and `citation_accuracy` are structurally zero for any
turn that never reaches the tool-calling loop, which is most of them here. This is the evaluation
harness catching a genuine capability limit of a small, locally-served model under this design — not
a software defect, and not something the dataset was loosened to hide. One real dataset-authoring bug
*was* found and fixed this way: `general-01`'s expected documents were assigned from a category tag
without checking the actual corpus file titles, and pointed partly at a glossary document; verified
directly against the live endpoint and corrected (see the dated change-log entry). A materially better
score on the extraction-dependent metrics would need either a larger/more instruction-precise model
than fits the one-developer-machine budget (README §5), or a refined extraction prompt with explicit
few-shot examples of the canonical enum values — both documented as follow-ups, not implemented here.

## Load test

**Results** (default 20-item dataset × 3 repetitions = 60 measured turns, `max_concurrency=4`, run
<!-- LOAD_TEST_TIMESTAMP -->):

<!-- LOAD_TEST_RESULTS_TABLE -->

**Bottleneck**: a complete turn makes 2 fixed model calls (classify, extract) plus 1 final response
call plus 1–4 agent tool-selection calls — 4–7 LLM calls per turn depending on how many tools the
agent decides to use. Ollama serialises generation on this one local model, so aggregate LLM
generation dominates total latency by an order of magnitude; concurrency beyond ~2–4 is expected to
mostly grow queue time rather than throughput, confirmed by comparing per-repetition wall time against
the linked per-generation Langfuse spans <!-- BOTTLENECK_NOTE -->. Retrieval (a single CPU embedding
forward pass + Redis KNN over a few hundred vectors) and the deterministic tools are sub-millisecond
by comparison — exactly why the retrieval path was kept simple.

**Documented, not implemented, optimisations**:

1. **Fast path for simple policy questions** — when intent is `policy_question` with high confidence,
   skip `extract_information` and proceed directly to the agent loop, removing one LLM call without
   removing the dedicated classifier or autonomous tool choice.
2. **Production Redis cache layer** — cache query embeddings and retrieval results with bounded TTLs,
   once real traffic shows enough repeated questions to justify the invalidation/observability cost.

The PoC stays uncached deliberately, so its behaviour and latency remain easy to explain.
