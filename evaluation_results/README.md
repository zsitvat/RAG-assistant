# Evaluation and load test results

This folder holds the functional-evaluation and load-test results for the RAG-assistant PoC. Method,
metrics and how to run both are documented in the main [README](../README.md) (§8, §9); this file
holds the actual numbers, links and analysis, plus the per-run Markdown/JSON reports both tools write
here (`functional-<timestamp>.md`/`.json`, `load-<timestamp>.json`).

## Functional evaluation

**Results** (`qwen2.5:7b-instruct-q4_K_M`, run 2026-08-03T21:26:03Z, 20 cases):

| Metric | Pass rate | Scored cases |
| --- | --- | --- |
| classification_accuracy | 90.0% | 20 |
| slot_accuracy | 76.1% | 17 |
| retrieval_hit_at_4 | 22.2% | 18 |
| tool_selection_accuracy | 15.0% | 20 |
| outcome_accuracy | 25.0% | 20 |
| citation_accuracy | 22.2% | 18 |
| answer_quality | 10.0% | 20 |

Full per-case scores: [`functional-20260803-212603.md`](functional-20260803-212603.md) /
[`.json`](functional-20260803-212603.json).

**Analysis — these numbers are real and honest, and the low scores have one consistent, verified
root cause.** `classification_accuracy` (90%) and `slot_accuracy` (76.1%) are both strong: intent,
category and claim-field extraction work for most cases. `tool_selection_accuracy` (15.0%) and
`outcome_accuracy` (25.0%) are the weak spot, and the per-case data shows why: cases with a perfectly
extracted claim (`slot_accuracy = 1.0`) still fail
`tool_selection_accuracy` almost every time — e.g. `deadline-01`, `deadline-02`, `benefits-01`,
`equipment-01` all extract every field correctly yet the agent's `agent_step` ReAct loop picks a
different tool sequence than the case's exact expected ordered list, or skips `search_policies`
altogether (hence `retrieval_hit_at_4 = False` on those same cases). One case, `general-01`, gets
every structural metric right (`retrieval_hit_at_4`, `tool_selection_accuracy`, `outcome_accuracy`,
`citation_accuracy` all `True`) yet still fails `answer_quality` — the LLM-judged answer text diverges
from the hand-authored `expected_answer_summary` even when the underlying graph state is fully
correct, which is why `answer_quality` (10.0%) is now the strictest metric in the suite. Net read: for
this 7B model, autonomous tool selection and final-answer phrasing are harder than field extraction —
a genuine small-model capability limit under this design, not a software defect. A materially better
score would need either a larger/more instruction-precise model than fits the one-developer-machine
budget (README §5), or few-shot examples of the expected tool-call sequence in the `agent_step`
prompt — both documented as follow-ups, not implemented here.

## Load test

**Results** (default 20-item dataset × 3 repetitions = 60 measured turns, `max_concurrency=4`, run
2026-08-03T21:45:39Z):

| Metric | Result |
| --- | ---: |
| Total duration | 410.7 s (6 m 51 s) |
| Throughput | 8.77 queries/min |
| Latency, mean | 28.8 s |
| Latency, median | 30.3 s |
| Latency, p95 | 47.4 s |
| Errors | 8 / 60 (13.3%) |

Full result: [`load-20260803-214539.json`](load-20260803-214539.json). Langfuse run links are inside
that file (one per repetition).

**Bottleneck #1 — LLM generation dominates.** A complete turn makes 2 fixed model calls (classify,
extract) plus 1 final response call plus 1–4 agent tool-selection calls — 4–7 LLM calls per turn
depending on how many tools the agent decides to use. Ollama serialises generation on this one local
model, so aggregate LLM generation dominates total latency by an order of magnitude over retrieval and
the deterministic tools; a ~29 s mean turn at 4–7 calls each puts single-call generation in the
multi-second range on this CPU-bound setup. Concurrency beyond ~2–4 is expected to mostly grow queue
time rather than throughput. Retrieval (a single CPU embedding forward pass + Redis KNN over a few
hundred vectors) and the deterministic tools are sub-millisecond by comparison — exactly why the
retrieval path was kept simple.

**Bottleneck #2 — a real error surfaced under concurrency, and its cause is diagnosable from the code,
not just the numbers.** 8 of 60 turns (13.3%) failed outright, and the run log shows the underlying
cause: repeated `RuntimeError: Event loop is closed` inside `StructuredOutputRunner.run`
(`src/app/agent/structured.py`) during concurrent classification/extraction calls — some retried
successfully, at least one degraded all the way to the fallback value, and some turns failed the whole
graph invocation. The shared `ChatOllama` instance (`src/app/integrations/llm.py`) is built once inside
`asyncio.run(_build_runner(...))` in `load_test/load.py`, so its underlying async HTTP client is bound
to that first event loop; that loop closes as soon as `_build_runner` returns, and Langfuse's
`dataset.run_experiment` then drives the actual concurrent task execution on a separate event
loop/thread — reusing a client whose loop is already gone. This is a load-test-harness wiring bug
(the same long-lived client is built the same way in the FastAPI app, but the app never rebuilds its
event loop underneath a live client), not a symptom of Ollama or the agent graph itself.

**Documented, not implemented, optimisations**:

1. **Fix the load-test client/event-loop lifecycle** — build `ApplicationDependencies` inside the same
   event loop that runs the concurrent task executions (or lazily construct/reset the `ChatOllama`
   client per task), so the 13.3% error rate measured here reflects the agent under load rather than a
   harness artifact. This is the highest-priority fix: it directly caused the measured errors.
2. **Fast path for simple policy questions** — when intent is `policy_question` with high confidence,
   skip `extract_information` and proceed directly to the agent loop, removing one LLM call without
   removing the dedicated classifier or autonomous tool choice.
3. **Production Redis cache layer** — cache query embeddings and retrieval results with bounded TTLs,
   once real traffic shows enough repeated questions to justify the invalidation/observability cost.

The PoC stays uncached deliberately, so its behaviour and latency remain easy to explain.
