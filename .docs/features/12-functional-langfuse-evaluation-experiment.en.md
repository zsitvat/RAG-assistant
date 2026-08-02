# 12 — Functional Langfuse evaluation experiment

## What it does

A reviewer can synchronise a repository-owned, 20-case functional dataset to Langfuse, run it as an
experiment against the deployed API, and get both a Langfuse-linked trace/score per case and a local
Markdown + JSON report with aggregate pass rates per metric.

`python -m llm_eval.run_eval` (optionally `--node intent` for a classifier-only pass) is the one command
that ties this together: validate the dataset → idempotently sync it to Langfuse → run it as a
`dataset.run_experiment(...)` → write `evaluation_results/functional-<timestamp>.md` / `.json`.

## How it works

### Dataset (`llm_eval/dataset.json`, `llm_eval/model.py`)

20 hand-authored cases span general policy, all six expense categories, a clarification
(one-way/round-trip ambiguity), a deadline still open and one expired, a missing-receipt case and an
out-of-scope question. Every case's `expected_amount_huf`/`expected_decision` was verified against
the real `ReimbursementCalculator`/`RuleChecker` output for its exact claim fields before being
committed — not hand-computed — so a failing case means the agent diverged from the deterministic
rule engine, not that the dataset itself is wrong.

`EvalCase` (Pydantic) validates one case's shape: `expected_slots` keys must be real `ExpenseClaim`
fields, `expected_tools` must be real tool names, and a `needs_info` decision cannot also expect tool
calls (the graph short-circuits to `ask_clarification` before any tool runs). `EvalDataset.load()`
additionally rejects duplicate ids and references to a category or document id the loaded
`RuleCatalogue` doesn't know about — all of this runs before any network call, so a broken dataset
never reaches Langfuse or the API.

### Evaluation endpoint (`POST /admin/eval`, `src/app/api/routes/evaluation.py`)

Separate from `/admin/ingest`/`/admin/stats` (`src/app/api/routes/admin.py`) — evaluation is its own
route module. `EvaluationRequest` carries `{thread_id, message, reference_date, dataset_item_id?,
experiment_name?}`; `reference_date` is injected into `AgentState["reference_date"]` (new field) so
`check_rules`'s deadline math is deterministic regardless of when the eval actually runs — the tool
now reads `runtime.state.get("reference_date") or reference_date_provider()`, preferring the pinned
value. `dataset_item_id`/`experiment_name`, when present, get folded into the turn's Langfuse trace
metadata (`Observability.trace_config` now accepts `tags` and arbitrary extra metadata kwargs) purely
for cross-referencing; the SDK's own experiment-item span (see below) is what Langfuse actually links
to the dataset run.

`AgentService.evaluate()` mirrors `respond()`/`stream()` but projects graph state into
`EvaluationResponse` instead of `ChatResponse`: `intent`, `category`, `decision`, `claim`,
`missing_slots` (via `RequiredSlotTable`, same table `route_after_extraction` uses),
`tool_calls` (ordered names from the current turn's `AIMessage.tool_calls`), `calculation`,
`findings`, `retrieved_doc_ids` (raw top-k `search_policies` hits, for retrieval hit@4),
`cited_doc_ids` (the deduplicated, context-budgeted citations the answer was actually given to cite,
for citation accuracy — a different, smaller set than `retrieved_doc_ids`), `degraded`, and `answer`
(the final generated text, `state["messages"][-1].content`, same as `ChatResponse.answer`) — the one
field the `answer_quality` metric needs, and the only prose the six deterministic metrics still never
touch.

### `degraded`: a per-turn, non-sticky flag (`src/app/agent/state.py`, `src/app/agent/nodes.py`)

New `AgentState["degraded"]` field. `classify_intent` runs first on every turn and always sets it
(`True`/`False`) from whether `StructuredOutputRunner` fell back to its default; every later node in
the same turn only ever *sets it to `True`* (never explicitly resets it to `False`), so a degradation
anywhere in the turn survives, but it never leaks into the *next* turn on the same thread — because
`classify_intent`'s unconditional first write on the new turn overwrites it. `StructuredOutputRunner.run()`
now returns a `StructuredResult(value, degraded)` `NamedTuple` instead of a bare value, so callers
know whether they got a real answer or the caller-supplied fallback (previously this was only visible
as a warning log line with no signal reaching the caller).

### Metrics (`llm_eval/metrics.py`)

Seven `EvaluationMetrics` methods, each a Langfuse `EvaluatorFunction`: `classification_accuracy`,
`slot_accuracy` (share of expected slots matching, tolerant of JSON `int`/`float`/ISO-date-string
drift against the claim), `retrieval_hit_at_4`, `tool_selection_accuracy` (exact ordered-list
equality — the strictest and most sensitive-to-model-behavior metric), `outcome_accuracy` (decision,
plus calculated amount when one is expected), `citation_accuracy`, and `answer_quality` (below). A
metric returns `[]` (Langfuse's documented "skip scoring this item" shape — a bare
`Evaluation(value=None, ...)` fails the SDK's own score validation) when a case has nothing to score
against, e.g. `slot_accuracy` on a case with no `expected_slots` — the report excludes a metric
entirely from a case's scores rather than counting it as a failure, and it never contributes to that
metric's pass-rate denominator.

### Answer quality judge (`llm_eval/judge.py`, `EvaluationMetrics.answer_quality`)

The six metrics above score structured graph state; none of them ever look at the generated answer's
prose. `answer_quality` is the one metric that does, and the only non-deterministic one: it sends the
question, `output["answer"]` and the case's `expected_answer_summary` (a short, hand-authored
reference of what a correct answer must state — new required `EvalCase` field, present on all 20
cases) to `JUDGE_PROMPT` through a `StructuredOutputRunner`, which returns a typed
`AnswerJudgeVerdict{correct: bool, reasoning: str}`; the verdict's `reasoning` becomes the Langfuse
score's comment, so a failed case's judge rationale is visible without re-reading the trace.

The judge chat model is built from `EVAL_JUDGE_MODEL`, a setting independent of `LLM_MODEL`
(`build_chat_model(settings, settings.eval_judge_model)` — `build_chat_model` now takes an optional
model-name override for exactly this). It defaults to the same tag as `LLM_MODEL` so the metric runs
out of the box on this PoC's single pulled model, but pointing it at a second, genuinely different
model is materially more meaningful: a model grading its own answers risks not catching its own
systematic mistakes. Missing an answer entirely scores `False` without calling the judge; a judge
call failure degrades to `AnswerJudgeVerdict(correct=False, reasoning="judge model unavailable")` via
`StructuredOutputRunner`'s existing retry-then-fallback behaviour, the same pattern `classify_intent`
and `extract_information` already use.

### Langfuse sync (`llm_eval/dataset_sync.py`)

`LangfuseDatasetSync.sync()` creates the `test-dataset` dataset on a `NotFoundError` from
`get_dataset`, then calls `create_dataset_item(id=case.id, ...)` per case — the SDK upserts by `id`,
so re-running sync after editing a case (or adding a new one) is idempotent and never duplicates
items.

### Runner and reports (`llm_eval/run_eval.py`, `llm_eval/report.py`)

`EvaluationRunner.run()` syncs, fetches the Langfuse dataset, and calls
`langfuse_dataset.run_experiment(task=..., evaluators=[...], max_concurrency=4)`. The SDK itself
supplies bounded concurrency, per-item tracing linked to the dataset run, per-evaluator score
creation, and — critically for "one failing item does not abort the run" — isolates a raising task
into a logged failure that's simply excluded from `item_results`; `_http_task`/`_intent_task` both
additionally catch their own errors and return `{"error": ...}` instead of raising, so a failed case
still gets its own traced span and shows up in the report with its trace id, rather than silently
vanishing. `EvaluationReport` aggregates a percentage per metric and writes both a human-readable
Markdown table (with a "Failure notes" section linking each failed case to its trace id) and a
machine-readable JSON file with the same data, under `evaluation_results/`.

`--node intent` swaps the HTTP task for an in-process one that only runs `classify_intent`'s
`StructuredOutputRunner` directly (built once in `EvaluationRunner.__init__`, not per item) and scores
only `classification_accuracy` — useful because intent errors cascade through the whole turn, so
isolating the classifier is the fastest way to check it didn't regress.

## How to use

```bash
# Requires LANGFUSE_ENABLED=true and real credentials in .env — this is a real network call.
uv run python -m llm_eval.run_eval
uv run python -m llm_eval.run_eval --node intent
```

## Key files

| File | Responsibility |
| --- | --- |
| `llm_eval/dataset.json` | 20 version-controlled functional test cases, each with `expected_answer_summary` |
| `llm_eval/model.py` | `EvalCase`, `EvalDataset`, `DatasetValidationError` |
| `llm_eval/metrics.py` | `EvaluationMetrics` — the seven Langfuse evaluator functions |
| `llm_eval/judge.py` | `AnswerJudgeVerdict`, `JUDGE_PROMPT` — the `answer_quality` judge |
| `llm_eval/dataset_sync.py` | `LangfuseDatasetSync` — idempotent dataset upsert |
| `llm_eval/report.py` | `EvaluationReport` — aggregate + per-case Markdown/JSON report |
| `llm_eval/run_eval.py` | `EvaluationRunner`, CLI entry point |
| `src/app/settings.py` | `EVAL_JUDGE_MODEL` — independent judge-model configuration |
| `src/app/integrations/llm.py` | `build_chat_model(settings, model=None)` — optional model-name override |
| `src/app/api/routes/evaluation.py` | `POST /admin/eval` |
| `src/app/api/schemas.py` | `EvaluationRequest`, `EvaluationResponse` (including `answer`) |
| `src/app/agent/service.py` | `AgentService.evaluate()` and its projection helpers |
| `src/app/agent/state.py` | `AgentState["reference_date"]`, `AgentState["degraded"]` |
| `src/app/agent/structured.py` | `StructuredResult` — `(value, degraded)` from `StructuredOutputRunner.run()` |

`llm_eval/` has no unit tests — it is a standalone CLI script validated by running it against a live
Redis/Ollama/Langfuse stack, not app logic exercised by the deployed request path (technical design
§13.4).
