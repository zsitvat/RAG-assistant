# 12 — Functional Langfuse evaluation experiment

## What it does

A reviewer can synchronise a repository-owned, 20-case functional dataset to Langfuse, run it as an
experiment against the deployed API, and get both a Langfuse-linked trace/score per case and a local
Markdown + JSON report with aggregate pass rates per metric.

`python -m eval.run_eval` (optionally `--node intent` for a classifier-only pass) is the one command
that ties this together: validate the dataset → idempotently sync it to Langfuse → run it as a
`dataset.run_experiment(...)` → write `.docs/eval/functional-<timestamp>.md` / `.json`.

## How it works

### Dataset (`eval/dataset.json`, `eval/model.py`)

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

### Evaluation endpoint (`POST /admin/eval`, `app/api/routes/evaluation.py`)

Separate from `/admin/ingest`/`/admin/stats` (`app/api/routes/admin.py`) — evaluation is its own
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
for citation accuracy — a different, smaller set than `retrieved_doc_ids`), and `degraded`. No answer
text and no answer-prose parsing anywhere in this path.

### `degraded`: a per-turn, non-sticky flag (`app/agent/state.py`, `app/agent/nodes.py`)

New `AgentState["degraded"]` field. `classify_intent` runs first on every turn and always sets it
(`True`/`False`) from whether `StructuredOutputRunner` fell back to its default; every later node in
the same turn only ever *sets it to `True`* (never explicitly resets it to `False`), so a degradation
anywhere in the turn survives, but it never leaks into the *next* turn on the same thread — because
`classify_intent`'s unconditional first write on the new turn overwrites it. `StructuredOutputRunner.run()`
now returns a `StructuredResult(value, degraded)` `NamedTuple` instead of a bare value, so callers
know whether they got a real answer or the caller-supplied fallback (previously this was only visible
as a warning log line with no signal reaching the caller).

### Metrics (`eval/metrics.py`)

Six `EvaluationMetrics` methods, each a Langfuse `EvaluatorFunction`: `classification_accuracy`,
`slot_accuracy` (share of expected slots matching, tolerant of JSON `int`/`float`/ISO-date-string
drift against the claim), `retrieval_hit_at_4`, `tool_selection_accuracy` (exact ordered-list
equality — the strictest and most sensitive-to-model-behavior metric), `outcome_accuracy` (decision,
plus calculated amount when one is expected), `citation_accuracy`. A metric returns `[]` (Langfuse's
documented "skip scoring this item" shape — a bare `Evaluation(value=None, ...)` fails the SDK's own
score validation) when a case has nothing to score against, e.g. `slot_accuracy` on a case with no
`expected_slots` — the report excludes a metric entirely from a case's scores rather than counting
it as a failure, and it never contributes to that metric's pass-rate denominator.

### Langfuse sync (`eval/langfuse_sync.py`)

`LangfuseDatasetSync.sync()` creates the `rag-assistant-functional` dataset on a `NotFoundError` from
`get_dataset`, then calls `create_dataset_item(id=case.id, ...)` per case — the SDK upserts by `id`,
so re-running sync after editing a case (or adding a new one) is idempotent and never duplicates
items.

### Runner and reports (`eval/run_eval.py`, `eval/report.py`)

`EvaluationRunner.run()` syncs, fetches the Langfuse dataset, and calls
`langfuse_dataset.run_experiment(task=..., evaluators=[...], max_concurrency=4)`. The SDK itself
supplies bounded concurrency, per-item tracing linked to the dataset run, per-evaluator score
creation, and — critically for "one failing item does not abort the run" — isolates a raising task
into a logged failure that's simply excluded from `item_results`; `_http_task`/`_intent_task` both
additionally catch their own errors and return `{"error": ...}` instead of raising, so a failed case
still gets its own traced span and shows up in the report with its trace id, rather than silently
vanishing. `EvaluationReport` aggregates a percentage per metric and writes both a human-readable
Markdown table (with a "Failure notes" section linking each failed case to its trace id) and a
machine-readable JSON file with the same data, under `.docs/eval/`.

`--node intent` swaps the HTTP task for an in-process one that only runs `classify_intent`'s
`StructuredOutputRunner` directly (built once in `EvaluationRunner.__init__`, not per item) and scores
only `classification_accuracy` — useful because intent errors cascade through the whole turn, so
isolating the classifier is the fastest way to check it didn't regress.

## How to use

```bash
# Requires LANGFUSE_ENABLED=true and real credentials in .env — this is a real network call.
uv run python -m eval.run_eval
uv run python -m eval.run_eval --node intent
```

## Key files

| File | Responsibility |
| --- | --- |
| `eval/dataset.json` | 20 version-controlled functional test cases |
| `eval/model.py` | `EvalCase`, `EvalDataset`, `DatasetValidationError` |
| `eval/metrics.py` | `EvaluationMetrics` — the six Langfuse evaluator functions |
| `eval/langfuse_sync.py` | `LangfuseDatasetSync` — idempotent dataset upsert |
| `eval/report.py` | `EvaluationReport` — aggregate + per-case Markdown/JSON report |
| `eval/run_eval.py` | `EvaluationRunner`, CLI entry point |
| `app/api/routes/evaluation.py` | `POST /admin/eval` |
| `app/api/schemas.py` | `EvaluationRequest`, `EvaluationResponse` |
| `app/agent/service.py` | `AgentService.evaluate()` and its projection helpers |
| `app/agent/state.py` | `AgentState["reference_date"]`, `AgentState["degraded"]` |
| `app/agent/structured.py` | `StructuredResult` — `(value, degraded)` from `StructuredOutputRunner.run()` |
| `tests/test_eval_dataset.py`, `tests/test_eval_metrics.py`, `tests/test_eval_langfuse_sync.py`, `tests/test_eval_report.py`, `tests/test_eval_run_eval.py`, `tests/test_agent_service_evaluate.py`, `tests/test_api_contracts.py` | unit tests (mocked Langfuse; no network) |
