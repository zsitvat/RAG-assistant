# 2026-08-02 17:00 — Functional Langfuse evaluation experiment (task 12)

## What changed

- Added `eval/` package: `dataset.json` (20 verified functional cases), `model.py`
  (`EvalCase`/`EvalDataset` validation), `metrics.py` (six deterministic Langfuse evaluators),
  `langfuse_sync.py` (idempotent dataset upsert), `report.py` (Markdown/JSON report), `run_eval.py`
  (`python -m eval.run_eval [--node intent]` CLI).
- Added `POST /admin/eval` in a new `app/api/routes/evaluation.py` route module (kept separate from
  `app/api/routes/admin.py`'s corpus-admin endpoints), with new `EvaluationRequest`/`EvaluationResponse`
  schemas in `app/api/schemas.py`.
- Added `AgentService.evaluate()`, projecting graph state into `EvaluationResponse` (intent, category,
  decision, claim, missing slots, ordered tool calls, calculation, findings, retrieved vs. cited
  document ids, degraded) without any answer-prose parsing.
- Added `AgentState["reference_date"]` (pinned per evaluation request, read by `check_rules` in
  preference to the runtime date provider) and `AgentState["degraded"]` (set by `classify_intent` on
  every turn, only ever raised to `True` by later same-turn nodes — never sticky across turns).
- `StructuredOutputRunner.run()` now returns `StructuredResult(value, degraded)` instead of a bare
  value, so callers can tell a real answer from a fallback; previously this was only a warning log line.
- `Observability.trace_config()` now accepts `tags` and arbitrary extra metadata kwargs, used to tag
  evaluation traces distinctly from chat traces and to attach `dataset_item_id`/`experiment_name`.
- Fixed a real bug found while diagnosing the containerized deployment: `chat_model.bind(temperature=0)`
  is invalid for `ChatOllama` (`temperature` is a constructor field, not a per-call kwarg) and was
  silently causing every real-model chat turn to fail after a full retry-with-backoff cycle
  (~60-90s) before falling back to the generic unavailable message. Fixed by constructing
  `ChatOllama(..., temperature=0)` directly in `app/integrations/llm.py::build_chat_model`.
- Also fixed: `MarkdownChunker` now prepends the section heading to every chunk's `page_content`
  (not just the first split of a section), and `app/api/routes/health.py`'s inline readiness-check
  functions were extracted into a dedicated `app/integrations/readiness.py::ReadinessChecker` class.

## Why

- The functional evaluation needs a deterministic `reference_date` per case (§13.1) rather than the
  process's real clock, so a deadline-check case's expected day-count stays correct regardless of
  when the eval actually runs.
- `degraded` gives the eval harness (and any future dashboard) a cheap per-turn confidence signal
  without re-deriving it from log lines.
- Evaluation and load-test are a materially different concern from corpus administration
  (ingest/stats); splitting the route module keeps each file's responsibility narrow.
