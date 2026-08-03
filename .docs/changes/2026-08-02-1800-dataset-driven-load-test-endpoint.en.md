# 2026-08-02 18:00 — Dataset-driven load-test endpoint (task 13)

## What changed

- Added `app/evaluation/` package: `LoadTestRunner`, `LoadTestValidationError`
  (`app/evaluation/load.py`). Replays a named Langfuse dataset `repetitions` times through
  `AgentService.respond()` (the same module `/chat` uses, not a recursive HTTP call), bounded by
  `max_concurrency` via the Langfuse SDK's own `dataset.run_experiment(...)`.
- Added `POST /admin/load-test` in `app/api/routes/evaluation.py`, with new `LoadTestRequest`/
  `LoadTestResult` schemas in `app/api/schemas.py`. Rejects with `503` when Langfuse is disabled and
  `422` when the resolved run (dataset size × repetitions, or `max_concurrency`) falls outside the
  documented 50–200 / 1–4 bounds.
- Added `observability: Observability` to `ApplicationDependencies` and a `get_observability`
  provider, so routes can check Langfuse availability without going through `AgentService`.
- Added `tests/test_load_test_runner.py` (fake Langfuse dataset: repetition/concurrency forwarding,
  fresh per-item-and-repetition thread ids, shared `load_run_id` metadata, percentile/throughput math,
  partial-failure isolation, invalid-totals rejection) plus HTTP and OpenAPI-contract coverage in
  `tests/test_api.py` / `tests/test_api_contracts.py`.

## Why

- Design explicitly rules out a second, application-owned load generator: the Langfuse experiment
  runner already supplies bounded concurrency, tracing and dataset-run linking, so `LoadTestRunner`'s
  job is only the task function and the aggregate math.
- The per-item task must be `async def` bridging to a thread with `asyncio.to_thread`, not a bare sync
  call: `dataset.run_experiment()` runs on its own event loop (spun up by the SDK's
  `run_async_safely` when called from inside FastAPI's already-running loop), and a synchronous task
  there would serialize every "concurrent" item, silently defeating `max_concurrency`.
