# 13 — Dataset-driven load test

## What it does

A reviewer can run `python -m load_test.load` with a dataset name, repetition count and max
concurrency, and get back aggregate latency/throughput/error numbers plus a direct link to each
repetition's Langfuse dataset-run for bottleneck analysis. It is intentionally a standalone CLI
script, not an API endpoint: there is no job queue, progress endpoint or cancellation, and the
invocation blocks in the terminal until every repetition completes.

This used to be a synchronous `POST /admin/load-test` endpoint invoked inside the live FastAPI
worker. It moved to a separate process because that design shared fate with real traffic: a crash or
resource exhaustion triggered by the synthetic load could take down the same worker serving `/chat`,
and the aggregated result existed only in that one request handler's memory until the final HTTP
response — a mid-run crash lost the whole result, salvageable only from whatever per-item traces had
already reached Langfuse. As a standalone script, `main()` builds its own `ApplicationDependencies`
(the same container the FastAPI lifespan builds) and its own `AgentService` instance, so it stresses
the same shared Ollama/Redis backends real traffic uses without running inside the same OS process —
a crash here cannot take `/chat` down with it.

## How it works

### `LoadTestResult` and `LoadTestValidationError` (`load_test/load.py`)

```python
class LoadTestResult(BaseModel):
    load_run_id: str
    dataset_name: str
    query_count: int
    max_concurrency: int
    total_duration_ms: int
    throughput_queries_per_minute: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    error_count: int
    dataset_run_urls: list[str]
```

The default 20-item functional dataset × 3 repetitions resolves to 60 measured turns, the documented
default. `max_concurrency` is CLI-argument-constrained to 1–4; the *resolved total*
(`len(dataset.items) * repetitions`, only known after fetching the dataset) is checked against 50–200
inside `LoadTestRunner.run()`, raising `LoadTestValidationError` when out of bounds — there is no
separate request schema to validate this ahead of time, since it depends on live dataset size.

### `LoadTestRunner` (`load_test/load.py`)

For each of `repetitions` (a plain sequential loop — repetitions themselves don't overlap, only items
*within* one repetition do), calls `dataset.run_experiment(name=f"{load_run_id}-rep{n}", task=...,
max_concurrency=..., metadata={"load_run_id": load_run_id})`. The Langfuse SDK itself supplies the
concurrency gate, per-item tracing and dataset-run linking — `LoadTestRunner` supplies only the task
function and the aggregate math, per the design note not to build a second load generator.

The per-item task is `async def task(*, item, **_): ...` — it must be async (not the sync
`AgentService.respond()` directly) because `dataset.run_experiment()` runs on its own event loop
(`langfuse._client.utils.run_async_safely` detects whether one is already running and spins up its
own thread+loop if so); a synchronous task there would serialize every "concurrent" item on that one
loop and silently defeat `max_concurrency`. The task bridges to a thread with `asyncio.to_thread(...)`
— this is what actually lets N graph invocations run concurrently against Ollama. Each task builds a
fresh `thread_id` from `{load_run_id}-rep{n}-{item.id}` (unique per item **and** repetition, so no two
measured turns share LangGraph conversation state) and times the complete `agent_service.respond()`
call with `time.monotonic()`; a raised exception is caught and returned as `{"error": ...}` rather
than propagating, so one failed item doesn't drop out of the aggregate silently.

`LoadTestRunner` scans every repetition's `item_results`, splitting `output` into a latency sample
(`"elapsed_ms"`) or an error count (`"error"`), then computes `total_duration_ms` (wall-clock across
all repetitions), throughput (`query_count / minutes`), mean/median (via `statistics`) and p95 (a
plain sorted-list index — no numpy dependency) — and returns the one `LoadTestResult`, plus the list
of `dataset_run_url`s the SDK returned per repetition.

### CLI entry point (`load_test/load.py::main`)

Parses `--dataset-name`/`--repetitions`/`--max-concurrency`, builds an `Observability` first and
exits with a clear message if Langfuse isn't enabled and configured — before doing anything
expensive. Only then does it build the full `ApplicationDependencies` (Redis connection, ingestion,
checkpointer, agent graph — the same construction the FastAPI lifespan performs) via
`asyncio.run(...)`, wraps the resulting `AgentService` in a `LoadTestRunner`, and runs it. The
`LoadTestResult` is written as JSON to `evaluation_results/load-<timestamp>.json` — the same
shared results directory `llm_eval/run_eval.py` writes its functional-evaluation reports to — and
also printed to the terminal along with the Langfuse run links.

## How to use

```bash
# Requires LANGFUSE_ENABLED=true and real credentials, and a reachable Redis/Ollama — this is a real
# network call and real concurrent load against the configured Ollama model.
uv run python -m load_test.load                                        # defaults: 3 reps, concurrency 4
uv run python -m load_test.load --dataset-name test-dataset --repetitions 5 --max-concurrency 2
```

Expect concurrency beyond ~2–4 to mostly grow queue time rather than throughput for a local 7B model
served by Ollama, since Ollama serialises generation internally — the linked per-repetition Langfuse
traces are where to confirm this against actual per-node/generation timings, not the aggregate alone.

## Key files

| File | Responsibility |
| --- | --- |
| `load_test/load.py` | `LoadTestRunner`, `LoadTestResult`, `LoadTestValidationError`, CLI `main()` |
| `src/app/dependencies.py` | `ApplicationDependencies.build()` — reused directly, not duplicated |
| `load_test/test_load.py` | `LoadTestRunner` unit tests against a fake Langfuse dataset |
