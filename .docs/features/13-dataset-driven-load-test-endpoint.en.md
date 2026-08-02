# 13 — Dataset-driven load-test endpoint

## What it does

A reviewer can trigger `POST /admin/load-test` with a dataset name, repetition count and max
concurrency, and get back aggregate latency/throughput/error numbers plus a direct link to each
repetition's Langfuse dataset-run for bottleneck analysis. The endpoint is intentionally synchronous:
there is no job queue, progress endpoint or cancellation, so callers must allow a long request timeout.

## How it works

### Request/response (`app/api/schemas.py`)

```python
class LoadTestRequest(BaseModel):
    dataset_name: str = "rag-assistant-functional"
    repetitions: int = 3            # Field(ge=1)
    max_concurrency: int = 4        # Field(ge=1, le=4)

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
default. `max_concurrency` is schema-constrained to 1–4; the *resolved total* (`len(dataset.items) *
repetitions`, only known after fetching the dataset) is checked against 50–200 inside
`LoadTestRunner`, not in the request schema, since it depends on live dataset size.

### `POST /admin/load-test` (`app/api/routes/evaluation.py`)

Rejects with `503` when `Observability.enabled` is `False` (Langfuse disabled or misconfigured) —
normal chat is unaffected, since this check is local to this one route. Otherwise runs
`LoadTestRunner.run()` off the event loop via `run_in_threadpool` (same bridging pattern as
`/chat`/`/admin/eval`) and translates a `LoadTestValidationError` (bad `max_concurrency` or an
out-of-bounds resolved total) into `422`.

### `LoadTestRunner` (`app/evaluation/load.py`)

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
(plain stdlib, not Starlette's `run_in_threadpool`, since that assumes an anyio-initialized loop which
the SDK's own thread doesn't guarantee) — this is what actually lets N graph invocations run
concurrently against Ollama. Each task builds a fresh `thread_id` from
`{load_run_id}-rep{n}-{item.id}` (unique per item **and** repetition, so no two measured turns share
LangGraph conversation state) and times the complete `agent_service.respond()` call with
`time.monotonic()`; a raised exception is caught and returned as `{"error": ...}` rather than
propagating, so one failed item doesn't drop out of the aggregate silently.

`LoadTestRunner` scans every repetition's `item_results`, splitting `output` into a latency sample
(`"elapsed_ms"`) or an error count (`"error"`), then computes `total_duration_ms` (wall-clock across
all repetitions), throughput (`query_count / minutes`), mean/median (via `statistics`) and p95 (a
plain sorted-list index — no numpy dependency) — and returns the one `LoadTestResult`, plus the list
of `dataset_run_url`s the SDK returned per repetition.

### Dependency wiring (`app/dependencies.py`)

`ApplicationDependencies` now also holds `observability: Observability` (previously only reachable
indirectly through `AgentService`), with a new `get_observability` FastAPI provider — needed so the
route can check `.enabled` and `LoadTestRunner` can call `.client.get_dataset(...)` without
constructing a second Langfuse client.

## How to use

```bash
# Requires LANGFUSE_ENABLED=true and real credentials — this is a real network call and real
# concurrent load against the configured Ollama model.
curl -X POST http://127.0.0.1:8000/admin/load-test \
  -H 'Content-Type: application/json' \
  -d '{"dataset_name": "rag-assistant-functional", "repetitions": 3, "max_concurrency": 4}'
```

Expect concurrency beyond ~2–4 to mostly grow queue time rather than throughput for a local 7B model
served by Ollama, since Ollama serialises generation internally — the linked per-repetition Langfuse
traces are where to confirm this against actual per-node/generation timings, not the aggregate alone.

## Key files

| File | Responsibility |
| --- | --- |
| `app/api/schemas.py` | `LoadTestRequest`, `LoadTestResult` |
| `app/api/routes/evaluation.py` | `POST /admin/load-test` — 503/422 handling, threadpool bridge |
| `app/evaluation/load.py` | `LoadTestRunner`, `LoadTestValidationError` |
| `app/dependencies.py` | `ApplicationDependencies.observability`, `get_observability` |
| `tests/test_load_test_runner.py` | `LoadTestRunner` unit tests against a fake Langfuse dataset |
| `tests/test_api.py` | `/admin/load-test` 503-when-disabled HTTP test |
| `tests/test_api_contracts.py` | `LoadTestResult` contract snapshot, distinct-OpenAPI-schema check |
