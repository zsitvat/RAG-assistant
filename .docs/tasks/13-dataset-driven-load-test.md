# 13 — Dataset-driven load test

**What to build:** A reviewer can trigger one simple standalone script that replays the Langfuse functional dataset under bounded concurrency and returns latency, throughput, error totals, and direct links to the resulting Langfuse experiment runs.

**Blocked by:** 12 — Functional Langfuse evaluation experiment.

**Status:** ready-for-agent

**Design references:** Technical design §10.1, §11 tracing requirements, §14, and the Langfuse load failure in §15.

**Technical notes:**

- Use the official Langfuse dataset experiment runner as the concurrency and trace orchestration mechanism; the application supplies only the task function and aggregate calculation.
- Run as a standalone CLI script (`load_test/load.py`) that builds its own dependency graph and its own agent-service instance, rather than as an endpoint inside the live FastAPI worker — a crash or resource exhaustion during the load run must not take real chat traffic down with it.
- Invoke the same agent module as chat rather than recursively calling the HTTP server from its own process.
- Measure monotonic elapsed time around each complete graph invocation and aggregate successful and failed item results after all repetitions finish.
- Generate a shared load-run identifier and unique thread identifiers before execution so traces can be grouped without sharing conversation state.
- Keep the script intentionally synchronous for the PoC. Its typed result and blocking terminal invocation make the absence of background job semantics explicit.
- Treat Langfuse instrumentation as part of the measured configuration and use linked generation/node spans for bottleneck attribution rather than creating a separate timing subsystem.

- [ ] The synchronous load-test script accepts a dataset name, repetition count, and maximum concurrency with documented defaults.
- [ ] Validation ensures the resolved run contains between 50 and 200 measured turns and allows concurrency only from one through four.
- [ ] The script exits with a clear message when Langfuse is disabled, unauthenticated, or unreachable, before building any application dependencies, while leaving normal chat unaffected since it never touches the live process.
- [ ] The implementation fetches the named Langfuse dataset and uses the official experiment runner rather than maintaining a separate local query bank or HTTP load-generator script.
- [ ] Each dataset item receives a fresh thread identifier, and all repetitions share one generated load-run identifier in trace and experiment metadata.
- [ ] The default 20-item dataset runs three times to produce 60 measured turns.
- [ ] The task measures complete graph invocation time through the same agent module used by chat and returns enough per-item information for aggregate timing.
- [ ] The Langfuse runner enforces the requested concurrency, isolates item failures, creates dataset-run links, and records normal graph/model observations for bottleneck analysis.
- [ ] The response reports query count, concurrency, total duration, queries per minute, mean latency, median latency, p95 latency, error count, and all dataset-run URLs.
- [ ] The returned result identifies the dataset and shared load-run identifier so the aggregate can always be reconciled with its traces.
- [ ] The design explicitly excludes a background queue, progress polling, cancellation, network-latency measurement, and `/chat` transport overhead; the script blocks in the terminal until every repetition completes.
- [ ] Tests with a fake Langfuse dataset verify repetition, concurrency forwarding, fresh thread identities, shared metadata, percentile calculation, partial failures, and invalid totals.
- [ ] A dummy-model integration run completes quickly and produces a valid aggregate response before the full Ollama load run is attempted.
- [ ] The official run demonstrates and documents that concurrency above the local model's useful range primarily increases queue latency rather than throughput.
