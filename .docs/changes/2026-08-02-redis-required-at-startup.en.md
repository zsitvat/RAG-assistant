# 2026-08-02 — Redis required at startup

## What changed

- Made FastAPI startup fail when Redis or its vector store is unavailable.
- Removed the runtime in-memory checkpointer and empty-retrieval fallbacks.
- Added dependency and lifespan regression tests for fail-fast startup.

## Why

Chat requires both grounded policy retrieval and durable conversation state, so serving the API in
a degraded Redis-free mode would expose behavior the application does not support.
