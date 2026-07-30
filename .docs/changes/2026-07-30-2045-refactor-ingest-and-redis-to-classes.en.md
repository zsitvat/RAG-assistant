# 2026-07-30 20:45 UTC — Split ingest.py into cohesive classes; RedisIndex wraps the connection

## What changed

- Split `app/rag/ingest.py` (previously one file holding every ingestion concern) into:
  `app/rag/errors.py` (`IngestionError`), `app/rag/docx_converter.py`
  (`DocxToMarkdownConverter`), `app/rag/docx_loader.py` (`DocxMarkdownLoader`),
  `app/rag/chunker.py` (`MarkdownChunker`), `app/rag/rule_metadata.py`
  (`RuleMetadataResolver`), `app/rag/manifest.py` (`CorpusManifestBuilder`). `app/rag/ingest.py`
  now only holds `PolicyCorpusIngestor` (the orchestrator), `connect_and_ingest` (the FastAPI
  lifespan's entry point) and the CLI entry point.
- `app/integrations/redis.py`: replaced `build_redis_client()` + three free functions
  (`read_manifest`, `write_manifest`, `get_index_stats`) with one `RedisIndex` class — constructed
  directly from a `redis_url` (no separate factory function or classmethod), exposing `ping()`,
  `read_manifest()`, `write_manifest()`, `get_index_stats()`, and a `.client` property for the rare
  caller that needs the raw `redis.Redis` connection.
- Updated every call site to thread `RedisIndex` through instead of a raw `redis.Redis`:
  `app/dependencies.py::get_redis_client`, `app/api/routes/health.py::_check_redis`,
  `app/api/routes/admin.py`, `app/rag/ingest.py::connect_and_ingest`/`PolicyCorpusIngestor.run()`.
- Updated tests accordingly (`tests/test_ingest.py`, `tests/test_run_ingest.py`,
  `tests/test_redis_integration.py`); `tests/test_health_readiness.py`/`tests/test_admin.py` needed
  no changes (duck-typed mocks / real HTTP wiring).

## Why

Two rounds of review feedback on the same code:

1. The original `app/rag/ingest.py` (free functions for conversion, chunking, rule metadata,
   manifest and orchestration) didn't match the project's object-oriented code-architecture
   preference — classes were introduced for each concern in response.
2. Once introduced, cramming six-plus classes into one file was itself flagged as too complex to
   navigate — each class now lives in its own small module, `ingest.py` keeping only the top-level
   orchestrator.

Separately, `app/integrations/redis.py`'s client-factory-plus-free-functions shape was asked to
become one class wrapping the connection together with its lifecycle behaviour, so "a few Redis
things" stay in one place instead of being threaded through as a bare `redis.Redis` plus helper
functions. A `@classmethod` factory (`RedisIndex.connect(url)`) was tried first and rejected in
favour of a plain constructor taking the URL directly, per instruction to avoid `classmethod` here.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`,
`pytest --cov=app --cov-report=term-missing` — all clean (47 passed, 97% coverage, including the
Redis-Stack integration tests). Live smoke test against the running `redis-stack-dev` container:
`/ready`, `/admin/stats`, `/admin/ingest` all correct after the restructuring.
