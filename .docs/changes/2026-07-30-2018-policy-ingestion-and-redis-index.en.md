# 2026-07-30 20:18 UTC — Policy ingestion and Redis index visibility (task 2)

## What changed

- Added `rules.yaml`: the deterministic rule catalogue, hand-authored from the actual English
  `.docx` policy corpus (limits, rates, approval tiers, deadlines all verified against source text —
  several numbers differ from the technical design's illustrative example, e.g. the commuting
  personal-vehicle minimum distance is 10 km not 5, and the monthly cap is 40,000 HUF not 60,000).
- Added `app/rules/model.py` (`RuleCatalogue`, `DocumentMeta`, `SectionMeta`, `RuleDefinition`,
  `ApprovalTier`, `SubmissionRules`, `CategoryRules`) and `app/rules/loader.py`
  (`load_rule_catalogue`/`get_rule_catalogue`), validating document/section/rule cross-references at
  load time.
- Added `app/rag/ingest.py`: `DocxToMarkdownConverter` (python-docx → Markdown, preserving
  headings/lists/tables), `DocxMarkdownLoader` (LangChain `BaseLoader`), `MarkdownChunker`
  (header-aware splitting, table-atomic, short-section-merge-forward, 800/120 size guard),
  `RuleMetadataResolver` (attaches `section_id`/`rule_ids`/`categories`, validates that every
  rules.yaml anchor resolves to a real heading), `CorpusManifestBuilder`, and `PolicyCorpusIngestor`
  (orchestrates load → chunk → validate → embed → upsert, skipping embed/upsert when the manifest
  is unchanged).
- Added `app/rag/index_schema.py` (Redis index name, key prefix, vector/distance/algorithm settings,
  metadata schema) and `app/rag/store.py` (`E5Embeddings` — adds the `query:`/`passage:` prefixes
  `intfloat/multilingual-e5-small` needs — and the `RedisVectorStore` factory).
- Added `app/integrations/redis.py` (raw client, manifest read/write, index stats) — raw Redis
  access is limited to these lifecycle concerns; all writes/searches go through the LangChain
  integration.
- Added `app/api/routes/admin.py` (`POST /admin/ingest`, `GET /admin/stats`), returning 503 when
  Redis is unavailable rather than a raw 500.
- `app/main.py`: lifespan now calls `app.rag.ingest.connect_and_ingest()`, which pings Redis, builds
  the vector store and index only if reachable, and degrades to `redis_client=None` otherwise —
  the dummy-backend/no-Redis local dev story from task 1 still works unchanged.
- `app/api/routes/health.py`: `/ready`'s `redis` check now pings the real client instead of the
  task-1 placeholder; overall `ready` is `false` when Redis is unreachable.
- `app/ui.py`: sidebar now calls `GET /admin/stats` and renders indexed-chunk count and
  per-category counts, with no Redis-specific logic in the UI.
- `app/core/logging.py`: quiets noisy third-party loggers (`httpcore`, `httpx2`, `sentence_transformers`,
  `transformers`, `redisvl`, `huggingface_hub`, `filelock`, `urllib3`, `asyncio`) to `WARNING`
  regardless of the configured `LOG_LEVEL` — these were flooding the JSON log output once the
  embedding model and Redis client were wired in.
- Added `pyproject.toml` `[[tool.uv.index]]`/`[tool.uv.sources]` pinning `torch` to the CPU-only
  wheel index (`download.pytorch.org/whl/cpu`), avoiding ~1.5 GB of unused CUDA packages; all
  dependencies are now pinned to exact versions (`==`) rather than lower bounds.
- Tests: `tests/test_rules.py`, `tests/test_ingest.py`, `tests/test_run_ingest.py` (manifest
  decision branches via mocks), `tests/test_admin.py`, `tests/test_health_readiness.py`, and
  `tests/test_redis_integration.py` (real Redis 8, skipped automatically if unreachable) —
  covers full ingest, idempotent re-run, dimension-mismatch rebuild, category-filtered similarity
  search and index stats against a live index. 47 tests, 97% coverage.

## Why

Executing task 2 of the ordered development plan: make the fictional policy corpus searchable in
Redis 8 with stable citation identities, and make that index observable through the application
rather than only through raw Redis inspection.

Two things were discovered and corrected during implementation, beyond the plan's own scope:

- **Rebuilding via the raw Redis client's `FT.DROPINDEX` left the vector store unable to write a
  searchable index again.** `langchain_redis.RedisVectorStore` holds its own `redisvl.SearchIndex`
  object, constructed once; dropping the index through a separate raw client doesn't inform that
  object the schema is gone, so its `add_texts` calls kept writing unindexed hashes. The fix is to
  call `vector_store.index.create(overwrite=True, drop=True)` for rebuilds — the vector store's own
  API for "wipe and recreate" — instead of a bare `FT.DROPINDEX` through `app/integrations/redis.py`
  (that function was removed).
- **The design's illustrative `rules.yaml` numbers do not match the actual corpus text.** The design
  document's example (§4.4) was written before the real `.docs/sources/en/*.docx` content existed
  in its final form. `rules.yaml` was authored by reading every source document directly rather than
  copying the design's placeholder figures, so the eventual consistency check (§13.4, a later task)
  has real numbers to verify against.

`app/rag/ingest.py` was also restructured from a flat module of free functions into a small set of
cohesive classes (`DocxToMarkdownConverter`, `MarkdownChunker`, `RuleMetadataResolver`,
`CorpusManifestBuilder`, `PolicyCorpusIngestor`) per the project's object-oriented code-architecture
preference; `connect_and_ingest` remains a module-level function because it is the FastAPI lifespan's
entry point.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`,
`pytest --cov=app --cov-report=term-missing` — all clean (47 passed, 97% coverage, including the
Redis integration tests run against the pinned `redis:8.8.1` Compose service).
Live smoke test: `uvicorn app.main:app` against that container — `/ready` reports `redis: ok`,
`/admin/stats` reports 83 chunks across all seven categories, `/admin/ingest` correctly reports
`reused` on a second call.
