# 2026-08-02 14:00 — Self-describing chunks and a dedicated readiness service

## What changed

- `MarkdownChunker.chunk()` now prepends the section heading to every resulting chunk's
  `page_content`, not just `metadata["section"]`. Previously only the first split of a section
  carried its heading (via `strip_headers` behavior on the header splitter); every other chunk —
  including every piece of a long section split into multiple chunks by the character splitter —
  had no textual indication of what it was about.
- Updated `tests/test_ingest.py` assertions (`test_table_is_kept_whole_even_when_it_exceeds_chunk_size`,
  `test_long_prose_section_is_split_into_multiple_overlapping_chunks`) to match the heading-prefixed
  `page_content`, and re-ran ingestion against the dev Redis index.
- Extracted the `/ready` endpoint's business logic (Redis ping + vector-dimension check, LLM backend
  check) out of `app/api/routes/health.py` into a new `ReadinessChecker` class in
  `app/integrations/readiness.py`. The route now only wires dependencies and delegates.
- Moved the corresponding unit tests from `tests/test_health_readiness.py` (previously importing
  `app.api.routes.health._check_redis` directly) to call `ReadinessChecker` instead.

## Why

- Embeddings and retrieval both operate on raw `page_content`. Without the section heading in the
  text itself, the embedding model and the retriever had no way to relate a chunk's content to the
  policy section it belongs to — only the final LLM-facing context block (built separately in
  `RagNodes._build_context`) carried that label.
- Route handlers should stay thin; readiness-check logic is business logic and belongs on a
  dedicated class per the project's architecture conventions, not as module-level functions in a
  FastAPI route file.
