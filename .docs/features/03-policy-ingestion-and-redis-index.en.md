# Feature: Policy ingestion and Redis index visibility

Implements task
[`02-policy-ingestion-and-redis-index-visibility.md`](../tasks/02-policy-ingestion-and-redis-index-visibility.md).

## What it does

Converts the fictional English policy corpus (`.docs/sources/en/*.docx`) into a searchable Redis
Stack vector index, attaches stable citation/rule metadata to every chunk, and exposes ingestion and
index state through the application (`/admin/ingest`, `/admin/stats`, the Streamlit sidebar, and
`/ready`) rather than only through raw Redis inspection.

## How it works

### The rule catalogue (`app/rules/`)

`rules.yaml` (repository root) is the deterministic rule catalogue: per-document category and
section-anchor declarations, and per-category rule definitions (limits, rates, approval tiers,
deadlines). Every number in it was read directly from the source `.docx` files, not copied from the
technical design's illustrative example — several values differ (see the deviation note below).
`app/rules/model.py` defines the typed Pydantic models (`RuleCatalogue`, `DocumentMeta`,
`SectionMeta`, `RuleDefinition`, `ApprovalTier`, `SubmissionRules`, `CategoryRules`) and validates,
at load time, that every rule's `doc_ref` resolves to a declared document and section, that document
categories are never empty, and that rule ids are unique. `app/rules/loader.py` exposes
`load_rule_catalogue(path)` and a cached `get_rule_catalogue()`; a missing or malformed catalogue
raises `RuleCatalogueError` before the application starts serving requests.

### Ingestion (`app/rag/ingest.py`)

- **`DocxToMarkdownConverter`** walks a `.docx` file's paragraphs and tables in document order
  (`python-docx`'s `iter_inner_content`) and maps Word styles to Markdown: `Title` → document `#`
  heading (falls back to the file stem), `Heading 1..3` → `#`/`##`/`###`, `List Bullet`/`List Number`
  → `-`/`1.` items, tables → GitHub-style Markdown tables, everything else (images, headers/footers,
  comments, tracked changes) dropped.
- **`DocxMarkdownLoader`** is a LangChain `BaseLoader` yielding one `Document` per source file
  (`doc_id` = the stable `00`–`07` filename prefix, `doc_title`, `source_path`).
- **`MarkdownChunker`** runs `MarkdownHeaderTextSplitter` (h1–h3) then merges sections shorter than
  200 characters into the *following* sibling (a trailing short section — no sibling to merge into —
  keeps its own heading rather than being absorbed into an unrelated earlier section). Within each
  (possibly merged) section, table blocks are kept atomic; only prose runs are size-guarded through
  `RecursiveCharacterTextSplitter` (800 chars, 120 overlap).
- **`RuleMetadataResolver`** attaches `section_id` (resolved by matching the chunk's heading text
  against `rules.yaml`'s declared section anchors for that `doc_id`), `rule_ids` (rules whose
  `doc_ref` points at that section), and `categories` (the whole document's category list) to every
  chunk; it raises `IngestionError` for an unknown `doc_id` and, separately,
  `validate_anchors_resolve()` raises if a declared rules.yaml anchor's heading never actually
  appears in the ingested corpus.
- **`CorpusManifestBuilder`** hashes every source `.docx` plus `rules.yaml` together with the
  chunking parameters, embedding model name/revision and vector dimension.
- **`PolicyCorpusIngestor.run()`** loads+chunks the corpus, compares the freshly computed manifest
  against the one stored in Redis, and only embeds/upserts when they differ — reporting
  `built`/`rebuilt`/`reused`. A **rebuild** calls `vector_store.index.create(overwrite=True,
  drop=True)` (the LangChain integration's own index object) before re-upserting, not a raw
  `FT.DROPINDEX` — see the deviation note.
- `connect_and_ingest(settings, rule_catalogue)` is the one module-level function (the FastAPI
  lifespan's entry point): pings Redis, and if reachable builds the embeddings/vector store and runs
  the ingestor; returns `(None, None)` if Redis is unreachable so the rest of the app can still run
  in dummy/offline mode.
- `python -m app.rag.ingest` is the standalone CLI entry point.

### The vector store (`app/rag/index_schema.py`, `app/rag/store.py`)

`app/rag/index_schema.py` holds the Redis index identity and schema as plain constants:
`INDEX_NAME` (`idx:chunks`), `KEY_PREFIX` (`chunk`), `TOP_K`, `VECTOR_DIMENSION` (384), distance
metric (`COSINE`), algorithm (`HNSW`), vector datatype (`FLOAT32`), and the metadata schema (`doc_id`,
`section_id`, `categories`, `rule_ids` as `TAG` fields — the LangChain integration's default `|`
separator matches the design's tag format with no extra configuration; `section` as `TEXT NOSTEM`).
`app/rag/store.py` holds `E5Embeddings` (a thin `HuggingFaceEmbeddings` subclass adding the
`query:`/`passage:` prefixes `intfloat/multilingual-e5-small` needs) and `build_vector_store()`,
which constructs a `langchain_redis.RedisVectorStore` from that schema. Chunk keys are
`chunk:{doc_id}:{chunk_index}` (`add_texts(..., keys=[f"{doc_id}:{chunk_index}"])`, prefixed
automatically by the integration's `key_prefix`).

### Redis lifecycle (`app/integrations/redis.py`)

Raw Redis access is limited to what the LangChain integration doesn't own: a plain client factory,
`manifest:corpus` read/write (as JSON), and `get_index_stats()` (chunk count + per-category counts,
read via `FT.INFO` + a `categories`-only `FT.SEARCH`). All chunk writes and similarity searches go
through `RedisVectorStore`.

### API and UI

- `POST /admin/ingest` and `GET /admin/stats` (`app/api/routes/admin.py`) return `503` with a clear
  `detail` when Redis is unavailable, instead of a raw 500.
- `GET /ready` (`app/api/routes/health.py`) now pings the real Redis client; overall `ready` is
  `false` when Redis is unreachable (previously a fixed `not_configured` placeholder from task 1).
- The FastAPI lifespan (`app/main.py`) calls `connect_and_ingest()`; if Redis is unreachable it logs a
  warning and leaves `app.state.redis_client`/`vector_store` as `None` rather than failing startup —
  the task-1 "runs without Redis" dummy-backend story still holds.
- The Streamlit sidebar (`app/ui.py`) calls `GET /admin/stats` and renders indexed-chunk count and
  per-category counts; it contains no Redis-specific logic.

## How to use

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:7.4.0-v6 redis-stack-server --appendonly yes
uv run python -m app.rag.ingest          # one-off CLI ingest
# or just start the API — its lifespan ingests automatically:
uv run uvicorn app.main:app --port 8000
curl -X POST http://127.0.0.1:8000/admin/ingest
curl http://127.0.0.1:8000/admin/stats
```

Run the Redis-Stack integration test (skipped automatically if unreachable):

```bash
TEST_REDIS_URL=redis://127.0.0.1:6379/0 uv run pytest tests/test_redis_integration.py -v
```

## Key files

| File | Responsibility |
| --- | --- |
| `rules.yaml` | deterministic rule catalogue, hand-authored from the real corpus |
| `app/rules/model.py` | `RuleCatalogue` typed models + cross-reference validation |
| `app/rules/loader.py` | `load_rule_catalogue()`, cached `get_rule_catalogue()` |
| `app/rag/ingest.py` | conversion, chunking, rule metadata, manifest, `PolicyCorpusIngestor` |
| `app/rag/index_schema.py` | Redis index name, key prefix, vector/schema constants |
| `app/rag/store.py` | `E5Embeddings`, `RedisVectorStore` factory |
| `app/integrations/redis.py` | raw client, manifest read/write, index stats |
| `app/api/routes/admin.py` | `/admin/ingest`, `/admin/stats` |
| `app/api/routes/health.py` | real Redis check in `/ready` |
| `app/ui.py` | sidebar index stats |
| `tests/test_rules.py`, `tests/test_ingest.py`, `tests/test_run_ingest.py`, `tests/test_admin.py`, `tests/test_health_readiness.py` | unit tests (no Redis required) |
| `tests/test_redis_integration.py` | integration tests against a real Redis Stack |

## Deliberate deviations from the technical design

- **Rebuild uses `vector_store.index.create(overwrite=True, drop=True)`, not a raw `FT.DROPINDEX`.**
  `langchain_redis.RedisVectorStore` owns a `redisvl.SearchIndex` constructed once; dropping the
  index through a separate raw Redis client leaves that object unaware the schema is gone, so its
  `add_texts` silently writes unindexed hashes afterward. Using the vector store's own recreate API
  keeps the LangChain integration and Redis in agreement.
- **`rules.yaml` numbers were read from the actual corpus, not copied from the design's example.**
  The technical design's §4.4 sample catalogue was illustrative and predates the final source
  documents. Notable differences: the commuting personal-vehicle minimum distance is 10 km (design
  example said 5), the monthly commuting cap is 40,000 HUF (design example said 60,000), and
  `benefits` is three separate rules (recreational/training/sport, each with its own annual budget
  and reimbursement ratio) rather than one combined 300,000 HUF pool.
- **`app/rag/ingest.py` is a set of cohesive classes, not a flat module of functions** — per the
  project's object-oriented code-architecture preference. `connect_and_ingest` is the only remaining
  module-level function, kept because it is the FastAPI lifespan's required entry point.
