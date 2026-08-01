# 2026-08-01 06:33 UTC — Grounded RAG subgraph

## What changed

- Added `app/rag/retriever.py`: `PolicyRetriever` wraps a `RedisVectorStore` and exposes
  `search(query, category)`, calling `similarity_search_with_score` with the
  `@categories:{<category>|general}` tag filter built per call and converting cosine distance to
  `similarity = 1 - distance`. The category filter is a search-time argument, not construction
  state — one `PolicyRetriever` instance serves every category, so there is no retriever-factory
  indirection.
- Added `app/rag/state.py`: the dedicated `RagState` LangGraph contract (`question`, `category`,
  `result`).
- Added `app/rag/graph.py`: `RagNodes` (`retrieve_documents`, `build_context`, bound to one injected
  `PolicyRetriever`) and
  `build_rag_graph(retriever)`, compiling a two-node `retrieve_documents -> build_context`
  `StateGraph`. `retrieve_documents` retries once without the category filter when the filtered
  search returns nothing. `build_context` sorts results by similarity, dedups by `(doc_id, section)`,
  and emits `[S1]`, `[S2]`, ... numbered blocks (always including the top result even if it alone
  exceeds the budget) up to `CONTEXT_TOKEN_BUDGET` (~1,800 tokens, approximated as 4 chars/token).
- Added `app/rag/tool.py`: `build_search_policies_tool(rag_graph)` returns a
  `response_format="content_and_artifact"` LangChain tool — content is the assembled context (or an
  explicit "no relevant policy information found" string), artifact is the full `RagResult`.
- Added `RetrievedResult`, `Citation`, `RagResult` (with a `confidence` property over
  `results[0].similarity`) to `app/rag/model.py`.
- Added `TOP_K`-adjacent constants to `app/rag/index_schema.py`: `MIN_CONFIDENCE_THRESHOLD` (0.8) and
  `CONTEXT_TOKEN_BUDGET` (1800).
- Added `langgraph==1.2.10` as an explicit direct dependency (previously only a transitive
  dependency of `langchain`).
- Moved `rules.yaml` to `config/rules.yaml`; updated `app/rules/loader.py::DEFAULT_RULES_PATH` and
  `app/rag/ingest.py::RULES_PATH`.
- Tests: `tests/test_retriever.py`, `tests/test_rag_graph.py`, `tests/test_rag_tool.py` (fake
  retrievers, no Redis), plus category-coverage and low-confidence tests appended to
  `tests/test_redis_integration.py`.

## Why

Task 3 (`03-grounded-rag-subgraph.md`). The subgraph is built and compiled independently from the
main agent, taking only a question and optional category and returning one typed `RagResult`, so it
stays reusable and network/Redis-free at construction time — only the injected retriever factory
touches Redis, and only when a node actually runs.

`similarity_search_with_relevance_scores()` raises `NotImplementedError` on this vector store (no
`relevance_score_fn` configured), so `PolicyRetriever` computes similarity itself from
`similarity_search_with_score()`'s raw cosine distance. `MIN_CONFIDENCE_THRESHOLD = 0.8` was
calibrated empirically against the live corpus: on-topic questions across all seven categories score
0.875-0.937 top-1 similarity, off-topic questions score 0.708-0.786 — a clean gap.

`rules.yaml` moved into `config/` to keep repository-root config files grouped rather than scattered
loose at the top level.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`,
`pytest --cov=app --cov-report=term-missing` — all clean (70 passed, 97% coverage, including the
Redis 8 integration tests covering all seven rule categories). Live smoke test against the
running Compose Redis service: the compiled graph returns grounded, cited, correctly-ranked
evidence for representative questions in every category, and flags an off-topic question below the
confidence threshold.
