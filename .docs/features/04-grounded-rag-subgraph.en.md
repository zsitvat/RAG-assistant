# Feature: Grounded RAG subgraph

Implements task [`03-grounded-rag-subgraph.md`](../tasks/03-grounded-rag-subgraph.md).

## What it does

Turns a policy question (plus an optional category) into ranked, source-backed evidence: a compiled
LangGraph `StateGraph` retrieves the relevant chunks from the Redis vector store, ranks them by
cosine similarity, and assembles a bounded, numbered, citable context block. It is compiled
independently of the main agent and exposed to it as a `search_policies` LangChain tool.

## How it works

### State (`app/rag/state.py`) and nodes (`app/rag/graph.py`)

`RagState` is a `TypedDict` with three keys: `question` and `category` in, `result: RagResult` out.
It lives separately from graph construction and node behaviour. `RagNodes` holds the injected
`Retriever` and provides the two nodes:

- **`retrieve_documents`** calls `retriever.search(question, category)`. If the category filter
  returns nothing, it retries once with `retriever.search(question, None)` (unfiltered) and the
  returned `RagResult.category` reflects the fallback. Results are turned into `RetrievedResult`
  objects and sorted by similarity, descending.
- **`build_context`** walks the sorted results, skipping any whose `(doc_id, section)` was already
  emitted (citation deduplication — a section split across multiple chunks only cites once), and
  emits `[S1] doc_title › section` blocks up to `CONTEXT_TOKEN_BUDGET` (≈1,800 tokens, approximated
  as 4 characters/token). The top-ranked result is always included even if it alone exceeds the
  budget, so a single relevant chunk is never dropped entirely. Each emitted block gets a matching
  `Citation`.

`build_rag_graph(retriever)` wires the two nodes into `START -> retrieve_documents ->
build_context -> END` and compiles the graph. Constructing or importing this module does no network
or Redis work — the retriever is only called when a node runs.

### Retrieval (`app/rag/retriever.py`)

`Retriever` is a `langchain_core.retrievers.BaseRetriever` subclass wrapping a `RedisVectorStore`,
so retrieval goes through the LangChain retriever interface (`.invoke()`) rather than a direct
vector-store call from application code; this also makes retrieval appear as a distinct "retriever"
observation type in Langfuse traces. `search(query, category)` is a thin convenience wrapper calling
`self.invoke(query, category=category)`. Its `_get_relevant_documents` implementation calls
`similarity_search_with_score(query, k=TOP_K, filter=...)` and converts the raw cosine **distance**
each result carries into `similarity = 1 - distance`, stored in `Document.metadata["similarity"]`.
`similarity_search_with_relevance_scores()` (and the built-in `VectorStoreRetriever`'s
`search_type="similarity_score_threshold"`, which relies on the same method) is not usable here — it
raises `NotImplementedError` because `RedisVectorStore` has no `_select_relevance_score_fn`
configured — so this conversion is done directly rather than relying on that LangChain convenience
path. The category filter, built fresh on every call, is `@categories:{<category>|general}`
(RediSearch TAG OR-syntax) when a category is given, matching chunks tagged with either the active
category or `general`; without a category, no filter is applied. The category is a search-time
argument rather than construction state, so one `Retriever` instance (injected into
`build_rag_graph`) serves every category — tests supply a fake with the same `search(query,
category)` shape, and only the real Redis integration constructs the genuine one.

### Result models (`app/rag/model.py`)

- **`RetrievedResult`**: `content`, `similarity`, `doc_id`, `doc_title`, `section_id`, `section`,
  `categories`, `rule_ids`, `source_path` — everything needed to trace a result back to its source.
- **`Citation`**: `marker` (`"S1"`, `"S2"`, ...), `doc_id`, `doc_title`, `section` — one per block
  actually included in the context, never for a result that was dropped.
- **`RagResult`**: `results`, `category`, `context`, `citations`, plus a `confidence` property reading
  `results[0].similarity` (0.0 when there are no results) — there is no separate confidence field to
  drift out of sync with the results.

### Confidence threshold (`app/rag/index_schema.py`)

`MIN_CONFIDENCE_THRESHOLD = 0.8` was calibrated against the live corpus and embedding model: one
representative on-topic question per category (general, meal, equipment, travel, commuting, mileage,
benefits) scores 0.875-0.937 top-1 similarity; off-topic questions ("what is the weather like on
mars today", "who won the football world cup in 2018", ...) score 0.708-0.786. A downstream consumer
(a later task) compares `RagResult.confidence` against this threshold to decide whether to answer or
say the policy does not cover the question — this task only guarantees the value is present and
correctly derived, not the response text.

### Tool exposure (`app/rag/tool.py`)

`build_search_policies_tool(rag_graph)` returns a `@tool(response_format="content_and_artifact")`
LangChain tool named `search_policies` taking `question` and an optional `category`. Its content is
the assembled context string (or `"No relevant policy information found."` when there are no results —
an explicit, non-fabricated result rather than an empty string reaching the model unexplained); its
artifact is the complete `RagResult`, available to the agent's evaluation/citation-checking code
without re-parsing the content string.

## How to use

```python
from app.rag.graph import build_rag_graph
from app.rag.retriever import Retriever
from app.rag.tool import build_search_policies_tool

graph = build_rag_graph(Retriever(vector_store))
result = graph.invoke({"question": "how much can I claim for a business meal?", "category": "meal"})["result"]
print(result.context, result.citations, result.confidence)

search_policies = build_search_policies_tool(graph)
```

## Key files

| File | Responsibility |
| --- | --- |
| `app/rag/state.py` | `RagState` LangGraph state contract |
| `app/rag/graph.py` | `RagNodes`, `build_rag_graph` |
| `app/rag/retriever.py` | `Retriever` |
| `app/rag/tool.py` | `build_search_policies_tool` (`search_policies`) |
| `app/rag/model.py` | `RetrievedResult`, `Citation`, `RagResult` |
| `app/rag/index_schema.py` | `MIN_CONFIDENCE_THRESHOLD`, `CONTEXT_TOKEN_BUDGET` (alongside `TOP_K`) |
| `tests/test_retriever.py` | filter/similarity-conversion unit tests (mocked vector store) |
| `tests/test_rag_graph.py` | filtered/unfiltered/fallback/empty/ranking/budget/dedup node tests |
| `tests/test_rag_tool.py` | content-and-artifact tool behaviour |
| `tests/test_redis_integration.py` | per-category grounded-evidence and low-confidence tests against Redis 8 |
