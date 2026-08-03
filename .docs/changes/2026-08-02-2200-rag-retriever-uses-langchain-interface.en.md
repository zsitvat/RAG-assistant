# 2026-08-02 22:00 — RAG retrieval uses the LangChain retriever interface

## What changed

- `app.rag.retriever.Retriever` now subclasses `langchain_core.retrievers.BaseRetriever` and
  implements `_get_relevant_documents`, so retrieval goes through `.invoke()` rather than calling
  `RedisVectorStore.similarity_search_with_score` directly from application code. `Retriever.search()`
  is now a thin wrapper around `self.invoke(query, category=category)`, keeping the existing call
  site in `RagNodes` unchanged.
- `TOP_K` in `app.rag.index_schema` corrected from 5 to 4, matching technical design §3's `k=4`
  retrieval and the `retrieval_hit_at_4` evaluation metric name.

## Why

- Technical design and task 03's checklist required the LangChain retriever interface, not a direct
  vector-store call; the direct-call form also meant retrieval never surfaced as a distinct LangChain
  "retriever" observation type.
- `TOP_K = 5` silently mismatched the design's `k=4` and the eval metric's own "hit@4" name.

## Notes

- `RedisVectorStore` does not implement `_select_relevance_score_fn`, so the built-in
  `VectorStoreRetriever` (`search_type="similarity_score_threshold"`) would raise `NotImplementedError`
  when asked for scores. The custom `BaseRetriever` subclass keeps the existing
  `similarity_search_with_score` call and cosine-distance-to-similarity conversion, so citation scores
  are unaffected.
- Full test suite (317 tests) re-run against the live Redis/Ollama stack after this change; all pass.
