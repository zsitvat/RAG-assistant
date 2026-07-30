# 03 — Grounded RAG subgraph

**What to build:** A policy question can be sent to a standalone compiled LangGraph subgraph and produce ranked policy evidence, bounded context, and checkable citations. The subgraph uses the LangChain Redis retriever and remains reusable independently of the main agent.

**Blocked by:** 02 — Policy ingestion and Redis index visibility.

**Status:** ready-for-agent

**Design references:** Technical design §4.2–§4.3, §8, and the retrieval failure modes in §15.

**Technical notes:**

- Give the subgraph its own small `RagState` and compile it independently from the main agent; the seam is question/category in and one typed `RagResult` out.
- Supply a retriever factory rather than constructing Redis or embeddings inside graph nodes, allowing fake retrievers in unit tests and the Redis adapter in integration tests.
- Keep embeddings and raw vectors outside graph state. The retriever encapsulates query embedding and KNN execution, while state carries only information consumed by another node.
- Build citations from the exact documents admitted to the bounded context so the answer cannot cite a retrieved chunk that was dropped from the prompt.
- Expose retrieval to the agent through a content-and-artifact LangChain tool: concise evidence for the model, complete structured hits for evaluation.
- Keep query rewriting, LLM relevance grading, reranking, and multi-strategy escalation out of this PoC; if retrieval evaluation fails, improvements are considered in that order.

- [ ] The RAG workflow is a separately compiled LangGraph `StateGraph` with distinct retrieval and context-building nodes.
- [ ] Its public state contract accepts only a question and optional category and returns one typed RAG result.
- [ ] Retrieval uses the LangChain retriever interface with top-four dense search rather than direct RediSearch commands in application code.
- [ ] Similarity is normalised from cosine distance consistently, and the confidence/low-relevance decision derives from the highest-ranked hit rather than a duplicated state field.
- [ ] A category-filtered query includes both the active category and general policies; a query without a category searches the complete corpus.
- [ ] An empty filtered result is retried once without the category, while an empty or low-confidence final result is represented explicitly rather than fabricated.
- [ ] Retrieved hits preserve similarity, document identity, section identity, categories, rule identifiers, and source information.
- [ ] Context building emits numbered source blocks within the token budget and produces citation objects that map exactly to included blocks.
- [ ] The assembled context is limited to approximately 1,800 tokens and uses stable `[S1]`, `[S2]`, and subsequent markers with document title and section labels.
- [ ] The policy-search LangChain tool returns a compact model-facing summary and the complete typed RAG result as its artifact.
- [ ] Constructing or importing the subgraph performs no network or Redis work; retrievers are supplied through the graph factory.
- [ ] Focused tests cover filtered, unfiltered, fallback, empty, ranking, token-budget, and citation-deduplication behaviour.
- [ ] A standalone invocation against the ingested Redis index returns relevant, source-backed evidence for at least one policy question in each configured category.
- [ ] An irrelevant top result below the configured threshold yields the documented uncovered-policy response path instead of allowing response generation to imply policy support.
