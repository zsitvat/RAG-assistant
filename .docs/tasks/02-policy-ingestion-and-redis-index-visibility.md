# 02 — Policy ingestion and Redis index visibility

**What to build:** A reviewer can ingest the fictional English policy documents into Redis Stack and inspect the resulting index through the application. The slice preserves headings and tables, attaches stable policy metadata, validates the deterministic rule catalogue, and makes repeated ingestion safe and observable.

**Blocked by:** 01 — Runnable application shell.

**Status:** ready-for-agent

**Design references:** Technical design §4.1–§4.4, §10.1 admin endpoints, §12 startup ingestion, and the Redis/index failure modes in §15.

**Technical notes:**

- Represent the catalogue as one validated `RuleCatalogue` dependency shared by ingestion and later deterministic tools; ingestion must never parse ad hoc YAML fragments.
- Implement the Word normaliser behind LangChain's loader interface and emit `Document` objects before applying the header-aware and recursive splitters.
- Treat stable document/section identifiers and rule references as data contracts. Human-readable headings may change, but citation and rule identities must not.
- Use the LangChain Redis vector-store integration for writes and searches. Keep raw Redis access limited to lifecycle concerns such as manifests, checkpoint namespaces, health, and index replacement where the integration requires it.
- Compute the manifest before mutation and make rebuild decisions explicit so interrupted or mismatched ingestion cannot silently mix vector generations.
- Use the documented English corpus as one index and keep the normalised Markdown in memory; source documents remain unchanged.

- [ ] The rule catalogue is parsed into typed models once during startup, and missing or malformed rules fail before the application begins serving requests.
- [ ] The corpus contains the eight stable document identities `00` through `07`, covering glossary/index, general reimbursement, travel, commuting, personal vehicle, benefits, receipts/approvals, and FAQ/examples.
- [ ] The Word loader implements the LangChain loader interface and converts titles, heading levels, paragraphs, lists, and tables into predictable Markdown-backed documents.
- [ ] Images, headers, footers, comments, tracked changes, and unsupported Word elements are deliberately dropped, while document title falls back to the source name when no title style exists.
- [ ] Header-aware splitting keeps policy sections meaningful, preserves tables as whole blocks, merges unusably short sections, and applies the configured size guard only where needed.
- [ ] Splitting uses heading levels one through three, an 800-character size guard, 120-character overlap, and approximately 200 characters as the short-section merge threshold.
- [ ] Every chunk carries stable document, section, category, rule, ordering, and source metadata suitable for retrieval and citations.
- [ ] Ingestion rejects unknown document identifiers, empty categories, unresolved heading anchors, and invalid rule references with actionable errors.
- [ ] The pinned multilingual embedding model and LangChain Redis vector-store integration create a searchable index with the expected vector dimension and metadata schema.
- [ ] Embeddings use the pinned `intfloat/multilingual-e5-small` revision at 384 dimensions with the required query/passage prefixes configured once in the embedding adapter.
- [ ] Redis stores chunk hashes under stable identities, indexes text plus document/section/category/rule metadata, and uses cosine HNSW vectors with a category tag separator compatible with the retriever filter.
- [ ] Redis key ownership follows the design: stable chunk hashes and `idx:chunks` for retrieval, one corpus manifest, and a separate checkpoint namespace owned by LangGraph persistence.
- [ ] A corpus manifest includes source content, chunking settings, embedding model, embedding revision, and vector dimension; matching manifests make ingestion a no-op and mismatches trigger a safe rebuild.
- [ ] The ingest endpoint reports whether it built, rebuilt, or reused the index, and the statistics endpoint reports chunk totals and category coverage.
- [ ] Corpus vectors are embedded and upserted in bounded batches of 128 and are never recomputed during a user request.
- [ ] Corpus/index and manifest keys have no expiry, while checkpoint keys use their separately owned retention policy rather than inheriting corpus lifetime.
- [ ] The Streamlit sidebar can display index readiness and category counts without containing Redis-specific logic.
- [ ] Unit tests cover document conversion, tables, chunk boundaries, metadata validation, and manifest decisions.
- [ ] An integration test against Redis Stack proves a complete ingest, idempotent re-run, and dimension-mismatch rebuild; vector search is not faked with an incompatible Redis substitute.
- [ ] Redis unavailability, failed rebuild, and an invalid index manifest produce an actionable startup/admin failure rather than an apparently empty knowledge base.
