# 11 — Containerised clean-start runtime

**What to build:** A reviewer can clone the repository, provide the documented environment values, run one Compose command, and reach a healthy Streamlit assistant backed by FastAPI, Redis Stack, and Ollama. Startup prepares models and policy data automatically and fails clearly when a required dependency cannot become ready.

**Blocked by:** 02 — Policy ingestion and Redis index visibility; 09 — Streaming chat experience and thread controls; 10 — Langfuse observability, prompt resolution, and operational logging.

**Status:** ready-for-agent

**Design references:** Technical design §1.2, Redis durability in §4.3, configuration in §11, §12–§12.1, and infrastructure failure modes in §15.

**Technical notes:**

- Build one application image and select API or UI behaviour by command, eliminating dependency drift between two Python runtimes.
- Keep model and embedding acquisition at different lifecycle stages: bake embedding weights into the image, but persist the larger Ollama model in its named runtime volume.
- Gate startup in dependency order—Redis, Ollama, model availability, catalogue/index validation, ingestion, then Uvicorn readiness.
- Use Redis Stack rather than plain Redis because vector indexing requires RediSearch; share the deployment while isolating vector, manifest, and checkpoint key namespaces.
- Run as a non-root user and validate writable mounts before serving so permission failures are immediate and diagnosable.
- Keep the official runtime to one Uvicorn worker: local Ollama serialises generation, so additional workers do not remove the actual bottleneck.

- [ ] One multi-stage Python image serves both API and UI commands and runs the application as a non-root user.
- [ ] The runtime and development dependency inputs produce fully pinned lock files with hashes, and installation rejects unverified dependency drift.
- [ ] The Python base image, Redis Stack image, Ollama image, LLM quantisation tag, and embedding revision are explicitly pinned for reproducibility.
- [ ] Embedding weights are downloaded during the image build so startup does not fetch them on the first user request.
- [ ] Compose defines healthy API, UI, Redis Stack, and Ollama services with the intended ports, named data volumes, source mounts, log mounts, and bounded container logging.
- [ ] Redis enables durable append-only storage and provides both vector indexing and LangGraph checkpoint persistence.
- [ ] Redis Stack is documented as a mature general-purpose datastore chosen for both vector search and application state, including the project owner's production experience with Redis-backed systems.
- [ ] API startup waits for Redis and Ollama, ensures the configured model is available, runs idempotent ingestion, verifies log-directory permissions, and starts serving only after readiness succeeds.
- [ ] The UI waits for API readiness and does not present a broken chat while backend dependencies are unavailable.
- [ ] Model, index, and log volumes survive container recreation while the fictional source corpus remains read-only inside the runtime.
- [ ] Langfuse remains an external optional cloud integration and does not add another runtime container.
- [ ] Dependency timeout, model-pull failure, ingest failure, index-dimension mismatch, and unwritable-log failures terminate or report readiness with an actionable reason.
- [ ] Redis loss is documented as losing rebuildable index data and open conversations; startup rebuilds the corpus index but does not pretend lost checkpoints can be restored.
- [ ] Readiness fails while Redis or Ollama is unreachable, while liveness continues to report whether the process itself is running.
- [ ] A clean-clone smoke test builds the images, starts the stack, ingests the corpus, answers a grounded question, resumes a thread, and shuts down without losing persisted data.
- [ ] The clean-start path fits the documented single-machine budget: approximately 5 GB for the quantised 7B model plus the small embedding model and supporting services.
