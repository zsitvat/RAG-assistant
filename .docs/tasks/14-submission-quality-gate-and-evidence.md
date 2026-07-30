# 14 — Submission quality gate and evidence

**What to build:** A reviewer receives a reproducible, documented, and evidence-backed submission whose implementation matches the accepted technical design, passes automated checks, starts from a clean clone, and clearly distinguishes PoC behaviour from production recommendations.

**Blocked by:** 11 — Containerised clean-start runtime; 12 — Functional Langfuse evaluation experiment; 13 — Dataset-driven load-test endpoint.

**Status:** ready-for-agent

**Design references:** Technical design §1 requirement traceability, §4.5 production rule extraction, §14 optimisation conclusions, §16 milestones, §17 rationale, §18 PoC boundaries, and the final GDPR/EU AI Act notes.

**Technical notes:**

- Treat quality configuration as executable submission infrastructure: pinned tools, deterministic commands, correctly classified source/tests, and CI secrets isolated from repository content.
- Run checks from narrow to broad—static checks, unit/integration tests, coverage, container build, clean-start smoke, evaluation, load run, then external quality gate.
- Build README claims from recorded outputs and linked Langfuse runs rather than expected numbers from the design.
- Perform specification and engineering reviews as separate passes so requirement coverage cannot hide maintainability, security, or failure-handling defects.
- Keep production recommendations explicitly outside implemented scope, especially PII/content safety, authentication, audit, GDPR, EU AI Act classification, and high-risk controls.
- Use the technical design's requirement table as the coverage source: each requirement must point to a completed ticket, executable behaviour, test/evaluation evidence, or an explicitly documented production-only exclusion.

- [ ] Ruff lint and formatting checks pass over application, evaluation, and test code without blanket suppressions added only to make the gate green.
- [ ] Bandit completes with no unexplained security findings, and any accepted exception is narrow and documented with its concrete rationale.
- [ ] The complete pytest suite passes against pinned dependencies and produces terminal and XML coverage reports for the application code.
- [ ] Sonar configuration classifies source and test code correctly, reads coverage output, excludes generated or fictional data appropriately, and passes the configured SonarCloud quality gate.
- [ ] Continuous integration runs dependency installation, lint, formatting, security, tests, coverage, and Sonar submission in a deterministic order with secrets supplied only by the CI environment.
- [ ] A clean-clone verification builds and starts the Compose stack, checks readiness, exercises grounded policy, reimbursement, clarification/resume, unsupported, streaming, reset, and failure paths, and records the result.
- [ ] The official functional Langfuse experiment is executed and its local summary, aggregate scores, model/prompt identity, and experiment link are retained as submission evidence.
- [ ] The official 50–200 turn load run is executed through the admin endpoint and its throughput, latency distribution, errors, bottleneck evidence, and Langfuse links are recorded.
- [ ] The measured bottleneck is compared with graph step and generation spans, and the README proposes the documented fast-path and Redis-cache optimisations without claiming they were implemented.
- [ ] The README explains the problem, user journeys, LangChain/LangGraph architecture, RAG subgraph, deterministic tools, Redis choice and production experience, model trade-offs, setup, configuration, run commands, evaluation method, and results.
- [ ] Architecture documentation states that LangChain owns documents, splitting, embeddings, retrieval, prompts, chat models, messages, structured output, and tools, while LangGraph owns state, routing, tool execution, checkpointing, and streaming.
- [ ] The rationale records the 7B local hardware constraint, multilingual embedding and best-effort Hungarian capability over an English corpus, ReAct tool autonomy, Langfuse requirement, SSE choice, SonarCloud choice, prompt fallback, Redis choice, RedisSaver, and hand-authored PoC catalogue.
- [ ] The README and UI consistently state that the policies are fictional and that answers are not real company policy, tax advice, or legal advice.
- [ ] Production-only notes cover uploaded-document rule extraction, PII redaction/content safety, authentication, auditability, GDPR, EU AI Act assessment, human oversight, and other explicitly excluded controls without presenting the PoC as compliant.
- [ ] Production catalogue generation is described as uploaded-document structure detection/normalisation, section-level extraction into the same typed schema, verbatim source-number validation, human diff review, and versioning before runtime use.
- [ ] PII guidance covers redaction before prompts, persistence, and observability; Microsoft Presidio for self-hosting; Azure-native PII detection and content safety for Azure; and inspection of uploaded documents, input, and generated output.
- [ ] The PoC boundary list explicitly covers authentication/authorisation, multi-tenancy, financial/ERP integration, live FX, receipt/OCR input, rule effective dates, audit trail, personal-data handling, horizontal scaling/rate limiting, prompt-injection hardening, and independently localised policy corpora.
- [ ] GDPR is identified as requiring a dedicated production assessment and controls; the submission does not claim legal compliance.
- [ ] The EU AI Act gap analysis covers AI-literacy training, user disclosure of AI interaction, formal employment/high-risk classification assessment, risk and data governance, technical documentation and logs, human oversight, accuracy/robustness/cybersecurity evidence, post-market monitoring, incident handling, worker/representative notice, and AI-output identification where required.
- [ ] Compliance notes state that provider/deployer role, intended use, and influence on employment decisions require qualified legal and compliance review.
- [ ] No credentials, personal data, model cache, runtime logs, generated secrets, or machine-specific artifacts are committed.
- [ ] A specification review maps every assignment requirement to working behaviour and evidence, and an engineering review leaves no unresolved high-severity finding before human approval.
- [ ] A final coverage audit maps every technical-design section §1–§18 to one or more completed tickets and confirms that no in-scope contract, constant, failure mode, test obligation, or operational requirement is left without implementation evidence.
- [ ] Every design item intentionally not implemented is present in the documented PoC-boundary or production-recommendation section rather than silently omitted.
- [ ] Human-review evidence covers README comprehension, clean startup, normal/clarification/unsupported/failure paths, streamed sources and steps, selected Langfuse traces, functional scores, load results, bottleneck conclusions, and optimisation proposals.
