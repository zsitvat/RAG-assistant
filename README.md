# RAG-assistant

Agentic RAG prototype: a corporate expense reimbursement and employee benefits assistant, built with
LangChain + LangGraph, a local open-source LLM (Ollama), Redis 8, FastAPI and Streamlit.

> **This assistant describes a fictional company.** Every policy, limit, rate and rule it cites is
> made up for this prototype. Nothing it says is real company policy, and nothing it says is tax or
> legal advice.

## Contents

1. [The problem and why this topic](#1-the-problem-and-why-this-topic)
2. [User journeys](#2-user-journeys)
3. [Architecture](#3-architecture)
4. [Redis: single datastore, and what that costs](#4-redis-single-datastore-and-what-that-costs)
5. [Model trade-offs](#5-model-trade-offs)
6. [Production rule-catalogue extraction (not built here)](#6-production-rule-catalogue-extraction-not-built-here)
7. [Setup, configuration and run commands](#7-setup-configuration-and-run-commands)
8. [Evaluation method](#8-evaluation-method)
9. [Load test method](#9-load-test-method)
10. [PoC boundaries (deliberately out of scope)](#10-poc-boundaries-deliberately-out-of-scope)
11. [Recommendations and ideas (not implemented)](#11-recommendations-and-ideas-not-implemented)
12. [Key design decisions](#12-key-design-decisions)
13. [What's committed, what isn't](#13-whats-committed-what-isnt)

## 1. The problem and why this topic

An agentic RAG chatbot that answers employees' questions about expense reimbursement and benefits
against a fictional company's internal policies: whether an expense is reimbursable, how much can be
claimed, which supporting documents are required, whether manager approval is needed, and what the
next step is.

The topic was chosen deliberately close to a consulting/audit environment's own subject matter
(financial processes, internal controls, governance) while also being a genuinely good fit for the
assignment's technical requirements: it is document-based (needs RAG for "what does the policy say"),
decision-heavy (limits, eligibility, deadlines, approval thresholds) and calculation-heavy (needs
deterministic tools, not LLM arithmetic). See
[`.docs/plan/01-idea-plan.en.md`](.docs/plan/01-idea-plan.en.md) for the full problem statement and
[`.docs/plan/02-technical-design.en.md`](.docs/plan/02-technical-design.en.md) for the complete
implementation reference this README summarises.

**Why agentic RAG instead of plain RAG.** A typical question — *"I commute to work by car, I live 32
km from the office, and I come in three times a week. How much allowance can I get this month?"* —
cannot be answered by retrieving one passage. The system has to recognise the category, extract the
data, ask back when something is ambiguous (is 32 km one-way or round trip?), find the applicable
rule, check eligibility and calculate the amount. That needs intent recognition, state management,
conditional routing, autonomous tool use and clarification — agentic behaviour, not a single retrieval
step. The one non-negotiable behaviour: **when a question is incomplete, the assistant asks — it never
guesses.**

## 2. User journeys

- **Grounded policy question** — "What is the reimbursement limit for a business meal?" → retrieval,
  a cited answer, no calculation.
- **Complete reimbursement request** — category, amount and supporting facts all present in one
  message → the agent searches policy, calculates the amount, checks eligibility/caps/approval, and
  answers with a deterministic decision (`eligible` / `partially_eligible` / `not_eligible`) and
  citations.
- **Clarification and resume** — an ambiguous or incomplete claim (e.g. an unstated one-way/round-trip
  distance) gets a targeted follow-up question; the user's next message in the same thread resumes
  the same claim rather than starting over.
- **Deadline check** — "Is it still within the deadline to submit this expense?" evaluated against a
  pinned or current reference date.
- **Unsupported / out-of-scope** — tax/legal advice or unrelated requests get an explicit
  out-of-scope response with a disclaimer, never a fabricated policy answer.
- **Streamed steps and sources** — the Streamlit UI shows step-by-step progress (`Intent classified`
  → `Details extracted` → `Policies searched` → `Rules checked` → `Answer generated`) and the cited
  sources live, then collapses them under the answer once the turn completes.

## 3. Architecture

```mermaid
flowchart TB
    UI[Streamlit UI] -->|"POST /chat, /chat/stream"| API[FastAPI service]
    API -->|invoke/stream| G
    subgraph G[Main agent graph - LangGraph]
        direction TB
        N1[1 classify_intent] --> N2[2 extract_information]
        N2 -->|incomplete| N3[3 ask_clarification]
        N2 -->|complete| N4[4 agent_step]
        N2 -->|unsupported| N7[7 out_of_scope]
        N4 -->|tool call| N5[5 execute_tools]
        N5 --> N4
        N4 -->|"no tool call / budget spent"| N6[6 generate_response]
    end
    N5 -.->|search_policies| R[[RAG subgraph]]
    R -.->|category filter + KNN| VS[(Redis 8 vector index)]
    N5 -.->|calculate / check_rules| RY[(rules.yaml)]
    G -.->|LLM calls| LLM[Ollama local model]
    G -.->|traces, spans, scores| LF[Langfuse]
    G <-->|checkpoint| CP[(Redis - LangGraph checkpointer)]
```

**LangChain owns**: documents and splitting (`DocxToMarkdownConverter`, `MarkdownChunker`),
embeddings (`E5Embeddings`), the Redis vector store and retriever, prompts (`ChatPromptTemplate`),
the chat model abstraction (`ChatOllama`/`FakeListChatModel`), messages, structured output
(`with_structured_output`) and tools (`@tool`, `bind_tools`, `ToolNode`).

**LangGraph owns**: state (`AgentState`, checkpointed per conversation thread), routing (conditional
edges), tool execution (`ToolNode`), checkpointing (`AsyncRedisSaver`) and streaming
(`astream(stream_mode=["updates", "messages"])`).

The agent runs entirely inside the FastAPI service; the Streamlit UI is a thin HTTP client with no
graph imports, so the agent stays independently callable (curl, the eval harness, the load-test
script, another front-end).

**Two-layer knowledge design** — the single most important design decision in the project:

- **Prose policies** (`.docs/sources/en/*.docx`) are the RAG corpus: they answer "what does the rule
  say" and provide citations.
- **A machine-readable rule catalogue** (`rules.yaml`) drives the deterministic tools: limits, rates,
  thresholds, deadlines. The LLM never invents or copies a policy number into a tool call — the
  deterministic tool selects the applicable rule from the validated claim and catalogue, then
  computes. A consistency test keeps both in sync, so a citation always backs the number that was
  calculated.

`rules.yaml` is **hand-authored** in this PoC on purpose: with the numbers fixed, a wrong calculation
has exactly one possible cause (the tool), so the eval's calculation metric can't confuse bad
extraction with a bad tool. §4 of the technical design records what a production version would do
instead (§6 below).

### 3.1 Main agent graph (7 nodes)

1. `classify_intent` — structured-output intent + optional category classification.
2. `extract_information` — merges expense-claim fields extracted from the conversation.
3. `ask_clarification` — asks for exactly one missing required slot, deterministically (a slot table
   keyed by `(intent, category)`, not a prompt instruction) — ends the turn; the checkpointed claim
   resumes on the user's next message in the same thread.
4. `agent_step` — a ReAct tool-calling loop: the model decides, each iteration, whether to call a
   tool and which one (`search_policies`, `calculate`, `check_rules`) via `bind_tools()`. There is no
   static intent→tool table — tool selection is the one piece of the design that most directly
   demonstrates autonomous decision-making, bounded by deterministic guardrails (a 4-step budget,
   duplicate-call reuse, disable-after-two-argument-errors).
5. `execute_tools` — a generic LangGraph `ToolNode`.
6. `generate_response` — grounds the final answer in the gathered tool evidence and derives the
   deterministic `decision`.
7. `out_of_scope` — an explicit disclaimer for unsupported requests, never a fabricated answer.

### 3.2 RAG subgraph

Deliberately simple, and reusable outside the main graph: query embedding → similarity search with a
category-tag filter → context assembly with `[S1]`/`[S2]`-style citations, budgeted to a token limit.
The complexity of the project lives in the agentic workflow and the tools, not in the retriever.
Section headings are prepended to every chunk's `page_content` (not only metadata), so a chunk is
self-describing for both embedding and retrieval even split across multiple pieces.

### 3.3 Deterministic tools

- **`calculate`** — pure-function reimbursement math per category (meal per-person cap, travel
  subtype caps, mileage rate, commuting pass/ticket/vehicle formulas, equipment passthrough, benefits
  budget/ratio) with rounding, caps and excess reported explicitly.
- **`check_rules`** — eligibility, document/receipt requirements, approval-tier thresholds and the
  submission deadline, evaluated against the validated rule catalogue and a pinned or current
  reference date.

Both read the current claim through LangGraph's injected `ToolRuntime` rather than the model
re-stating it as a tool argument, and both return typed pydantic results (`CalculationResult`,
`list[Finding]`) as tool artifacts — never free text the model would have to parse back.

## 4. Redis: single datastore, and what that costs

Redis 8 holds both the policy vector index (`RediSearch`) and LangGraph's conversation checkpoints —
one mature, well-understood database rather than a specialised vector database plus a separate
key-value store. This reflects real production experience running Redis-backed systems: fewer moving
parts, one healthcheck, shared state if the UI is ever scaled horizontally. The cost, stated plainly:
Redis Search's in-memory index holds the whole corpus, and the LangChain Redis integration still
depends on Redis Search index/schema compatibility even though it encapsulates `FT.CREATE`, KNN query
construction and vector serialisation. For a corpus of a few hundred chunks this is a good trade; a
much larger corpus would be worth evaluating behind the same LangChain vector-store interfaces.
`langgraph-checkpoint-redis`'s `AsyncRedisSaver` is used rather than a hand-rolled checkpointer;
isolated graph unit tests use LangGraph's in-memory saver so they stay Redis-free.

## 5. Model trade-offs

| Model | Pros | Cons |
| --- | --- | --- |
| `qwen2.5:7b-instruct-q4_K_M` (chosen) | best structured-output/tool-calling reliability at this size; usable Hungarian capability | 4–5 GB RAM, slow on pure CPU |
| `llama3.1:8b-instruct` | strong reasoning on English | weaker Hungarian, larger, slower |
| `qwen2.5:3b-instruct` | 2–3× faster, fits small machines | more extraction errors |
| Dummy LangChain test chat model | deterministic CI/tests, no Ollama needed | canned answers only |

The 7B ceiling is a direct consequence of running everything (Ollama + Redis + the API) on one
developer machine, no paid API. The knowledge base is English-only with one Redis index — Hungarian
conversation is a **best-effort model capability** (the multilingual `intfloat/multilingual-e5-small`
embedder plus Qwen's own multilingual instruction-following), not a claim that the knowledge base
itself is localised; the official evaluation stays English. Temperature is `0` for classification,
extraction and tool selection.

## 6. Production rule-catalogue extraction (not built here)

`rules.yaml` is hand-authored for this PoC. A non-PoC version would generate the catalogue from
uploaded policy documents instead: detect and normalise their structure (real uploads rarely share
this PoC's clean, structured source pack), run an offline LLM pass proposing rules into the same
pydantic schema with the exact `doc_ref` each came from, accept a rule only if its number appears
verbatim in the cited section (the same consistency check this PoC already runs as a test, reused as
a gate), and require a human diff review before the catalogue is versioned. At runtime nothing
changes — the tools still read a validated catalogue, never raw LLM output. Left out here because an
extraction pipeline plus review workflow is a project of its own, and none of it changes the agentic
behaviour this assignment grades.

## 7. Setup, configuration and run commands

Requires [`uv`](https://docs.astral.sh/uv/) (installs Python 3.12 automatically) and Docker.

```bash
uv sync --dev
cp .env.example .env
```

`Settings` (`src/app/settings.py`) loads `.env` automatically; the committed `.env.example` documents
every setting and contains no credentials.

### Clean-start containerised runtime (recommended)

```bash
docker compose up
```

One image serves both processes (`api`, `ui` — same build, each with its own `command:` in
`docker-compose.yml`). This brings up Redis 8 (+ Redis Insight at <http://127.0.0.1:5540>), Ollama
(a one-shot `ollama-pull` service pulls the configured model before the API starts), the API and the
Streamlit UI. Compose's `depends_on` health/completion conditions order the startup; the API's own
FastAPI lifespan ingests the corpus on boot, skipping the work when the stored manifest already
matches. Once `api` reports healthy:

- Streamlit UI: <http://127.0.0.1:8501>
- API + Swagger docs: <http://127.0.0.1:8000/docs>
- `GET /health` — liveness; `GET /ready` — Redis reachable + index dimension match, LLM responding.

### Local development (dummy backend, no Ollama)

```bash
docker compose up -d redis redisinsight
LLM_BACKEND=dummy PYTHONPATH=src uv run uvicorn app.main:app --port 8000
# in a second terminal
API_BASE_URL=http://127.0.0.1:8000 PYTHONPATH=src uv run streamlit run src/app/ui.py
```

### Quality gate

```bash
uv run ruff check .
uv run ruff format --check .
uv run bandit -c pyproject.toml -r src/app load_test
uv run pytest --cov=app --cov-report=term-missing --cov-report=xml
```

`make check` runs all four. The same checks run in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) in that order (lint → format → security →
test+coverage → SonarQube Cloud), with `SONAR_TOKEN`/`SONAR_ORGANIZATION`/`SONAR_PROJECT_KEY`
supplied only by CI secrets/variables — fork PRs run everything except the Sonar submission. Locally:

```bash
cp sonar-project.properties.example sonar-project.properties   # gitignored, fill in your own keys
export SONAR_TOKEN="<your-token>"
make sonar
```

### Functional evaluation and load test

```bash
# Requires LANGFUSE_ENABLED=true and real credentials in .env, and the API reachable at API_BASE_URL.
uv run python -m llm_eval.run_eval               # full 20-case functional evaluation
uv run python -m llm_eval.run_eval --node intent # classifier-only fast pass

# Requires LANGFUSE_ENABLED=true and real credentials in .env; runs in-process against a live Redis/Ollama.
uv run python -m load_test.load                                          # defaults: 3 reps, concurrency 4
uv run python -m load_test.load --repetitions 5 --max-concurrency 2      # override the defaults
```

## 8. Evaluation method

**Dataset** — `llm_eval/dataset.json`, 20 version-controlled cases spanning general policy, all six
expense categories (meal, travel, commuting, mileage, equipment, benefits), a one-way/round-trip
clarification, a still-open and an expired submission deadline, a missing-receipt case and an
out-of-scope question. Every case's `expected_amount_huf`/`expected_decision` was verified against
the real `ReimbursementCalculator`/`RuleChecker` output for its exact claim fields before being
committed, not hand-computed — so a case failing means the agent diverged from the deterministic rule
engine, not that the dataset is wrong.

**Metrics** (`llm_eval/metrics.py`, seven Langfuse evaluators): classification accuracy (intent +
category), slot accuracy (share of expected `ExpenseClaim` fields matching exactly), retrieval hit@4,
tool-selection accuracy (exact ordered tool-call list — the strictest, most model-behavior-sensitive
metric), outcome accuracy (decision + calculated amount), citation accuracy (an expected document
actually placed in the answer's citation context, not merely retrieved) — six deterministic checks
over structured graph state, no answer-prose parsing among them — plus **answer quality**, an
LLM-as-judge check of the actual generated answer text against a hand-authored
`expected_answer_summary` per case (`llm_eval/judge.py`). The judge runs on `EVAL_JUDGE_MODEL`,
independently configurable from `LLM_MODEL` (defaults to the same tag so it works out of the box,
but pointing it at a genuinely different model gives a materially more meaningful judgement, since a
model grading its own answers risks not catching its own systematic mistakes).

**Runner** — `python -m llm_eval.run_eval` idempotently syncs the dataset to a Langfuse dataset
(`test-dataset`, upserted by stable case id) and runs it as a Langfuse
`dataset.run_experiment(...)`, which supplies concurrency, per-item tracing/dataset-run linking and
per-metric score recording; the application supplies only the task function (one `POST /admin/eval`
per case, pinning `reference_date` for deterministic deadline math) and the metric functions. One
failing case never aborts the run. A local Markdown + JSON report lands under `evaluation_results/`.

**Results, scores and analysis:** see [`evaluation_results/README.md`](evaluation_results/README.md).

## 9. Load test method

`python -m load_test.load` is a standalone CLI script, not an API endpoint: it builds its own copy of
the application dependency graph (the same `ApplicationDependencies.build()` the FastAPI lifespan
uses) and replays a named Langfuse dataset through its own `AgentService` instance, under bounded
concurrency supplied by the same Langfuse experiment runner. Running it as a separate process — not
an endpoint inside the live `uvicorn` worker — means a crash or resource exhaustion during the load
test cannot take real `/chat` traffic down with it, and per-item results are traced to Langfuse as
they complete rather than existing only in one process's memory until a final HTTP response. It
measures the complete graph invocation only, not `/chat` transport overhead or network latency. The
aggregate result is written to `evaluation_results/load-<timestamp>.json` — the same shared results
directory the functional evaluation writes its reports to — as well as printed to the terminal. The
PoC stays uncached deliberately, so its behaviour and latency remain easy to explain.

**Results, bottleneck analysis and optimisation proposals:** see
[`evaluation_results/README.md`](evaluation_results/README.md).

## 10. PoC boundaries (deliberately out of scope)

| Not in scope | Why, and what it would take |
| --- | --- |
| Authentication / authorisation | No user identity; "am *I* eligible" is answered from what the user types, not an HR record. The API host port is loopback-only, but that is a network-placement convenience, not access control. A production version should give Streamlit a service token and let an authorised Swagger user supply a separate token through Swagger's `Authorize` flow, plus SSO on the UI and tenure/budget read from HR. |
| Multi-tenancy | One fictional company, one policy set, one `rules.yaml`. Tenant-scoped indices and catalogue would be the shape. |
| Real financial/ERP integration | No booking, no submission, no ERP call — tells you what's claimable, doesn't file it. |
| Live FX rates | Fixed fictional rates in `rules.yaml`. A real system needs a rate provider plus a rate-date rule. |
| Receipt/OCR input | Amounts come from the conversation, not an uploaded invoice; `ExpenseClaim` is already the seam for a future extraction step. |
| Rule versioning / effective dates | One current catalogue; no "which rule applied last March". |
| Audit trail | Langfuse traces and 7-day operational logs are for debugging, not audit — no tamper resistance or per-user attribution. |
| Personal-data handling & content safety | Conversations sit in Redis with a 24h TTL, no encryption/redaction/export-delete flow. See §11 below. |
| Horizontal scale / rate limiting | One `uvicorn` process, one Ollama, no queue, no per-client limits — state is already in Redis so more API workers is a compose change; the LLM is the actual constraint. |
| Ingestion runs inside the API process | Corpus/rule-catalogue ingestion (`src/app/rag/ingest.py`) runs inline: once at startup, and again on demand through `POST /admin/ingest` on the same process serving `/chat`. A production system would more likely run it as its own pipeline or service, triggered by a content change or a schedule, so a slow embedding run or a bad corpus change can't block or crash the request-serving API — the same reason the load test (§9) is a standalone script rather than an in-process endpoint. |
| Prompt-injection hardening | The corpus is trusted because we wrote it; retrieved context isn't treated as untrusted input. |
| Localised (non-English) policy corpora | One English corpus; Hungarian conversation is a model capability, not a second indexed language. |

None of these change the graph, the tools or the retrieval path — each is an integration or
operational concern layered around the same seams (`ExpenseClaim` as the tools' single input
contract, `rules.yaml` as the single source of numbers, the API as the single entry point).

## 11. Recommendations and ideas (not implemented)

**Personal data and content safety.** A production system should add a PII detection and redaction
layer before prompts, persistence and observability, with controlled re-identification only where the
business flow requires it: Microsoft Presidio for a self-hosted deployment, or Azure-native PII
detection together with Azure AI Content Safety (or the model endpoint's own content filters) on
Azure. The same policy should inspect uploaded documents, user input and generated output, recording
blocked/redacted events in an audit trail without storing the sensitive value itself.

**GDPR** requires a dedicated production assessment and controls; this submission does not claim
legal compliance.

**EU AI Act** — indicative engineering-level gaps, not a legal determination:

- no documented AI-literacy/role-specific training for people operating or supervising the system;
- no compliance control informing employees they are interacting with an AI system;
- no formal assessment of whether production use stays an informational assistant or materially
  influences employment/benefits decisions (which could bring employment-related high-risk
  classification into scope);
- if high-risk: no complete risk-management system, data-governance process, technical
  documentation, compliant event logging, human-oversight procedure, accuracy/robustness/cybersecurity
  evidence, post-market monitoring or serious-incident process;
- no process for notifying affected employees/workers' representatives before deployment;
- no assessed mechanism for machine-readable identification of AI-generated output where required.

The organisation's role (provider, deployer or both), intended use and actual influence on employment
decisions must be assessed by qualified legal and compliance specialists before any production use —
this project makes no compliance claim.

## 12. Key design decisions

Full rationale for each of these in [§17 of the technical design](.docs/plan/02-technical-design.en.md).

- **Tool selection is a ReAct loop**, not a static intent→tool table or a single planner call — the
  one piece of the design that most directly demonstrates autonomous decision-making, bounded by
  deterministic guardrails (step budget, duplicate-call reuse, argument-retry-then-disable).
- **Langfuse is required**, not an optional bonus layer, for the official eval and load runs — the
  per-case scores, prompt-version linkage and per-generation bottleneck spans are exactly the evidence
  the assignment asks for.
- **SSE streaming** is kept alongside plain `/chat` because watching steps and sources appear live is
  a materially better demonstration of agentic behaviour than a blocking response.
- **SonarCloud + Bandit + Ruff + CI** are kept even though the brief only grades code by inspection,
  for an objective, citable quality signal and automatic regression detection.
- **Prompt resolution stays dual-path** (Langfuse-hosted with an embedded fallback) under the same
  `LANGFUSE_ENABLED` switch that already governs tracing, rather than two separate toggles.
- **Redis as the single datastore** — see §4 above.

## 13. What's committed, what isn't

No credentials, personal data, embedding/model caches, runtime logs or machine-specific artifacts are
committed — `.env`, `.venv`, `logs/`, `.sonar/`, `sonar-project.properties` and coverage/cache
directories are all gitignored (see [`.gitignore`](.gitignore)); `.env.example` and
`sonar-project.properties.example` are the committed templates.
