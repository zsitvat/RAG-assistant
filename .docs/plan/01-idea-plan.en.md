# Idea Plan – Agentic RAG Assistant (high level)

English version of [01-idea-plan.hu.md](01-idea-plan.hu.md).

## 1. The chosen topic

**Corporate expense reimbursement and employee benefits assistant.**

An agentic RAG chatbot that answers employees' questions about expense reimbursement and
benefits based on the internal policies of a fictional company: whether an expense is
reimbursable, how much can be claimed, which supporting documents are required, whether
manager approval is needed, and what the next step is.

The prototype is not connected to any real financial system and does not provide tax or legal
advice – the policies describe a fictional company and serve demonstration purposes only.

## 2. Why I chose this topic

The topic was deliberately chosen to be close to the activity of the company that issued the
assignment: financial processes, internal controls, governance and compliance are core themes in
a consulting / audit environment. I wanted a use case that is related to this in substance, not
just a technical demo.

Beyond that, the problem is:

- real and common – an administrative process that exists at practically every larger company;
- document-based – answers come from internal policies, so it naturally requires RAG;
- decision-heavy – limits, eligibility, deadlines, approval thresholds;
- calculation-heavy – so it needs deterministic tools, not just LLM text generation.

## 3. What user need it serves

The employee asks in natural language and gets, in a single answer:

1. whether the expense is reimbursable / whether they are eligible;
2. how much can be reimbursed;
3. which documents are required;
4. whether approval is needed;
5. what the next step is, and which policy rule it is based on.

This removes the need to read through policies manually and reduces the number of incorrect
claims.

## 4. Why agentic RAG instead of plain RAG

Plain RAG retrieves the relevant passage and answers from it. However, typical questions are
multi-step, and the answer is not readily present in the document. Example:

> "I commute to work by car, I live 32 km from the office, and I come in three times a week.
> How much allowance can I get this month?"

The system has to recognise the category, extract the data, ask back (is 32 km one-way or the
total distance?), find the applicable rule, check eligibility, and then calculate the amount.
This requires intent recognition, state management, conditional routing, tool use and
clarification questions – i.e. agentic behaviour.

Key behaviour: when the question is incomplete, the chatbot **does not guess** – it asks for the
missing data.

## 5. Covered categories (scoped down to a manageable prototype)

1. **Business expenses** – meals, accommodation, taxi, parking.
2. **Commuting** – public transport pass, own car, mileage reimbursement.
3. **Work equipment** – monitor, headphones, office equipment.
4. **Employee benefits** – holiday / recreation / training allowance, home office costs.

## 6. Solution outline

- **Main agentic workflow (LangGraph):** intent classification → information extraction → missing
  information check (clarification) → autonomous tool loop → grounded response generation.
- **Dedicated RAG subgraph:** deliberately simple – query embedding → similarity search with tag
  filtering by expense category → context building with citations. A standalone, reusable
  module; the complexity of the project lives in the agentic workflow and the tools, not in the retriever.
- **Tools (deterministic Python, not the LLM):** expense and reimbursement calculator, rule
  checker and deadline calculator.
- **Data source:** fictional company policies as `.docx` documents under
  `.docs/sources/<lang>/`, converted to Markdown at ingest so the heading structure survives; reviewed by
  hand for internal consistency, small in volume – the emphasis is on quality processing. The structured
  rule catalogue (`rules.yaml`) is hand-authored for the prototype; a production version would extract it
  from the documents themselves.
- **Language:** an English and a Hungarian knowledge base, one Redis index each, and a config setting
  (`ACTIVE_LANG`) decides which one the assistant uses. Both indices are built with the same
  multilingual embedding model, since Hungarian has no strong language-specific retrieval embedder and a
  shared model keeps the two languages' eval numbers comparable.
- **Storage:** Redis (Redis Stack) as the single datastore – vector indices for the policy chunks
  and LangGraph conversation checkpoints.
- **Model:** locally runnable open-source LLM (no paid API); dummy LLM fallback if needed.
- **Service split:** the agent runs as a FastAPI service; Streamlit is a thin client over HTTP, so the
  agent is also callable by the eval harness or curl.
- **UI:** a focused Streamlit chat showing the assistant answer, completion date and response time,
  plus a streamed, expandable summary of the used sources and completed workflow steps. Detailed
  tool arguments, timings and retrieval diagnostics remain in Langfuse.
- **Runtime:** containerised – Dockerfile and docker-compose (`api`, `ui`, `redis`, `ollama`) in the
  repository root. Application logs go to both stdout and daily rotating files retained for seven
  days.

## 7. Evaluation outline

- **Functional:** a test set of 10–20 questions across the categories above (including an
  incomplete question, exceeding a limit, a prohibited expense type, and a complex case).
  Measured dimensions: intent accuracy, information extraction, document retrieval, tool
  selection, calculation, final decision, source attribution. The version-controlled JSON dataset
  stays in the repository and is synchronised to Langfuse for experiment runs and scoring.
- **Performance:** 50–200 queries, latency metrics (mean, median, p95), bottleneck identification
  (expected to be LLM generation), and 1–2 concrete optimisation proposals. The local load runner
  generates the traffic; its requests are traced and grouped as one Langfuse load run.

## 8. Expected outcome

A working prototype that demonstrates, on a realistic corporate administrative problem,
LangGraph-based agentic behaviour, modular RAG, deterministic tool use, state management and
autonomous decision-making – with source-backed, verifiable answers.

Note on scope: the implementation keeps the graph deliberately small while retaining separate intent
classification, autonomous tool selection, modular RAG, deterministic calculation and checkpointed
multi-turn state. The goal is to demonstrate the assignment requirements clearly without adding
workflow stages that exist only for architectural depth.
