# Feature: First policy-question agent journey

Implements task
[`04-first-policy-question-agent-journey.md`](../tasks/04-first-policy-question-agent-journey.md).

## What it does

A seven-node LangGraph `StateGraph` turns one chat message into a grounded, cited answer (or a
deterministic clarification question, or a fictional-policy refusal). It classifies intent and
category, extracts an incremental expense claim, routes unsupported/incomplete requests
deterministically, runs an autonomous tool-calling loop (`search_policies`, `calculate`,
`check_rules`) with hard guardrails, and derives a typed eligibility decision from the tools' own
artifacts before generating the final answer. `POST /chat` exposes it as a blocking HTTP endpoint.

## How it works

### State (`src/app/agent/state.py`) and domain models (`src/app/agent/model.py`)

`AgentState` is a five-key `TypedDict`: `messages` (LangChain messages, `add_messages`-reduced),
`intent`, `category`, `claim` (`ExpenseClaim`), `decision`. `ExpenseClaim` carries every documented
incremental field (category, expense_type, amount_huf, headcount, expense_date, distance_km,
distance_is_one_way, commute_days_per_month, non_reimbursable_amount, has_receipt,
approval_obtained, annual_budget_used_huf), all optional so it fills in incrementally
across turns. `ExpenseClaim.merged_with(update)` keeps a field's old value unless the update
supplies a new one — the one place slot merging happens.

### The seven nodes (`src/app/agent/nodes.py`, `src/app/agent/graph.py`)

| # | Node | Writes |
| --- | --- | --- |
| 1 | `classify_intent` | `intent`, `category` (structured output, one repair retry, typed fallback) |
| 2 | `extract_information` | `claim`, `decision=None` — merges into the pending claim only when the prior turn's `decision` was `needs_info` and the category is compatible; otherwise replaces it |
| 3 | `ask_clarification` | one deterministic question for the top missing slot, `decision=needs_info` |
| 4 | `agent_step` | an `AIMessage`, possibly carrying one tool call (`bind_tools`) |
| 5 | `execute_tools` | LangGraph's prebuilt `ToolNode` (`handle_tool_errors=True`, so a tool's own exception becomes an error `ToolMessage` rather than crashing the graph) |
| 6 | `generate_response` | `decision` (derived from `check_rules` findings) + the final `AIMessage` |
| 7 | `out_of_scope` | the fixed refusal message, `decision=out_of_scope` |

`build_agent_graph(nodes, checkpointer=None)` wires `START → classify_intent → extract_information
→[route_after_extraction]→ {ask_clarification→END | agent_step | out_of_scope→END}`, and
`agent_step →[route_after_agent]→ {execute_tools→agent_step | agent_step (duplicate-call loop-back)
| generate_response→END}`. `route_after_extraction` is a conditional edge straight off
`extract_information` — there is no separate router node. Compiling defaults to LangGraph's
in-memory checkpointer.

### Required slots and clarification (`src/app/agent/slots.py`)

`RequiredSlotTable` is the exact (intent, category) → required-fields table from design §6.2
(`policy_question` needs nothing; `document_requirements` needs `category`; `expense_check`/`meal`
needs `amount_huf`+`headcount`; and so on). `route_after_extraction` sends
`unsupported` straight to `out_of_scope` before any tool ever runs; otherwise it asks for the first
missing slot or proceeds to `agent_step`. An unmapped (intent, category) pair has no required slots
and falls through to the agent, matching the design's stated trade-off.

`src/app/agent/static_texts.py` holds every clarification question, refusal, and system-state message as a
fixed string rather than letting a node generate it. This much hardcoding is mainly a consequence of
running a small chat model: a larger, more capable model could likely be trusted to phrase these
reliably on its own, without the risk of inconsistent or unclear wording that motivates fixing them
here today.

### The autonomous loop and its guardrails (`src/app/agent/message_history.py`, `src/app/agent/nodes.py`)

`MessageHistory` slices `messages` to the suffix starting at the latest `HumanMessage` and derives
every loop guardrail from that slice alone:

- **Step budget** — `agent_step` refuses to call the model once `agent_step_count() >=
  MAX_AGENT_STEPS` (4), instead emitting an empty `AIMessage` that routes straight to
  `generate_response`. If the loop was cut short this way, `generate_response` deterministically
  appends a note that the answer may be based on incomplete evidence — not left to the prompt alone.
- **Tool arg-error disabling** — `agent_step` excludes any tool whose `tool_error_count(name)` has
  reached `MAX_TOOL_ARG_ERRORS` (2) from the next `bind_tools()` call, so a twice-failed tool is
  simply not offered again this turn.
- **Duplicate-call reuse** — before letting a proposed tool call reach `execute_tools`,
  `find_duplicate_call` looks for an identical `(name, args)` call already made this turn; if found,
  `agent_step` fabricates the matching `ToolMessage` from the prior result itself (skipping
  re-execution) and logs a warning.
- **Recursion backstop** — `RECURSION_LIMIT` (12) is passed as LangGraph's own `recursion_limit` at
  invoke time, the hard backstop behind the step budget.
- **Ollama unreachable** — `agent_step` and `generate_response` invoke the model through
  `Runnable.with_retry()` (3 attempts, exponential backoff with jitter) and return a fixed
  `LLM_UNAVAILABLE_MESSAGE` if every attempt fails, instead of crashing the turn or fabricating an
  answer. `generate_response` recognises that sentinel on the incoming message and passes it through
  rather than overwriting it with its own "no evidence" message.
- **No tool artifact** — if a turn reaches `generate_response` without a single `ToolMessage`,
  it refuses to present a policy-dependent conclusion and says so explicitly, without calling the
  model at all.

`_bind_tools` also absorbs `bind_tools()` raising `NotImplementedError` (the dummy/`FakeListChatModel`
backend has no tool-calling support) by falling back to the plain model — so dummy mode still
completes every turn, just without ever calling a tool.

**Model input is filtered, not just guardrail counting.** `classify_intent`, `extract_information`,
`agent_step` and `generate_response` all send the model `MessageHistory.model_context()` rather than
the raw `state["messages"]`. For every completed previous request it keeps only that request's
`HumanMessage` and final (non-tool-calling) `AIMessage`, dropping the `ToolMessage`s and intermediate
tool-calling `AIMessage`s it produced; the current request (from the latest `HumanMessage` onward) is
kept in full, since the active tool loop needs every message in it. This keeps enough conversational
context to resolve references to earlier turns without letting the model treat a previous request's
tool evidence as current, and without the prompt growing with old tool payloads forever.
`MessageHistory.messages()` (unfiltered current-request slice) remains the input for guardrails,
duplicate-call detection, source collection and decision derivation — only the model-facing calls use
`model_context()`.

### Tools (`src/app/agent/calculator.py`, `src/app/agent/rule_checker.py`, `src/app/agent/deadline.py`, `src/app/agent/tools.py`)

- **`search_policies`** — reused from the RAG subgraph (task 3); the agent may pass a category
  explicitly, or defaults to the classifier's.
- **`calculate`** (`ReimbursementCalculator`) — category dispatch over the loaded `RuleCatalogue`:
  meal (`cap = limit_per_person × headcount`), travel (per-`expense_type` cap or submitted amount),
  mileage (`distance × (2 if one-way) × rate`), commuting (monthly distance × rate, capped at the
  flat monthly maximum — see feature 09), equipment (full amount), benefits
  (`min(amount, remaining budget) × reimbursement_ratio`). Missing required fields raise
  `CalculationInputError`, which `ToolNode` turns into an error `ToolMessage`.
  Its artifact is the compact `CalculationResult` from `src/app/agent/model.py`: `amount_huf`, optional
  `cap_huf`, `excess_huf`, and `warnings`.
- **`check_rules`** (`RuleChecker`) — prohibited-item reminder (meal), approval-tier check (against
  `submission.approval_tiers`), receipt presence, commuting minimum-distance eligibility, benefits
  annual-budget exhaustion and approval-above threshold, an explicit not-applicable finding for
  benefits tenure (not modelled by `ExpenseClaim`), and submission-deadline status (delegates to
  `DeadlineChecker`, which takes an explicit `reference_date` — never `date.today()` internally).
- Both `calculate` and `check_rules` read the current `claim` through LangGraph's injected
  `ToolRuntime.state` rather than accepting it as model-facing arguments — the model calls them with
  no arguments at all, so there is no second extraction-by-tool-call.

### Structured output (`src/app/agent/structured.py`)

`StructuredOutputRunner` composes `prompt | chat_model.with_structured_output(schema)`, tries it
once, retries once with the validation/parse error appended to the conversation, and returns a
caller-supplied typed fallback if both attempts fail — logging a warning each time. It also treats
`with_structured_output()` itself raising `NotImplementedError` (the dummy backend) as an immediate
fallback rather than a crash.

### Deriving the decision (`src/app/agent/nodes.py::AgentNodes._derive_decision`)

`generate_response` collects every `check_rules` `Finding` and calculation artifact from the current
request's `ToolMessage`s: any `fail` → `not_eligible`; otherwise a rule warning, calculation warning,
or positive `excess_huf` → `partially_eligible`; otherwise findings present → `eligible`. With no
rule findings, `decision=None` because a pure policy question has no eligibility decision.

### HTTP surface (`src/app/agent/service.py`, `src/app/api/routes/chat.py`, `src/app/api/schemas.py`)

`AgentService.respond(thread_id, message)` invokes the compiled graph (`recursion_limit=20`),
times the call, and projects the result into `ChatResponse` (`answer` = `messages[-1]`,
`generated_at` UTC, `response_time_ms`, the deterministic `decision`, deduplicated `sources` built from every current-request
`search_policies` `ToolMessage`'s `RagResult.citations`, and stable `steps` labels — "Request
understood", "Information extracted", then any of "Policies searched"/"Rules checked"/"Amount
calculated" that actually ran, then "Answer prepared"). `POST /chat` (`src/app/api/routes/chat.py`) is a
thin transport wrapper, running the synchronous call via `run_in_threadpool`.

### Wiring (`src/app/dependencies.py`)

`ApplicationDependencies.build(settings)` constructs everything once at startup: the chat model
bound at two temperatures (`0` for classification/extraction/tool-selection, `0.2` for the final
answer), the calculator and rule checker from the loaded `RuleCatalogue`, the RAG graph (built from
a `Retriever`), the three tools, `AgentNodes`, and the compiled agent graph wrapped in
`AgentService`. If Redis is unreachable, `build()` raises and FastAPI startup fails because grounded
policy retrieval and durable conversation state are required. `src/app/main.py`'s lifespan calls only
`ApplicationDependencies.build()`.

## How to use

```bash
uv run uvicorn app.main:app --port 8000
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo", "message": "What is the business meal limit per person?"}'
```

## Key files

| File | Responsibility |
| --- | --- |
| `src/app/agent/model.py` | `Intent`, `Decision`, `ExpenseClaim`, `CalculationResult`, `IntentClassification` |
| `src/app/agent/state.py` | `AgentState`, loop-budget constants |
| `src/app/agent/message_history.py` | `MessageHistory` — latest-request messages, guardrail derivations, and `model_context()` for filtered model input |
| `src/app/agent/slots.py` | `RequiredSlotTable` |
| `src/app/agent/structured.py` | `StructuredOutputRunner` |
| `src/app/agent/prompts.py` | the four embedded prompt templates |
| `src/app/agent/deadline.py` | `DeadlineChecker` |
| `src/app/agent/calculator.py` | `ReimbursementCalculator`, `CalculationInputError` |
| `src/app/agent/rule_checker.py` | `RuleChecker`, `Finding` |
| `src/app/agent/tools.py` | `build_calculate_tool`, `build_check_rules_tool`, `build_tools` |
| `src/app/agent/nodes.py` | `AgentNodes` — all custom nodes and routing functions |
| `src/app/agent/static_texts.py` | fixed clarification/refusal/system-state strings, kept deterministic mainly because the current chat model is small |
| `src/app/agent/graph.py` | `build_agent_graph` |
| `src/app/agent/service.py` | `AgentService` — graph output → `ChatResponse` |
| `src/app/api/routes/chat.py` | `POST /chat` |
| `src/app/api/schemas.py` | `ChatRequest`, `ChatResponse`, `ChatSource` (added to the existing file) |
| `src/app/dependencies.py` | `ApplicationDependencies` — all application wiring |
| `tests/fakes.py` | `ScriptedChatModel` — test double supporting `bind_tools`/`with_structured_output` |
| `src/app/agent/tests/test_graph.py` | full-graph journeys: happy path, unsupported, clarification, loop budget, duplicate reuse, tool-error disabling, no-artifact refusal, LLM-unavailable |
| `src/app/agent/tests/test_calculator.py`, `src/app/agent/tests/test_rule_checker.py`, `src/app/agent/tests/test_deadline.py` | per-category arithmetic and rule-check unit tests |
| `src/app/agent/tests/test_slots.py`, `src/app/agent/tests/test_message_history.py`, `src/app/agent/tests/test_nodes_model_context.py`, `src/app/agent/tests/test_structured.py`, `src/app/agent/tests/test_tools.py`, `src/app/agent/tests/test_service.py` | focused unit tests for each supporting module |

## Deliberate deviations from the technical design

- **`/chat` runs the graph synchronously inside `run_in_threadpool`**, not `graph.ainvoke`. The
  nodes and tools are plain synchronous Python; a full async conversion is deferred to task 9, which
  needs `graph.astream` regardless.
- **`calculate` and `check_rules` take no model-facing arguments** — both read `claim` from
  `ToolRuntime.state`, avoiding a second extraction-by-tool-call for either.
- **Travel expense sub-types are selected via `expense_type` string values**
  (`accommodation_domestic`/`accommodation_international`/`meal_per_diem_domestic`/
  `meal_per_diem_international`), since `ExpenseClaim`'s documented fields have no explicit
  domestic/international flag. Commuting's public-transport pass/ticket rules (`R-COMM-03`/
  `R-COMM-04`) are not yet implemented — the required-slot table only covers the personal-vehicle
  case; task 7 owns the pass/ticket variant.
- **The compiled graph defaults to LangGraph's in-memory checkpointer**, not a Redis-backed
  `RedisSaver`. Durable cross-restart thread persistence and the thread-reset endpoint are grouped
  under "thread controls" in task 9, not task 4.
- **Benefits tenure eligibility is not verified** — `ExpenseClaim` has no employment-start-date
  field; `RuleChecker` reports an explicit `not_applicable` finding instead.
