# 2026-08-01 07:37 UTC — First policy-question agent journey

## What changed

- Added `app/agent/`: the seven-node main agent graph.
  - `model.py`: `Intent`, `Decision`, `ExpenseClaim` (all documented incremental fields, plus
    `merged_with()` for clarification-turn slot merging), compact `CalculationResult`
    (`amount_huf`, `cap_huf`, `excess_huf`, `warnings`), and `IntentClassification`.
  - `state.py`: `AgentState` (`messages`, `intent`, `category`, `claim`, `decision`),
    `MAX_AGENT_STEPS` (4), `RECURSION_LIMIT` (20), `MAX_TOOL_ARG_ERRORS` (2).
  - `current_request.py`: `CurrentRequest` — `messages()` (the slice from the latest
    `HumanMessage`), `agent_step_count()`, `tool_error_count(name)`, `find_duplicate_call(name, args)`.
  - `slots.py`: `RequiredSlotTable` — the (intent, category) → required-fields lookup (design §6.2).
  - `structured.py`: `StructuredOutputRunner` — one repair retry, then a typed fallback; also
    absorbs `with_structured_output`/`bind_tools` `NotImplementedError` from the dummy backend.
  - `prompts.py`: the four embedded prompt templates (classify-intent, extract-information,
    agent-step, generate-response).
  - `deadline.py`: `DeadlineChecker` (explicit `reference_date`, never `date.today()` internally).
  - `calculator.py`: `ReimbursementCalculator` — per-category formulas (meal, travel, mileage,
    commuting, equipment, benefits) against the loaded `RuleCatalogue`; `CalculationInputError` for
    missing fields. The calculator returns only values consumed by the agent and evaluation instead
    of duplicating formula lines, per-person limits, or applied rule identifiers.
  - `rule_checker.py`: `RuleChecker` — prohibited-item reminder, approval-tier check, receipt check,
    commuting minimum-distance, benefits budget/tenure, submission deadline (delegates to
    `DeadlineChecker`).
  - `tools.py`: `build_calculate_tool`, `build_check_rules_tool` (both read the claim from
    `ToolRuntime.state`, no model-facing arguments), `build_tools()` assembling all three tools
    alongside `app.rag.tool.build_search_policies_tool`.
  - `nodes.py`: `AgentNodes` — `classify_intent`, `extract_information`, `ask_clarification`,
    `agent_step`, `generate_response`, `out_of_scope`, plus the
    `route_after_extraction`/`route_after_agent` conditional-edge functions. `execute_tools` is
    LangGraph's prebuilt `ToolNode`.
  - `graph.py`: `build_agent_graph(nodes, checkpointer=None)` compiles the seven-node graph
    (defaults to an in-memory checkpointer; a durable one is a later task's concern).
  - `service.py`: `AgentService.run_turn(thread_id, message)` — the one place graph
    messages/artifacts become the public `TurnResponse` (deduplicated sources, stable step labels).
- Added `app/api/routes/chat.py`: blocking `POST /chat`, registered in `app/api/router.py`.
- Added `TurnSource`, `TurnResponse`, `ChatRequest` to `app/api/schemas.py`.
- Consolidated all application wiring into `app/dependencies.py`'s `ApplicationDependencies`
  dataclass (`.build(settings)`/`.from_request(request)`), replacing the construction that used to
  live in `app/main.py`'s lifespan; `app/main.py` now only calls `ApplicationDependencies.build()`.
- Added `app/rag/retriever.py::NullPolicyRetriever` — used when Redis is unreachable at startup so
  the agent graph still constructs and `search_policies` degrades to "no results" instead of the
  app failing to start.
- Tests: `tests/test_calculator.py`, `tests/test_rule_checker.py`, `tests/test_deadline.py`,
  `tests/test_slots.py`, `tests/test_current_request.py`, `tests/test_structured.py`, `tests/test_tools.py`,
  `tests/test_agent_graph.py`, `tests/test_agent_service.py`, `tests/test_dependencies.py`, plus a
  `/chat` smoke test in `tests/test_api.py`. `tests/fakes.py` adds `ScriptedChatModel`, a test-only
  chat model whose `bind_tools`/`with_structured_output` are deterministic (dummy/fake LangChain
  chat models raise `NotImplementedError` for both, so a custom double was needed to exercise the
  ReAct loop, guardrails and structured-output paths without a real Ollama server).

## Why

Task 4 (`04-first-policy-question-agent-journey.md`). The graph, state and tools are built for real
(not stubbed), since tasks 5-8 layer category-specific correctness onto the same
calculator/rule-checker rather than rewriting them; task 4's own acceptance surface is the
policy-question and unsupported-refusal journeys plus the surrounding guardrails.

Two gaps found via a first full self-review against the task checklist and fixed before considering
the task done:

- **Ollama unreachable → clear failure, not a fabricated answer.** `agent_step` and
  `generate_response` now call the model through `Runnable.with_retry()` (3 attempts, exponential
  backoff with jitter — "retry twice" after the first attempt) and return a fixed
  `LLM_UNAVAILABLE_MESSAGE` if every attempt fails, rather than letting the exception crash the
  turn. `generate_response` recognises that sentinel on the incoming message and passes it through
  unchanged instead of overwriting it with its own "no evidence" message.
- **Exhausted tool budget → the answer must say evidence is incomplete.** `generate_response` checks
  whether `agent_step_count()` already reached `MAX_AGENT_STEPS` and, if so, deterministically
  appends a fixed note to the generated answer — this doesn't rely on the prompt instruction alone.
- **Repeated identical tool call → reuse is now logged.** `agent_step` logs a warning when it
  substitutes a duplicate call's previous result instead of re-executing it (design §6.3's
  "record a warning").

## Deliberate deviations from the technical design

- **`/chat` is synchronous under the hood**, wrapped in `run_in_threadpool` (matching the existing
  `/admin/ingest`/`/admin/stats` pattern) rather than using `graph.ainvoke`. The graph's nodes and
  tools are plain synchronous Python; converting the whole call chain to async is deferred to task 9
  (`streaming-chat-experience`), which needs `graph.astream` regardless.
- **`calculate` and `check_rules` both read the claim from `ToolRuntime.state`**, with no
  model-facing arguments, rather than `check_rules` also accepting claim fields + rule ids as tool
  arguments as one design passage suggested — this avoids a second extraction-by-tool-call for
  `check_rules` too, consistent with `calculate`'s stated rationale.
- **Travel's per-night/per-diem caps are selected via `expense_type` values
  (`accommodation_domestic`, `accommodation_international`, `meal_per_diem_domestic`,
  `meal_per_diem_international`)** rather than a separate domestic/international claim field, since
  `ExpenseClaim`'s documented field list has no such field. Commuting's pass/ticket rules
  (`R-COMM-03`/`R-COMM-04`) are not yet implemented in the calculator — the required-slot table for
  `calculation`/`commuting` only asks for the personal-vehicle fields, and the pass/ticket variant is
  left for task 7 (`commuting-and-mileage-clarification-journey`) to define precisely.
- **The main graph's checkpointer defaults to LangGraph's `InMemorySaver`**, not the design's
  Redis-backed `RedisSaver`. Durable, cross-restart thread persistence and the `DELETE
  /threads/{id}` reset endpoint are grouped with "thread controls" in task 9, not task 4's checklist.
- **Benefits tenure eligibility (`eligible_after_months`) is not verified** — `ExpenseClaim` has no
  employment-start-date field. `RuleChecker` emits an explicit `not_applicable` finding rather than
  silently skipping the check.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`,
`pytest --cov=app --cov-report=term-missing` — all clean (127 passed, 97% coverage). Manual smoke
test: `POST /chat` against a running app (dummy LLM backend, real local Redis) returns a well-formed
`TurnResponse`; against a `ScriptedChatModel` simulating real tool-calling, the full
classify→extract→check→agent_step→execute_tools→generate_response path produces a grounded, cited
answer, and an unsupported question is refused before any tool is ever called.
