# Feature: Commuting and mileage clarification journey

Implements task
[`07-commuting-and-mileage-clarification-journey.md`](../tasks/07-commuting-and-mileage-clarification-journey.md).

## What it does

Completes the commuting category (public-transport pass and individual ticket alongside personal
vehicle), moves conversation state onto a Redis-backed LangGraph checkpointer with a 24-hour expiry
and a thread-reset endpoint, and makes an unanswerable distance question terminate with both
conditional outcomes instead of asking forever.

## How it works

### Commuting transport modes (`app/agent/calculator.py`)

`_calculate_commuting` now dispatches on `expense_type` to three private calculations:

| Mode | `expense_type` | Formula | Catalogue rule |
| --- | --- | --- | --- |
| Personal vehicle | anything else (e.g. `own_car`) | `min(one-way×2 × office days × rate, monthly cap)` | `R-COMM-02` |
| Monthly/30-day pass | `pass` | `min(price × 80%, 30,000)` | `R-COMM-03` |
| Individual ticket | `ticket` | `min(spend × 80%, 3,000 × office days)` | `R-COMM-04` |

Mileage (`R-MILE-01`, 45 HUF/km) is unchanged and already powertrain-independent — doc 04 states
the same rate applies to every powertrain, so an EV claim needs no separate branch.

**Corrected: the monthly cap is no longer pro-rated.** The previous implementation multiplied the
40,000 HUF monthly maximum by `min(office_days, 20) / 20` using a hardcoded
`HYBRID_WORK_FULL_TIME_DAYS = 20` Python constant that appears in neither `rules.yaml` nor the
corpus. Doc 03 §4 states a flat "Monthly maximum HUF 40,000", so pro-rating the cap silently
under-paid hybrid workers. Hybrid work is already accounted for the way the policy actually
describes it (§5: "Personal-vehicle reimbursement may be calculated only for actual office days") —
`commute_days_per_month` scales the distance, not the cap. The doc 03 §7 worked example (18 km,
10 office days → 10,800 HUF) is now a regression test.

### Mode-aware required slots (`app/agent/slots.py`)

`RequiredSlotTable` is keyed by `(intent, category)`, which cannot express "a pass claim needs a
price but a car claim needs a distance". Commuting now takes a second lookup step: the base entry
requires `expense_type` (the transport-mode declaration doc 03 §2 asks for), and
`_commuting_mode_slots` appends the mode's own slots once it is known. An unrecognised mode falls
back to the personal-vehicle slots.

### Redis-backed conversation state (`app/integrations/checkpointer.py`)

`build_checkpointer(redis_url)` returns a `RedisSaver` (from `langgraph-checkpoint-redis`) with a
`default_ttl` of 1,440 minutes and `refresh_on_read`, writing under the `checkpoint:*` /
`checkpoint_write:*` prefixes — distinct from the corpus's `chunk:*` and `build_info:corpus` keys.
`ApplicationDependencies` holds it so both the agent graph and the reset endpoint share one
instance. Application startup fails when Redis is unreachable; only isolated graph tests compile
with the in-memory default.

**A Redis round-trip restores `ExpenseClaim` as a plain dict, not a model.** `RedisSaver` stores
channel values as JSON and does not revive LangChain's `{"lc": 2, "type": "constructor", ...}`
envelope on load, so `state["claim"]` is a `dict` after any resumed turn — which would have made
`merged_with()` raise and `RequiredSlotTable.missing()` (a `getattr` scan) report every slot as
missing, i.e. an endless clarification loop. `ExpenseClaim.from_state()` is the single coercion
point that accepts a model, a plain dict, or the serialized envelope; every read of `state["claim"]`
in `nodes.py` and `tools.py` goes through it. This only reproduces against real Redis, which is why
`app/integrations/tests/test_checkpointer_integration.py` exists separately from the in-memory graph tests.

### Thread reset (`app/api/routes/chat.py`)

`DELETE /threads/{thread_id}` calls `checkpointer.delete_thread(...)` in a worker thread and returns
`ThreadResetResponse`. The next message on that thread starts with empty state.

### Refusing to disambiguate a distance (`app/agent/nodes.py`)

`CurrentRequest.was_already_asked(question)` looks for the same fixed clarification string earlier in
the thread. When `ask_clarification` is about to repeat itself and `distance_is_one_way` is the only
remaining gap, `_conditional_distance_answer` calls
`ReimbursementCalculator.calculate_both_directions()` and answers with both readings
(`CONDITIONAL_DISTANCE_ANSWER`) rather than looping. `decision` stays `needs_info` so a later answer
can still resolve it, but the turn ends with a useful amount either way.

## Key files

| File | Responsibility |
| --- | --- |
| `app/agent/calculator.py` | commuting mode dispatch, pass/ticket/vehicle formulas, `calculate_both_directions` |
| `app/agent/slots.py` | mode-aware commuting slot resolution |
| `app/agent/model.py` | `ExpenseClaim.from_state` checkpoint coercion |
| `app/agent/current_request.py` | `was_already_asked` |
| `app/agent/nodes.py` | conditional-outcome path, claim coercion at every state read |
| `app/agent/rule_checker.py` | minimum distance reported `not_applicable` for pass/ticket claims |
| `app/integrations/checkpointer.py` | `RedisSaver` with the 24 h TTL and `checkpoint:*` namespace |
| `app/api/routes/chat.py` | `DELETE /threads/{thread_id}` |
| `tests/journeys/test_commuting_and_mileage_journeys.py` | clarification, two-turn resume, category switch, refusal |
| `app/integrations/tests/test_checkpointer_integration.py` | restart persistence, TTL/namespace, reset, two workers on one thread |
| `tests/journeys/test_commuting_rule_document_consistency.py` | every commuting/mileage number traced verbatim to the corpus |
