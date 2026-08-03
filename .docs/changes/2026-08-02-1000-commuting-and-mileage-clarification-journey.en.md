# 2026-08-02 10:00 UTC — Commuting and mileage clarification journey

## What changed

- `app/agent/calculator.py`: `_calculate_commuting` now dispatches on `expense_type` to
  `_calculate_commuting_pass` (`R-COMM-03`: 80% of the price, 30,000 HUF monthly cap),
  `_calculate_commuting_ticket` (`R-COMM-04`: 80% of the spend, 3,000 HUF × office days) or
  `_calculate_commuting_vehicle`. Removed the hardcoded `HYBRID_WORK_FULL_TIME_DAYS = 20` constant
  and the monthly-cap pro-rating it drove. Added `calculate_both_directions` and restored the
  general `_first_rule_with` catalogue lookup.
- `app/agent/slots.py`: commuting resolves its required slots in two steps — the base entry asks for
  `expense_type`, then `_commuting_mode_slots` appends the declared mode's own slots. Added
  `("expense_check", "mileage")` and `("expense_check", "commuting")` entries.
- `app/agent/model.py`: added `ExpenseClaim.from_state`, accepting a model, a plain dict, or
  LangChain's serialized-constructor envelope.
- `app/agent/nodes.py`: every read of `state["claim"]` goes through `ExpenseClaim.from_state`;
  `AgentNodes` takes a `ReimbursementCalculator`; `ask_clarification` answers with
  `_conditional_distance_answer` instead of repeating a question the thread already asked.
- `app/agent/current_request.py`: added `was_already_asked`.
- `app/agent/messages.py`: added `CONDITIONAL_DISTANCE_ANSWER`.
- `app/agent/rule_checker.py`: the minimum-distance rule now reports `not_applicable` for `pass`
  and `ticket` claims instead of being silently skipped.
- `app/agent/tools.py`: both tools coerce the runtime claim through `ExpenseClaim.from_state`.
- Added `app/integrations/checkpointer.py` (`RedisSaver`, 24 h TTL, `checkpoint:*` namespace), wired
  through `ApplicationDependencies.checkpointer` + `get_checkpointer`, and added
  `DELETE /threads/{thread_id}` with `ThreadResetResponse`.
- Added `langgraph-checkpoint-redis==0.5.1`.
- Tests: new `tests/test_commuting_and_mileage_journeys.py`,
  `tests/test_checkpointer_integration.py`, `tests/test_commuting_rule_document_consistency.py`;
  extended `tests/test_calculator.py`, `tests/test_slots.py`, `tests/test_api.py`.
- Added `.docs/features/09-commuting-and-mileage-clarification-journey.en.md`.

## Why

Task 7 (`07-commuting-and-mileage-clarification-journey.md`). Three real defects surfaced:

- **The monthly commuting cap was pro-rated by an invented constant.** `HYBRID_WORK_FULL_TIME_DAYS`
  existed only in Python — not in `rules.yaml`, not in the corpus. Doc 03 §4 states a flat
  40,000 HUF monthly maximum, so the pro-rating under-paid every hybrid worker. Hybrid work is
  already handled the way the policy describes it (§5), through `commute_days_per_month` scaling the
  distance. The doc 03 §7 worked example is now a regression test.
- **`R-COMM-03`/`R-COMM-04` were dead catalogue data.** A pass or ticket claim fell through to the
  personal-vehicle branch and demanded a distance it does not have.
- **A Redis checkpoint restored `ExpenseClaim` as a plain dict.** `RedisSaver` writes channel values
  as JSON and does not revive LangChain's constructor envelope, so a resumed turn would have raised
  from `merged_with()` and reported every slot as missing (`getattr` on a dict) — an endless
  clarification loop. This is invisible under `InMemorySaver`, so it only appeared once the
  Redis-backed integration test existed.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` — all clean
(225 passed), no regressions in tasks 1-6 or 8.
