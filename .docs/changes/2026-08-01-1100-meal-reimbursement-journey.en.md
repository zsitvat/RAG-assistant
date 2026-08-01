# 2026-08-01 11:00 UTC — Meal reimbursement journey

## What changed

- `app/agent/calculator.py`: `_calculate_meal` now adds a warning describing the excluded-item
  amount deducted before the per-person cap, distinct from `excess_huf` (which is only the amount
  still over the cap). Added `_meal_limit_rule()` so a missing catalogue limit returns a
  lower-confidence result with `cap_huf=None` instead of inventing a cap or leaking
  `StopIteration`.
- `app/agent/rule_checker.py`: the focused `DocumentChecker` now emits a source-linked,
  category-specific required-document finding. Expense checks compare `provided_documents` with
  the configured list; document questions list requirements without lowering eligibility.
- `config/rules.yaml`: added the missing `"personal consumption"` excluded item to `R-MEAL-02`,
  matching the actual policy document text.
- Tests: `tests/test_calculator.py` (+5: below-cap, exactly-at-cap, excluded-item warning, half-up
  rounding, missing-catalogue-limit), `tests/test_rule_checker.py` (+1: required-documents finding),
  `tests/test_tools.py` (fixed an index-based assertion that broke once required-documents became
  the first finding), new `tests/test_meal_reimbursement_journey.py` (scripted compiled-graph
  integration journey for the reference dinner), new
  `tests/test_meal_rule_document_consistency.py` (meal rules' `doc_ref`s resolve to indexed
  sections; the per-person limit and excluded items appear verbatim in the referenced document).
- Added `.docs/features/06-meal-reimbursement-journey.en.md`.

## Why

Task 5 (`05-meal-reimbursement-journey.md`). An audit against its checklist found the meal
calculation and rule-checker mostly already worked (built as part of task 4's general machinery),
but four gaps: excluded items weren't distinguished from cap-excess anywhere in the output, a
missing catalogue limit would crash instead of failing cleanly, required documents were declared in
`rules.yaml` but never surfaced to the user, and there was no compiled-graph integration test proving the full
  meal graph journey (amount, cap, excess, decision, findings, citations) together, nor a test proving the
applied numbers trace back to the real policy text.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` — all clean
after the follow-up review (177 passed, 21 skipped).
