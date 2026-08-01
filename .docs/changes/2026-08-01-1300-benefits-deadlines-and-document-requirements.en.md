# 2026-08-01 13:00 UTC — Benefits, deadlines, and document requirements

## What changed

- `app/agent/model.py`: added tenure and provided-document facts; moved the Pydantic
  `Finding` schema beside the other agent schemas.
- `app/agent/slots.py`: `("expense_check", "benefits")` now requires `tenure_months`.
- `app/agent/rule_checker.py`: split rule checks into focused collaborators. Dedicated
  `R-BEN-TENURE` and `R-BEN-CARRY-OVER` catalogue rules provide stable IDs and references;
  document checks now compare receipt presence and category attachments, and deadline-only requests work
  without a category.
- Tests: `tests/test_calculator.py` (+2: unused and exhausted benefit budgets),
  `tests/test_deadline.py` (+3: day 29/30/31 boundary), `tests/test_rule_checker.py` (tenure
  missing/below/at-threshold, carry-over surfaced; replaced the old not-applicable test),
  `tests/test_slots.py` (updated for the new required `tenure_months` slot), new
  `tests/test_benefits_deadline_document_journeys.py` (a full benefit-amount journey; a
  compiled-graph benefit, deadline-only and document-only journeys proving the agent calls only
  `search_policies`+`check_rules`, never `calculate`), new
  `tests/test_benefits_rule_document_consistency.py` (benefit rules' annual
  allowances, six-month tenure requirement, and no-carry-over rule traced verbatim to the real
  policy document).
- Added `.docs/features/08-benefits-deadlines-and-document-requirements.en.md`.

## Why

Task 8 (`08-benefits-deadlines-and-document-requirements.md`). `ExpenseClaim` had no tenure field,
so benefit eligibility could never actually be checked — every benefits claim got a hardcoded
`not_applicable` regardless of the facts. The task explicitly asks to model tenure and distinguish
insufficient information from ineligibility, and to surface the no-carry-over rule as an explicit
finding rather than leaving it implied. The deadline calculation itself was already correct from
task 4; this task adds boundary tests and compiled-graph proofs of the short tool paths for
deadline questions.

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` — all clean
after the follow-up review (177 passed, 21 skipped).
