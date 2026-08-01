# Feature: Benefits, deadlines, and document requirements

Implements task
[`08-benefits-deadlines-and-document-requirements.md`](../tasks/08-benefits-deadlines-and-document-requirements.md).

## What it does

Models employee tenure explicitly so benefit eligibility can actually be checked instead of always
reporting "not applicable", surfaces the no-carry-over rule as an explicit finding, and locks in the
day-30 submission-deadline boundary and the agent's ability to take a short tool path (search +
check_rules, no calculate) for deadline and document questions. No new tools, no new calculator
categories — this task is entirely about the rule checker and the claim model.

## How it works

### Tenure (`app/agent/model.py`, `app/agent/rule_checker.py::RuleChecker._check_tenure`)

`ExpenseClaim.tenure_months: int | None` is new. `RequiredSlotTable` now requires it for
`("expense_check", "benefits")` alongside `expense_type`/`amount_huf`/`annual_budget_used_huf`, so
a benefits claim without it routes to `ask_clarification` instead of silently skipping the check.
The source-linked `R-BEN-TENURE` rule distinguishes three outcomes using its
`eligible_after_months` value (6 months for all three benefit types, per
`.docs/sources/en/05_Employee_and_Recreational_Benefits.docx` §2: "A benefit may be claimed after at
least six months of continuous employment"):

- `tenure_months` missing → `warning` ("insufficient information", not a rejection)
- `tenure_months < eligible_after_months` → `fail` (ineligible)
- otherwise → `pass`

This replaces the old `_tenure_not_tracked_finding()`, which unconditionally returned
`not_applicable` regardless of whether tenure was knowable — it no longer needs to exist as a
concept now that the claim can actually carry the fact.

### No carry-over (`RuleChecker._check_carry_over`)

The dedicated `R-BEN-CARRY-OVER` rule stores `carry_over: false` with its eligibility-section
reference (§2: "An unused annual allowance may not be carried forward to the next year"). Its
`pass` finding states the applied policy explicitly without duplicating an ID or `doc_ref` in Python.

### Deadline boundary (`app/agent/deadline.py`, unchanged; tests added)

`DeadlineChecker` was already correct; this task adds explicit day-29/30/31 tests locking in that
day 30 (the last day within the 30-day window) is `due_soon` (a warning, not a rejection — it is
within the window, but close to the edge), and day 31 is the first `expired` day.

### Short tool path for deadline/document questions

`agent_step`'s ReAct loop lets the model call only the relevant subset of tools. Separate journey
tests prove both `deadline_check` and `document_requirements` use `search_policies` + `check_rules`
without `calculate`. Deadline checking no longer requires an expense category, and document
questions list source-linked requirements without treating an informational list as an eligibility
warning.

### Receipt and attachment facts

`ExpenseClaim` records `provided_documents` alongside `has_receipt`. `DocumentChecker` distinguishes
an absent receipt, an incomplete attachment set, and a complete set. Specific document types are
represented once in `provided_documents`, avoiding a second overlapping receipt field. Category
requirement IDs and references live in `rules.yaml`.

## Deliberate deviations

- **Benefits' non-eligible sub-items (alcohol, minibar, another adult's cost, etc., per doc 05 §4)
  are not modeled or deducted.** The task's calculation checklist item lists exactly five factors —
  annual budget, used budget, remaining amount, requested amount, no-carry-over — and doesn't ask
  for this; `non_reimbursable_amount`-style subtraction for benefits (mirroring meal's pattern) is
  left as a possible future refinement, not implemented here to stay in scope.

## Key files

| File | Responsibility |
| --- | --- |
| `app/agent/model.py` | tenure, provided documents, and source-linked `Finding` schema |
| `app/agent/slots.py` | benefits now requires `tenure_months` |
| `app/agent/rule_checker.py` | focused eligibility, document, approval, and deadline checkers |
| `tests/test_calculator.py` | unused and exhausted benefit budget |
| `tests/test_deadline.py` | day 29/30/31 boundary |
| `tests/test_rule_checker.py` | tenure warning/fail/pass, carry-over surfaced |
| `tests/test_slots.py` | benefits required-slot coverage including tenure |
| `tests/test_benefits_deadline_document_journeys.py` | benefit, deadline-only, and document-only graph journeys |
| `tests/test_benefits_rule_document_consistency.py` | benefit rule `doc_ref`s resolve; annual allowances, the six-month tenure rule, and the no-carry-over rule appear verbatim in the referenced policy document |
