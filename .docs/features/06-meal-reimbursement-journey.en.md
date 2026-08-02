# Feature: Meal reimbursement journey

Implements task [`05-meal-reimbursement-journey.md`](../tasks/05-meal-reimbursement-journey.md).

## What it does

Tightens the meal category's reimbursement calculation and rule-checking to the task's exact
requirements, on top of the general agent machinery already built for task 4 (see
[`05-first-policy-question-agent-journey.en.md`](05-first-policy-question-agent-journey.en.md)):
the calculator, rule checker, tools, and ReAct loop are unchanged in shape — this task hardens the
meal-specific arithmetic, adds a required-documents finding shared by every category, and proves
both against the real policy corpus and a full scripted conversation.

## How it works

### Meal calculation (`src/app/agent/calculator.py::ReimbursementCalculator._calculate_meal`)

Unchanged formula (`min(amount_huf - non_reimbursable_amount, limit_per_person_huf * headcount)`,
half-up rounding), with two additions:

- **Excluded items are now distinguished from the cap-excess amount.** `CalculationResult` still
  holds only `amount_huf`, `cap_huf`, `excess_huf`, and `warnings` (per the task's exact schema), so
  the distinction is carried in `warnings`: when `non_reimbursable_amount > 0`, a warning states the
  excluded amount that was deducted before the cap was applied, separate from `excess_huf` (which
  reports only the amount still over the cap after that deduction).
- **A missing catalogue limit produces a lower-confidence result.** `_meal_limit_rule()` is an
  explicit, statically inspectable lookup. When no limit is configured, the calculator returns the
  eligible submitted amount with `cap_huf=None` and a warning instead of inventing a number or
  raising an unhandled `StopIteration`.

### Required documents (`src/app/agent/rule_checker.py::RuleChecker._check_required_documents`)

Every category in `config/rules.yaml` declares `required_documents` (e.g. meal:
`[invoice, business_purpose_note, participant_list]`), but nothing read that list before this task.
`DocumentChecker.check_requirements` emits a category-specific, source-linked finding. For a
document question it lists the requirements without lowering eligibility. For an expense check it
compares `provided_documents` with the catalogue: a complete set passes and only missing or unknown
documents warn. This prevents an unconditional reminder from making every complete claim
`partially_eligible`.

### Data fix (`config/rules.yaml`)

`R-MEAL-02`'s `excluded_items` was missing `"personal consumption"`, which the actual policy text
(`.docs/sources/en/01_General_Expense_Reimbursement_Policy.docx`, §4) lists alongside alcohol,
tobacco, and tips. Added so the rule-checker's warning message and the new consistency test both
match the source document.

## How to use

Nothing new is exposed at the API surface — this is purely calculation/rule-check correctness
inside the existing `POST /chat` journey from task 4. A meal expense_check request still flows
`classify_intent → extract_information → agent_step ⇄ execute_tools → generate_response`.

## Key files

| File | Responsibility |
| --- | --- |
| `src/app/agent/calculator.py` | `_calculate_meal`, `_meal_limit_rule` |
| `src/app/agent/rule_checker.py` | focused document, approval, eligibility and deadline checkers |
| `config/rules.yaml` | `R-MEAL-02` excluded-items fix |
| `src/app/agent/tests/test_calculator.py` | below-cap, exactly-at-cap, excluded-item warning, half-up rounding, missing-catalogue-limit |
| `src/app/agent/tests/test_rule_checker.py` | required-documents finding |
| `tests/journeys/test_meal_reimbursement_journey.py` | compiled-graph integration journey for the scripted "reference dinner" request |
| `tests/journeys/test_meal_rule_document_consistency.py` | proves every meal rule's `doc_ref` resolves to an indexed section, and that the per-person limit and excluded items appear verbatim in the referenced policy document |
