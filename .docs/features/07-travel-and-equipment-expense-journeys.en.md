# Feature: Travel and equipment expense journeys

Implements task
[`06-travel-and-equipment-expense-journeys.md`](../tasks/06-travel-and-equipment-expense-journeys.md).

## What it does

Extends the calculator and rule checker (unchanged interfaces, no new model-facing tools) so
accommodation, taxi, parking, and equipment claims get correct deterministic amounts, a
travel-specific approval rule that the generic submission tiers could not express, explicit
rejection of prohibited expense types, and a build-time gate against a category with no retrievable
policy evidence at all.

## How it works

### Travel-specific approval (`src/app/agent/rule_checker.py::ApprovalChecker`)

The actual policy (`.docs/sources/en/02_Business_Travel_and_Accommodation_Policy.docx`, §1) reads:
"An international trip or an estimated total cost above HUF 150,000 also requires approval from the
department head" — a threshold-OR-international rule that the generic `submission.approval_tiers`
(50,000 / 150,000 / unlimited) cannot represent, since it neither uses those exact tiers nor reacts
to trip country. `RuleChecker._check_approval` now dispatches: `category == "travel"` uses
`ApprovalChecker` uses `R-TRAVEL-01.department_head_approval_above_huf` together with the explicit
`claim.is_international_trip` fact. Domestic trips require line-manager approval; international
trips or amounts above HUF 150,000 require department-head approval. A missing approval warns and an
explicitly absent approval fails, including below the threshold. Equipment reads its own
`R-EQUIP-01.approval_tiers` instead of relying on a coincidentally identical global table.

### Prohibited expense types (`RuleChecker._check_prohibited_expense_type`)

Distinct from meal's excluded-amount handling, a `travel` claim whose `expense_type` matches a
category rule's `excluded_items` is rejected outright. Explicit `is_business_related=False` also
fails travel or equipment without overwriting the real expense subtype with the word `personal`.
Two rules back this: `R-TRAVEL-04` (`excluded_items: [fine, minibar]`, doc 01 §6
"Non-reimbursable items" — a newly added section anchor) and `R-EQUIP-02`
(`business_use_required: true`, doc 01 §5, same section as `R-EQUIP-01`). The calculator is
unaffected — per the design's calculation/decision split, `_calculate_travel`/`_calculate_equipment`
still return the submitted amount; only the rule-checker `Finding` (and the decision it feeds into)
communicates the rejection. `EXTRACT_INFORMATION_SYSTEM` was extended to normalize these into
`expense_type` values (`"fine"`, `"minibar"`) while preserving business purpose and trip scope in
their own typed claim fields.

### Required documents

Already generic since task 5's `_check_required_documents` reads `required_documents` for whichever
category is on the claim — travel and equipment needed no rule-checker change to get this.

### Category-coverage consistency gate (`src/app/rag/rule_metadata.py::RuleMetadataResolver`)

`validate_categories_reachable(chunks)` — called from
`CorpusIngestor.load_and_chunk` right after the existing `validate_anchors_resolve` — raises
`IngestionError` if any `Category` in the rule catalogue has zero chunks carrying it in
`metadata["categories"]`, and if a configured rule reference cannot be retrieved under its own
category. Referenced sections inherit the rule's category in addition to document-level categories,
so `R-TRAVEL-04` remains reachable through a travel filter even though it lives in document 01.

## How to use

No new API surface — this refines the existing `POST /chat` journey (task 4) for the `travel` and
`equipment` categories.

## Key files

| File | Responsibility |
| --- | --- |
| `src/app/agent/rule_checker.py` | focused `ApprovalChecker` and `EligibilityChecker` collaborators |
| `src/app/agent/prompts.py` | `EXTRACT_INFORMATION_SYSTEM` — taxi/parking/fine/minibar/personal expense_type normalization |
| `src/rules_config/rules.yaml` | `R-TRAVEL-04`, `R-EQUIP-02`, new `01#non-reimbursable-items` section anchor |
| `src/app/rag/rule_metadata.py` | `validate_categories_reachable` |
| `src/app/rag/ingest.py` | wires the new validation into `load_and_chunk` |
| `src/app/agent/tests/test_calculator.py` | accommodation international, meal per-diem domestic/international, parking |
| `src/app/agent/tests/test_rule_checker.py` | approval boundary tests, travel approval (domestic/international), prohibited expense types |
| `src/app/agent/tests/test_slots.py` | travel/equipment required-slot coverage |
| `src/app/rag/tests/test_ingest.py` | `validate_categories_reachable` pass/fail cases |
| `tests/journeys/test_travel_and_equipment_journeys.py` | graph journeys plus a full domestic-accommodation request through `POST /chat` |
| `tests/journeys/test_travel_equipment_rule_document_consistency.py` | every travel/equipment rule's `doc_ref` resolves, and its numeric values/excluded items appear verbatim in the referenced policy document |
