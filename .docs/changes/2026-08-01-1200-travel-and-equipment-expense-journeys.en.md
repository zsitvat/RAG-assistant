# 2026-08-01 12:00 UTC — Travel and equipment expense journeys

## What changed

- `app/agent/rule_checker.py`: `ApprovalChecker` reads explicit trip scope and category approval
  rules; `EligibilityChecker` rejects prohibited types and explicit non-business use without
  encoding business purpose in `expense_type`.
- `config/rules.yaml`: added `R-TRAVEL-04` (`excluded_items: [fine, minibar]`, business use required)
  and `R-EQUIP-02` (business use required); added a new `01#non-reimbursable-items` section
  anchor (doc 01 §6) that `R-TRAVEL-04` references.
- `app/agent/prompts.py`: `EXTRACT_INFORMATION_SYSTEM` now normalizes `"taxi"`, `"parking"`,
  `"fine"` and `"minibar"` while extracting business use and international trip scope into separate
  fields.
- `app/rag/rule_metadata.py`: added `RuleMetadataResolver.validate_categories_reachable(chunks)`,
  raising `IngestionError` if any category or configured rule has no retrievable indexed evidence.
  Referenced sections inherit the rule category, including cross-document references. Wired into
  `PolicyCorpusIngestor.load_and_chunk` (`app/rag/ingest.py`) after `validate_anchors_resolve`.
- Tests: extended `tests/test_calculator.py` (accommodation international, meal per-diem
  domestic/international, parking), `tests/test_rule_checker.py` (approval boundary tests at the
  base tier, travel approval for domestic/international trips, prohibited expense-type fails for
  travel and equipment), `tests/test_slots.py` (travel/equipment required slots),
  `tests/test_ingest.py` (`validate_categories_reachable` pass/fail); fixed
  `tests/test_tools.py::test_check_rules_tool_reads_the_claim_from_state_and_returns_findings`,
  which asserted on `artifact[0]` positionally and broke once `_check_required_documents` (task 5)
  became the first finding. New `tests/test_travel_and_equipment_journeys.py` (compiled-graph
  journeys plus a domestic accommodation request through `POST /chat`) and
  `tests/test_travel_equipment_rule_document_consistency.py` (rule numeric values and excluded
  items traced verbatim to the real policy documents).
- Added `.docs/features/07-travel-and-equipment-expense-journeys.en.md`.

## Why

Task 6 (`06-travel-and-equipment-expense-journeys.md`). An audit found the calculator's travel/
equipment arithmetic already correct from task 4, but the rule checker had no travel-specific
approval logic (silently applying the wrong generic thresholds — doc 02 §1's actual rule is
"international OR above 150,000", which the generic tiers cannot express) and no way to reject
fines, minibar charges, or personal expenses at all (an uncapped `expense_type` was previously just
reimbursed in full with a warning). Also added the category-coverage consistency gate the task
explicitly asks for, and closed the remaining test gaps (boundary tests, prohibited-type tests,
graph/API journeys and document-consistency proof).

## Quality gates run

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, `pytest -q` — all clean
after the follow-up review (177 passed, 21 skipped).
