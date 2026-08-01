# 2026-08-01 19:02 UTC — Rule clarity and journey corrections

## What changed

- Split `RuleChecker` into focused document, approval, eligibility, and deadline collaborators while
  retaining one injected facade for the tool layer.
- Added explicit claim facts for business use, international trip scope, and provided documents.
  Required-slot routing asks only for decision-critical meal, travel, and equipment facts instead of
  carrying unused descriptions or encoding business purpose in `expense_type`.
- Made required-document, receipt, approval, deadline, benefit-tenure, and carry-over findings use
  stable catalogue IDs and resolvable document references. Complete document sets pass; unknown or
  incomplete sets warn.
- Corrected approval behavior: domestic travel still requires line-manager approval, international
  travel uses an explicit boolean rather than a subtype suffix, and equipment uses its own tiers.
- Made deadline-only checks work without a category. Calculation excess/warnings now contribute to
  `partially_eligible`, and `ChatResponse` exposes the deterministic decision.
- Replaced the string-based meal-rule lookup with `_meal_limit_rule`; a missing catalogue cap returns
  a lower-confidence result rather than raising or inventing a limit.
- Strengthened ingestion metadata so a referenced section inherits the rule category, and validation
  proves configured rule evidence is retrievable under that category.
- Consolidated repeated journey-test fixtures, added document-only coverage, and added a complete
  travel request through `POST /chat`. That API test also exposed and corrected the graph recursion
  backstop from 10 to the documented 20.

## Why

A review of the new meal, travel, equipment, benefits, deadline, and document journeys found several
places where code names overstated guarantees, prompt-specific strings leaked into deterministic
logic, or informational warnings accidentally changed eligibility. These changes make the data
needed for a decision explicit, keep policy references in one validated catalogue, and distinguish
compiled-graph integration tests from HTTP endpoint coverage.

## Verification

`ruff check .`, `ruff format --check .`, `bandit -c pyproject.toml -r app`, and the complete pytest
suite — all clean (177 passed, 21 skipped).
