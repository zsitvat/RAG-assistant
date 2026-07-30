# 06 — Travel and equipment expense journeys

**What to build:** Employees can check accommodation, taxi, travel parking, and work-equipment expenses through the same agent journey and receive the correct deterministic amount, eligibility findings, approval requirement, required documents, and source-backed explanation.

**Blocked by:** 05 — Meal reimbursement journey.

**Status:** ready-for-agent

**Design references:** Technical design §4.4, §5 `ExpenseClaim`, §6.2–§6.3, §7.1–§7.2, and relevant failure modes in §15.

**Technical notes:**

- Preserve a canonical category plus travel subtype in the claim: retrieval filters on the category, while deterministic rule selection uses the subtype.
- Extend the existing calculator dispatch internally rather than registering separate model-facing tools for accommodation, taxi, parking, or equipment.
- Distinguish calculation from decision: uncapped travel can return the submitted amount while the rule checker still rejects personal purpose, prohibited charges, missing documents, or absent approval.
- Represent rule outcomes as typed findings with stable status values and resolvable `doc_ref` values so response generation and evaluation consume the same facts.
- Reuse the existing claim extraction and missing-slot machinery; category additions must not create a second extraction schema inside tool arguments.
- Keep approval checking based on the catalogue's strict greater-than threshold semantics, separate from whether an amount is arithmetically reimbursable.

- [ ] Classification normalises accommodation, taxi, and business-travel parking to the travel category while preserving the subtype needed by deterministic rules.
- [ ] Travel calculation selects the applicable catalogue rule by subtype and applies a cap only when the policy defines one.
- [ ] A travel item without a numeric cap returns the submitted amount as the calculation result while leaving final eligibility and approval to the rule checker.
- [ ] Fines, minibar charges, personal travel, and other prohibited or non-business items produce explicit fail findings rather than being silently included.
- [ ] Equipment calculation reports the submitted amount while the rule checker independently determines business eligibility and manager approval.
- [ ] Required receipts and supporting documents are returned for each supported journey and remain linked to resolvable policy references.
- [ ] The common approval threshold is applied exactly as documented, including a boundary test at the threshold and immediately above it.
- [ ] The same calculator and rule-checker interfaces introduced for meals are extended without adding category-specific tools to the model-visible registry.
- [ ] Missing subtype, amount, business-purpose, or item information routes to a focused clarification or typed tool error according to the required-slot contract.
- [ ] Unit tests cover capped and uncapped travel, taxi, parking, accommodation, prohibited charges, equipment below and above approval threshold, and absent documentation.
- [ ] Unit tests also cover personal/non-business-purpose travel rejection distinctly from other prohibited charges, and the missing-subtype/amount/business-purpose clarification routing path.
- [ ] End-to-end tests verify that representative travel and equipment questions produce deterministic tool artifacts, decisions, and citations through the chat endpoint, including at least one rejected case (personal purpose or prohibited charge).
- [ ] A category with no reachable indexed policy chunk fails the consistency gate rather than remaining silently unusable.
