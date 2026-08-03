# 08 — Benefits, deadlines, and document requirements

**What to build:** Employees can ask about benefit budgets, eligibility tenure, claim deadlines, receipts, and required supporting documents and receive deterministic answers without unnecessary arithmetic or retrieval steps.

**Blocked by:** 07 — Commuting and mileage clarification journey.

**Status:** ready-for-agent

**Design references:** Technical design §4.4, §6.2–§6.3, §7.1–§7.3, and cap/deadline failure handling in §15.

**Technical notes:**

- Treat benefit budget calculation, eligibility checking, deadline calculation, and document lookup as separate deterministic operations behind the existing tool seam.
- Inject the reference date into deadline logic from the evaluation/request context; never read wall-clock date inside the pure calculation.
- Let the agent omit irrelevant tools: a deadline or document question may call only the rule checker, while a requested benefit amount may require search, calculation, and checking.
- Keep finding statuses and references machine-readable so final decisions and evaluation metrics do not depend on model wording.
- Model missing tenure, budget usage, date, or receipt facts explicitly in `ExpenseClaim` and route them through the shared clarification mechanism.
- Delegate deadline evaluation from rule checking to one pure deadline implementation so all callers share the same inclusive boundary and injected-time semantics.

- [x] Benefit calculations apply annual budget, already-used budget, remaining amount, requested amount, and no-carry-over rules deterministically.
- [x] Benefit eligibility checks the configured tenure requirement and distinguishes insufficient information from ineligibility.
- [x] Deadline calculation accepts an injected reference date, computes elapsed and remaining days, and returns within-deadline, due-soon, or expired status reproducibly.
- [x] Catalogue rules for no carry-over and minimum benefit tenure are surfaced as explicit applied rules rather than being implied only in answer prose.
- [x] End-to-end tests verify representative benefit, deadline, and document-requirement answers, tool sequences, decisions, and citations.
