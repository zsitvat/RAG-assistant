# 05 — Meal reimbursement journey

**What to build:** An employee can describe a business meal and receive a deterministic reimbursable amount, effective cap, excess, receipt requirements, approval finding, and supporting citation. The model chooses when to calculate but does not copy claim fields or policy numbers into the calculator call.

**Blocked by:** 04 — First policy-question agent journey.

**Status:** ready-for-agent

**Design references:** Technical design §2 two-layer knowledge model, §4.4, §6.2–§6.3, and §7.1–§7.4.

**Technical notes:**

- Place calculation behind one `ReimbursementCalculator.calculate(claim)` interface and inject the validated catalogue at construction; category-specific functions remain private implementation details.
- Wrap the module with an argument-free LangChain tool. The injected `ToolRuntime` reads the already extracted claim from graph state and stays absent from the model-visible schema.
- Return `CalculationResult` as the tool artifact and a short arithmetic summary as content, avoiding both prose parsing and large model transcripts.
- Use half-up rounding and integer HUF outputs. Keep the typed calculation artifact limited to the amount, effective cap, excess, and warnings.
- Keep rule checking separate from arithmetic so eligibility, receipt, approval, and document findings can evolve without altering formula code.
- Select numeric policy values from the validated catalogue, not from retrieved prose or model arguments; retrieval supplies evidence and citations while deterministic tools supply authoritative calculation inputs.

- [x] The reimbursement calculator is a deep module whose interface accepts a validated expense claim and whose rule catalogue dependency is supplied at construction.
- [x] `CalculationResult` lives in `src/app/agent/model.py` and contains only `amount_huf`, `cap_huf`, `excess_huf`, and `warnings`.
- [x] A missing cap is represented by `None`, and excess is zero when no cap applies.
- [x] Meal calculation applies the per-person limit, headcount, excluded-item amount, policy cap, amount-over-cap calculation, integer-HUF convention, and half-up rounding exactly as documented.
- [x] Explicitly excluded items such as alcohol and minibar are distinguished from otherwise eligible amounts above the policy cap.
- [x] The LangChain calculator adapter exposes no model-supplied business arguments; it reads the current claim from the hidden `ToolRuntime`.
- [ ] A schema test proves that the model sees an empty calculator argument object and cannot override the extracted claim or catalogue values.
- [x] Missing or inconsistent meal fields produce a typed tool error and never trigger a guessed headcount or amount.
- [x] A meal request missing amount or headcount routes to `ask_clarification` with a focused question and a checkpointed partial claim, resumable in the same thread, rather than reaching the calculator first.
- [x] The rule checker reports eligibility, required documents, receipt status, manager-approval threshold, and cited rule identifiers for the meal claim.
- [x] A complete meal request normally produces the observable tool order search, calculate, and rule check, while the graph remains free to stop safely when evidence is unavailable.
- [x] Unit tests cover below-cap, exactly-at-cap, above-cap, excluded-item, rounding, missing-input, and approval-threshold cases.
- [x] A missing catalogue limit or unresolved rule produces a warning/lower-confidence result rather than an invented number.
- [x] An end-to-end scripted test reproduces the reference dinner example and returns the expected reimbursement, cap, excess, decision, findings, and citations.
- [x] A consistency test proves every applied numeric rule appears verbatim in its referenced policy section and every rule-checker identifier resolves to an indexed citation anchor.
