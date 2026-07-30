# 05 — Meal reimbursement journey

**What to build:** An employee can describe a business meal and receive a deterministic reimbursable amount, policy cap, excluded amount explanation, receipt requirements, approval finding, and supporting citation. The model chooses when to calculate but does not copy claim fields or policy numbers into the calculator call.

**Blocked by:** 04 — First policy-question agent journey.

**Status:** ready-for-agent

**Design references:** Technical design §2 two-layer knowledge model, §4.4, §6.2–§6.3, and §7.1–§7.4.

**Technical notes:**

- Place calculation behind one `ReimbursementCalculator.calculate(claim)` interface and inject the validated catalogue at construction; category-specific functions remain private implementation details.
- Wrap the module with an argument-free LangChain tool. The injected `ToolRuntime` reads the already extracted claim from graph state and stays absent from the model-visible schema.
- Return `CalculationResult` as the tool artifact and a short arithmetic summary as content, avoiding both prose parsing and large model transcripts.
- Use decimal-safe half-up rounding and integer HUF outputs. Record conversion rates and each material formula in typed calculation-breakdown lines.
- Keep rule checking separate from arithmetic so eligibility, receipt, approval, and document findings can evolve without altering formula code.
- Select numeric policy values from the validated catalogue, not from retrieved prose or model arguments; retrieval supplies evidence and citations while deterministic tools supply authoritative calculation inputs.

- [ ] The reimbursement calculator is a deep module whose interface accepts a validated expense claim and whose rule catalogue dependency is supplied at construction.
- [ ] The calculation result uses the domain names `reimbursable_amount_huf`, `applied_policy_cap_huf`, `amount_over_policy_cap_huf`, `applied_per_person_limit_huf`, `calculation_breakdown`, `applied_rule_ids`, and `warnings`.
- [ ] A missing cap is represented by `None`, amount over cap is zero when no cap applies, and separately excluded items remain distinguishable in the calculation breakdown.
- [ ] Meal calculation applies the per-person limit, headcount, excluded-item amount, policy cap, amount-over-cap calculation, integer-HUF convention, and half-up rounding exactly as documented.
- [ ] Explicitly excluded items such as alcohol and minibar are distinguished from otherwise eligible amounts above the policy cap.
- [ ] The LangChain calculator adapter exposes no model-supplied business arguments; it reads the current claim from the hidden `ToolRuntime`.
- [ ] A schema test proves that the model sees an empty calculator argument object and cannot override the extracted claim or catalogue values.
- [ ] Missing or inconsistent meal fields produce a typed tool error and never trigger a guessed headcount, amount, or business purpose.
- [ ] A meal request missing amount, headcount, or business-relatedness routes to `ask_clarification` with a focused question and a checkpointed partial claim, resumable in the same thread, rather than reaching the calculator first.
- [ ] The rule checker reports eligibility, required documents, receipt status, manager-approval threshold, and cited rule identifiers for the meal claim.
- [ ] A complete meal request normally produces the observable tool order search, calculate, and rule check, while the graph remains free to stop safely when evidence is unavailable.
- [ ] Unit tests cover below-cap, exactly-at-cap, above-cap, excluded-item, foreign-currency, rounding, missing-input, and approval-threshold cases.
- [ ] Fixed fictional EUR and USD conversion rates come from the catalogue, and each converted calculation records the rate and formula used.
- [ ] A missing catalogue limit or unresolved rule produces a warning/lower-confidence result rather than an invented number.
- [ ] An end-to-end scripted test reproduces the reference dinner example and returns the expected reimbursement, decision, calculation breakdown, findings, and citations.
- [ ] A consistency test proves every applied numeric rule appears verbatim in its referenced policy section and every returned rule identifier resolves to an indexed citation anchor.
