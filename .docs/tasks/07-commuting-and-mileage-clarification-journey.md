# 07 — Commuting and mileage clarification journey

**What to build:** An employee can ask about commuting or business mileage, be asked for genuinely missing distance information, resume the same conversation later, and receive a deterministic allowance backed by the applicable policy. A new expense in the same thread starts cleanly instead of inheriting the previous claim.

**Blocked by:** 06 — Travel and equipment expense journeys.

**Status:** ready-for-agent

**Design references:** Technical design §5, §6.2, §6.4–§6.5, §7.1–§7.3, Redis checkpointing in §4.3, and clarification failures in §15.

**Technical notes:**

- Encode category-specific required slots in deterministic application logic. An optional Boolean such as one-way distance is missing when `None`, not false.
- End an incomplete turn at the clarification node and rely on the compiled graph's Redis checkpointer for resumption; do not block an HTTP request waiting for user input.
- Merge new extraction into a previous claim only when the checkpointed decision is `needs_info` and the intent/category remain compatible.
- Use current-turn message slicing for tool-loop guards while allowing the classifier, extractor, and final-answer prompt to see conversational history.
- Keep commuting and mileage formulas inside the calculator module and expose only the resulting amount, effective cap, excess, and warnings in the typed result.
- Use `RedisSaver` in the running application and `InMemorySaver` only in isolated graph tests; the checkpointer is an infrastructure adapter, not a second conversation-memory implementation.

- [ ] Own-car commuting calculates monthly round-trip distance, attendance days, rate, monthly cap, and hybrid-work pro-rating from catalogue rules.
- [ ] Public-pass commuting applies the reimbursement ratio, monthly cap, integer-HUF convention, and required-document rules.
- [ ] Business mileage distinguishes one-way and round-trip distance and selects the transport rate deterministically.
- [ ] Minimum-distance eligibility and transport-mode constraints produce deterministic rule findings.
- [ ] Missing `distance_is_one_way`, commute days, or distance is treated as missing information and never inferred by the model.
- [ ] The clarification node ends the turn with `needs_info`, a focused question, and a checkpointed partial claim.
- [ ] Redis-backed LangGraph checkpointing restores the pending claim by thread identifier after a later request or application restart and applies the configured expiry.
- [ ] Conversation checkpoints expire after 24 hours and use a key namespace distinct from corpus and manifest data.
- [ ] Claim merging occurs only for a compatible clarification response; a new intent, category, or expense replaces the old claim.
- [ ] Thread reset removes the stored conversation state and makes the next message behave as a new conversation.
- [ ] If the user refuses to provide a required distance fact, the response presents explicit one-way and round-trip conditional outcomes instead of looping indefinitely.
- [ ] Loop counters, duplicate-tool detection, projected artifacts, and decision derivation inspect only the current turn.
- [ ] Unit and integration tests cover own-car, public-pass, EV/mileage, cap, minimum distance, one-way ambiguity, two-turn resume, incompatible follow-up, reset, and checkpoint expiry behaviour.
- [ ] Several Streamlit workers can address the same Redis-backed thread without introducing process-local conversation truth.
