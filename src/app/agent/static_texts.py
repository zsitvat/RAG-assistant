"""Fixed, non-LLM-generated user-facing strings, so clarification questions, refusals, and
system-state messages stay deterministic and testable rather than varying with model wording. This
much hardcoding is mainly needed because the current chat model is small; a larger, more capable
model could likely generate this text reliably on its own instead."""

CLARIFICATION_QUESTIONS: dict[str, str] = {
    "category": (
        "Which expense category is this about — meal, travel, equipment, commuting, mileage, "
        "or benefits?"
    ),
    "amount_huf": "What is the total amount, in HUF?",
    "headcount": "How many people were included?",
    "non_reimbursable_amount": (
        "How much of the total was for excluded items such as alcohol, tobacco, or tips? Enter 0 "
        "if there were none."
    ),
    "expense_type": (
        "What type of expense is this (for example accommodation, per diem, or transport)?"
    ),
    "is_business_related": "Was the expense incurred for a documented business purpose?",
    "is_international_trip": "Was this a domestic or international trip?",
    "distance_km": "What is the distance, in kilometers?",
    "distance_is_one_way": "Is that distance one-way or round-trip?",
    "commute_days_per_month": "How many days per month do you commute?",
    "annual_budget_used_huf": (
        "How much of your annual benefit budget have you already used this year?"
    ),
    "tenure_months": "How many months have you been continuously employed?",
    "expense_date": "What date was the expense incurred?",
}
DEFAULT_CLARIFICATION_QUESTION = "Could you share a few more details?"

OUT_OF_SCOPE_MESSAGE = (
    "I can help with company expense reimbursement and benefits policy questions, but I can't "
    "provide tax or legal advice or help with anything outside that scope. The policies I know "
    "are company policies and are not tax or legal advice."
)
NO_TOOL_ARTIFACT_MESSAGE = (
    "I do not have enough verified evidence to answer that policy-dependent question yet. Please "
    "rephrase or ask again so I can look up the relevant policy first."
)
LLM_UNAVAILABLE_MESSAGE = (
    "The language model is currently unreachable, even after retrying. Please try again shortly."
)
CONDITIONAL_DISTANCE_ANSWER = (
    "You have not told me whether that distance is one-way or a round trip, so here is both:\n"
    "- if it is one-way (the return journey is counted): {one_way}\n"
    "- if it is already the round-trip distance: {round_trip}\n"
    "Tell me which one applies and I can confirm a single amount."
)
INCOMPLETE_EVIDENCE_NOTE = (
    "\n\n(Note: I stopped gathering evidence after the maximum number of policy lookups for this "
    "request, so this answer may be based on incomplete information.)"
)
