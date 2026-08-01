CLARIFICATION_QUESTIONS: dict[str, str] = {
    "category": (
        "Which expense category is this about — meal, travel, equipment, commuting, mileage, "
        "or benefits?"
    ),
    "amount_huf": "What is the total amount, in HUF?",
    "headcount": "How many people were included?",
    "expense_type": (
        "What type of expense is this (for example accommodation, per diem, or transport)?"
    ),
    "distance_km": "What is the distance, in kilometers?",
    "distance_is_one_way": "Is that distance one-way or round-trip?",
    "commute_days_per_month": "How many days per month do you commute?",
    "annual_budget_used_huf": (
        "How much of your annual benefit budget have you already used this year?"
    ),
    "expense_date": "What date was the expense incurred?",
}
DEFAULT_CLARIFICATION_QUESTION = "Could you share a few more details?"

OUT_OF_SCOPE_MESSAGE = (
    "I can help with company expense reimbursement and benefits policy questions, but I can't "
    "provide tax or legal advice or help with anything outside that scope. The policies I know "
    "describe a fictional company, are not a real company's rules, and are not tax or legal advice."
)
NO_TOOL_ARTIFACT_MESSAGE = (
    "I do not have enough verified evidence to answer that policy-dependent question yet. Please "
    "rephrase or ask again so I can look up the relevant policy first."
)
LLM_UNAVAILABLE_MESSAGE = (
    "The language model is currently unreachable, even after retrying. Please try again shortly."
)
INCOMPLETE_EVIDENCE_NOTE = (
    "\n\n(Note: I stopped gathering evidence after the maximum number of policy lookups for this "
    "request, so this answer may be based on incomplete information.)"
)
