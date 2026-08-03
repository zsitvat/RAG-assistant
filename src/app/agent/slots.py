from app.agent.model import ExpenseClaim, Intent
from app.rules.model import Category

# Maps each (intent, category) pair to the ExpenseClaim fields the agent must have
# before it can evaluate the rules, so the graph knows which follow-up questions to ask.
_REQUIRED_SLOTS: dict[tuple[Intent, Category | None], list[str]] = {
    ("policy_question", None): [],
    ("document_requirements", None): ["category"],
    ("expense_check", "meal"): [
        "amount_huf",
        "headcount",
        "is_business_related",
        "non_reimbursable_amount",
    ],
    ("expense_check", "travel"): [
        "expense_type",
        "amount_huf",
        "is_business_related",
        "is_international_trip",
    ],
    ("expense_check", "equipment"): ["amount_huf", "is_business_related"],
    ("calculation", "mileage"): ["distance_km", "distance_is_one_way"],
    ("expense_check", "mileage"): ["distance_km", "distance_is_one_way"],
    ("calculation", "commuting"): ["expense_type"],
    ("expense_check", "commuting"): ["expense_type"],
    ("expense_check", "benefits"): [
        "expense_type",
        "amount_huf",
        "annual_budget_used_huf",
        "tenure_months",
    ],
    ("deadline_check", None): ["expense_date"],
}

# Commuting has no fixed slot set: the required fields depend on the declared transport
# mode (pass/ticket vs. own vehicle), so these are added on top of _REQUIRED_SLOTS.
_COMMUTING_MODE_SLOTS: dict[str, list[str]] = {
    "pass": ["amount_huf"],
    "ticket": ["amount_huf", "commute_days_per_month"],
}
_COMMUTING_VEHICLE_SLOTS = ["distance_km", "distance_is_one_way", "commute_days_per_month"]


class RequiredSlotTable:
    """Looks up the required slots for an (intent, category) pair and reports which are missing."""

    def missing(self, intent: Intent, category: Category | None, claim: ExpenseClaim) -> list[str]:
        """Returns the required slot names that are still unset on the claim."""
        required = _REQUIRED_SLOTS.get((intent, category), _REQUIRED_SLOTS.get((intent, None), []))
        if category == "commuting":
            required = required + self._commuting_mode_slots(claim.expense_type)
        return [slot for slot in required if getattr(claim, slot, None) is None]

    @staticmethod
    def _commuting_mode_slots(expense_type: str | None) -> list[str]:
        """Returns the slots the declared commuting transport mode needs."""
        if expense_type is None:
            return []
        return _COMMUTING_MODE_SLOTS.get(expense_type, _COMMUTING_VEHICLE_SLOTS)
