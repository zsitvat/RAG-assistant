from app.agent.model import ExpenseClaim, Intent
from app.rules.model import Category

_REQUIRED_SLOTS: dict[tuple[Intent, Category | None], list[str]] = {
    ("policy_question", None): [],
    ("document_requirements", None): ["category"],
    ("expense_check", "meal"): ["amount_huf", "headcount"],
    ("expense_check", "travel"): ["expense_type", "amount_huf"],
    ("expense_check", "equipment"): ["amount_huf"],
    ("calculation", "mileage"): ["distance_km", "distance_is_one_way"],
    ("calculation", "commuting"): ["distance_km", "distance_is_one_way", "commute_days_per_month"],
    ("expense_check", "benefits"): [
        "expense_type",
        "amount_huf",
        "annual_budget_used_huf",
    ],
    ("deadline_check", None): ["expense_date"],
}


class RequiredSlotTable:
    """Looks up the required slots for an (intent, category) pair and reports which are missing."""

    def missing(self, intent: Intent, category: Category | None, claim: ExpenseClaim) -> list[str]:
        """Returns the required slot names that are still unset on the claim."""
        required = _REQUIRED_SLOTS.get((intent, category), _REQUIRED_SLOTS.get((intent, None), []))
        return [slot for slot in required if getattr(claim, slot, None) is None]
