from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.rules.model import Category

Intent = Literal[
    "policy_question",
    "document_requirements",
    "expense_check",
    "calculation",
    "deadline_check",
    "unsupported",
]

Decision = Literal["eligible", "partially_eligible", "not_eligible", "needs_info", "out_of_scope"]
FindingStatus = Literal["pass", "fail", "warning", "not_applicable"]


class ExpenseClaim(BaseModel):
    """Holds the expense details extracted from the conversation so far."""

    category: Category | None = None
    expense_type: str | None = None
    amount_huf: float | None = None
    headcount: int | None = None
    expense_date: date | None = None
    distance_km: float | None = None
    distance_is_one_way: bool | None = None
    commute_days_per_month: int | None = None
    non_reimbursable_amount: float | None = None
    has_receipt: bool | None = None
    approval_obtained: bool | None = None
    annual_budget_used_huf: float | None = None
    tenure_months: int | None = None
    is_business_related: bool | None = None
    is_international_trip: bool | None = None
    provided_documents: list[str] | None = None

    @classmethod
    def from_state(cls, value: "ExpenseClaim | dict | None") -> "ExpenseClaim":
        """Rebuilds a claim from graph state; a Redis checkpoint restores it as a plain dict."""
        if isinstance(value, cls):
            return value
        if not value:
            return cls()
        return cls.model_validate(value.get("kwargs", value) if value.get("lc") else value)

    def merged_with(self, update: "ExpenseClaim") -> "ExpenseClaim":
        """New non-null fields win; fields the update leaves unset keep their prior value."""
        return self.model_copy(update=update.model_dump(exclude_none=True))


class CalculationResult(BaseModel):
    """Holds the reimbursable amount and any cap or warnings from a category calculation."""

    amount_huf: int
    cap_huf: int | None = None
    excess_huf: int = 0
    warnings: list[str] = []

    def compact_summary(self) -> str:
        """Renders the calculation result as a short human-readable summary."""
        parts = [f"reimbursable {self.amount_huf} HUF"]
        if self.cap_huf is not None:
            parts.append(f"cap {self.cap_huf}")
        if self.excess_huf:
            parts.append(f"excess {self.excess_huf}")
        if self.warnings:
            parts.append(f"warnings: {'; '.join(self.warnings)}")
        return ", ".join(parts)


class Finding(BaseModel):
    """Records the outcome of checking a claim against one configured rule."""

    rule_id: str
    status: FindingStatus
    message: str
    doc_ref: str | None = None


class IntentClassification(BaseModel):
    """Structured output of the classify_intent node."""

    intent: Intent
    category: Category | None = None
