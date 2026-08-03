from datetime import date
from functools import cache
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, create_model

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

    category: Category | None = Field(
        default=None,
        description="The expense or benefit category (meal, equipment, travel, "
        "commuting, mileage, benefits, general).",
    )
    expense_type: str | None = Field(
        default=None,
        description="The specific subtype of the expense within its category, e.g. 'taxi', "
        "'own_car', 'recreational'.",
    )
    amount_huf: float | None = Field(
        default=None,
        description="The claimed amount in Hungarian forint, as stated, never converted from "
        "another currency.",
    )
    headcount: int | None = Field(
        default=None,
        description="Total number of people covered by the expense, including the claimant.",
    )
    expense_date: date | None = Field(
        default=None, description="The date the expense was incurred or the claim relates to."
    )
    distance_km: float | None = Field(
        default=None, description="The commuting or travel distance in kilometers, as stated."
    )
    distance_is_one_way: bool | None = Field(
        default=None,
        description="Whether distance_km is a one-way distance rather than a round trip.",
    )
    commute_days_per_month: int | None = Field(
        default=None, description="Number of commuting days per month the ticket or pass covers."
    )
    non_reimbursable_amount: float | None = Field(
        default=None,
        description="The portion of amount_huf that is not reimbursable (e.g. alcohol, personal "
        "items), in HUF.",
    )
    has_receipt: bool | None = Field(
        default=None, description="Whether a receipt was provided for the expense."
    )
    approval_obtained: bool | None = Field(
        default=None, description="Whether prior approval was obtained for the expense."
    )
    annual_budget_used_huf: float | None = Field(
        default=None,
        description="The amount of the annual benefit budget already used, in HUF.",
    )
    tenure_months: int | None = Field(
        default=None, description="The employee's tenure at the company, in months."
    )
    is_business_related: bool | None = Field(
        default=None, description="Whether the expense was for a business purpose."
    )
    is_international_trip: bool | None = Field(
        default=None, description="Whether the trip was international rather than domestic."
    )
    provided_documents: list[str] | None = Field(
        default=None,
        description="Normalized short snake_case names of supporting documents the user says "
        "they have.",
    )

    _COMMON_EXTRACTION_FIELDS: ClassVar[tuple[str, ...]] = (
        "category",
        "expense_type",
        "amount_huf",
        "expense_date",
        "has_receipt",
        "approval_obtained",
        "provided_documents",
    )
    _CATEGORY_EXTRACTION_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "meal": ("headcount", "non_reimbursable_amount", "is_business_related"),
        "travel": ("is_business_related", "is_international_trip"),
        "equipment": ("is_business_related",),
        "commuting": ("distance_km", "distance_is_one_way", "commute_days_per_month"),
        "mileage": ("distance_km", "distance_is_one_way"),
        "benefits": ("annual_budget_used_huf", "tenure_months"),
        "general": (),
    }

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

    @classmethod
    @cache
    def extraction_schema(cls, category: Category | None) -> type["ExpenseClaim"]:
        """Builds the subset of fields relevant to one category, so a small extraction model
        only has to reason about fields that actually matter for the current request instead
        of every field every time. Falls back to the full schema when the category is unknown."""
        if category is None:
            return cls
        field_names = cls._COMMON_EXTRACTION_FIELDS + cls._CATEGORY_EXTRACTION_FIELDS.get(
            category, ()
        )
        fields = {
            name: (
                cls.model_fields[name].annotation,
                Field(default=None, description=cls.model_fields[name].description),
            )
            for name in field_names
        }
        return create_model(f"ExpenseClaimFor{category.capitalize()}", **fields)


class CalculationResult(BaseModel):
    """Holds the reimbursable amount and any cap or warnings from a category calculation."""

    amount_huf: int = Field(description="The reimbursable amount, in HUF.")
    cap_huf: int | None = Field(
        default=None, description="The policy cap that limited the reimbursable amount, in HUF."
    )
    excess_huf: int = Field(
        default=0, description="The portion of the claimed amount above the cap, in HUF."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Human-readable warnings about the calculation."
    )

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

    rule_id: str = Field(description="The id of the rule that was checked.")
    status: FindingStatus = Field(description="The outcome of checking the rule against the claim.")
    message: str = Field(description="A human-readable explanation of the finding.")
    doc_ref: str | None = Field(
        default=None, description="Reference to the policy document section backing this rule."
    )


class IntentClassification(BaseModel):
    """Structured output of the classify_intent node."""

    intent: Intent = Field(description="The classified intent of the latest employee request.")
    category: Category | None = Field(
        default=None, description="The expense or benefit category the request relates to."
    )
