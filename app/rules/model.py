from typing import Literal

from pydantic import BaseModel, model_validator

Category = Literal["general", "meal", "equipment", "travel", "commuting", "mileage", "benefits"]


class ApprovalTier(BaseModel):
    max_huf: int | None
    approver: str


class SectionMeta(BaseModel):
    headings: list[str]


class DocumentMeta(BaseModel):
    categories: list[Category]
    sections: dict[str, SectionMeta] = {}

    @model_validator(mode="after")
    def _categories_not_empty(self) -> "DocumentMeta":
        if not self.categories:
            raise ValueError("document categories must not be empty")
        return self


class RuleDefinition(BaseModel):
    id: str
    doc_ref: str | None = None

    # meal
    limit_per_person_huf: int | None = None
    excluded_items: list[str] | None = None

    # equipment
    approval_tiers: list[ApprovalTier] | None = None

    # travel
    department_head_approval_above_huf: int | None = None
    accommodation_limit_huf_per_night: dict[str, int] | None = None
    meal_per_diem_huf: dict[str, int] | None = None
    breakfast_reduction_huf: dict[str, int] | None = None

    # commuting / mileage
    min_one_way_km: float | None = None
    rate_huf_per_km: float | None = None
    monthly_cap_huf: int | None = None
    pass_reimbursement_ratio: float | None = None
    ticket_reimbursement_ratio: float | None = None
    daily_cap_huf: int | None = None

    # benefits
    benefit_type: str | None = None
    annual_budget_huf: int | None = None
    reimbursement_ratio: float | None = None
    approval_above_huf: int | None = None
    eligible_after_months: int | None = None
    carry_over: bool | None = None


class CategoryRules(BaseModel):
    rules: list[RuleDefinition] = []
    required_documents: list[str] = []


class SubmissionRules(BaseModel):
    deadline_days: int
    approval_tiers: list[ApprovalTier]


class RuleCatalogue(BaseModel):
    version: int
    currency: str
    fx_rates_fixed: dict[str, float]
    documents: dict[str, DocumentMeta]
    submission: SubmissionRules
    categories: dict[Category, CategoryRules]

    @model_validator(mode="after")
    def _validate_references(self) -> "RuleCatalogue":
        errors: list[str] = []

        for doc_id in self.documents:
            if not (len(doc_id) == 2 and doc_id.isdigit()):
                errors.append(f"invalid document id {doc_id!r}: must be two digits")

        seen_rule_ids: set[str] = set()
        for category_rules in self.categories.values():
            for rule in category_rules.rules:
                if rule.id in seen_rule_ids:
                    errors.append(f"duplicate rule id {rule.id!r}")
                seen_rule_ids.add(rule.id)
                errors.extend(self._validate_doc_ref(rule))

        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _validate_doc_ref(self, rule: RuleDefinition) -> list[str]:
        if rule.doc_ref is None:
            return []
        if "#" not in rule.doc_ref:
            return [f"rule {rule.id!r}: doc_ref {rule.doc_ref!r} must be '<doc_id>#<section_id>'"]

        doc_id, section_id = rule.doc_ref.split("#", 1)
        document = self.documents.get(doc_id)
        if document is None:
            return [f"rule {rule.id!r}: doc_ref points to unknown document {doc_id!r}"]
        if section_id not in document.sections:
            return [
                f"rule {rule.id!r}: doc_ref points to unresolved section "
                f"{section_id!r} in document {doc_id!r}"
            ]
        return []
