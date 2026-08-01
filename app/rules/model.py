from typing import Literal

from pydantic import BaseModel, model_validator

Category = Literal["general", "meal", "equipment", "travel", "commuting", "mileage", "benefits"]


class ApprovalTier(BaseModel):
    """Describes the approver required for amounts up to a threshold."""

    max_huf: int | None
    approver: str


class SectionMeta(BaseModel):
    """Lists the heading variants a rules.yaml section anchor may match in the corpus."""

    headings: list[str]


class DocumentMeta(BaseModel):
    """Describes a policy document's categories and known sections."""

    categories: list[Category]
    sections: dict[str, SectionMeta] = {}

    @model_validator(mode="after")
    def _categories_not_empty(self) -> "DocumentMeta":
        if not self.categories:
            raise ValueError("document categories must not be empty")
        return self


class RuleDefinition(BaseModel):
    """Describes a single eligibility, cap or approval rule for an expense category."""

    id: str
    doc_ref: str | None = None

    # meal
    limit_per_person_huf: int | None = None
    excluded_items: list[str] | None = None
    business_use_required: bool | None = None

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
    approval_required: bool | None = None
    approver: str | None = None
    eligible_after_months: int | None = None
    carry_over: bool | None = None


class CategoryRules(BaseModel):
    """Holds the rules and required documents for one expense category."""

    rules: list[RuleDefinition] = []
    required_documents: list[str] = []
    required_documents_rule_id: str | None = None
    required_documents_doc_ref: str | None = None

    @model_validator(mode="after")
    def _document_requirements_are_source_linked(self) -> "CategoryRules":
        if self.required_documents and (
            self.required_documents_rule_id is None or self.required_documents_doc_ref is None
        ):
            raise ValueError(
                "required_documents needs required_documents_rule_id and required_documents_doc_ref"
            )
        return self


class SubmissionRules(BaseModel):
    """Holds the submission deadline and approval tiers shared across categories."""

    deadline_days: int
    approval_tiers: list[ApprovalTier]
    deadline_rule_id: str = "SUBMISSION-DEADLINE"
    deadline_doc_ref: str | None = None
    approval_rule_id: str = "SUBMISSION-APPROVAL"
    approval_doc_ref: str | None = None
    receipt_rule_id: str = "SUBMISSION-DOCUMENTS"
    receipt_doc_ref: str | None = None


class RuleCatalogue(BaseModel):
    """Holds the full rule catalogue loaded from rules.yaml."""

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
            if category_rules.required_documents_rule_id:
                if category_rules.required_documents_rule_id in seen_rule_ids:
                    errors.append(
                        f"duplicate rule id {category_rules.required_documents_rule_id!r}"
                    )
                seen_rule_ids.add(category_rules.required_documents_rule_id)
                errors.extend(
                    self._validate_reference(
                        category_rules.required_documents_rule_id,
                        category_rules.required_documents_doc_ref,
                    )
                )

        submission_references = (
            (self.submission.deadline_rule_id, self.submission.deadline_doc_ref),
            (self.submission.approval_rule_id, self.submission.approval_doc_ref),
            (self.submission.receipt_rule_id, self.submission.receipt_doc_ref),
        )
        for rule_id, doc_ref in submission_references:
            if rule_id in seen_rule_ids:
                errors.append(f"duplicate rule id {rule_id!r}")
            seen_rule_ids.add(rule_id)
            errors.extend(self._validate_reference(rule_id, doc_ref))

        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _validate_doc_ref(self, rule: RuleDefinition) -> list[str]:
        return self._validate_reference(rule.id, rule.doc_ref)

    def _validate_reference(self, rule_id: str, doc_ref: str | None) -> list[str]:
        if doc_ref is None:
            return []
        if "#" not in doc_ref:
            return [f"rule {rule_id!r}: doc_ref {doc_ref!r} must be '<doc_id>#<section_id>'"]

        doc_id, section_id = doc_ref.split("#", 1)
        document = self.documents.get(doc_id)
        if document is None:
            return [f"rule {rule_id!r}: doc_ref points to unknown document {doc_id!r}"]
        if section_id not in document.sections:
            return [
                f"rule {rule_id!r}: doc_ref points to unresolved section "
                f"{section_id!r} in document {doc_id!r}"
            ]
        return []
