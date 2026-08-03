from datetime import date

from app.agent.deadline_check import DeadlineChecker
from app.agent.model import ExpenseClaim, Finding, Intent
from app.rules.lookup import first_matching
from app.rules.model import ApprovalTier, Category, RuleCatalogue, RuleDefinition

PUBLIC_TRANSPORT_COMMUTING_MODES = ("pass", "ticket")


class DocumentChecker:
    """Checks receipts and category-specific supporting documents."""

    def __init__(self, rules: RuleCatalogue) -> None:
        """Stores the rules containing document and receipt requirements."""
        self._rules = rules

    def check_requirements(self, claim: ExpenseClaim, assess_presence: bool) -> list[Finding]:
        """Lists requirements or compares them with the documents supplied by the employee."""

        if claim.category is None:
            return []
        category_rules = self._rules.categories.get(claim.category)
        if category_rules is None or not category_rules.required_documents:
            return []

        required = category_rules.required_documents
        rule_id = category_rules.required_documents_rule_id
        if not assess_presence:
            return [
                Finding(
                    rule_id=rule_id,
                    status="pass",
                    message=f"required supporting documents: {', '.join(required)}",
                    doc_ref=category_rules.required_documents_doc_ref,
                )
            ]

        if claim.provided_documents is None:
            status = "warning"
            message = f"document presence is unknown; required: {', '.join(required)}"
        else:
            provided = set(claim.provided_documents)
            missing = [document for document in required if document not in provided]
            status = "warning" if missing else "pass"
            message = (
                f"missing supporting documents: {', '.join(missing)}"
                if missing
                else "all category-specific supporting documents were provided"
            )
        return [
            Finding(
                rule_id=rule_id,
                status=status,
                message=message,
                doc_ref=category_rules.required_documents_doc_ref,
            )
        ]

    def check_receipt(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks whether a receipt or accepted substitute is present."""

        submission = self._rules.submission
        if claim.has_receipt is False:
            status, message = "fail", "no receipt or accepted substitute document was provided"
        elif claim.has_receipt is None:
            status, message = "warning", "receipt presence is unknown"
        else:
            status, message = "pass", "receipt or accepted substitute document was provided"
        return [
            Finding(
                rule_id=submission.receipt_rule_id,
                status=status,
                message=message,
                doc_ref=submission.receipt_doc_ref,
            )
        ]


class ApprovalChecker:
    """Checks the approver required by the claim category and amount."""

    def __init__(self, rules: RuleCatalogue) -> None:
        """Stores the rules containing category approval requirements."""
        self._rules = rules

    def check(self, claim: ExpenseClaim) -> list[Finding]:
        """Returns the applicable approval finding when the claim has an amount."""

        if claim.category is None or claim.amount_huf is None:
            return []
        if claim.category == "travel":
            return self._check_travel(claim)
        if claim.category == "benefits":
            return self._check_benefit(claim)

        tiers, rule_id, doc_ref = self._approval_tiers(claim.category)
        if not tiers:
            return []
        approver = self._required_approver(tiers, claim.amount_huf)
        return [self._approval_finding(rule_id, doc_ref, approver, claim.approval_obtained)]

    def _approval_tiers(self, category: Category) -> tuple[list[ApprovalTier], str, str | None]:
        """Returns category-specific tiers or the default submission tiers."""
        category_rules = self._rules.categories[category]
        configured = first_matching(category_rules.rules, lambda rule: rule.approval_tiers)
        if configured is not None:
            return configured.approval_tiers, configured.id, configured.doc_ref
        submission = self._rules.submission
        return submission.approval_tiers, submission.approval_rule_id, submission.approval_doc_ref

    @staticmethod
    def _required_approver(tiers: list[ApprovalTier], amount_huf: float) -> str:
        """Returns the approver for the first tier covering the claim amount."""
        ordered = sorted(
            tiers,
            key=lambda tier: tier.max_huf if tier.max_huf is not None else float("inf"),
        )
        return next(
            tier.approver for tier in ordered if tier.max_huf is None or amount_huf <= tier.max_huf
        )

    def _check_travel(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks travel approval according to trip scope and amount."""
        rule = first_matching(
            self._rules.categories["travel"].rules,
            lambda rule: rule.department_head_approval_above_huf is not None,
        )
        if rule is None:
            return []
        if (
            claim.is_international_trip is None
            and claim.amount_huf <= rule.department_head_approval_above_huf
        ):
            return [
                Finding(
                    rule_id=rule.id,
                    status="warning",
                    message="trip scope is unknown, so the required approver cannot be determined",
                    doc_ref=rule.doc_ref,
                )
            ]
        needs_department_head = (
            claim.is_international_trip is True
            or claim.amount_huf > rule.department_head_approval_above_huf
        )
        approver = "department head" if needs_department_head else "line manager"
        return [self._approval_finding(rule.id, rule.doc_ref, approver, claim.approval_obtained)]

    def _check_benefit(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks whether the selected benefit requires prior approval."""
        rule = first_matching(
            self._rules.categories["benefits"].rules,
            lambda rule: rule.benefit_type == claim.expense_type,
        )
        if rule is None:
            return []
        approval_required = rule.approval_required is True or (
            rule.approval_above_huf is not None and claim.amount_huf > rule.approval_above_huf
        )
        if not approval_required:
            return []
        return [
            self._approval_finding(
                rule.id,
                rule.doc_ref,
                rule.approver or "prior manager",
                claim.approval_obtained,
            )
        ]

    @staticmethod
    def _approval_finding(
        rule_id: str, doc_ref: str | None, approver: str, obtained: bool | None
    ) -> Finding:
        """Builds a finding for a required approver and approval status."""
        if obtained is True:
            status, message = "pass", f"required {approver} approval was obtained"
        elif obtained is False:
            status, message = "fail", f"required {approver} approval was not obtained"
        else:
            status, message = "warning", f"{approver} approval is required; status unknown"
        return Finding(rule_id=rule_id, status=status, message=message, doc_ref=doc_ref)


class EligibilityChecker:
    """Checks category-specific eligibility rules that do not perform reimbursement arithmetic."""

    def __init__(self, rules: RuleCatalogue) -> None:
        """Stores the rules containing category eligibility requirements."""
        self._rules = rules

    def check(self, claim: ExpenseClaim) -> list[Finding]:
        """Dispatches category-specific eligibility checks."""

        if claim.category == "meal":
            return self._check_meal(claim)
        if claim.category in ("travel", "equipment"):
            return self._check_business_expense(claim)
        if claim.category == "commuting":
            return self._check_commuting(claim)
        if claim.category == "benefits":
            return self._check_benefits(claim)
        return []

    def _check_meal(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks excluded meal items and documented business use."""
        rule = first_matching(
            self._rules.categories["meal"].rules, lambda rule: rule.excluded_items
        )
        if rule is None:
            return []
        if claim.non_reimbursable_amount is None:
            status, message = "warning", "the amount of excluded meal items is unknown"
        elif claim.non_reimbursable_amount > 0:
            status, message = (
                "warning",
                f"{claim.non_reimbursable_amount:g} HUF of excluded meal items must be deducted",
            )
        else:
            status, message = "pass", "no excluded meal-item amount was reported"
        return [
            Finding(rule_id=rule.id, status=status, message=message, doc_ref=rule.doc_ref),
            *self._check_business_use(claim),
        ]

    def _check_business_expense(self, claim: ExpenseClaim) -> list[Finding]:
        """Rejects prohibited expenses and checks their business use."""
        rules = self._rules.categories[claim.category].rules
        prohibited_rule = first_matching(
            rules,
            lambda rule: (
                rule.excluded_items
                and claim.expense_type is not None
                and claim.expense_type in rule.excluded_items
            ),
        )
        if prohibited_rule is not None:
            return [
                Finding(
                    rule_id=prohibited_rule.id,
                    status="fail",
                    message=(
                        f"{claim.expense_type!r} is not a reimbursable {claim.category} expense"
                    ),
                    doc_ref=prohibited_rule.doc_ref,
                )
            ]

        return self._check_business_use(claim)

    def _check_business_use(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks whether a claim satisfies its business-use requirement."""
        business_rule = first_matching(
            self._rules.categories[claim.category].rules,
            lambda rule: rule.business_use_required is True,
        )
        if business_rule is None:
            return []
        if claim.is_business_related is True:
            status, message = "pass", "the expense has a documented business purpose"
        elif claim.is_business_related is False:
            status, message = "fail", "personal or non-business expenses are not reimbursable"
        else:
            status, message = "warning", "the business purpose is unknown"
        return [
            Finding(
                rule_id=business_rule.id,
                status=status,
                message=message,
                doc_ref=business_rule.doc_ref,
            )
        ]

    def _check_commuting(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks minimum-distance eligibility for commuting claims."""
        rule = first_matching(
            self._rules.categories["commuting"].rules,
            lambda rule: rule.min_one_way_km is not None,
        )
        if rule is None:
            return []
        if claim.expense_type in PUBLIC_TRANSPORT_COMMUTING_MODES:
            return [
                Finding(
                    rule_id=rule.id,
                    status="not_applicable",
                    message=(
                        "the minimum-distance condition applies to personal-vehicle commuting, "
                        f"not to a {claim.expense_type} claim"
                    ),
                    doc_ref=rule.doc_ref,
                )
            ]
        if claim.distance_km is None or claim.distance_is_one_way is None:
            return []
        one_way_km = claim.distance_km if claim.distance_is_one_way else claim.distance_km / 2
        if one_way_km < rule.min_one_way_km:
            status = "fail"
            message = (
                f"one-way distance {one_way_km:g} km is below the "
                f"{rule.min_one_way_km:g} km minimum"
            )
        else:
            status, message = "pass", "meets minimum commuting-distance eligibility"
        return [Finding(rule_id=rule.id, status=status, message=message, doc_ref=rule.doc_ref)]

    def _check_benefits(self, claim: ExpenseClaim) -> list[Finding]:
        """Collects budget, tenure, and carry-over benefit findings."""
        return [
            *self._check_benefit_budget(claim),
            *self._check_benefit_tenure(claim),
            *self._check_benefit_carry_over(),
        ]

    def _check_benefit_budget(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks whether annual benefit budget remains available."""
        rule = self._benefit_allowance_rule(claim.expense_type)
        if rule is None or claim.annual_budget_used_huf is None:
            return []
        remaining = rule.annual_budget_huf - claim.annual_budget_used_huf
        status = "fail" if remaining <= 0 else "pass"
        message = (
            f"annual {rule.benefit_type} budget of {rule.annual_budget_huf} HUF is exhausted"
            if remaining <= 0
            else f"{remaining:g} HUF remains in the annual {rule.benefit_type} budget"
        )
        return [Finding(rule_id=rule.id, status=status, message=message, doc_ref=rule.doc_ref)]

    def _check_benefit_tenure(self, claim: ExpenseClaim) -> list[Finding]:
        """Checks employee tenure against the configured benefit threshold."""
        rule = first_matching(
            self._rules.categories["benefits"].rules,
            lambda rule: rule.eligible_after_months is not None,
        )
        if rule is None:
            return []
        if claim.tenure_months is None:
            status, message = "warning", "employee tenure was not provided"
        elif claim.tenure_months < rule.eligible_after_months:
            status = "fail"
            message = (
                f"{claim.tenure_months} months of tenure is below the "
                f"{rule.eligible_after_months}-month requirement"
            )
        else:
            status, message = "pass", "employee tenure meets the eligibility requirement"
        return [Finding(rule_id=rule.id, status=status, message=message, doc_ref=rule.doc_ref)]

    def _check_benefit_carry_over(self) -> list[Finding]:
        """Reports whether unused benefit budget carries into the next year."""
        rule = first_matching(
            self._rules.categories["benefits"].rules,
            lambda rule: rule.carry_over is not None,
        )
        if rule is None:
            return []
        message = (
            "unused benefit budget carries over to the next year"
            if rule.carry_over
            else "unused benefit budget does not carry over to the next year"
        )
        return [Finding(rule_id=rule.id, status="pass", message=message, doc_ref=rule.doc_ref)]

    def _benefit_allowance_rule(self, expense_type: str | None) -> RuleDefinition | None:
        """Returns the allowance rule matching a benefit type."""
        return first_matching(
            self._rules.categories["benefits"].rules,
            lambda rule: rule.benefit_type == expense_type,
        )


class SubmissionDeadlineChecker:
    """Adapts the pure deadline calculation into a source-linked rule finding."""

    def __init__(self, rules: RuleCatalogue, deadline_checker: DeadlineChecker) -> None:
        """Stores submission rules and the deadline calculator."""
        self._submission = rules.submission
        self._deadline_checker = deadline_checker

    def check(self, expense_date: date, reference_date: date) -> Finding:
        """Returns the deadline finding for the supplied expense and reference dates."""

        result = self._deadline_checker.check(expense_date, reference_date)
        status = {"within_deadline": "pass", "due_soon": "warning", "expired": "fail"}[
            result.status
        ]
        message = f"{result.days_remaining} days remain until the submission deadline"
        if result.status == "expired":
            message = (
                f"deadline expired {-result.days_remaining} days ago; {result.exception_procedure}"
            )
        return Finding(
            rule_id=self._submission.deadline_rule_id,
            status=status,
            message=message,
            doc_ref=self._submission.deadline_doc_ref,
        )


class RuleChecker:
    """Coordinates focused document, approval, eligibility, and deadline checks."""

    def __init__(self, rules: RuleCatalogue, deadline_checker: DeadlineChecker) -> None:
        """Builds the focused rule checkers coordinated by this facade."""
        self._documents = DocumentChecker(rules)
        self._approvals = ApprovalChecker(rules)
        self._eligibility = EligibilityChecker(rules)
        self._deadline = SubmissionDeadlineChecker(rules, deadline_checker)

    def check(
        self, claim: ExpenseClaim, reference_date: date, intent: Intent | None = None
    ) -> list[Finding]:
        """Evaluates only the rule groups relevant to the current request."""

        if intent == "deadline_check":
            return self._deadline_findings(claim, reference_date)
        if intent == "document_requirements":
            return self._documents.check_requirements(claim, assess_presence=False)

        findings: list[Finding] = []
        if claim.category is not None:
            findings.extend(self._documents.check_requirements(claim, assess_presence=True))
            findings.extend(self._documents.check_receipt(claim))
            findings.extend(self._approvals.check(claim))
            findings.extend(self._eligibility.check(claim))
        if claim.expense_date is not None:
            findings.extend(self._deadline_findings(claim, reference_date))
        if findings:
            return findings
        return [Finding(rule_id="-", status="not_applicable", message="no applicable rule facts")]

    def _deadline_findings(self, claim: ExpenseClaim, reference_date: date) -> list[Finding]:
        """Returns a deadline finding when the claim has an expense date."""
        if claim.expense_date is None:
            return []
        return [self._deadline.check(claim.expense_date, reference_date)]
