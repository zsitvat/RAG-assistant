from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.agent.deadline import DeadlineChecker
from app.agent.model import ExpenseClaim
from app.rules.model import RuleCatalogue

PROHIBITED_MEAL_ITEMS = ("alcohol", "tobacco", "tips")
SUBMISSION_DEADLINE_RULE_ID = "SUBMISSION-DEADLINE"


class Finding(BaseModel):
    """Records the outcome of checking a claim against a single rule."""

    rule_id: str
    status: Literal["pass", "fail", "warning", "not_applicable"]
    message: str
    doc_ref: str | None = None


class RuleChecker:
    """Checks a claim against eligibility, caps, approval, receipt and deadline rules."""

    def __init__(self, rules: RuleCatalogue, deadline_checker: DeadlineChecker) -> None:
        """Stores the rule catalogue and deadline checker used for evaluation."""
        self._rules = rules
        self._deadline_checker = deadline_checker

    def check(self, claim: ExpenseClaim, reference_date: date) -> list[Finding]:
        """Evaluates the claim against all applicable rules and returns the findings."""
        if claim.category is None:
            return [
                Finding(rule_id="-", status="not_applicable", message="claim has no category yet")
            ]

        findings = [
            *self._check_prohibited_items(claim),
            *self._check_approval_threshold(claim),
            *self._check_receipt(claim),
        ]
        if claim.category == "commuting":
            findings.extend(self._check_minimum_distance(claim))
        if claim.category == "benefits":
            findings.extend(self._check_annual_budget(claim))
            findings.append(self._tenure_not_tracked_finding())
        if claim.expense_date is not None:
            findings.append(self._check_deadline(claim.expense_date, reference_date))
        return findings

    def _check_prohibited_items(self, claim: ExpenseClaim) -> list[Finding]:
        if claim.category != "meal":
            return []
        rule = next((r for r in self._rules.categories["meal"].rules if r.excluded_items), None)
        if rule is None:
            return []
        return [
            Finding(
                rule_id=rule.id,
                status="warning",
                message=(
                    f"reminder: {', '.join(rule.excluded_items)} are excluded from reimbursement; "
                    "not verifiable from the claim fields alone"
                ),
                doc_ref=rule.doc_ref,
            )
        ]

    def _check_approval_threshold(self, claim: ExpenseClaim) -> list[Finding]:
        if claim.amount_huf is None:
            return []
        tiers = sorted(
            self._rules.submission.approval_tiers,
            key=lambda tier: tier.max_huf if tier.max_huf is not None else float("inf"),
        )
        base_tier = tiers[0]
        if claim.amount_huf <= (base_tier.max_huf or float("inf")):
            return [
                Finding(
                    rule_id="SUBMISSION-APPROVAL",
                    status="pass",
                    message=(
                        "amount is within the base approval tier; no additional approval needed"
                    ),
                )
            ]
        approver = next(
            tier.approver
            for tier in tiers
            if tier.max_huf is None or claim.amount_huf <= tier.max_huf
        )
        if claim.approval_obtained is True:
            status, message = "pass", f"{approver} approval was obtained as required"
        elif claim.approval_obtained is False:
            status, message = "fail", f"amount requires {approver} approval, which was not obtained"
        else:
            status, message = (
                "warning",
                f"amount requires {approver} approval; approval status unknown",
            )
        return [Finding(rule_id="SUBMISSION-APPROVAL", status=status, message=message)]

    def _check_receipt(self, claim: ExpenseClaim) -> list[Finding]:
        if claim.has_receipt is False:
            return [
                Finding(
                    rule_id="SUBMISSION-DOCUMENTS", status="fail", message="no receipt provided"
                )
            ]
        if claim.has_receipt is None:
            return [
                Finding(
                    rule_id="SUBMISSION-DOCUMENTS",
                    status="warning",
                    message="receipt status unknown",
                )
            ]
        return [
            Finding(
                rule_id="SUBMISSION-DOCUMENTS", status="pass", message="receipt confirmed present"
            )
        ]

    def _check_minimum_distance(self, claim: ExpenseClaim) -> list[Finding]:
        rule = next(
            (r for r in self._rules.categories["commuting"].rules if r.min_one_way_km is not None),
            None,
        )
        if rule is None or claim.distance_km is None or claim.distance_is_one_way is None:
            return []
        one_way_km = claim.distance_km if claim.distance_is_one_way else claim.distance_km / 2
        if one_way_km < rule.min_one_way_km:
            return [
                Finding(
                    rule_id=rule.id,
                    status="fail",
                    message=(
                        f"one-way distance {one_way_km:g} km is below the {rule.min_one_way_km} km "
                        "minimum eligibility threshold"
                    ),
                    doc_ref=rule.doc_ref,
                )
            ]
        return [
            Finding(rule_id=rule.id, status="pass", message="meets minimum distance eligibility")
        ]

    def _check_annual_budget(self, claim: ExpenseClaim) -> list[Finding]:
        rule = next(
            (
                r
                for r in self._rules.categories["benefits"].rules
                if r.benefit_type == claim.expense_type
            ),
            None,
        )
        if rule is None or claim.annual_budget_used_huf is None:
            return []
        remaining = rule.annual_budget_huf - claim.annual_budget_used_huf
        findings = []
        if remaining <= 0:
            findings.append(
                Finding(
                    rule_id=rule.id,
                    status="fail",
                    message=(
                        f"annual {rule.benefit_type} budget of {rule.annual_budget_huf} HUF "
                        "is exhausted"
                    ),
                    doc_ref=rule.doc_ref,
                )
            )
        else:
            findings.append(
                Finding(
                    rule_id=rule.id,
                    status="pass",
                    message=f"{remaining} HUF remains in the annual {rule.benefit_type} budget",
                    doc_ref=rule.doc_ref,
                )
            )
        if (
            rule.approval_above_huf is not None
            and claim.amount_huf is not None
            and claim.amount_huf > rule.approval_above_huf
            and claim.approval_obtained is not True
        ):
            findings.append(
                Finding(
                    rule_id=rule.id,
                    status="warning",
                    message=(
                        f"amount exceeds {rule.approval_above_huf} HUF and requires prior approval"
                    ),
                    doc_ref=rule.doc_ref,
                )
            )
        return findings

    @staticmethod
    def _tenure_not_tracked_finding() -> Finding:
        return Finding(
            rule_id="BENEFITS-TENURE",
            status="not_applicable",
            message=(
                "employee tenure is not captured by the current claim model; the "
                "eligible_after_months requirement cannot be verified automatically"
            ),
        )

    def _check_deadline(self, expense_date: date, reference_date: date) -> Finding:
        result = self._deadline_checker.check(expense_date, reference_date)
        status = {"within_deadline": "pass", "due_soon": "warning", "expired": "fail"}[
            result.status
        ]
        message = f"{result.days_remaining} days remaining until the submission deadline"
        if result.status == "expired":
            message = (
                f"deadline expired {-result.days_remaining} days ago; {result.exception_procedure}"
            )
        return Finding(rule_id=SUBMISSION_DEADLINE_RULE_ID, status=status, message=message)
