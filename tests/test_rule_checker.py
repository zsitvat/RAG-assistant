from datetime import date

import pytest

from app.agent.deadline import DeadlineChecker
from app.agent.model import ExpenseClaim
from app.agent.rule_checker import RuleChecker
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()
REFERENCE_DATE = date(2026, 8, 1)


@pytest.fixture
def checker() -> RuleChecker:
    return RuleChecker(CATALOGUE, DeadlineChecker(CATALOGUE.submission.deadline_days))


def _status(findings, rule_id):
    return next(f.status for f in findings if f.rule_id == rule_id)


def test_meal_warns_about_excluded_items(checker):
    findings = checker.check(ExpenseClaim(category="meal", amount_huf=1000), REFERENCE_DATE)

    assert _status(findings, "R-MEAL-02") == "warning"


def test_approval_threshold_passes_below_base_tier(checker):
    findings = checker.check(ExpenseClaim(category="equipment", amount_huf=20000), REFERENCE_DATE)

    assert _status(findings, "SUBMISSION-APPROVAL") == "pass"


def test_approval_threshold_fails_when_required_but_not_obtained(checker):
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=80000, approval_obtained=False),
        REFERENCE_DATE,
    )

    assert _status(findings, "SUBMISSION-APPROVAL") == "fail"


def test_approval_threshold_passes_when_obtained(checker):
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=80000, approval_obtained=True), REFERENCE_DATE
    )

    assert _status(findings, "SUBMISSION-APPROVAL") == "pass"


def test_receipt_missing_fails(checker):
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=1000, has_receipt=False), REFERENCE_DATE
    )

    assert _status(findings, "SUBMISSION-DOCUMENTS") == "fail"


def test_commuting_below_minimum_distance_fails(checker):
    findings = checker.check(
        ExpenseClaim(category="commuting", distance_km=5, distance_is_one_way=True), REFERENCE_DATE
    )

    assert _status(findings, "R-COMM-01") == "fail"


def test_commuting_at_or_above_minimum_distance_passes(checker):
    findings = checker.check(
        ExpenseClaim(category="commuting", distance_km=10, distance_is_one_way=True), REFERENCE_DATE
    )

    assert _status(findings, "R-COMM-01") == "pass"


def test_benefits_exhausted_budget_fails(checker):
    findings = checker.check(
        ExpenseClaim(
            category="benefits", expense_type="recreational", annual_budget_used_huf=120000
        ),
        REFERENCE_DATE,
    )

    assert _status(findings, "R-BEN-01") == "fail"


def test_benefits_tenure_is_flagged_as_not_applicable(checker):
    findings = checker.check(
        ExpenseClaim(category="benefits", expense_type="recreational", annual_budget_used_huf=0),
        REFERENCE_DATE,
    )

    assert _status(findings, "BENEFITS-TENURE") == "not_applicable"


def test_deadline_within_window_passes(checker):
    findings = checker.check(
        ExpenseClaim(category="meal", expense_date=date(2026, 7, 20)), REFERENCE_DATE
    )

    assert _status(findings, "SUBMISSION-DEADLINE") == "pass"


def test_deadline_expired_fails(checker):
    findings = checker.check(
        ExpenseClaim(category="meal", expense_date=date(2026, 1, 1)), REFERENCE_DATE
    )

    assert _status(findings, "SUBMISSION-DEADLINE") == "fail"


def test_no_category_returns_a_single_not_applicable_finding(checker):
    findings = checker.check(ExpenseClaim(), REFERENCE_DATE)

    assert len(findings) == 1
    assert findings[0].status == "not_applicable"
