from datetime import date

import pytest

from app.agent.deadline_check import DeadlineChecker
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
    # Act
    findings = checker.check(ExpenseClaim(category="meal", amount_huf=1000), REFERENCE_DATE)

    # Assert
    assert _status(findings, "R-MEAL-02") == "warning"


def test_meal_excluded_item_amount_is_named_in_the_warning(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="meal", amount_huf=20000, non_reimbursable_amount=3000),
        REFERENCE_DATE,
    )

    # Assert
    finding = next(f for f in findings if f.rule_id == "R-MEAL-02")
    assert finding.status == "warning"
    assert "3000" in finding.message


def test_non_business_meal_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="meal",
            amount_huf=1000,
            non_reimbursable_amount=0,
            is_business_related=False,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-MEAL-03") == "fail"


def test_meal_reports_required_documents(checker):
    # Act
    findings = checker.check(ExpenseClaim(category="meal", amount_huf=1000), REFERENCE_DATE)

    # Assert
    finding = next(f for f in findings if f.rule_id == "MEAL-REQUIRED-DOCUMENTS")
    assert finding.status == "warning"
    assert finding.doc_ref == "01#business-meal-limit"
    assert "invoice" in finding.message
    assert "business_purpose_note" in finding.message
    assert "participant_list" in finding.message


def test_approval_threshold_passes_below_base_tier_when_approval_was_obtained(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=20000, approval_obtained=True),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-EQUIP-01") == "pass"


def test_approval_threshold_fails_below_base_tier_without_line_manager_approval(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=20000, approval_obtained=False),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-EQUIP-01") == "fail"


def test_approval_threshold_fails_when_required_but_not_obtained(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=80000, approval_obtained=False),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-EQUIP-01") == "fail"


def test_approval_threshold_passes_when_obtained(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=80000, approval_obtained=True), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "R-EQUIP-01") == "pass"


def test_approval_threshold_passes_exactly_at_the_base_tier_boundary(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=50000, approval_obtained=True),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-EQUIP-01") == "pass"


def test_approval_threshold_requires_approval_immediately_above_the_base_tier(checker):
    # Act
    findings = checker.check(ExpenseClaim(category="equipment", amount_huf=50001), REFERENCE_DATE)

    # Assert
    assert _status(findings, "R-EQUIP-01") == "warning"


def test_travel_domestic_trip_within_threshold_needs_only_line_manager(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="travel", expense_type="accommodation_domestic", amount_huf=100000),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-TRAVEL-01") == "warning"


def test_travel_domestic_trip_without_line_manager_approval_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="travel",
            expense_type="taxi",
            amount_huf=10000,
            is_international_trip=False,
            approval_obtained=False,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-TRAVEL-01") == "fail"


def test_travel_domestic_trip_above_threshold_requires_department_head(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="travel",
            expense_type="accommodation_domestic",
            amount_huf=160000,
            approval_obtained=False,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-TRAVEL-01") == "fail"


def test_travel_international_trip_requires_department_head_regardless_of_amount(checker):
    findings = checker.check(
        ExpenseClaim(
            category="travel",
            expense_type="accommodation_international",
            amount_huf=1000,
            is_international_trip=True,
            approval_obtained=True,
        ),
        REFERENCE_DATE,
    )

    assert _status(findings, "R-TRAVEL-01") == "pass"

    findings_without_approval = checker.check(
        ExpenseClaim(
            category="travel",
            expense_type="accommodation_international",
            amount_huf=1000,
            is_international_trip=True,
            approval_obtained=False,
        ),
        REFERENCE_DATE,
    )

    assert _status(findings_without_approval, "R-TRAVEL-01") == "fail"


def test_international_taxi_uses_explicit_trip_scope_for_approval(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="travel",
            expense_type="taxi",
            amount_huf=5000,
            is_international_trip=True,
            approval_obtained=False,
        ),
        REFERENCE_DATE,
    )

    # Assert
    approval = next(finding for finding in findings if finding.rule_id == "R-TRAVEL-01")
    assert approval.status == "fail"
    assert "department head" in approval.message


def test_travel_fine_is_an_explicit_fail(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="travel", expense_type="fine", amount_huf=5000), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "R-TRAVEL-04") == "fail"


def test_travel_minibar_is_an_explicit_fail(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="travel", expense_type="minibar", amount_huf=3000), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "R-TRAVEL-04") == "fail"


def test_travel_personal_expense_is_an_explicit_fail(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="travel",
            expense_type="taxi",
            amount_huf=2000,
            is_business_related=False,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-TRAVEL-04") == "fail"


def test_equipment_personal_use_is_an_explicit_fail(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="equipment",
            amount_huf=20000,
            is_business_related=False,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-EQUIP-02") == "fail"


def test_receipt_missing_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="equipment", amount_huf=1000, has_receipt=False), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "SUBMISSION-DOCUMENTS") == "fail"


def test_receipt_type_and_complete_document_set_pass(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="equipment",
            amount_huf=1000,
            has_receipt=True,
            provided_documents=["invoice", "managerial_approval"],
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "SUBMISSION-DOCUMENTS") == "pass"
    assert _status(findings, "EQUIPMENT-REQUIRED-DOCUMENTS") == "pass"


def test_commuting_below_minimum_distance_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="commuting", distance_km=5, distance_is_one_way=True), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "R-COMM-01") == "fail"


def test_commuting_at_or_above_minimum_distance_passes(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="commuting", distance_km=10, distance_is_one_way=True), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "R-COMM-01") == "pass"


def test_commuting_by_pass_is_not_applicable_to_the_minimum_distance_rule(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="commuting", expense_type="pass"), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "R-COMM-01") == "not_applicable"


def test_commuting_without_distance_facts_has_no_minimum_distance_finding(checker):
    # Act
    findings = checker.check(ExpenseClaim(category="commuting"), REFERENCE_DATE)

    # Assert
    assert "R-COMM-01" not in {f.rule_id for f in findings}


def test_benefits_exhausted_budget_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="benefits", expense_type="recreational", annual_budget_used_huf=120000
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-BEN-01") == "fail"


def test_benefits_without_reported_budget_usage_has_no_budget_finding(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="benefits", expense_type="recreational"), REFERENCE_DATE
    )

    # Assert
    assert "R-BEN-01" not in {f.rule_id for f in findings}


def test_benefits_unmatched_expense_type_has_no_approval_or_budget_finding(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="benefits", expense_type="unknown_benefit", amount_huf=1000),
        REFERENCE_DATE,
    )

    # Assert
    assert not {"R-BEN-01", "R-BEN-02", "R-BEN-03"} & {f.rule_id for f in findings}


def test_benefits_recreational_below_the_approval_threshold_has_no_approval_finding(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="benefits",
            expense_type="recreational",
            amount_huf=1000,
            annual_budget_used_huf=0,
        ),
        REFERENCE_DATE,
    )

    # Assert
    budget_finding = next(f for f in findings if f.rule_id == "R-BEN-01")
    assert "approval" not in budget_finding.message


def test_benefits_tenure_missing_is_a_warning_not_a_failure(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="benefits", expense_type="recreational", annual_budget_used_huf=0),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-BEN-TENURE") == "warning"


def test_benefits_tenure_below_requirement_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="benefits",
            expense_type="recreational",
            annual_budget_used_huf=0,
            tenure_months=5,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-BEN-TENURE") == "fail"


def test_benefits_tenure_at_exactly_the_requirement_passes(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="benefits",
            expense_type="recreational",
            annual_budget_used_huf=0,
            tenure_months=6,
        ),
        REFERENCE_DATE,
    )

    # Assert
    assert _status(findings, "R-BEN-TENURE") == "pass"


def test_benefits_no_carry_over_is_surfaced_as_an_explicit_finding(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(
            category="benefits",
            expense_type="recreational",
            annual_budget_used_huf=0,
            tenure_months=12,
        ),
        REFERENCE_DATE,
    )

    # Assert
    finding = next(f for f in findings if f.rule_id == "R-BEN-CARRY-OVER")
    assert finding.status == "pass"
    assert "does not carry over" in finding.message


def test_deadline_within_window_passes(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="meal", expense_date=date(2026, 7, 20)), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "SUBMISSION-DEADLINE") == "pass"


def test_deadline_expired_fails(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(category="meal", expense_date=date(2026, 1, 1)), REFERENCE_DATE
    )

    # Assert
    assert _status(findings, "SUBMISSION-DEADLINE") == "fail"


def test_deadline_does_not_require_an_expense_category(checker):
    # Act
    findings = checker.check(
        ExpenseClaim(expense_date=date(2026, 7, 20)), REFERENCE_DATE, "deadline_check"
    )

    # Assert
    assert _status(findings, "SUBMISSION-DEADLINE") == "pass"
    assert len(findings) == 1


def test_deadline_check_without_an_expense_date_returns_no_findings(checker):
    # Act
    findings = checker.check(ExpenseClaim(), REFERENCE_DATE, "deadline_check")

    # Assert
    assert findings == []


def test_document_requirements_without_a_category_returns_no_findings(checker):
    # Act
    findings = checker.check(ExpenseClaim(), REFERENCE_DATE, "document_requirements")

    # Assert
    assert findings == []


def test_no_category_returns_a_single_not_applicable_finding(checker):
    # Act
    findings = checker.check(ExpenseClaim(), REFERENCE_DATE)

    # Assert
    assert len(findings) == 1
    assert findings[0].status == "not_applicable"
