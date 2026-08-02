import pytest

from app.agent.calculator import CalculationInputError, ReimbursementCalculator
from app.agent.model import ExpenseClaim
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


@pytest.fixture
def calculator() -> ReimbursementCalculator:
    return ReimbursementCalculator(CATALOGUE)


def test_meal_caps_at_per_person_limit(calculator):
    claim = ExpenseClaim(
        category="meal", amount_huf=50000, headcount=3, non_reimbursable_amount=3000
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 45000
    assert result.cap_huf == 45000
    assert result.excess_huf == 2000


def test_meal_requires_amount_and_headcount(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(ExpenseClaim(category="meal", amount_huf=1000))


def test_meal_below_cap_reimburses_the_full_amount(calculator):
    claim = ExpenseClaim(category="meal", amount_huf=20000, headcount=2)

    result = calculator.calculate(claim)

    assert result.amount_huf == 20000
    assert result.cap_huf == 30000
    assert result.excess_huf == 0
    assert result.warnings == []


def test_meal_exactly_at_cap_reimburses_the_full_amount(calculator):
    claim = ExpenseClaim(category="meal", amount_huf=45000, headcount=3)

    result = calculator.calculate(claim)

    assert result.amount_huf == 45000
    assert result.cap_huf == 45000
    assert result.excess_huf == 0


def test_meal_excluded_item_amount_is_reported_as_a_warning(calculator):
    claim = ExpenseClaim(
        category="meal", amount_huf=20000, headcount=2, non_reimbursable_amount=3000
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 17000
    assert result.excess_huf == 0
    assert any("3000" in warning for warning in result.warnings)


def test_meal_rounds_half_up(calculator):
    claim = ExpenseClaim(category="meal", amount_huf=10000.5, headcount=1)

    result = calculator.calculate(claim)

    assert result.amount_huf == 10001


def test_meal_missing_catalogue_limit_returns_a_lower_confidence_result(calculator):
    meal_rules = CATALOGUE.categories["meal"]
    broken_meal_rules = meal_rules.model_copy(
        update={"rules": [r for r in meal_rules.rules if r.limit_per_person_huf is None]}
    )
    broken_catalogue = CATALOGUE.model_copy(
        update={"categories": {**CATALOGUE.categories, "meal": broken_meal_rules}}
    )
    broken_calculator = ReimbursementCalculator(broken_catalogue)

    result = broken_calculator.calculate(
        ExpenseClaim(category="meal", amount_huf=1000, headcount=1)
    )

    assert result.amount_huf == 1000
    assert result.cap_huf is None
    assert "confidence" in result.warnings[0]


def test_mileage_doubles_one_way_distance(calculator):
    claim = ExpenseClaim(category="mileage", distance_km=100, distance_is_one_way=True)

    result = calculator.calculate(claim)

    assert result.amount_huf == 9000


def test_mileage_round_trip_is_not_doubled(calculator):
    claim = ExpenseClaim(category="mileage", distance_km=200, distance_is_one_way=False)

    result = calculator.calculate(claim)

    assert result.amount_huf == 9000


def test_commuting_full_time_uses_full_monthly_cap(calculator):
    claim = ExpenseClaim(
        category="commuting", distance_km=15, distance_is_one_way=True, commute_days_per_month=22
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 19800
    assert result.cap_huf == 40000
    assert result.warnings == []


def test_commuting_hybrid_scales_distance_by_office_days_not_the_cap(calculator):
    claim = ExpenseClaim(
        category="commuting", distance_km=15, distance_is_one_way=True, commute_days_per_month=10
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 9000
    assert result.cap_huf == 40000
    assert result.excess_huf == 0


def test_commuting_reproduces_the_policy_worked_example(calculator):
    claim = ExpenseClaim(
        category="commuting", distance_km=18, distance_is_one_way=True, commute_days_per_month=10
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 10800
    assert result.cap_huf == 40000


def test_commuting_vehicle_caps_at_the_flat_monthly_maximum(calculator):
    claim = ExpenseClaim(
        category="commuting", distance_km=60, distance_is_one_way=True, commute_days_per_month=20
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 40000
    assert result.cap_huf == 40000
    assert result.excess_huf == 32000


def test_commuting_pass_applies_the_reimbursement_ratio(calculator):
    claim = ExpenseClaim(category="commuting", expense_type="pass", amount_huf=20000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 16000
    assert result.cap_huf == 30000
    assert result.excess_huf == 0


def test_commuting_pass_caps_at_the_monthly_maximum(calculator):
    claim = ExpenseClaim(category="commuting", expense_type="pass", amount_huf=50000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 30000
    assert result.cap_huf == 30000
    assert result.excess_huf == 10000


def test_commuting_ticket_caps_by_daily_limit_times_office_days(calculator):
    claim = ExpenseClaim(
        category="commuting", expense_type="ticket", amount_huf=30000, commute_days_per_month=8
    )

    result = calculator.calculate(claim)

    assert result.cap_huf == 24000
    assert result.amount_huf == 24000
    assert result.excess_huf == 0


def test_commuting_ticket_below_the_daily_cap_applies_the_ratio(calculator):
    claim = ExpenseClaim(
        category="commuting", expense_type="ticket", amount_huf=10000, commute_days_per_month=8
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 8000
    assert result.cap_huf == 24000


def test_mileage_uses_the_same_rate_for_every_powertrain(calculator):
    electric = ExpenseClaim(
        category="mileage", expense_type="electric", distance_km=250, distance_is_one_way=False
    )
    petrol = ExpenseClaim(
        category="mileage", expense_type="petrol", distance_km=250, distance_is_one_way=False
    )

    assert calculator.calculate(electric).amount_huf == 11250
    assert calculator.calculate(petrol).amount_huf == 11250


def test_equipment_reimburses_the_full_amount(calculator):
    claim = ExpenseClaim(category="equipment", amount_huf=80000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 80000
    assert result.cap_huf is None


def test_benefits_unused_budget_applies_the_full_reimbursement_ratio(calculator):
    claim = ExpenseClaim(
        category="benefits",
        expense_type="training",
        amount_huf=100000,
        annual_budget_used_huf=0,
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 80000
    assert result.cap_huf == 200000
    assert result.excess_huf == 0


def test_benefits_exhausted_budget_reimburses_nothing(calculator):
    claim = ExpenseClaim(
        category="benefits",
        expense_type="recreational",
        amount_huf=50000,
        annual_budget_used_huf=120000,
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 0
    assert result.cap_huf == 0
    assert result.excess_huf == 50000


def test_benefits_caps_at_remaining_budget(calculator):
    claim = ExpenseClaim(
        category="benefits",
        expense_type="recreational",
        amount_huf=50000,
        annual_budget_used_huf=90000,
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 30000
    assert result.excess_huf == 20000


def test_benefits_unknown_type_raises(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(
            ExpenseClaim(
                category="benefits", expense_type="unknown", amount_huf=1, annual_budget_used_huf=0
            )
        )


def test_travel_accommodation_caps_by_domestic_tier(calculator):
    claim = ExpenseClaim(category="travel", expense_type="accommodation_domestic", amount_huf=60000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 45000
    assert result.excess_huf == 15000


def test_travel_accommodation_caps_by_international_tier(calculator):
    claim = ExpenseClaim(
        category="travel", expense_type="accommodation_international", amount_huf=70000
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 55000
    assert result.excess_huf == 15000


def test_travel_meal_per_diem_domestic_returns_the_daily_limit(calculator):
    claim = ExpenseClaim(category="travel", expense_type="meal_per_diem_domestic", amount_huf=20000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 18000
    assert result.cap_huf == 18000


def test_travel_meal_per_diem_international_returns_the_daily_limit(calculator):
    claim = ExpenseClaim(
        category="travel", expense_type="meal_per_diem_international", amount_huf=35000
    )

    result = calculator.calculate(claim)

    assert result.amount_huf == 30000
    assert result.cap_huf == 30000


def test_travel_without_a_known_expense_type_returns_submitted_amount(calculator):
    claim = ExpenseClaim(category="travel", expense_type="taxi", amount_huf=12000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 12000
    assert result.cap_huf is None
    assert result.warnings


def test_travel_parking_returns_submitted_amount(calculator):
    claim = ExpenseClaim(category="travel", expense_type="parking", amount_huf=3000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 3000
    assert result.cap_huf is None


def test_missing_category_raises(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(ExpenseClaim())
