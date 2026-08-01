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


def test_commuting_hybrid_prorates_the_monthly_cap(calculator):
    claim = ExpenseClaim(
        category="commuting", distance_km=15, distance_is_one_way=True, commute_days_per_month=10
    )

    result = calculator.calculate(claim)

    assert result.cap_huf == 20000
    assert result.amount_huf == 9000
    assert result.warnings


def test_equipment_reimburses_the_full_amount(calculator):
    claim = ExpenseClaim(category="equipment", amount_huf=80000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 80000
    assert result.cap_huf is None


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


def test_travel_without_a_known_expense_type_returns_submitted_amount(calculator):
    claim = ExpenseClaim(category="travel", expense_type="taxi", amount_huf=12000)

    result = calculator.calculate(claim)

    assert result.amount_huf == 12000
    assert result.cap_huf is None
    assert result.warnings


def test_missing_category_raises(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(ExpenseClaim())
