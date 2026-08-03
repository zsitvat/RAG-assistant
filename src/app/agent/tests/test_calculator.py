import pytest

from app.agent.calculator import CalculationInputError, ReimbursementCalculator
from app.agent.model import ExpenseClaim
from app.rules.loader import load_rule_catalogue

CATALOGUE = load_rule_catalogue()


@pytest.fixture
def calculator() -> ReimbursementCalculator:
    return ReimbursementCalculator(CATALOGUE)


def test_meal_caps_at_per_person_limit(calculator):
    # Arrange
    # rules.yaml meal.limit_per_person_huf = 15000 (R-MEAL-01); cap = 15000 * headcount
    claim = ExpenseClaim(
        category="meal", amount_huf=50000, headcount=3, non_reimbursable_amount=3000
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 45000
    assert result.cap_huf == 45000
    assert result.excess_huf == 2000


def test_meal_requires_amount_and_headcount(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(ExpenseClaim(category="meal", amount_huf=1000))


def test_meal_below_cap_reimburses_the_full_amount(calculator):
    # Arrange
    # rules.yaml meal.limit_per_person_huf = 15000 (R-MEAL-01); cap = 15000 * headcount
    claim = ExpenseClaim(category="meal", amount_huf=20000, headcount=2)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 20000
    assert result.cap_huf == 30000
    assert result.excess_huf == 0
    assert result.warnings == []


def test_meal_exactly_at_cap_reimburses_the_full_amount(calculator):
    # Arrange
    # rules.yaml meal.limit_per_person_huf = 15000 (R-MEAL-01); cap = 15000 * headcount
    claim = ExpenseClaim(category="meal", amount_huf=45000, headcount=3)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 45000
    assert result.cap_huf == 45000
    assert result.excess_huf == 0


def test_meal_excluded_item_amount_is_reported_as_a_warning(calculator):
    # Arrange
    claim = ExpenseClaim(
        category="meal", amount_huf=20000, headcount=2, non_reimbursable_amount=3000
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 17000
    assert result.excess_huf == 0
    assert any("3000" in warning for warning in result.warnings)


def test_meal_rounds_half_up(calculator):
    # Arrange
    claim = ExpenseClaim(category="meal", amount_huf=10000.5, headcount=1)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 10001


def test_meal_missing_catalogue_limit_returns_a_lower_confidence_result(calculator):
    # Arrange
    meal_rules = CATALOGUE.categories["meal"]
    broken_meal_rules = meal_rules.model_copy(
        update={"rules": [r for r in meal_rules.rules if r.limit_per_person_huf is None]}
    )
    broken_catalogue = CATALOGUE.model_copy(
        update={"categories": {**CATALOGUE.categories, "meal": broken_meal_rules}}
    )
    broken_calculator = ReimbursementCalculator(broken_catalogue)

    # Act
    result = broken_calculator.calculate(
        ExpenseClaim(category="meal", amount_huf=1000, headcount=1)
    )

    # Assert
    assert result.amount_huf == 1000
    assert result.cap_huf is None
    assert "confidence" in result.warnings[0]


def test_mileage_doubles_one_way_distance(calculator):
    # Arrange
    # rules.yaml mileage.rate_huf_per_km = 45 (R-MILE-01); amount = 100 km * 2 * 45
    claim = ExpenseClaim(category="mileage", distance_km=100, distance_is_one_way=True)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 9000


def test_mileage_round_trip_is_not_doubled(calculator):
    # Arrange
    # rules.yaml mileage.rate_huf_per_km = 45 (R-MILE-01); amount = 200 km * 45, not doubled
    claim = ExpenseClaim(category="mileage", distance_km=200, distance_is_one_way=False)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 9000


def test_commuting_full_time_uses_full_monthly_cap(calculator):
    # Arrange
    # rules.yaml commuting.rate_huf_per_km = 30, monthly_cap_huf = 40000 (R-COMM-02)
    # amount = 15 km * 2 (round trip) * 22 office days * 30 = 19800, below the cap
    claim = ExpenseClaim(
        category="commuting", distance_km=15, distance_is_one_way=True, commute_days_per_month=22
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 19800
    assert result.cap_huf == 40000
    assert result.warnings == []


def test_commuting_hybrid_scales_distance_by_office_days_not_the_cap(calculator):
    # Arrange
    # rules.yaml commuting.rate_huf_per_km = 30 (R-COMM-02); amount = 15 km * 2 * 10 days * 30
    claim = ExpenseClaim(
        category="commuting", distance_km=15, distance_is_one_way=True, commute_days_per_month=10
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 9000
    assert result.cap_huf == 40000
    assert result.excess_huf == 0


def test_commuting_reproduces_the_policy_worked_example(calculator):
    # Arrange
    # rules.yaml commuting.rate_huf_per_km = 30 (R-COMM-02); amount = 18 km * 2 * 10 days * 30
    claim = ExpenseClaim(
        category="commuting", distance_km=18, distance_is_one_way=True, commute_days_per_month=10
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 10800
    assert result.cap_huf == 40000


def test_commuting_vehicle_caps_at_the_flat_monthly_maximum(calculator):
    # Arrange
    # rules.yaml commuting.monthly_cap_huf = 40000 (R-COMM-02)
    # raw = 60 km * 2 * 20 days * 30 = 72000, capped at 40000, excess = 32000
    claim = ExpenseClaim(
        category="commuting", distance_km=60, distance_is_one_way=True, commute_days_per_month=20
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 40000
    assert result.cap_huf == 40000
    assert result.excess_huf == 32000


def test_commuting_pass_applies_the_reimbursement_ratio(calculator):
    # Arrange
    # rules.yaml commuting.pass_reimbursement_ratio = 0.8, monthly_cap_huf = 30000 (R-COMM-03)
    claim = ExpenseClaim(category="commuting", expense_type="pass", amount_huf=20000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 16000
    assert result.cap_huf == 30000
    assert result.excess_huf == 0


def test_commuting_pass_caps_at_the_monthly_maximum(calculator):
    # Arrange
    # rules.yaml commuting.pass_reimbursement_ratio = 0.8, monthly_cap_huf = 30000 (R-COMM-03)
    # raw = 50000 * 0.8 = 40000, capped at 30000, excess = 10000
    claim = ExpenseClaim(category="commuting", expense_type="pass", amount_huf=50000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 30000
    assert result.cap_huf == 30000
    assert result.excess_huf == 10000


def test_commuting_ticket_caps_by_daily_limit_times_office_days(calculator):
    # Arrange
    # rules.yaml commuting.ticket_reimbursement_ratio = 0.8, daily_cap_huf = 3000 (R-COMM-04)
    # cap = 3000 * 8 days = 24000; raw = 30000 * 0.8 = 24000, so the cap and ratio coincide here
    claim = ExpenseClaim(
        category="commuting", expense_type="ticket", amount_huf=30000, commute_days_per_month=8
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.cap_huf == 24000
    assert result.amount_huf == 24000
    assert result.excess_huf == 0


def test_commuting_ticket_below_the_daily_cap_applies_the_ratio(calculator):
    # Arrange
    # rules.yaml commuting.ticket_reimbursement_ratio = 0.8, daily_cap_huf = 3000 (R-COMM-04)
    # amount = 10000 * 0.8 = 8000, below the 3000 * 8 days = 24000 cap
    claim = ExpenseClaim(
        category="commuting", expense_type="ticket", amount_huf=10000, commute_days_per_month=8
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 8000
    assert result.cap_huf == 24000


def test_mileage_uses_the_same_rate_for_every_powertrain(calculator):
    # Arrange
    # rules.yaml mileage.rate_huf_per_km = 45 (R-MILE-01); amount = 250 km * 45
    electric = ExpenseClaim(
        category="mileage", expense_type="electric", distance_km=250, distance_is_one_way=False
    )
    petrol = ExpenseClaim(
        category="mileage", expense_type="petrol", distance_km=250, distance_is_one_way=False
    )

    # Assert
    assert calculator.calculate(electric).amount_huf == 11250
    assert calculator.calculate(petrol).amount_huf == 11250


def test_equipment_reimburses_the_full_amount(calculator):
    # Arrange
    claim = ExpenseClaim(category="equipment", amount_huf=80000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 80000
    assert result.cap_huf is None


def test_benefits_unused_budget_applies_the_full_reimbursement_ratio(calculator):
    # Arrange
    # rules.yaml benefits.training: annual_budget_huf = 200000, reimbursement_ratio = 0.8 (R-BEN-02)
    claim = ExpenseClaim(
        category="benefits",
        expense_type="training",
        amount_huf=100000,
        annual_budget_used_huf=0,
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 80000
    assert result.cap_huf == 200000
    assert result.excess_huf == 0


def test_benefits_exhausted_budget_reimburses_nothing(calculator):
    # Arrange
    # rules.yaml benefits.recreational: annual_budget_huf = 120000 (R-BEN-01); used == budget
    claim = ExpenseClaim(
        category="benefits",
        expense_type="recreational",
        amount_huf=50000,
        annual_budget_used_huf=120000,
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 0
    assert result.cap_huf == 0
    assert result.excess_huf == 50000


def test_benefits_caps_at_remaining_budget(calculator):
    # Arrange
    # rules.yaml benefits.recreational: annual_budget_huf = 120000, reimbursement_ratio = 1.0
    # (R-BEN-01); remaining = 120000 - 90000 = 30000
    claim = ExpenseClaim(
        category="benefits",
        expense_type="recreational",
        amount_huf=50000,
        annual_budget_used_huf=90000,
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
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
    # Arrange
    # rules.yaml travel.accommodation_limit_huf_per_night.domestic = 45000 (R-TRAVEL-02)
    claim = ExpenseClaim(category="travel", expense_type="accommodation_domestic", amount_huf=60000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 45000
    assert result.excess_huf == 15000


def test_travel_accommodation_caps_by_international_tier(calculator):
    # Arrange
    # rules.yaml travel.accommodation_limit_huf_per_night.international = 55000 (R-TRAVEL-02)
    claim = ExpenseClaim(
        category="travel", expense_type="accommodation_international", amount_huf=70000
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 55000
    assert result.excess_huf == 15000


def test_travel_meal_per_diem_domestic_returns_the_daily_limit(calculator):
    # Arrange
    # rules.yaml travel.meal_per_diem_huf.domestic = 18000 (R-TRAVEL-03)
    claim = ExpenseClaim(category="travel", expense_type="meal_per_diem_domestic", amount_huf=20000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 18000
    assert result.cap_huf == 18000


def test_travel_meal_per_diem_international_returns_the_daily_limit(calculator):
    # Arrange
    # rules.yaml travel.meal_per_diem_huf.international = 30000 (R-TRAVEL-03)
    claim = ExpenseClaim(
        category="travel", expense_type="meal_per_diem_international", amount_huf=35000
    )

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 30000
    assert result.cap_huf == 30000


def test_travel_without_a_known_expense_type_returns_submitted_amount(calculator):
    # Arrange
    claim = ExpenseClaim(category="travel", expense_type="taxi", amount_huf=12000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 12000
    assert result.cap_huf is None
    assert result.warnings


def test_travel_parking_returns_submitted_amount(calculator):
    # Arrange
    claim = ExpenseClaim(category="travel", expense_type="parking", amount_huf=3000)

    # Act
    result = calculator.calculate(claim)

    # Assert
    assert result.amount_huf == 3000
    assert result.cap_huf is None


def test_missing_category_raises(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(ExpenseClaim())


def test_general_category_has_no_calculation_defined(calculator):
    with pytest.raises(CalculationInputError):
        calculator.calculate(ExpenseClaim(category="general", amount_huf=1000))
