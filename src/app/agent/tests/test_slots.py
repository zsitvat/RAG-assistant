from app.agent.model import ExpenseClaim
from app.agent.slots import RequiredSlotTable

TABLE = RequiredSlotTable()


def test_policy_question_needs_nothing():
    assert TABLE.missing("policy_question", None, ExpenseClaim()) == []


def test_document_requirements_needs_category():
    assert TABLE.missing("document_requirements", None, ExpenseClaim()) == ["category"]
    assert TABLE.missing("document_requirements", None, ExpenseClaim(category="meal")) == []


def test_meal_expense_check_reports_each_missing_slot():
    # Act
    missing = TABLE.missing("expense_check", "meal", ExpenseClaim(amount_huf=1000))

    # Assert
    assert missing == ["headcount", "is_business_related", "non_reimbursable_amount"]


def test_travel_expense_check_requires_decision_facts():
    # Act
    missing = TABLE.missing("expense_check", "travel", ExpenseClaim())

    # Assert
    assert missing == [
        "expense_type",
        "amount_huf",
        "is_business_related",
        "is_international_trip",
    ]


def test_equipment_expense_check_requires_business_use():
    # Act
    missing = TABLE.missing("expense_check", "equipment", ExpenseClaim())

    # Assert
    assert missing == ["amount_huf", "is_business_related"]


def test_mileage_requires_distance_direction():
    # Act
    missing = TABLE.missing("calculation", "mileage", ExpenseClaim(distance_km=10))

    # Assert
    assert missing == ["distance_is_one_way"]


def test_benefits_requires_the_benefit_type():
    # Arrange
    claim = ExpenseClaim(amount_huf=1000, annual_budget_used_huf=0, tenure_months=12)

    # Act
    missing = TABLE.missing("expense_check", "benefits", claim)

    # Assert
    assert missing == ["expense_type"]


def test_commuting_treats_ambiguous_direction_as_missing():
    # Arrange
    claim = ExpenseClaim(expense_type="own_car", distance_km=10, commute_days_per_month=20)

    # Act
    missing = TABLE.missing("calculation", "commuting", claim)

    # Assert
    assert missing == ["distance_is_one_way"]


def test_commuting_asks_for_the_transport_mode_first():
    # Act
    missing = TABLE.missing("calculation", "commuting", ExpenseClaim())

    # Assert
    assert missing == ["expense_type"]


def test_commuting_pass_only_needs_the_purchase_price():
    # Act
    missing = TABLE.missing("calculation", "commuting", ExpenseClaim(expense_type="pass"))

    # Assert
    assert missing == ["amount_huf"]


def test_commuting_ticket_needs_price_and_office_days():
    # Act
    missing = TABLE.missing("calculation", "commuting", ExpenseClaim(expense_type="ticket"))

    # Assert
    assert missing == ["amount_huf", "commute_days_per_month"]


def test_commuting_vehicle_needs_the_distance_facts():
    # Act
    missing = TABLE.missing("calculation", "commuting", ExpenseClaim(expense_type="own_car"))

    # Assert
    assert missing == ["distance_km", "distance_is_one_way", "commute_days_per_month"]


def test_unmapped_intent_category_has_no_required_slots():
    assert TABLE.missing("calculation", "meal", ExpenseClaim()) == []
