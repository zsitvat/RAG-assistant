from app.agent.model import ExpenseClaim
from app.agent.slots import RequiredSlotTable

TABLE = RequiredSlotTable()


def test_policy_question_needs_nothing():
    assert TABLE.missing("policy_question", None, ExpenseClaim()) == []


def test_document_requirements_needs_category():
    assert TABLE.missing("document_requirements", None, ExpenseClaim()) == ["category"]
    assert TABLE.missing("document_requirements", None, ExpenseClaim(category="meal")) == []


def test_meal_expense_check_reports_each_missing_slot():
    missing = TABLE.missing("expense_check", "meal", ExpenseClaim(amount_huf=1000))

    assert missing == ["headcount"]


def test_mileage_requires_distance_direction():
    missing = TABLE.missing("calculation", "mileage", ExpenseClaim(distance_km=10))

    assert missing == ["distance_is_one_way"]


def test_benefits_requires_the_benefit_type():
    claim = ExpenseClaim(amount_huf=1000, annual_budget_used_huf=0)

    missing = TABLE.missing("expense_check", "benefits", claim)

    assert missing == ["expense_type"]


def test_commuting_treats_ambiguous_direction_as_missing():
    claim = ExpenseClaim(distance_km=10, commute_days_per_month=20)

    missing = TABLE.missing("calculation", "commuting", claim)

    assert missing == ["distance_is_one_way"]


def test_unmapped_intent_category_has_no_required_slots():
    assert TABLE.missing("expense_check", "commuting", ExpenseClaim()) == []
