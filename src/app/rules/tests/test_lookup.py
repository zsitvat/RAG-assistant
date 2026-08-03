from app.rules.lookup import first_matching
from app.rules.model import RuleDefinition


def test_first_matching_returns_none_for_an_empty_list():
    # Act / Assert
    assert first_matching([], lambda rule: True) is None


def test_first_matching_returns_none_when_nothing_matches():
    # Arrange
    rules = [RuleDefinition(id="R-1"), RuleDefinition(id="R-2")]

    # Act / Assert
    assert first_matching(rules, lambda rule: rule.limit_per_person_huf is not None) is None


def test_first_matching_returns_the_first_of_several_matches():
    # Arrange
    rules = [
        RuleDefinition(id="R-1"),
        RuleDefinition(id="R-2", limit_per_person_huf=10000),
        RuleDefinition(id="R-3", limit_per_person_huf=20000),
    ]

    # Act
    rule = first_matching(rules, lambda rule: rule.limit_per_person_huf is not None)

    # Assert
    assert rule.id == "R-2"
