from datetime import date

from app.agent.deadline_check import DeadlineChecker

CHECKER = DeadlineChecker(deadline_days=30)


def test_day_9_is_well_within_the_deadline_window():
    # Act
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 10))

    # Assert
    assert result.status == "within_deadline"
    assert result.days_elapsed == 9
    assert result.days_remaining == 21


def test_day_26_is_due_soon():
    # Act
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 27))

    # Assert
    assert result.status == "due_soon"


def test_day_44_is_expired_and_includes_the_exception_procedure():
    # Act
    result = CHECKER.check(date(2026, 6, 1), date(2026, 7, 15))

    # Assert
    assert result.status == "expired"
    assert result.exception_procedure is not None


def test_day_29_is_within_the_deadline_window():
    # Act
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 30))

    # Assert
    assert result.days_elapsed == 29
    assert result.status == "due_soon"
    assert result.exception_procedure is None


def test_day_30_is_the_last_day_within_the_deadline():
    # Act
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 31))

    # Assert
    assert result.days_elapsed == 30
    assert result.days_remaining == 0
    assert result.status == "due_soon"
    assert result.exception_procedure is None


def test_day_31_is_expired():
    # Act
    result = CHECKER.check(date(2026, 7, 1), date(2026, 8, 1))

    # Assert
    assert result.days_elapsed == 31
    assert result.status == "expired"
    assert result.exception_procedure is not None
