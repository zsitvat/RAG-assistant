from datetime import date

from app.agent.deadline_check import DeadlineChecker

CHECKER = DeadlineChecker(deadline_days=30)


def test_within_deadline():
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 10))

    assert result.status == "within_deadline"
    assert result.days_elapsed == 9
    assert result.days_remaining == 21


def test_due_soon():
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 27))

    assert result.status == "due_soon"


def test_expired_includes_exception_procedure():
    result = CHECKER.check(date(2026, 6, 1), date(2026, 7, 15))

    assert result.status == "expired"
    assert result.exception_procedure is not None


def test_day_29_is_within_the_deadline_window():
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 30))

    assert result.days_elapsed == 29
    assert result.status == "due_soon"
    assert result.exception_procedure is None


def test_day_30_is_the_last_day_within_the_deadline():
    result = CHECKER.check(date(2026, 7, 1), date(2026, 7, 31))

    assert result.days_elapsed == 30
    assert result.days_remaining == 0
    assert result.status == "due_soon"
    assert result.exception_procedure is None


def test_day_31_is_expired():
    result = CHECKER.check(date(2026, 7, 1), date(2026, 8, 1))

    assert result.days_elapsed == 31
    assert result.status == "expired"
    assert result.exception_procedure is not None
