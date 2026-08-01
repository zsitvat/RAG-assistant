from datetime import date

from app.agent.deadline import DeadlineChecker

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
