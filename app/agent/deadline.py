from datetime import date
from typing import Literal

from pydantic import BaseModel

DUE_SOON_THRESHOLD_DAYS = 5
EXCEPTION_PROCEDURE = (
    "Submit a late-claim exception request to Finance with a written justification."
)


class DeadlineResult(BaseModel):
    """Outcome of a deadline check against a submission's reference date."""

    days_elapsed: int
    days_remaining: int
    status: Literal["within_deadline", "due_soon", "expired"]
    exception_procedure: str | None = None


class DeadlineChecker:
    """Determines whether a submission falls within, near, or past its deadline."""

    def __init__(self, deadline_days: int) -> None:
        """Stores the number of days allowed between expense and submission."""
        self._deadline_days = deadline_days

    def check(self, expense_date: date, reference_date: date) -> DeadlineResult:
        """Computes the deadline status for an expense relative to a reference date."""
        days_elapsed = (reference_date - expense_date).days
        days_remaining = self._deadline_days - days_elapsed
        if days_remaining < 0:
            status = "expired"
        elif days_remaining <= DUE_SOON_THRESHOLD_DAYS:
            status = "due_soon"
        else:
            status = "within_deadline"
        return DeadlineResult(
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            status=status,
            exception_procedure=EXCEPTION_PROCEDURE if status == "expired" else None,
        )
