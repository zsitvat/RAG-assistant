import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, model_validator

from app.agent.model import Decision, ExpenseClaim, Intent
from app.rules.model import Category, RuleCatalogue

KNOWN_TOOLS = {"search_policies", "calculate", "check_rules"}
KNOWN_SLOT_FIELDS = set(ExpenseClaim.model_fields)


class DatasetValidationError(RuntimeError):
    """Raised when the functional evaluation dataset fails validation."""


class EvalCase(BaseModel):
    """One functional evaluation case: a question plus its deterministic expected outcome."""

    id: str
    question: str
    reference_date: date
    expected_intent: Intent
    expected_category: Category | None = None
    expected_slots: dict[str, object] = {}
    expected_tools: list[str] = []
    expected_doc_ids: list[str] = []
    expected_amount_huf: float | None = None
    expected_decision: Decision | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "EvalCase":
        """Rejects unknown slot names, unknown tool names and an impossible decision shape."""
        errors: list[str] = []
        unknown_slots = sorted(set(self.expected_slots) - KNOWN_SLOT_FIELDS)
        if unknown_slots:
            errors.append(f"case {self.id!r}: unknown expected_slots fields {unknown_slots}")
        unknown_tools = sorted(set(self.expected_tools) - KNOWN_TOOLS)
        if unknown_tools:
            errors.append(f"case {self.id!r}: unknown expected_tools {unknown_tools}")
        if self.expected_decision == "needs_info" and self.expected_tools:
            errors.append(f"case {self.id!r}: a needs_info case cannot expect any tool calls")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class EvalDataset:
    """Loads and validates the version-controlled functional evaluation dataset."""

    def __init__(self, cases: list[EvalCase]) -> None:
        """Stores the validated cases."""
        self.cases = cases

    @classmethod
    def load(cls, path: Path, rule_catalogue: RuleCatalogue) -> "EvalDataset":
        """Loads dataset.json, validates every case, and rejects references to unknown facts."""
        raw_cases = json.loads(path.read_text())
        cases = [EvalCase.model_validate(item) for item in raw_cases]
        errors = cls._cross_reference_errors(cases, rule_catalogue)
        if errors:
            raise DatasetValidationError("; ".join(errors))
        return cls(cases)

    @staticmethod
    def _cross_reference_errors(cases: list[EvalCase], rule_catalogue: RuleCatalogue) -> list[str]:
        """Returns errors for duplicate ids and references to unknown categories or documents."""
        errors: list[str] = []
        seen_ids: set[str] = set()
        for case in cases:
            if case.id in seen_ids:
                errors.append(f"duplicate case id {case.id!r}")
            seen_ids.add(case.id)

            if case.expected_category is not None and case.expected_category not in (
                rule_catalogue.categories
            ):
                errors.append(f"case {case.id!r}: unknown category {case.expected_category!r}")

            unknown_docs = [d for d in case.expected_doc_ids if d not in rule_catalogue.documents]
            if unknown_docs:
                errors.append(f"case {case.id!r}: unknown document ids {unknown_docs}")
        return errors
