from datetime import date

from langfuse import Evaluation


class EvaluationMetrics:
    """Computes the six deterministic functional-evaluation scores as Langfuse evaluators."""

    def classification_accuracy(self, *, output: dict, expected_output: dict, **_: object):
        """Scores whether the classified intent, and category when expected, both match."""
        expected_category = expected_output.get("expected_category")
        intent_match = output.get("intent") == expected_output.get("expected_intent")
        category_match = expected_category is None or output.get("category") == expected_category
        return Evaluation(
            name="classification_accuracy",
            value=bool(intent_match and category_match),
            comment=f"intent={output.get('intent')!r} category={output.get('category')!r}",
        )

    def slot_accuracy(self, *, output: dict, expected_output: dict, **_: object):
        """Scores the share of expected slots whose extracted claim value matches exactly."""
        expected_slots: dict = expected_output.get("expected_slots") or {}
        if not expected_slots:
            return []
        claim = output.get("claim") or {}
        matched = sum(
            1
            for name, value in expected_slots.items()
            if self._slot_matches(claim.get(name), value)
        )
        total = len(expected_slots)
        return Evaluation(
            name="slot_accuracy", value=matched / total, comment=f"{matched}/{total} slots matched"
        )

    def retrieval_hit_at_4(self, *, output: dict, expected_output: dict, **_: object):
        """Scores whether at least one expected document appears in the retrieved top four."""
        expected_docs = expected_output.get("expected_doc_ids") or []
        if not expected_docs:
            return []
        retrieved = set(output.get("retrieved_doc_ids") or [])
        return Evaluation(name="retrieval_hit_at_4", value=bool(retrieved & set(expected_docs)))

    def tool_selection_accuracy(self, *, output: dict, expected_output: dict, **_: object):
        """Scores whether the ordered tool-call list equals the expected tool list exactly."""
        actual = output.get("tool_calls") or []
        expected = expected_output.get("expected_tools") or []
        return Evaluation(
            name="tool_selection_accuracy", value=actual == expected, comment=f"tool_calls={actual}"
        )

    def outcome_accuracy(self, *, output: dict, expected_output: dict, **_: object):
        """Scores whether the decision, and calculated amount when expected, both match."""
        decision_match = output.get("decision") == expected_output.get("expected_decision")
        expected_amount = expected_output.get("expected_amount_huf")
        if expected_amount is None:
            return Evaluation(name="outcome_accuracy", value=bool(decision_match))
        calculation = output.get("calculation") or {}
        amount_match = calculation.get("amount_huf") == expected_amount
        return Evaluation(name="outcome_accuracy", value=bool(decision_match and amount_match))

    def citation_accuracy(self, *, output: dict, expected_output: dict, **_: object):
        """Scores whether the answer's context cites at least one expected document."""
        expected_docs = expected_output.get("expected_doc_ids") or []
        if not expected_docs:
            return []
        cited = set(output.get("cited_doc_ids") or [])
        return Evaluation(name="citation_accuracy", value=bool(cited & set(expected_docs)))

    @classmethod
    def _slot_matches(cls, actual: object, expected: object) -> bool:
        """Compares one claim field to its expected value, tolerating JSON type drift."""
        if isinstance(expected, str) and cls._looks_like_date(expected):
            return str(actual) == expected
        if isinstance(expected, bool) or isinstance(actual, bool):
            return actual is expected
        if isinstance(expected, int | float) and isinstance(actual, int | float):
            return float(actual) == float(expected)
        return actual == expected

    @staticmethod
    def _looks_like_date(value: str) -> bool:
        """Reports whether a string parses as an ISO date, so it can compare a date field."""
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True
