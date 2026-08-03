from app.rag.model import Citation, RagResult, RetrievedResult

_RESULT_KWARGS = {
    "doc_id": "doc-1",
    "doc_title": "Meal Policy",
    "section_id": "s1",
    "section": "Daily limits",
    "categories": ["meal"],
    "rule_ids": ["meal.daily_cap"],
    "source_path": "meal.md",
    "content": "Meals are reimbursed up to ...",
    "similarity": 0.87,
}


def _lc_wrap(module: str, name: str, kwargs: dict) -> dict:
    """Builds the LangChain checkpoint envelope a pydantic object round-trips through."""
    return {"lc": 2, "type": "constructor", "id": [module, name], "kwargs": kwargs}


def test_from_artifact_returns_the_instance_unchanged():
    # Arrange
    result = RagResult(results=[RetrievedResult(**_RESULT_KWARGS)])

    # Act
    rebuilt = RagResult.from_artifact(result)

    # Assert
    assert rebuilt is result


def test_from_artifact_returns_empty_result_for_none():
    # Act
    rebuilt = RagResult.from_artifact(None)

    # Assert
    assert rebuilt == RagResult()


def test_from_artifact_unwraps_a_single_lc_envelope():
    # Arrange
    wrapped = _lc_wrap("app.rag.model", "RagResult", {"results": [], "context": "no evidence"})

    # Act
    rebuilt = RagResult.from_artifact(wrapped)

    # Assert
    assert rebuilt == RagResult(context="no evidence")


def test_from_artifact_unwraps_individually_wrapped_nested_results_and_citations():
    # Arrange: a checkpoint that couldn't revive RetrievedResult/Citation leaves each one
    # as its own lc envelope nested inside the already-unwrapped RagResult kwargs.
    wrapped = _lc_wrap(
        "app.rag.model",
        "RagResult",
        {
            "results": [_lc_wrap("app.rag.model", "RetrievedResult", _RESULT_KWARGS)],
            "citations": [
                _lc_wrap(
                    "app.rag.model",
                    "Citation",
                    {
                        "marker": "[1]",
                        "doc_id": "doc-1",
                        "doc_title": "Meal Policy",
                        "section": None,
                    },
                )
            ],
        },
    )

    # Act
    rebuilt = RagResult.from_artifact(wrapped)

    # Assert
    assert rebuilt.results == [RetrievedResult(**_RESULT_KWARGS)]
    assert rebuilt.citations == [
        Citation(marker="[1]", doc_id="doc-1", doc_title="Meal Policy", section=None)
    ]
