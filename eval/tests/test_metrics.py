from eval.metrics import EvaluationMetrics

METRICS = EvaluationMetrics()


def test_classification_accuracy_requires_intent_and_expected_category_to_match():
    expected = {"expected_intent": "expense_check", "expected_category": "meal"}

    correct = METRICS.classification_accuracy(
        output={"intent": "expense_check", "category": "meal"}, expected_output=expected
    )
    wrong_category = METRICS.classification_accuracy(
        output={"intent": "expense_check", "category": "travel"}, expected_output=expected
    )

    assert correct.value is True
    assert wrong_category.value is False


def test_classification_accuracy_ignores_category_when_none_is_expected():
    expected = {"expected_intent": "policy_question", "expected_category": None}

    result = METRICS.classification_accuracy(
        output={"intent": "policy_question", "category": "meal"}, expected_output=expected
    )

    assert result.value is True


def test_slot_accuracy_reports_the_share_of_matching_slots():
    expected = {"expected_slots": {"amount_huf": 48000, "headcount": 4}}

    result = METRICS.slot_accuracy(
        output={"claim": {"amount_huf": 48000.0, "headcount": 5}}, expected_output=expected
    )

    assert result.value == 0.5


def test_slot_accuracy_scores_nothing_when_no_slots_are_expected():
    result = METRICS.slot_accuracy(output={"claim": {}}, expected_output={"expected_slots": {}})

    assert result == []


def test_slot_accuracy_compares_dates_as_strings():
    expected = {"expected_slots": {"expense_date": "2026-07-10"}}

    result = METRICS.slot_accuracy(
        output={"claim": {"expense_date": "2026-07-10"}}, expected_output=expected
    )

    assert result.value == 1.0


def test_retrieval_hit_at_4_true_when_an_expected_document_was_retrieved():
    expected = {"expected_doc_ids": ["01", "06"]}

    hit = METRICS.retrieval_hit_at_4(
        output={"retrieved_doc_ids": ["03", "01"]}, expected_output=expected
    )
    miss = METRICS.retrieval_hit_at_4(
        output={"retrieved_doc_ids": ["03"]}, expected_output=expected
    )

    assert hit.value is True
    assert miss.value is False


def test_retrieval_hit_at_4_scores_nothing_when_no_documents_are_expected():
    result = METRICS.retrieval_hit_at_4(output={"retrieved_doc_ids": []}, expected_output={})

    assert result == []


def test_tool_selection_accuracy_requires_exact_ordered_match():
    expected = {"expected_tools": ["search_policies", "calculate", "check_rules"]}

    exact = METRICS.tool_selection_accuracy(
        output={"tool_calls": ["search_policies", "calculate", "check_rules"]},
        expected_output=expected,
    )
    reordered = METRICS.tool_selection_accuracy(
        output={"tool_calls": ["calculate", "search_policies", "check_rules"]},
        expected_output=expected,
    )

    assert exact.value is True
    assert reordered.value is False


def test_outcome_accuracy_checks_decision_and_amount_when_expected():
    expected = {"expected_decision": "eligible", "expected_amount_huf": 48000}

    correct = METRICS.outcome_accuracy(
        output={"decision": "eligible", "calculation": {"amount_huf": 48000}},
        expected_output=expected,
    )
    wrong_amount = METRICS.outcome_accuracy(
        output={"decision": "eligible", "calculation": {"amount_huf": 1}}, expected_output=expected
    )

    assert correct.value is True
    assert wrong_amount.value is False


def test_outcome_accuracy_checks_only_decision_when_no_amount_is_expected():
    expected = {"expected_decision": "needs_info", "expected_amount_huf": None}

    result = METRICS.outcome_accuracy(output={"decision": "needs_info"}, expected_output=expected)

    assert result.value is True


def test_citation_accuracy_true_only_when_a_cited_document_is_expected():
    expected = {"expected_doc_ids": ["01"]}

    cited = METRICS.citation_accuracy(output={"cited_doc_ids": ["01"]}, expected_output=expected)
    not_cited = METRICS.citation_accuracy(
        output={"cited_doc_ids": ["06"]}, expected_output=expected
    )

    assert cited.value is True
    assert not_cited.value is False


def test_citation_accuracy_scores_nothing_when_no_documents_are_expected():
    result = METRICS.citation_accuracy(output={"cited_doc_ids": []}, expected_output={})

    assert result == []
