import json
from pathlib import Path

import pytest

from app.rules.loader import load_rule_catalogue
from eval.model import DatasetValidationError, EvalCase, EvalDataset

CATALOGUE = load_rule_catalogue()
DATASET_PATH = Path("eval/dataset.json")


def _write(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(cases))
    return path


def _base_case(**overrides) -> dict:
    case = {
        "id": "case-1",
        "question": "What is the meal policy?",
        "reference_date": "2026-08-02",
        "expected_intent": "policy_question",
    }
    case.update(overrides)
    return case


def test_the_committed_dataset_has_20_cases_and_passes_validation():
    dataset = EvalDataset.load(DATASET_PATH, CATALOGUE)
    assert len(dataset.cases) == 20
    assert len({case.id for case in dataset.cases}) == 20


def test_dataset_load_rejects_duplicate_ids(tmp_path):
    path = _write(tmp_path, [_base_case(id="dup"), _base_case(id="dup")])

    with pytest.raises(DatasetValidationError, match="duplicate case id"):
        EvalDataset.load(path, CATALOGUE)


def test_dataset_load_rejects_an_unknown_category(tmp_path):
    path = _write(tmp_path, [_base_case(expected_category="not-a-real-category")])

    with pytest.raises(Exception, match="not-a-real-category"):
        EvalDataset.load(path, CATALOGUE)


def test_dataset_load_rejects_an_unknown_document_id(tmp_path):
    path = _write(tmp_path, [_base_case(expected_doc_ids=["99"])])

    with pytest.raises(DatasetValidationError, match="unknown document ids"):
        EvalDataset.load(path, CATALOGUE)


def test_eval_case_rejects_an_unknown_expected_slot_field():
    with pytest.raises(Exception, match="unknown expected_slots"):
        EvalCase.model_validate(_base_case(expected_slots={"not_a_real_field": 1}))


def test_eval_case_rejects_an_unknown_expected_tool():
    with pytest.raises(Exception, match="unknown expected_tools"):
        EvalCase.model_validate(_base_case(expected_tools=["not_a_real_tool"]))


def test_eval_case_rejects_a_needs_info_case_that_also_expects_tool_calls():
    with pytest.raises(Exception, match="cannot expect any tool calls"):
        EvalCase.model_validate(
            _base_case(expected_decision="needs_info", expected_tools=["search_policies"])
        )
