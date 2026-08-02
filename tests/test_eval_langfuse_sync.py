from unittest.mock import MagicMock

from langfuse.api.commons.errors.not_found_error import NotFoundError

from eval.langfuse_sync import DATASET_NAME, LangfuseDatasetSync
from eval.model import EvalCase


def _case(**overrides) -> EvalCase:
    payload = {
        "id": "case-1",
        "question": "What is the meal policy?",
        "reference_date": "2026-08-02",
        "expected_intent": "policy_question",
    }
    payload.update(overrides)
    return EvalCase.model_validate(payload)


def test_sync_creates_the_dataset_when_it_does_not_exist():
    client = MagicMock()
    client.get_dataset.side_effect = NotFoundError(body="not found")
    dataset = MagicMock(cases=[_case()])

    LangfuseDatasetSync(client).sync(dataset)

    client.create_dataset.assert_called_once_with(name=DATASET_NAME)


def test_sync_does_not_recreate_an_existing_dataset():
    client = MagicMock()
    dataset = MagicMock(cases=[_case()])

    LangfuseDatasetSync(client).sync(dataset)

    client.create_dataset.assert_not_called()


def test_sync_upserts_each_case_by_its_stable_id():
    client = MagicMock()
    case = _case(
        id="meal-01",
        expected_category="meal",
        expected_slots={"amount_huf": 48000},
        expected_tools=["search_policies"],
        expected_doc_ids=["01"],
        expected_amount_huf=48000,
        expected_decision="eligible",
    )
    dataset = MagicMock(cases=[case])

    LangfuseDatasetSync(client).sync(dataset)

    client.create_dataset_item.assert_called_once_with(
        dataset_name=DATASET_NAME,
        id="meal-01",
        input={"question": case.question, "reference_date": "2026-08-02"},
        expected_output={
            "expected_intent": "policy_question",
            "expected_category": "meal",
            "expected_slots": {"amount_huf": 48000},
            "expected_tools": ["search_policies"],
            "expected_doc_ids": ["01"],
            "expected_amount_huf": 48000,
            "expected_decision": "eligible",
        },
    )


def test_sync_is_idempotent_across_repeated_runs():
    client = MagicMock()
    dataset = MagicMock(cases=[_case()])

    LangfuseDatasetSync(client).sync(dataset)
    LangfuseDatasetSync(client).sync(dataset)

    assert client.create_dataset_item.call_count == 2
    first_call, second_call = client.create_dataset_item.call_args_list
    assert first_call == second_call
