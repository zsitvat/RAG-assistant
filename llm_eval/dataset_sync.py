from langfuse import Langfuse
from langfuse.api.commons.errors.not_found_error import NotFoundError

from llm_eval.model import EvalCase, EvalDataset

DATASET_NAME = "test-dataset"


class LangfuseDatasetSync:
    """Idempotently synchronises the local functional dataset to a Langfuse dataset."""

    def __init__(self, client: Langfuse, dataset_name: str = DATASET_NAME) -> None:
        """Stores the Langfuse client and the target dataset name."""
        self._client = client
        self._dataset_name = dataset_name

    def sync(self, dataset: EvalDataset) -> None:
        """Ensures the dataset exists, then upserts every case as a dataset item by stable id."""
        self._ensure_dataset_exists()
        for case in dataset.cases:
            self._client.create_dataset_item(
                dataset_name=self._dataset_name,
                id=case.id,
                input=self._input(case),
                expected_output=self._expected_output(case),
            )

    def _ensure_dataset_exists(self) -> None:
        """Creates the dataset if it does not already exist."""
        try:
            self._client.get_dataset(self._dataset_name)
        except NotFoundError:
            self._client.create_dataset(name=self._dataset_name)

    @staticmethod
    def _input(case: EvalCase) -> dict:
        """Builds the dataset item input the eval runner's task function consumes."""
        return {"question": case.question, "reference_date": case.reference_date.isoformat()}

    @staticmethod
    def _expected_output(case: EvalCase) -> dict:
        """Builds the dataset item expected output the metric evaluators consume."""
        return {
            "expected_intent": case.expected_intent,
            "expected_category": case.expected_category,
            "expected_slots": case.expected_slots,
            "expected_tools": case.expected_tools,
            "expected_doc_ids": case.expected_doc_ids,
            "expected_amount_huf": case.expected_amount_huf,
            "expected_decision": case.expected_decision,
        }
