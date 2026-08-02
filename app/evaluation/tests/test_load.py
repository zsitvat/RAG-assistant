import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.evaluation.load import LoadTestRunner, LoadTestValidationError


class _FakeItemResult:
    def __init__(self, item, output):
        self.item = item
        self.output = output


class _FakeDataset:
    """Mimics dataset.run_experiment() so LoadTestRunner can be tested without a real Langfuse
    call."""

    def __init__(self, items):
        self.items = items
        self.calls = []

    def run_experiment(self, *, name, task, max_concurrency, metadata):
        self.calls.append({"name": name, "max_concurrency": max_concurrency, "metadata": metadata})
        item_results = [_FakeItemResult(item, asyncio.run(task(item=item))) for item in self.items]
        return SimpleNamespace(
            dataset_run_url=f"https://fake.test/{name}", item_results=item_results
        )


def _items(n: int) -> list:
    return [SimpleNamespace(id=f"case-{i}", input={"question": f"question {i}"}) for i in range(n)]


def _observability(dataset: _FakeDataset) -> MagicMock:
    observability = MagicMock()
    observability.client.get_dataset.return_value = dataset
    return observability


def test_run_rejects_max_concurrency_outside_one_to_four():
    dataset = _FakeDataset(_items(20))
    runner = LoadTestRunner(MagicMock(), _observability(dataset))

    with pytest.raises(LoadTestValidationError, match="max_concurrency"):
        runner.run("rag-assistant-functional", repetitions=3, max_concurrency=5)


def test_run_rejects_a_resolved_total_below_50_measured_turns():
    dataset = _FakeDataset(_items(20))
    runner = LoadTestRunner(MagicMock(), _observability(dataset))

    with pytest.raises(LoadTestValidationError, match="measured turns"):
        runner.run("rag-assistant-functional", repetitions=1, max_concurrency=4)


def test_run_rejects_a_resolved_total_above_200_measured_turns():
    dataset = _FakeDataset(_items(20))
    runner = LoadTestRunner(MagicMock(), _observability(dataset))

    with pytest.raises(LoadTestValidationError, match="measured turns"):
        runner.run("rag-assistant-functional", repetitions=11, max_concurrency=4)


def test_run_forwards_repetitions_concurrency_and_a_shared_load_run_id():
    dataset = _FakeDataset(_items(20))
    agent_service = MagicMock()
    runner = LoadTestRunner(agent_service, _observability(dataset))

    result = runner.run("rag-assistant-functional", repetitions=3, max_concurrency=2)

    assert len(dataset.calls) == 3
    assert {call["max_concurrency"] for call in dataset.calls} == {2}
    load_run_ids = {call["metadata"]["load_run_id"] for call in dataset.calls}
    assert load_run_ids == {result.load_run_id}


def test_run_uses_a_fresh_thread_id_per_item_and_repetition():
    dataset = _FakeDataset(_items(2))
    agent_service = MagicMock()
    runner = LoadTestRunner(agent_service, _observability(dataset))

    runner.run("rag-assistant-functional", repetitions=25, max_concurrency=1)

    thread_ids = {call.args[0] for call in agent_service.respond.call_args_list}
    assert len(thread_ids) == len(agent_service.respond.call_args_list)


def test_run_counts_query_count_and_reports_dataset_run_urls():
    dataset = _FakeDataset(_items(20))
    agent_service = MagicMock()
    runner = LoadTestRunner(agent_service, _observability(dataset))

    result = runner.run("rag-assistant-functional", repetitions=3, max_concurrency=4)

    assert result.query_count == 60
    assert result.max_concurrency == 4
    assert result.dataset_name == "rag-assistant-functional"
    assert len(result.dataset_run_urls) == 3
    assert result.error_count == 0
    assert result.latency_mean_ms >= 0
    assert result.latency_p95_ms >= result.latency_median_ms >= 0


def test_run_isolates_item_failures_into_the_error_count():
    dataset = _FakeDataset(_items(20))
    agent_service = MagicMock()
    call_count = {"n": 0}

    def _respond(thread_id, question):
        call_count["n"] += 1
        if call_count["n"] % 5 == 0:
            raise RuntimeError("boom")
        return SimpleNamespace(decision=None)

    agent_service.respond.side_effect = _respond
    runner = LoadTestRunner(agent_service, _observability(dataset))

    result = runner.run("rag-assistant-functional", repetitions=3, max_concurrency=4)

    assert result.error_count == 12
    assert result.query_count == 60


def test_percentile_and_throughput_are_computed_from_raw_samples():
    assert LoadTestRunner._percentile([100, 200, 300, 400, 500], 95) == 500
    assert LoadTestRunner._percentile([], 95) == 0.0
    assert LoadTestRunner._throughput(60, 0) == 0.0
    assert LoadTestRunner._throughput(60, 60_000) == 60.0
