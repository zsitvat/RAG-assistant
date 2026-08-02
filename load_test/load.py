import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from app.agent.service import AgentService
from app.dependencies import ApplicationDependencies
from app.integrations.langfuse import Observability
from app.settings import Settings, get_settings

DEFAULT_DATASET_NAME = "test-dataset"
MIN_MEASURED_TURNS = 50
MAX_MEASURED_TURNS = 200
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 4
REPORT_DIR = Path("evaluation_results")


class LoadTestValidationError(ValueError):
    """Raised when the requested load-test parameters resolve outside the allowed bounds."""


class LoadTestResult(BaseModel):
    """Aggregates latency, throughput and error counts for one load-test run."""

    load_run_id: str
    dataset_name: str
    query_count: int
    max_concurrency: int
    total_duration_ms: int
    throughput_queries_per_minute: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    error_count: int
    dataset_run_urls: list[str]


class LoadTestRunner:
    """Replays a Langfuse dataset through the agent module chat uses, bounded by concurrency."""

    def __init__(self, agent_service: AgentService, observability: Observability) -> None:
        """Stores the agent service invoked per turn and the Langfuse client owning the dataset."""
        self._agent_service = agent_service
        self._observability = observability

    def run(self, dataset_name: str, repetitions: int, max_concurrency: int) -> LoadTestResult:
        """Runs the named dataset `repetitions` times and returns the aggregated load result."""
        if not (MIN_CONCURRENCY <= max_concurrency <= MAX_CONCURRENCY):
            raise LoadTestValidationError(
                f"max_concurrency must be between {MIN_CONCURRENCY} and {MAX_CONCURRENCY}, "
                f"got {max_concurrency}"
            )

        dataset = self._observability.client.get_dataset(dataset_name)
        query_count = len(dataset.items) * repetitions
        if not (MIN_MEASURED_TURNS <= query_count <= MAX_MEASURED_TURNS):
            raise LoadTestValidationError(
                f"resolved run of {query_count} measured turns ({len(dataset.items)} items x "
                f"{repetitions} repetitions) must be between {MIN_MEASURED_TURNS} and "
                f"{MAX_MEASURED_TURNS}"
            )

        load_run_id = f"load-{uuid.uuid4().hex[:12]}"
        latencies_ms: list[float] = []
        dataset_run_urls: list[str] = []
        error_count = 0

        start = time.monotonic()
        for repetition in range(repetitions):
            result = dataset.run_experiment(
                name=f"{load_run_id}-rep{repetition}",
                task=self._build_task(load_run_id, repetition),
                max_concurrency=max_concurrency,
                metadata={"load_run_id": load_run_id},
            )
            if result.dataset_run_url:
                dataset_run_urls.append(result.dataset_run_url)
            for item_result in result.item_results:
                output = item_result.output
                if isinstance(output, dict) and "error" in output:
                    error_count += 1
                elif isinstance(output, dict) and "elapsed_ms" in output:
                    latencies_ms.append(output["elapsed_ms"])
        total_duration_ms = (time.monotonic() - start) * 1000

        return LoadTestResult(
            load_run_id=load_run_id,
            dataset_name=dataset_name,
            query_count=query_count,
            max_concurrency=max_concurrency,
            total_duration_ms=round(total_duration_ms),
            throughput_queries_per_minute=self._throughput(query_count, total_duration_ms),
            latency_mean_ms=round(statistics.mean(latencies_ms), 1) if latencies_ms else 0.0,
            latency_median_ms=round(statistics.median(latencies_ms), 1) if latencies_ms else 0.0,
            latency_p95_ms=round(self._percentile(latencies_ms, 95), 1),
            error_count=error_count,
            dataset_run_urls=dataset_run_urls,
        )

    def _build_task(self, load_run_id: str, repetition: int) -> Callable:
        """Builds the per-repetition task that times one complete graph invocation."""

        async def task(*, item, **_kwargs) -> dict:
            """Runs one dataset item through the same agent module used by chat, timed."""
            thread_id = f"{load_run_id}-rep{repetition}-{item.id}"
            start = time.monotonic()
            try:
                await asyncio.to_thread(
                    self._agent_service.respond, thread_id, item.input["question"]
                )
                return {"elapsed_ms": (time.monotonic() - start) * 1000}
            except Exception as exc:
                return {"error": str(exc)}

        return task

    @staticmethod
    def _throughput(query_count: int, total_duration_ms: float) -> float:
        """Returns queries per minute, or 0.0 when there is no measurable duration."""
        if total_duration_ms <= 0:
            return 0.0
        return round(query_count / (total_duration_ms / 1000 / 60), 2)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Returns the given percentile of a latency sample, or 0.0 when empty."""
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1)
        return ordered[max(0, index)]


async def _build_runner(settings: Settings, observability: Observability) -> LoadTestRunner:
    """Builds the full application dependency graph and wraps its agent service for load testing."""
    dependencies = await ApplicationDependencies.build(settings)
    return LoadTestRunner(dependencies.agent_service, observability)


def main() -> None:
    """CLI entry point: `python -m load_test.load [--dataset-name ...] [--repetitions N] \
[--max-concurrency N]`."""
    parser = argparse.ArgumentParser(
        description="Replay a Langfuse dataset through the live agent graph under bounded "
        "concurrency to measure latency and throughput."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    args = parser.parse_args()

    settings = get_settings()
    observability = Observability.build(settings)
    if observability.client is None:
        raise SystemExit(
            "Langfuse must be enabled and configured (LANGFUSE_ENABLED=true plus credentials) "
            "to run the load test."
        )

    runner = asyncio.run(_build_runner(settings, observability))
    try:
        result = runner.run(args.dataset_name, args.repetitions, args.max_concurrency)
    except LoadTestValidationError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"load-{timestamp}.json"
    report_path.write_text(json.dumps(result.model_dump(), indent=2))

    print(json.dumps(result.model_dump(), indent=2))
    print(f"Wrote {report_path}")
    for url in result.dataset_run_urls:
        print(f"Langfuse run: {url}")


if __name__ == "__main__":
    main()
