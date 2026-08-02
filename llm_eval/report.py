import json
from pathlib import Path

METRIC_ORDER = [
    "classification_accuracy",
    "slot_accuracy",
    "retrieval_hit_at_4",
    "tool_selection_accuracy",
    "outcome_accuracy",
    "citation_accuracy",
    "answer_quality",
]


class EvaluationReport:
    """Builds and writes the local Markdown and JSON summary for one evaluation run."""

    def __init__(self, run_name: str, model_name: str, dataset_run_url: str | None) -> None:
        """Stores the run identity and metadata shown in the report header."""
        self._run_name = run_name
        self._model_name = model_name
        self._dataset_run_url = dataset_run_url

    def build(self, item_results: list) -> dict:
        """Aggregates per-case evaluations into percentages and a per-case outcome table."""
        cases = [self._case_summary(item_result) for item_result in item_results]
        aggregates = self._aggregate(cases)
        return {
            "run_name": self._run_name,
            "model": self._model_name,
            "dataset_run_url": self._dataset_run_url,
            "case_count": len(cases),
            "aggregates": aggregates,
            "cases": cases,
        }

    @staticmethod
    def _case_summary(item_result) -> dict:
        """Extracts one case's id, per-metric scores and failure detail from its item result."""
        item = item_result.item
        case_id = item.id if hasattr(item, "id") else item.get("metadata", {}).get("id")
        output = item_result.output
        failed = isinstance(output, dict) and "error" in output
        return {
            "id": case_id,
            "trace_id": item_result.trace_id,
            "failed": failed,
            "error": output.get("error") if failed else None,
            "scores": {
                evaluation.name: evaluation.value
                for evaluation in item_result.evaluations
                if evaluation.value is not None
            },
        }

    @staticmethod
    def _aggregate(cases: list[dict]) -> dict:
        """Computes the pass percentage for each metric across cases that reported it."""
        aggregates: dict[str, dict] = {}
        for metric in METRIC_ORDER:
            values = [case["scores"][metric] for case in cases if metric in case["scores"]]
            if not values:
                aggregates[metric] = {"percent": None, "scored_cases": 0}
                continue
            passed = sum(1 for value in values if value is True or value == 1)
            aggregates[metric] = {
                "percent": round(100 * passed / len(values), 1),
                "scored_cases": len(values),
            }
        return aggregates

    def write(self, report: dict, output_dir: Path, timestamp: str) -> tuple[Path, Path]:
        """Writes the Markdown and JSON reports and returns their paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"functional-{timestamp}.json"
        md_path = output_dir / f"functional-{timestamp}.md"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        md_path.write_text(self._render_markdown(report))
        return md_path, json_path

    def _render_markdown(self, report: dict) -> str:
        """Renders the report as a summary table, per-case rows and failure notes."""
        lines = [
            f"# Functional evaluation — {report['run_name']}",
            "",
            f"Model: `{report['model']}`  \nCases: {report['case_count']}",
        ]
        if report["dataset_run_url"]:
            lines.append(f"Langfuse run: {report['dataset_run_url']}")
        lines += [
            "",
            "## Aggregate scores",
            "",
            "| Metric | Pass rate | Scored cases |",
            "| --- | --- | --- |",
        ]
        for metric in METRIC_ORDER:
            aggregate = report["aggregates"][metric]
            percent = "n/a" if aggregate["percent"] is None else f"{aggregate['percent']}%"
            lines.append(f"| {metric} | {percent} | {aggregate['scored_cases']} |")

        lines += ["", "## Per-case results", "", "| Case | Result |", "| --- | --- |"]
        for case in report["cases"]:
            summary = (
                f"ERROR: {case['error']}"
                if case["failed"]
                else ", ".join(f"{k}={v}" for k, v in case["scores"].items())
            )
            lines.append(f"| {case['id']} | {summary} |")

        failures = [case for case in report["cases"] if case["failed"]]
        if failures:
            lines += ["", "## Failure notes", ""]
            for case in failures:
                lines.append(f"- **{case['id']}** (trace `{case['trace_id']}`): {case['error']}")
        return "\n".join(lines) + "\n"
