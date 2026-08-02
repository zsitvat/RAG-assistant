import json
from types import SimpleNamespace

from langfuse import Evaluation

from eval.report import EvaluationReport


def _item_result(case_id: str, output, evaluations, trace_id="trace-1"):
    return SimpleNamespace(
        item=SimpleNamespace(id=case_id),
        output=output,
        evaluations=evaluations,
        trace_id=trace_id,
    )


def test_build_aggregates_pass_rates_across_scored_cases():
    item_results = [
        _item_result(
            "case-1", {"decision": "eligible"}, [Evaluation(name="outcome_accuracy", value=True)]
        ),
        _item_result(
            "case-2",
            {"decision": "not_eligible"},
            [Evaluation(name="outcome_accuracy", value=False)],
        ),
    ]
    report = EvaluationReport(
        run_name="run-1", model_name="qwen2.5:7b", dataset_run_url=None
    ).build(item_results)

    assert report["case_count"] == 2
    assert report["aggregates"]["outcome_accuracy"] == {"percent": 50.0, "scored_cases": 2}


def test_build_excludes_none_valued_evaluations_from_the_aggregate():
    item_results = [
        _item_result("case-1", {}, [Evaluation(name="slot_accuracy", value=None)]),
    ]
    report = EvaluationReport(
        run_name="run-1", model_name="qwen2.5:7b", dataset_run_url=None
    ).build(item_results)

    assert report["aggregates"]["slot_accuracy"] == {"percent": None, "scored_cases": 0}


def test_build_marks_a_failed_item_without_aborting_the_report():
    item_results = [
        _item_result("case-1", {"error": "connection refused"}, []),
        _item_result(
            "case-2", {"decision": "eligible"}, [Evaluation(name="outcome_accuracy", value=True)]
        ),
    ]
    report = EvaluationReport(
        run_name="run-1", model_name="qwen2.5:7b", dataset_run_url=None
    ).build(item_results)

    failed = next(case for case in report["cases"] if case["id"] == "case-1")
    assert failed["failed"] is True
    assert failed["error"] == "connection refused"
    assert report["case_count"] == 2


def test_write_produces_a_markdown_and_json_report_with_failure_notes(tmp_path):
    item_results = [
        _item_result("case-1", {"error": "boom"}, [], trace_id="trace-abc"),
    ]
    writer = EvaluationReport(
        run_name="run-1", model_name="qwen2.5:7b", dataset_run_url="https://cloud.langfuse.com/x"
    )
    report = writer.build(item_results)

    md_path, json_path = writer.write(report, tmp_path, "20260802-1000")

    assert md_path.exists() and json_path.exists()
    markdown = md_path.read_text()
    assert "case-1" in markdown
    assert "trace-abc" in markdown
    assert "boom" in markdown
    assert json.loads(json_path.read_text())["run_name"] == "run-1"
