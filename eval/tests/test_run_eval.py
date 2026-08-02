from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.integrations.langfuse import Observability
from app.settings import Settings
from eval.run_eval import EvaluationRunner


def _runner() -> EvaluationRunner:
    settings = Settings(llm_backend="dummy", langfuse_enabled=False)
    return EvaluationRunner(settings, Observability(None))


def test_http_task_posts_the_case_and_returns_the_parsed_response():
    runner = _runner()
    item = SimpleNamespace(id="meal-01", input={"question": "hi?", "reference_date": "2026-08-02"})
    fake_response = MagicMock()
    fake_response.json.return_value = {"intent": "policy_question"}
    fake_response.raise_for_status.return_value = None

    with patch("eval.run_eval.httpx2.post", return_value=fake_response) as post:
        result = runner._http_task(item=item)

    assert result == {"intent": "policy_question"}
    assert post.call_args.kwargs["json"]["thread_id"] == "eval-meal-01"
    assert post.call_args.kwargs["json"]["dataset_item_id"] == "meal-01"


def test_http_task_returns_an_error_marker_instead_of_raising_on_failure():
    import httpx2

    runner = _runner()
    item = SimpleNamespace(id="meal-01", input={"question": "hi?", "reference_date": "2026-08-02"})

    with patch("eval.run_eval.httpx2.post", side_effect=httpx2.ConnectError("refused")):
        result = runner._http_task(item=item)

    assert "error" in result
    assert "refused" in result["error"]


def test_intent_task_returns_the_classifier_output():
    runner = _runner()
    fake_result = SimpleNamespace(value=SimpleNamespace(intent="expense_check", category="meal"))
    runner._classify_runner = MagicMock(run=MagicMock(return_value=fake_result))
    item = SimpleNamespace(id="meal-01", input={"question": "hi?", "reference_date": "2026-08-02"})

    result = runner._intent_task(item=item)

    assert result == {"intent": "expense_check", "category": "meal"}


def test_intent_task_returns_an_error_marker_on_failure():
    runner = _runner()
    runner._classify_runner = MagicMock(run=MagicMock(side_effect=RuntimeError("boom")))
    item = SimpleNamespace(id="meal-01", input={"question": "hi?", "reference_date": "2026-08-02"})

    result = runner._intent_task(item=item)

    assert result == {"error": "boom"}
