import argparse
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from langchain_core.messages import HumanMessage

from app.agent.langfuse_prompt_library import PromptLibrary
from app.agent.model import IntentClassification
from app.agent.structured import StructuredOutputRunner
from app.integrations.langfuse import Observability
from app.integrations.llm import build_chat_model
from app.rules.loader import load_rule_catalogue
from app.settings import Settings, get_settings
from llm_eval.dataset_sync import DATASET_NAME, LangfuseDatasetSync
from llm_eval.judge import JUDGE_PROMPT, AnswerJudgeVerdict
from llm_eval.metrics import EvaluationMetrics
from llm_eval.model import EvalDataset
from llm_eval.report import EvaluationReport

DATASET_PATH = Path(__file__).parent / "dataset.json"
REPORT_DIR = Path("evaluation_results")
REQUEST_TIMEOUT_SECONDS = 180


class EvaluationRunner:
    """Syncs the dataset, runs it as a Langfuse experiment, and writes the local reports.

    The six deterministic metrics always evaluate whatever LLM_MODEL the live endpoint is
    configured with; there is no independent reference model to confirm a low score there is a
    real capability limit rather than a harness/prompt defect. The answer_quality judge is the one
    exception: it runs on EVAL_JUDGE_MODEL, independently configurable from LLM_MODEL, to avoid
    the tested model grading its own answers. See technical design §13.3 "Known limitation".
    """

    def __init__(self, settings: Settings, observability: Observability) -> None:
        """Stores settings and the observability adapter used for sync, scoring and prompts."""
        self._settings = settings
        self._client = observability.client
        self._metrics = EvaluationMetrics(
            answer_judge=StructuredOutputRunner(
                build_chat_model(settings, settings.eval_judge_model),
                JUDGE_PROMPT,
                AnswerJudgeVerdict,
            )
        )
        self._classify_runner = StructuredOutputRunner(
            build_chat_model(settings),
            PromptLibrary(observability).get("classify_intent").template,
            IntentClassification,
        )

    def run(self, *, node: str | None) -> dict:
        """Syncs the dataset, runs the full or single-node experiment, and returns the report."""
        catalogue = load_rule_catalogue()
        dataset = EvalDataset.load(DATASET_PATH, catalogue)
        LangfuseDatasetSync(self._client).sync(dataset)
        langfuse_dataset = self._client.get_dataset(DATASET_NAME)

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_name = f"functional-{node or 'full'}-{timestamp}"
        task = self._intent_task if node == "intent" else self._http_task
        evaluators = (
            [self._metrics.classification_accuracy]
            if node == "intent"
            else [
                self._metrics.classification_accuracy,
                self._metrics.slot_accuracy,
                self._metrics.retrieval_hit_at_4,
                self._metrics.tool_selection_accuracy,
                self._metrics.outcome_accuracy,
                self._metrics.citation_accuracy,
                self._metrics.answer_quality,
            ]
        )

        result = langfuse_dataset.run_experiment(
            name=run_name, task=task, evaluators=evaluators, max_concurrency=4
        )

        report_writer = EvaluationReport(
            run_name=result.run_name,
            model_name=self._settings.llm_model,
            dataset_run_url=result.dataset_run_url,
        )
        report = report_writer.build(result.item_results)
        md_path, json_path = report_writer.write(report, REPORT_DIR, timestamp)
        report["report_paths"] = {"markdown": str(md_path), "json": str(json_path)}
        return report

    def _http_task(self, *, item, **kwargs) -> dict:
        """Posts one case to the deployed API and returns the parsed evaluation response."""
        try:
            response = httpx2.post(
                f"{self._settings.api_base_url}/admin/eval",
                json={
                    "thread_id": f"eval-{item.id}",
                    "message": item.input["question"],
                    "reference_date": item.input["reference_date"],
                    "dataset_item_id": item.id,
                    "experiment_name": DATASET_NAME,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except httpx2.HTTPError as exc:
            return {"error": str(exc)}

    def _intent_task(self, *, item, **kwargs) -> dict:
        """Runs only the classifier in-process, for the fast intent-only evaluation mode."""
        try:
            context = [HumanMessage(content=item.input["question"])]
            result = self._classify_runner.run(
                context, fallback=IntentClassification(intent="policy_question")
            )
            return {"intent": result.value.intent, "category": result.value.category}
        except Exception as exc:
            return {"error": str(exc)}


def main() -> None:
    """CLI entry point: `python -m llm_eval.run_eval [--node intent]`."""
    parser = argparse.ArgumentParser(description="Run the functional evaluation dataset.")
    parser.add_argument("--node", choices=["intent"], default=None)
    args = parser.parse_args()

    settings = get_settings()
    observability = Observability.build(settings)
    if observability.client is None:
        raise SystemExit(
            "Langfuse must be enabled and configured (LANGFUSE_ENABLED=true plus credentials) "
            "to run the functional evaluation."
        )

    report = EvaluationRunner(settings, observability).run(node=args.node)
    print(f"Wrote {report['report_paths']['markdown']}")
    if report["dataset_run_url"]:
        print(f"Langfuse run: {report['dataset_run_url']}")


if __name__ == "__main__":
    main()
