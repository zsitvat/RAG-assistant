from collections.abc import Callable
from datetime import date

from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime

from app.agent.calculator import ReimbursementCalculator
from app.agent.model import CalculationResult, ExpenseClaim, Finding
from app.agent.rule_checker import RuleChecker
from app.rag.tool import build_search_policies_tool

CALCULATE_DESCRIPTION = (
    "Compute the reimbursable amount for the current claim. Never do arithmetic yourself. "
    "Search the policies first so the final answer has supporting evidence."
)
CHECK_RULES_DESCRIPTION = (
    "Check eligibility, caps, approval thresholds, receipt requirements and the submission "
    "deadline against the rule catalogue for the current claim."
)


def build_calculate_tool(calculator: ReimbursementCalculator) -> BaseTool:
    """Builds the tool that computes the reimbursable amount for the current claim."""

    @tool(response_format="content_and_artifact", description=CALCULATE_DESCRIPTION)
    def calculate(runtime: ToolRuntime) -> tuple[str, CalculationResult]:
        """Calculates the reimbursement result for the claim held in the graph state."""
        claim = ExpenseClaim.model_validate(runtime.state["claim"])
        result = calculator.calculate(claim)
        return result.compact_summary(), result

    return calculate


def build_check_rules_tool(
    rule_checker: RuleChecker, reference_date_provider: Callable[[], date]
) -> BaseTool:
    """Builds the tool that checks the current claim against the rule catalogue."""

    @tool(response_format="content_and_artifact", description=CHECK_RULES_DESCRIPTION)
    def check_rules(runtime: ToolRuntime) -> tuple[str, list[Finding]]:
        """Checks the claim held in the graph state against all applicable rules."""
        claim = ExpenseClaim.model_validate(runtime.state["claim"])
        findings = rule_checker.check(claim, reference_date_provider(), runtime.state.get("intent"))
        summary = "; ".join(f"{f.rule_id}:{f.status}" for f in findings) or "no applicable rules"
        return summary, findings

    return check_rules


def build_tools(
    rag_graph: CompiledStateGraph,
    calculator: ReimbursementCalculator,
    rule_checker: RuleChecker,
    reference_date_provider: Callable[[], date],
) -> list[BaseTool]:
    """Builds the full set of tools available to the agent."""
    return [
        build_search_policies_tool(rag_graph),
        build_calculate_tool(calculator),
        build_check_rules_tool(rule_checker, reference_date_provider),
    ]
