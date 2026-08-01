import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from app.agent.current_request import CurrentRequest
from app.agent.messages import (
    CLARIFICATION_QUESTIONS,
    DEFAULT_CLARIFICATION_QUESTION,
    INCOMPLETE_EVIDENCE_NOTE,
    LLM_UNAVAILABLE_MESSAGE,
    NO_TOOL_ARTIFACT_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
)
from app.agent.model import CalculationResult, Decision, ExpenseClaim, IntentClassification
from app.agent.prompts import (
    AGENT_STEP_PROMPT,
    CLASSIFY_INTENT_PROMPT,
    EXTRACT_INFORMATION_PROMPT,
    GENERATE_RESPONSE_PROMPT,
)
from app.agent.slots import RequiredSlotTable
from app.agent.state import MAX_AGENT_STEPS, MAX_TOOL_ARG_ERRORS, AgentState
from app.agent.structured import StructuredOutputRunner

logger = logging.getLogger(__name__)


class AgentNodes:
    """Implements the LangGraph node callbacks that drive the agent's conversation flow."""

    def __init__(
        self,
        structured_chat_model: BaseChatModel,
        response_chat_model: BaseChatModel,
        tools: list[BaseTool],
    ) -> None:
        """Stores the chat models and tools used by the graph nodes."""
        self._tools = tools
        self._agent_step_model = structured_chat_model
        self._response_model = response_chat_model
        self._classify_runner = StructuredOutputRunner(
            structured_chat_model, CLASSIFY_INTENT_PROMPT, IntentClassification
        )
        self._extract_runner = StructuredOutputRunner(
            structured_chat_model, EXTRACT_INFORMATION_PROMPT, ExpenseClaim
        )
        self._slot_table = RequiredSlotTable()

    @property
    def tools(self) -> list[BaseTool]:
        """Returns the tools available to the agent step."""
        return self._tools

    def classify_intent(self, state: AgentState) -> AgentState:
        """Classifies the user's intent and, if applicable, the expense category."""
        context = CurrentRequest(state["messages"]).model_context()
        classification = self._classify_runner.run(
            context, fallback=IntentClassification(intent="policy_question")
        )
        return {"intent": classification.intent, "category": classification.category}

    def extract_information(self, state: AgentState) -> AgentState:
        """Extracts and merges expense claim fields from the conversation so far."""
        previous_claim = state.get("claim") or ExpenseClaim()
        previous_decision = state.get("decision")
        category = state.get("category")

        context = CurrentRequest(state["messages"]).model_context()
        extracted = self._extract_runner.run(context, fallback=ExpenseClaim())
        if category is not None:
            extracted = extracted.model_copy(update={"category": category})

        is_continuation = previous_decision == "needs_info" and (
            category is None or category == previous_claim.category
        )
        claim = previous_claim.merged_with(extracted) if is_continuation else extracted
        return {"claim": claim, "decision": None}

    def route_after_extraction(self, state: AgentState) -> str:
        """Routes to clarification, the agent step, or out-of-scope handling."""
        if state["intent"] == "unsupported":
            return "out_of_scope"
        missing = self._slot_table.missing(state["intent"], state.get("category"), state["claim"])
        return "ask_clarification" if missing else "agent_step"

    def ask_clarification(self, state: AgentState) -> AgentState:
        """Asks the user for the next missing required slot."""
        missing = self._slot_table.missing(state["intent"], state.get("category"), state["claim"])
        question = CLARIFICATION_QUESTIONS.get(missing[0], DEFAULT_CLARIFICATION_QUESTION)
        return {"messages": [AIMessage(content=question)], "decision": "needs_info"}

    def agent_step(self, state: AgentState) -> AgentState:
        """Invokes the tool-calling model for one reasoning step, reusing duplicate calls."""
        request = CurrentRequest(state["messages"])
        if request.agent_step_count() >= MAX_AGENT_STEPS:
            return {"messages": [AIMessage(content="")]}

        available_tools = [
            t for t in self._tools if request.tool_error_count(t.name) < MAX_TOOL_ARG_ERRORS
        ]
        model = self._bind_tools(available_tools)
        response = self._invoke_with_retry(AGENT_STEP_PROMPT | model, request.model_context())
        if response is None:
            return {"messages": [AIMessage(content=LLM_UNAVAILABLE_MESSAGE)]}

        if not response.tool_calls:
            return {"messages": [response]}

        call = response.tool_calls[0]
        duplicate = request.find_duplicate_call(call["name"], call["args"])
        if duplicate is None:
            return {"messages": [response]}

        logger.warning("reusing prior result for repeated identical call to %s", call["name"])
        reused = ToolMessage(
            content=duplicate.content,
            artifact=duplicate.artifact,
            tool_call_id=call["id"],
            name=call["name"],
            status="success",
        )
        return {"messages": [response, reused]}

    @staticmethod
    def _invoke_with_retry(runnable: Runnable, messages: list) -> AIMessage | None:
        try:
            return runnable.with_retry().invoke({"messages": messages})
        except Exception:
            logger.warning("chat model call failed after retrying with backoff")
            return None

    def _bind_tools(self, available_tools: list[BaseTool]) -> BaseChatModel:
        if not available_tools:
            return self._agent_step_model
        try:
            return self._agent_step_model.bind_tools(available_tools)
        except NotImplementedError:
            return self._agent_step_model

    def route_after_agent(self, state: AgentState) -> str:
        """Routes to tool execution, another agent step, or response generation."""
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            return "agent_step"
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "execute_tools"
        return "generate_response"

    def generate_response(self, state: AgentState) -> AgentState:
        """Generates the final answer from the gathered tool evidence and derives the decision."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.content == LLM_UNAVAILABLE_MESSAGE:
            return {"decision": None}

        request = CurrentRequest(state["messages"])
        request_messages = request.messages()
        tool_messages = [m for m in request_messages if isinstance(m, ToolMessage)]

        if not tool_messages:
            return {"messages": [AIMessage(content=NO_TOOL_ARTIFACT_MESSAGE)], "decision": None}

        decision = self._derive_decision(tool_messages)
        answer = self._invoke_with_retry(
            GENERATE_RESPONSE_PROMPT | self._response_model, request.model_context()
        )
        if answer is None:
            return {"messages": [AIMessage(content=LLM_UNAVAILABLE_MESSAGE)], "decision": decision}

        if request.agent_step_count() >= MAX_AGENT_STEPS:
            answer = answer.model_copy(
                update={"content": answer.content + INCOMPLETE_EVIDENCE_NOTE}
            )
        return {"messages": [answer], "decision": decision}

    @staticmethod
    def _derive_decision(tool_messages: list[ToolMessage]) -> Decision | None:
        findings = [
            finding
            for message in tool_messages
            if message.name == "check_rules" and isinstance(message.artifact, list)
            for finding in message.artifact
        ]
        if not findings:
            return None
        if any(finding.status == "fail" for finding in findings):
            return "not_eligible"
        calculations = [
            message.artifact
            for message in tool_messages
            if message.name == "calculate" and isinstance(message.artifact, CalculationResult)
        ]
        if any(finding.status == "warning" for finding in findings) or any(
            result.excess_huf > 0 or result.warnings for result in calculations
        ):
            return "partially_eligible"
        return "eligible"

    def out_of_scope(self, _state: AgentState) -> AgentState:
        """Responds with the out-of-scope message for unsupported intents."""
        return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)], "decision": "out_of_scope"}
