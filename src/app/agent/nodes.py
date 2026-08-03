import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from app.agent.calculator import CalculationInputError, ReimbursementCalculator
from app.agent.message_history import MessageHistory
from app.agent.model import CalculationResult, Decision, ExpenseClaim, IntentClassification
from app.agent.langfuse_prompt_library import PromptLibrary
from app.agent.slots import RequiredSlotTable
from app.agent.graph_state import MAX_AGENT_STEPS, MAX_TOOL_ARG_ERRORS, AgentState
from app.agent.static_texts import (
    CLARIFICATION_QUESTIONS,
    CONDITIONAL_DISTANCE_ANSWER,
    DEFAULT_CLARIFICATION_QUESTION,
    INCOMPLETE_EVIDENCE_NOTE,
    LLM_UNAVAILABLE_MESSAGE,
    NO_TOOL_ARTIFACT_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
)
from app.agent.structured import StructuredOutputRunner
from app.integrations.langfuse import Observability

logger = logging.getLogger(__name__)


class AgentNodes:
    """Implements the LangGraph node callbacks that drive the agent's conversation flow."""

    def __init__(
        self,
        structured_chat_model: BaseChatModel,
        response_chat_model: BaseChatModel,
        tools: list[BaseTool],
        calculator: ReimbursementCalculator,
        prompts: PromptLibrary | None = None,
    ) -> None:
        """Stores the chat models, tools, calculator and resolved prompts used by the nodes."""
        self._tools = tools
        self._calculator = calculator
        self._agent_step_model = structured_chat_model
        self._response_model = response_chat_model
        self._prompts = prompts or PromptLibrary(Observability(None))
        self._classify_runner = StructuredOutputRunner(
            structured_chat_model, self._prompt("classify_intent"), IntentClassification
        )
        self._extract_runner = StructuredOutputRunner(
            structured_chat_model, self._prompt("extract_information"), ExpenseClaim
        )
        self._slot_table = RequiredSlotTable()

    def _prompt(self, name: str):
        """Returns the resolved chat template for a prompt name."""
        return self._prompts.get(name).template

    @property
    def tools(self) -> list[BaseTool]:
        """Returns the tools available to the agent step."""
        return self._tools

    async def classify_intent(self, state: AgentState) -> AgentState:
        """Classifies the user's intent and, if applicable, the expense category."""
        context = MessageHistory(state["messages"]).model_context()
        result = await self._classify_runner.run(
            context, fallback=IntentClassification(intent="policy_question")
        )
        return {
            "intent": result.value.intent,
            "category": result.value.category,
            "degraded": result.degraded,
        }

    async def extract_information(self, state: AgentState) -> AgentState:
        """Extracts and merges expense claim fields from the conversation so far."""
        previous_claim = ExpenseClaim.from_state(state.get("claim"))
        previous_decision = state.get("decision")
        category = state.get("category")

        context = MessageHistory(state["messages"]).model_context()
        result = await self._extract_runner.run(context, fallback=ExpenseClaim())
        extracted = result.value
        if category is not None:
            extracted = extracted.model_copy(update={"category": category})

        is_continuation = previous_decision == "needs_info" and (
            category is None or category == previous_claim.category
        )
        claim = previous_claim.merged_with(extracted) if is_continuation else extracted
        return {
            "claim": claim,
            "decision": None,
            "degraded": state.get("degraded", False) or result.degraded,
        }

    def route_after_extraction(self, state: AgentState) -> str:
        """Routes to clarification, the agent step, or out-of-scope handling."""
        if state["intent"] == "unsupported":
            return "out_of_scope"
        missing = self._slot_table.missing(
            state["intent"], state.get("category"), ExpenseClaim.from_state(state.get("claim"))
        )
        return "ask_clarification" if missing else "agent_step"

    def ask_clarification(self, state: AgentState) -> AgentState:
        """Asks for the next missing slot, or answers conditionally if it was already asked."""
        missing = self._slot_table.missing(
            state["intent"], state.get("category"), ExpenseClaim.from_state(state.get("claim"))
        )
        question = CLARIFICATION_QUESTIONS.get(missing[0], DEFAULT_CLARIFICATION_QUESTION)
        if MessageHistory(state["messages"]).was_already_asked(question):
            conditional = self._conditional_distance_answer(state, missing)
            if conditional is not None:
                return conditional
        return {"messages": [AIMessage(content=question)], "decision": "needs_info"}

    def _conditional_distance_answer(
        self, state: AgentState, missing: list[str]
    ) -> AgentState | None:
        """Answers with both distance readings once the user has declined to disambiguate."""
        if missing != ["distance_is_one_way"]:
            return None
        try:
            outcomes = self._calculator.calculate_both_directions(
                ExpenseClaim.from_state(state.get("claim"))
            )
        except CalculationInputError:
            logger.warning("cannot resolve the distance ambiguity from the current claim")
            return None
        answer = CONDITIONAL_DISTANCE_ANSWER.format(
            one_way=outcomes[True].compact_summary(),
            round_trip=outcomes[False].compact_summary(),
        )
        return {"messages": [AIMessage(content=answer)], "decision": "needs_info"}

    async def agent_step(self, state: AgentState) -> AgentState:
        """Invokes the tool-calling model for one reasoning step, reusing duplicate calls."""
        history = MessageHistory(state["messages"])
        if history.agent_step_count() >= MAX_AGENT_STEPS:
            return {"messages": [AIMessage(content="")]}

        available_tools = [
            t for t in self._tools if history.tool_error_count(t.name) < MAX_TOOL_ARG_ERRORS
        ]
        model = self._bind_tools(available_tools)
        response = await self._invoke_with_retry(
            self._prompt("agent_step") | model, history.model_context()
        )
        if response is None:
            return {"messages": [AIMessage(content=LLM_UNAVAILABLE_MESSAGE)]}

        if not response.tool_calls:
            return {"messages": [response]}

        call = response.tool_calls[0]
        duplicate = history.find_duplicate_call(call["name"], call["args"])
        if duplicate is None:
            return {"messages": [response]}

        logger.warning(f"reusing prior result for repeated identical call to {call['name']}")
        reused = ToolMessage(
            content=duplicate.content,
            artifact=duplicate.artifact,
            tool_call_id=call["id"],
            name=call["name"],
            status="success",
        )
        return {"messages": [response, reused]}

    @staticmethod
    async def _invoke_with_retry(runnable: Runnable, messages: list) -> AIMessage | None:
        """Invokes a model with retry handling and returns None after failure."""
        try:
            return await runnable.with_retry().ainvoke({"messages": messages})
        except Exception as e:
            logger.warning(
                f"chat model call failed after retrying with backoff: {type(e).__name__}: {e}"
            )
            return None

    def _bind_tools(self, available_tools: list[BaseTool]) -> BaseChatModel:
        """Binds the available tools when the chat model supports tool calling."""
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

    async def generate_response(self, state: AgentState) -> AgentState:
        """Generates the final answer from the gathered tool evidence and derives the decision."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.content == LLM_UNAVAILABLE_MESSAGE:
            return {"decision": None, "degraded": True}

        history = MessageHistory(state["messages"])
        request_messages = history.messages()
        tool_messages = [m for m in request_messages if isinstance(m, ToolMessage)]

        if not tool_messages:
            return {
                "messages": [AIMessage(content=NO_TOOL_ARTIFACT_MESSAGE)],
                "decision": None,
                "degraded": True,
            }

        decision = self._derive_decision(tool_messages)
        answer = await self._invoke_with_retry(
            self._prompt("generate_response") | self._response_model, history.model_context()
        )
        if answer is None:
            return {
                "messages": [AIMessage(content=LLM_UNAVAILABLE_MESSAGE)],
                "decision": decision,
                "degraded": True,
            }

        if history.agent_step_count() >= MAX_AGENT_STEPS:
            answer = answer.model_copy(
                update={"content": answer.content + INCOMPLETE_EVIDENCE_NOTE}
            )
            return {"messages": [answer], "decision": decision, "degraded": True}
        return {"messages": [answer], "decision": decision}

    @staticmethod
    def _derive_decision(tool_messages: list[ToolMessage]) -> Decision | None:
        """Derives claim eligibility from rule findings and calculation results."""
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
