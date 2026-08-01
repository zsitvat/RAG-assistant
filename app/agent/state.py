from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agent.model import Decision, ExpenseClaim, Intent
from app.rules.model import Category

MAX_AGENT_STEPS = 4
RECURSION_LIMIT = 10
MAX_TOOL_ARG_ERRORS = 2


class AgentState(TypedDict, total=False):
    """Holds the conversation messages and extracted claim data threaded through the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    intent: Intent
    category: Category | None
    claim: ExpenseClaim
    decision: Decision | None
