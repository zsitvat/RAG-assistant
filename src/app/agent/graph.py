from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.agent.nodes import AgentNodes
from app.agent.graph_state import AgentState


def build_agent_graph(
    nodes: AgentNodes, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Assembles and compiles the agent's LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("extract_information", nodes.extract_information)
    graph.add_node("ask_clarification", nodes.ask_clarification)
    graph.add_node("agent_step", nodes.agent_step)
    graph.add_node("execute_tools", ToolNode(nodes.tools, handle_tool_errors=True))
    graph.add_node("generate_response", nodes.generate_response)
    graph.add_node("out_of_scope", nodes.out_of_scope)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "extract_information")
    graph.add_conditional_edges(
        "extract_information",
        nodes.route_after_extraction,
        {
            "ask_clarification": "ask_clarification",
            "agent_step": "agent_step",
            "out_of_scope": "out_of_scope",
        },
    )
    graph.add_edge("ask_clarification", END)
    graph.add_conditional_edges(
        "agent_step",
        nodes.route_after_agent,
        {
            "execute_tools": "execute_tools",
            "generate_response": "generate_response",
            "agent_step": "agent_step",
        },
    )
    graph.add_edge("execute_tools", "agent_step")
    graph.add_edge("generate_response", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
