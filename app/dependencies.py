from dataclasses import dataclass
from datetime import date

from fastapi import Request
from langchain_redis import RedisVectorStore

from app.agent.calculator import ReimbursementCalculator
from app.agent.deadline import DeadlineChecker
from app.agent.graph import build_agent_graph
from app.agent.nodes import AgentNodes
from app.agent.rule_checker import RuleChecker
from app.agent.service import AgentService
from app.agent.tools import build_tools
from app.core.config import Settings
from app.integrations.llm import build_chat_model
from app.integrations.redis import RedisIndex
from app.rag.graph import build_rag_graph
from app.rag.ingest import connect_and_ingest
from app.rag.retriever import NullPolicyRetriever, PolicyRetriever
from app.rules.loader import get_rule_catalogue as load_rule_catalogue
from app.rules.model import RuleCatalogue


@dataclass(frozen=True, slots=True)
class ApplicationDependencies:
    """Holds application-scoped dependencies."""

    settings: Settings
    rule_catalogue: RuleCatalogue
    redis_index: RedisIndex | None
    vector_store: RedisVectorStore | None
    agent_service: AgentService

    @staticmethod
    def build(settings: Settings) -> "ApplicationDependencies":
        """Builds the application dependency container."""

        # Chat model
        chat_model = build_chat_model(settings)

        # Rules
        rule_catalogue = load_rule_catalogue()

        # Redis
        redis_index, vector_store = connect_and_ingest(settings, rule_catalogue)

        # Retriever and RAG graph
        retriever = (
            PolicyRetriever(vector_store) if vector_store is not None else NullPolicyRetriever()
        )
        rag_graph = build_rag_graph(retriever)

        # Policy tools
        calculator = ReimbursementCalculator(rule_catalogue)
        rule_checker = RuleChecker(
            rule_catalogue, DeadlineChecker(rule_catalogue.submission.deadline_days)
        )
        tools = build_tools(rag_graph, calculator, rule_checker, date.today)

        # Agent graph
        nodes = AgentNodes(
            chat_model.bind(temperature=0),
            chat_model.bind(temperature=0),
            tools,
        )
        return ApplicationDependencies(
            settings=settings,
            rule_catalogue=rule_catalogue,
            redis_index=redis_index,
            vector_store=vector_store,
            agent_service=AgentService(build_agent_graph(nodes)),
        )

    @staticmethod
    def from_request(request: Request) -> "ApplicationDependencies":
        """Returns the container attached to the current application."""

        return request.app.state.dependencies


def get_settings(request: Request) -> Settings:
    """Provides application settings."""

    return ApplicationDependencies.from_request(request).settings


def get_rule_catalogue(request: Request) -> RuleCatalogue:
    """Provides the rule catalogue."""

    return ApplicationDependencies.from_request(request).rule_catalogue


def get_redis_index(request: Request) -> RedisIndex | None:
    """Provides the Redis index when available."""

    return ApplicationDependencies.from_request(request).redis_index


def get_vector_store(request: Request) -> RedisVectorStore | None:
    """Provides the vector store when available."""

    return ApplicationDependencies.from_request(request).vector_store


def get_agent_service(request: Request) -> AgentService:
    """Provides the agent service."""

    return ApplicationDependencies.from_request(request).agent_service
