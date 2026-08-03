from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from langchain_redis import RedisVectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.service import AgentService
from app.dependencies import (
    ApplicationDependencies,
    get_agent_service,
    get_checkpointer,
    get_observability,
    get_redis_index,
    get_rule_catalogue,
    get_settings,
    get_vector_store,
)
from app.integrations.langfuse import Observability
from app.integrations.redis import RedisIndex
from app.rules.model import RuleCatalogue
from app.settings import Settings


def test_providers_read_the_typed_application_container():
    # Arrange
    dependencies = ApplicationDependencies(
        settings=MagicMock(spec=Settings),
        rule_catalogue=MagicMock(spec=RuleCatalogue),
        redis_index=MagicMock(spec=RedisIndex),
        vector_store=MagicMock(spec=RedisVectorStore),
        checkpointer=MagicMock(spec=BaseCheckpointSaver),
        agent_service=MagicMock(spec=AgentService),
        observability=MagicMock(spec=Observability),
    )
    request = cast(
        Request,
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(dependencies=dependencies))),
    )

    # Assert
    assert get_settings(request) is dependencies.settings
    assert get_rule_catalogue(request) is dependencies.rule_catalogue
    assert get_redis_index(request) is dependencies.redis_index
    assert get_vector_store(request) is dependencies.vector_store
    assert get_checkpointer(request) is dependencies.checkpointer
    assert get_agent_service(request) is dependencies.agent_service
    assert get_observability(request) is dependencies.observability


async def test_build_fails_when_redis_is_unreachable():
    # Arrange
    # langfuse explicitly disabled: this test only exercises the Redis-unreachable path
    # and must not depend on (or contact) a developer's real .env Langfuse credentials.
    settings = Settings(
        llm_backend="dummy", redis_url="redis://127.0.0.1:1/0", langfuse_enabled=False
    )

    # Act & Assert
    with pytest.raises(RuntimeError, match="Redis is required but unavailable at startup"):
        await ApplicationDependencies.build(settings)
