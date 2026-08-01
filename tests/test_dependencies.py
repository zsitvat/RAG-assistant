from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from langchain_redis import RedisVectorStore

from app.agent.service import AgentService
from app.core.config import Settings
from app.dependencies import (
    ApplicationDependencies,
    get_agent_service,
    get_redis_index,
    get_rule_catalogue,
    get_settings,
    get_vector_store,
)
from app.integrations.redis import RedisIndex
from app.rules.model import RuleCatalogue


def test_providers_read_the_typed_application_container():
    dependencies = ApplicationDependencies(
        settings=MagicMock(spec=Settings),
        rule_catalogue=MagicMock(spec=RuleCatalogue),
        redis_index=MagicMock(spec=RedisIndex),
        vector_store=MagicMock(spec=RedisVectorStore),
        agent_service=MagicMock(spec=AgentService),
    )
    request = cast(
        Request,
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(dependencies=dependencies))),
    )

    assert get_settings(request) is dependencies.settings
    assert get_rule_catalogue(request) is dependencies.rule_catalogue
    assert get_redis_index(request) is dependencies.redis_index
    assert get_vector_store(request) is dependencies.vector_store
    assert get_agent_service(request) is dependencies.agent_service


def test_build_raises_when_redis_is_unreachable():
    settings = Settings(llm_backend="dummy", redis_url="redis://127.0.0.1:1/0")

    with pytest.raises(RuntimeError):
        ApplicationDependencies.build(settings)
