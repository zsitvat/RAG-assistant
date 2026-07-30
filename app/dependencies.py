from fastapi import Request
from langchain_core.language_models import BaseChatModel
from langchain_redis import RedisVectorStore

from app.core.config import Settings
from app.integrations.redis import RedisIndex
from app.rules.model import RuleCatalogue


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_chat_model(request: Request) -> BaseChatModel:
    return request.app.state.chat_model


def get_rule_catalogue(request: Request) -> RuleCatalogue:
    return request.app.state.rule_catalogue


def get_redis_client(request: Request) -> RedisIndex:
    return request.app.state.redis_client


def get_vector_store(request: Request) -> RedisVectorStore:
    return request.app.state.vector_store
