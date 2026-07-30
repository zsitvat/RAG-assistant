from fastapi import Request
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_chat_model(request: Request) -> BaseChatModel:
    return request.app.state.chat_model
