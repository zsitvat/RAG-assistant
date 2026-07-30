from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_ollama import ChatOllama

from app.core.config import Settings

DUMMY_RESPONSES = [
    "This is a response from the dummy LLM backend; no model is loaded.",
]


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_backend == "dummy":
        return FakeListChatModel(responses=DUMMY_RESPONSES)
    return ChatOllama(base_url=settings.ollama_base_url, model=settings.llm_model)
