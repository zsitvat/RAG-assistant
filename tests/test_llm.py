from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_ollama import ChatOllama

from app.core.config import Settings
from app.integrations.llm import build_chat_model


def test_dummy_backend_returns_invokable_chat_model():
    model = build_chat_model(Settings(llm_backend="dummy"))
    response = model.invoke([HumanMessage("Hello")])
    assert isinstance(response, AIMessage)
    assert response.content


def test_ollama_backend_returns_configured_chat_ollama():
    settings = Settings(
        llm_backend="ollama",
        ollama_base_url="http://ollama:11434",
        llm_model="qwen2.5:7b-instruct-q4_K_M",
    )
    model = build_chat_model(settings)
    assert isinstance(model, ChatOllama)
    assert model.model == settings.llm_model
    assert model.base_url == settings.ollama_base_url


def test_generic_fake_chat_model_emits_scripted_tool_call_then_text():
    """Demonstrates the framework-native fake used by later graph/tool tests."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="search_policies",
                            args={"query": "meal limit"},
                            id="call_1",
                        )
                    ],
                ),
                "Meals are capped at 15,000 HUF per person.",
            ]
        )
    )

    first = model.invoke([HumanMessage("What is the meal limit?")])
    assert first.tool_calls[0]["name"] == "search_policies"

    second = model.invoke([HumanMessage("thanks")])
    assert second.content == "Meals are capped at 15,000 HUF per person."
