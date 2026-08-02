from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from app.agent.structured import StructuredOutputRunner
from app.tests.fakes import ScriptedChatModel

PROMPT = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])


class _Schema(BaseModel):
    value: int


async def test_run_returns_the_structured_result_on_first_success():
    model = ScriptedChatModel(
        chat_responses=iter([]), structured_responses=iter([_Schema(value=1)])
    )
    runner = StructuredOutputRunner(model, PROMPT, _Schema)

    result = await runner.run([HumanMessage(content="x")], fallback=_Schema(value=0))

    assert result.value.value == 1
    assert result.degraded is False


async def test_run_retries_once_then_returns_the_repaired_result():
    class _FlakyModel(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            runnable = super().with_structured_output(schema)
            calls = {"n": 0}
            original_invoke = runnable.invoke

            def invoke(input, config=None, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ValueError("bad output")
                return original_invoke(input, config, **kw)

            runnable.invoke = invoke
            return runnable

    model = _FlakyModel(chat_responses=iter([]), structured_responses=iter([_Schema(value=2)]))
    runner = StructuredOutputRunner(model, PROMPT, _Schema)

    result = await runner.run([HumanMessage(content="x")], fallback=_Schema(value=0))

    assert result.value.value == 2
    assert result.degraded is False


async def test_run_falls_back_when_both_attempts_fail():
    class _AlwaysFailingModel(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            class _Raiser:
                def invoke(self, input, config=None, **kw):
                    raise ValueError("always broken")

            return _Raiser()

    model = _AlwaysFailingModel(chat_responses=iter([]), structured_responses=iter([]))
    runner = StructuredOutputRunner(model, PROMPT, _Schema)

    result = await runner.run([HumanMessage(content="x")], fallback=_Schema(value=99))

    assert result.value.value == 99
    assert result.degraded is True


async def test_run_falls_back_immediately_when_structured_output_is_unsupported():
    class _UnsupportedModel(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            raise NotImplementedError

    model = _UnsupportedModel(chat_responses=iter([]), structured_responses=iter([]))
    runner = StructuredOutputRunner(model, PROMPT, _Schema)

    result = await runner.run([HumanMessage(content="x")], fallback=_Schema(value=7))

    assert result.value.value == 7
    assert result.degraded is True
