from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

JUDGE_SYSTEM = """You grade whether a chatbot's final answer to an employee expense/benefits \
question correctly conveys the expected outcome.

You are given the employee's question, the chatbot's answer, and a reference description of what a \
correct answer must convey — the decision, amount and cited documents already verified against a \
deterministic rule engine.

Mark correct=true only if the answer's stated decision and amount (when the reference mentions \
one) match the reference, and it does not fabricate a citation or a number absent from the \
reference. Wording, language and phrasing may differ freely from the reference — only the \
substance must match. Mark correct=false and explain why otherwise."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", JUDGE_SYSTEM), MessagesPlaceholder("messages")]
)


class AnswerJudgeVerdict(BaseModel):
    """One judge model's verdict on whether a final answer matches its expected summary."""

    correct: bool
    reasoning: str
