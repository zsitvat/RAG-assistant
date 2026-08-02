from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class MessageHistory:
    """Provides messages and tool-call facts for the latest user request."""

    def __init__(self, messages: list[BaseMessage]) -> None:
        """Stores the full message history to analyze."""
        self._messages = messages

    def messages(self) -> list[BaseMessage]:
        """Returns the messages from the latest HumanMessage onward."""
        for index in range(len(self._messages) - 1, -1, -1):
            if isinstance(self._messages[index], HumanMessage):
                return self._messages[index:]
        return list(self._messages)

    def was_already_asked(self, question: str) -> bool:
        """Reports whether the assistant already asked this exact question earlier in the thread."""
        return any(
            isinstance(message, AIMessage) and message.content == question
            for message in self._messages[:-1]
        )

    def agent_step_count(self) -> int:
        """Counts the agent tool-calling steps taken so far in the current request."""
        return sum(
            1
            for message in self.messages()
            if isinstance(message, AIMessage) and message.tool_calls
        )

    def tool_error_count(self, tool_name: str) -> int:
        """Counts the errored calls to the given tool in the current request."""
        return sum(
            1
            for message in self.messages()
            if isinstance(message, ToolMessage)
            and message.name == tool_name
            and message.status == "error"
        )

    def find_duplicate_call(self, tool_name: str, args: dict) -> ToolMessage | None:
        """Finds a prior successful call to the tool with the same name and arguments, if any."""
        request_messages = self.messages()
        matching_call_ids = {
            call["id"]
            for message in request_messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
            if call["name"] == tool_name and call["args"] == args
        }
        for message in request_messages:
            if (
                isinstance(message, ToolMessage)
                and message.tool_call_id in matching_call_ids
                and message.status == "success"
            ):
                return message
        return None

    def model_context(self) -> list[BaseMessage]:
        """Builds model input: prior requests condensed to human+final AI, current kept in full."""
        requests = self._split_into_requests()
        if not requests:
            return list(self._messages)

        context: list[BaseMessage] = []
        for request_messages in requests[:-1]:
            context.extend(self._condense(request_messages))
        context.extend(requests[-1])
        return context

    def _split_into_requests(self) -> list[list[BaseMessage]]:
        """Splits the full message history into per-request slices starting at each HumanMessage."""
        boundaries = [
            index
            for index, message in enumerate(self._messages)
            if isinstance(message, HumanMessage)
        ]
        if not boundaries:
            return []
        boundaries.append(len(self._messages))
        return [
            self._messages[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)
        ]

    @staticmethod
    def _condense(request_messages: list[BaseMessage]) -> list[BaseMessage]:
        """Reduces a completed request to its HumanMessage and final non-tool-calling AIMessage."""
        final_answer = next(
            (
                message
                for message in reversed(request_messages)
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            None,
        )
        condensed = [request_messages[0]]
        if final_answer is not None:
            condensed.append(final_answer)
        return condensed
