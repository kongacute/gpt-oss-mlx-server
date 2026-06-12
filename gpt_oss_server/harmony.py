from collections.abc import Sequence

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    DeveloperContent,
    ToolDescription,
    load_harmony_encoding,
)

from .schemas import ChatMessage


ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.DEVELOPER,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}

REASONING_MAP = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


def message_text(message: ChatMessage) -> str:
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content

    parts: list[str] = []
    for item in message.content:
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


class HarmonyAdapter:
    def __init__(self) -> None:
        self.encoding: HarmonyEncoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS
        )

    def render_prompt(
        self,
        messages: Sequence[ChatMessage],
        reasoning_effort: str | None = None,
        tools: Sequence[dict] | None = None,
    ) -> list[int]:
        harmony_messages = [self._system_message(reasoning_effort)]
        if tools:
            harmony_messages.append(self._developer_tools_message(tools))
        harmony_messages.extend(
            self._to_harmony_message(message) for message in messages
        )
        conversation = Conversation.from_messages(harmony_messages)
        return self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

    def parse_completion(self, tokens: Sequence[int], raw_text: str) -> str:
        try:
            messages = self.encoding.parse_messages_from_completion_tokens(
                tokens, Role.ASSISTANT, strict=False
            )
        except Exception:
            return raw_text

        final_parts: list[str] = []
        fallback_parts: list[str] = []
        for message in messages:
            text = self._content_text(message.content)
            if not text:
                continue
            if message.channel == "final" or message.channel is None:
                final_parts.append(text)
            fallback_parts.append(text)
        return "".join(final_parts or fallback_parts) or raw_text

    def _to_harmony_message(self, message: ChatMessage) -> Message:
        role = ROLE_MAP[message.role]
        author = Author(role=role, name=message.name)
        return Message(author=author, content=[TextContent(text=message_text(message))])

    @staticmethod
    def _system_message(reasoning_effort: str | None) -> Message:
        content = SystemContent.new()
        if reasoning_effort is not None:
            content = content.with_reasoning_effort(REASONING_MAP[reasoning_effort])
        return Message.from_role_and_contents(Role.SYSTEM, [content])

    @staticmethod
    def _developer_tools_message(tools: Sequence[dict]) -> Message:
        function_tools: list[ToolDescription] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            if not isinstance(function, dict) or "name" not in function:
                continue
            function_tools.append(
                ToolDescription.new(
                    name=str(function["name"]),
                    description=str(function.get("description", "")),
                    parameters=function.get("parameters"),
                )
            )

        content = DeveloperContent.new()
        if function_tools:
            content = content.with_function_tools(function_tools)
        return Message.from_role_and_contents(Role.DEVELOPER, [content])

    @staticmethod
    def _content_text(contents: Sequence[object]) -> str:
        parts: list[str] = []
        for content in contents:
            text = getattr(content, "text", None)
            if text is not None:
                parts.append(text)
        return "".join(parts)
