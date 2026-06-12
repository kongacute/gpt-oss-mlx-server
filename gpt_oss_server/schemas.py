from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = ""
    name: str | None = None
    reasoning: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    reasoning: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ResponseRequest(BaseModel):
    model: str | None = None
    input: str | list[dict[str, Any]] | None = None
    messages: list[ChatMessage] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = False
    reasoning: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponseTextFormat(BaseModel):
    type: str = "text"


class ResponseText(BaseModel):
    format: ResponseTextFormat = Field(default_factory=ResponseTextFormat)


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseReasoningSummary(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseReasoningItem(BaseModel):
    id: str
    type: Literal["reasoning"] = "reasoning"
    summary: list[ResponseReasoningSummary]


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["completed"] = "completed"
    content: list[ResponseOutputText]


ResponseOutputItem = ResponseOutputMessage | ResponseReasoningItem


class ResponseObject(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed"] = "completed"
    model: str
    output: list[ResponseOutputItem]
    output_text: str
    usage: ResponseUsage
    reasoning: dict[str, Any] | None = None
    text: ResponseText = Field(default_factory=ResponseText)
