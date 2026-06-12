import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import ServerConfig
from .engine import MLXEngine
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummary,
    ResponseRequest,
    ResponseUsage,
    Usage,
)


def create_app(config: ServerConfig, engine: MLXEngine | None = None) -> FastAPI:
    app = FastAPI(title="gpt-oss-server")
    active_engine = engine or MLXEngine(config)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": active_engine.model_id}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": active_engine.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.get("/v1/models/{model_id:path}")
    def retrieve_model(model_id: str) -> dict[str, object]:
        if model_id != active_engine.model_id:
            raise HTTPException(status_code=404, detail="Model not found")
        return {
            "id": active_engine.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }

    @app.post("/v1/chat/completions", response_model_exclude_none=True)
    def chat_completions(request: ChatCompletionRequest):
        _ensure_temperature_not_set(request.temperature)
        _ensure_text_only_messages(request.messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.model or active_engine.model_id

        if request.stream:
            return StreamingResponse(
                _stream_chunks(active_engine, request, completion_id, created, model),
                media_type="text/event-stream",
            )

        result = active_engine.complete(request)
        usage = Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        )
        response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result.text,
                        reasoning=result.reasoning or None,
                    ),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=usage,
        )
        return response.model_dump(exclude_none=True)

    @app.post("/v1/responses", response_model_exclude_none=True)
    def responses(request: ResponseRequest):
        _ensure_temperature_not_set(request.temperature)
        chat_request = _response_to_chat_request(request, active_engine.model_id)
        _ensure_text_only_messages(chat_request.messages)
        response_id = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.model or active_engine.model_id

        if request.stream:
            return StreamingResponse(
                _stream_response_events(
                    active_engine, chat_request, response_id, created, model
                ),
                media_type="text/event-stream",
            )

        result = active_engine.complete(chat_request)
        output_items = _response_output_items(result.reasoning, result.text)
        response = ResponseObject(
            id=response_id,
            created_at=created,
            model=model,
            output=output_items,
            output_text=result.text,
            usage=ResponseUsage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            reasoning=_response_reasoning_block(request.reasoning, result.reasoning),
        )
        return response.model_dump(exclude_none=True)

    return app


def _stream_chunks(
    engine: MLXEngine,
    request: ChatCompletionRequest,
    completion_id: str,
    created: int,
    model: str,
) -> Iterator[str]:
    first = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))
        ],
    )
    yield _sse(first.model_dump(exclude_none=True))

    for delta in engine.stream(request):
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(
                        content=delta.text or None,
                        reasoning=delta.reasoning or None,
                    ),
                    finish_reason=delta.finish_reason,
                )
            ],
        )
        yield _sse(chunk.model_dump(exclude_none=True))
    yield "data: [DONE]\n\n"


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _stream_response_events(
    engine: MLXEngine,
    request: ChatCompletionRequest,
    response_id: str,
    created: int,
    model: str,
) -> Iterator[str]:
    yield _sse(
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        }
    )

    sequence_number = 1
    output_text = []
    reasoning_text = []
    for delta in engine.stream(request):
        if delta.reasoning:
            reasoning_text.append(delta.reasoning)
            yield _sse(
                {
                    "type": "response.reasoning_text.delta",
                    "sequence_number": sequence_number,
                    "item_id": f"rs_{response_id}",
                    "output_index": 0,
                    "delta": delta.reasoning,
                }
            )
            sequence_number += 1
        if delta.text:
            output_text.append(delta.text)
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence_number,
                    "item_id": f"msg_{response_id}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta.text,
                }
            )
            sequence_number += 1
        if delta.finish_reason:
            text = "".join(output_text)
            yield _sse(
                {
                    "type": "response.completed",
                    "sequence_number": sequence_number,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created,
                        "status": "completed",
                        "model": model,
                        "output_text": text,
                        "reasoning": {"content": "".join(reasoning_text)}
                        if reasoning_text
                        else None,
                    },
                }
            )
    yield "data: [DONE]\n\n"


def _response_to_chat_request(
    request: ResponseRequest, default_model: str
) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    if request.instructions:
        messages.append(ChatMessage(role="developer", content=request.instructions))

    if request.messages is not None:
        messages.extend(request.messages)
    elif isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        messages.extend(_response_input_to_messages(request.input))
    else:
        raise HTTPException(status_code=400, detail="Responses input is required")

    reasoning_effort = None
    if request.reasoning:
        reasoning_effort = request.reasoning.get("effort")
        if reasoning_effort == "none":
            reasoning_effort = None

    return ChatCompletionRequest(
        model=request.model or default_model,
        messages=messages,
        max_tokens=request.max_output_tokens or request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        reasoning_effort=reasoning_effort,
        reasoning=request.reasoning,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )


def _response_output_items(reasoning: str, text: str) -> list[object]:
    output: list[object] = []
    if reasoning:
        output.append(
            ResponseReasoningItem(
                id=f"rs_{uuid.uuid4().hex}",
                summary=[ResponseReasoningSummary(text=reasoning)],
            )
        )
    output.append(
        ResponseOutputMessage(
            id=f"msg_{uuid.uuid4().hex}",
            content=[ResponseOutputText(text=text)],
        )
    )
    return output


def _response_reasoning_block(
    request_reasoning: dict[str, object] | None, generated_reasoning: str
) -> dict[str, object] | None:
    if not request_reasoning and not generated_reasoning:
        return None
    reasoning = dict(request_reasoning or {})
    if generated_reasoning:
        reasoning["content"] = generated_reasoning
    return reasoning


def _response_input_to_messages(items: list[dict[str, object]]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for item in items:
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "message" and isinstance(role, str):
            messages.append(
                ChatMessage(role=role, content=_response_content_text(item.get("content")))
            )
        elif item_type == "input_text":
            messages.append(ChatMessage(role="user", content=str(item.get("text", ""))))
        elif role == "user":
            messages.append(ChatMessage(role="user", content=_response_content_text(item)))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported text-only Responses input item: {item_type}",
            )
    return messages


def _response_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in {"input_text", "output_text", "text"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported multimodal content type: {part_type}",
                )
            parts.append(str(part.get("text", "")))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return _response_content_text([content])
    return ""


def _ensure_text_only_messages(messages: list[ChatMessage]) -> None:
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for item in message.content:
            item_type = item.get("type")
            if item_type not in {"text", "input_text", "output_text"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported multimodal content type: {item_type}",
                )


def _ensure_temperature_not_set(temperature: float | None) -> None:
    if temperature is not None:
        raise HTTPException(
            status_code=400,
            detail="temperature is not configurable for GPT-OSS reasoning models",
        )
