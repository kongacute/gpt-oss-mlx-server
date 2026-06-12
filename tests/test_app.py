from collections.abc import Iterator

from fastapi.testclient import TestClient

from gpt_oss_server.app import create_app
from gpt_oss_server.config import ServerConfig
from gpt_oss_server.engine import CompletionResult, TokenDelta


class FakeEngine:
    model_id = "openai/gpt-oss-test"

    def complete(self, request):
        # Verifies request conversion keeps a user message for both APIs.
        assert any(message.role == "user" for message in request.messages)
        return CompletionResult(
            text="hello",
            reasoning="brief reasoning",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=1,
        )

    def stream(self, request) -> Iterator[TokenDelta]:
        # Verifies SSE chunking without loading a real MLX model.
        yield TokenDelta(text="", reasoning="brief ")
        yield TokenDelta(text="", reasoning="reasoning")
        yield TokenDelta(text="hel")
        yield TokenDelta(text="lo")
        yield TokenDelta(text="", finish_reason="stop")


def client() -> TestClient:
    config = ServerConfig(model_id="openai/gpt-oss-test")
    return TestClient(create_app(config, engine=FakeEngine()))


def test_models_lists_active_model():
    # Model listing lets OpenAI clients discover configured local GPT-OSS model.
    response = client().get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "openai/gpt-oss-test"


def test_model_retrieve_returns_active_model():
    # Model retrieve matches OpenAI SDK checks against the configured model id.
    response = client().get("/v1/models/openai/gpt-oss-test")

    assert response.status_code == 200
    assert response.json()["id"] == "openai/gpt-oss-test"


def test_chat_completion_returns_openai_shape():
    # Chat completions require no API key and return assistant message plus usage.
    response = client().post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello",
        "reasoning": "brief reasoning",
    }
    assert payload["usage"]["total_tokens"] == 4


def test_chat_completion_rejects_multimodal_content():
    # This server is intentionally text-only; image/audio/file parts are rejected.
    response = client().post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "x"}],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "Unsupported multimodal content type" in response.json()["detail"]


def test_chat_completion_rejects_temperature():
    # GPT-OSS reasoning models should keep model sampling defaults, not request temp.
    response = client().post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0,
        },
    )

    assert response.status_code == 400
    assert "temperature is not configurable" in response.json()["detail"]


def test_chat_completion_streams_sse_chunks():
    # Streaming emits OpenAI-style SSE data chunks and final [DONE] sentinel.
    with client().stream(
        "POST",
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "chat.completion.chunk" in body
    assert '"reasoning":"brief "' in body
    assert '"content":"hel"' in body
    assert "data: [DONE]" in body


def test_responses_create_returns_text_output():
    # Responses API is the preferred GPT-OSS-compatible API for text generation.
    response = client().post(
        "/v1/responses",
        json={
            "input": "hi",
            "instructions": "be brief",
            "reasoning": {"effort": "low"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["output_text"] == "hello"
    assert payload["reasoning"]["content"] == "brief reasoning"
    assert payload["output"][0]["type"] == "reasoning"
    assert payload["output"][1]["content"][0]["type"] == "output_text"
    assert payload["usage"]["total_tokens"] == 4


def test_responses_accepts_message_input_shape():
    # GPT-OSS provider smoke tests send Responses message arrays, not only strings.
    response = client().post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["output_text"] == "hello"


def test_responses_rejects_temperature():
    # Responses rejects temp changes for parity with closed-source reasoning GPTs.
    response = client().post(
        "/v1/responses",
        json={"input": "hi", "temperature": 1},
    )

    assert response.status_code == 400
    assert "temperature is not configurable" in response.json()["detail"]


def test_responses_streams_events():
    # Responses streaming emits output_text deltas plus completion sentinel.
    with client().stream(
        "POST",
        "/v1/responses",
        json={"stream": True, "input": "hi"},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "response.output_text.delta" in body
    assert "response.reasoning_text.delta" in body
    assert '"delta":"hel"' in body
    assert "data: [DONE]" in body
