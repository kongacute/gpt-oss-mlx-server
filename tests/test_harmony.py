from gpt_oss_server.harmony import HarmonyAdapter
from gpt_oss_server.schemas import ChatMessage


def test_harmony_prompt_includes_reasoning_effort():
    # GPT-OSS uses Harmony system metadata to control reasoning effort.
    adapter = HarmonyAdapter()
    tokens = adapter.render_prompt(
        [ChatMessage(role="user", content="hi")], reasoning_effort="low"
    )

    rendered = adapter.encoding.decode_utf8(tokens)
    assert "Reasoning: low" in rendered
    assert "<|start|>user<|message|>hi" in rendered


def test_harmony_prompt_includes_function_tools():
    # Function tools must enter Harmony so GPT-OSS can emit tool calls.
    adapter = HarmonyAdapter()
    tokens = adapter.render_prompt(
        [ChatMessage(role="user", content="weather?")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )

    rendered = adapter.encoding.decode_utf8(tokens)
    assert "get_weather" in rendered
    assert "Get weather" in rendered
