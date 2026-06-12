from gpt_oss_server.config import ServerConfig
from gpt_oss_server.engine import MLXEngine
from gpt_oss_server.schemas import ChatCompletionRequest, ChatMessage


def test_engine_sampler_uses_model_generation_config(monkeypatch):
    # Default sampling should come from model generation_config, not server constants.
    captured = {}
    engine = MLXEngine(ServerConfig(model_id="fake-model"))
    engine._model = object()
    engine._tokenizer = object()
    engine._generation_config = {"do_sample": True, "temperature": 0.8, "top_p": 0.9}

    def fake_sampler(**kwargs):
        captured["sampler_kwargs"] = kwargs
        return "sampler"

    monkeypatch.setattr("gpt_oss_server.engine.make_sampler", fake_sampler)

    kwargs = engine._generation_kwargs(
        ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
    )

    assert kwargs == {"sampler": "sampler"}
    assert captured["sampler_kwargs"] == {"temp": 0.8, "top_p": 0.9}


def test_engine_request_top_p_overrides_only_that_setting(monkeypatch):
    # Request top_p may override top_p, while model temperature remains unchanged.
    captured = {}
    engine = MLXEngine(ServerConfig(model_id="fake-model"))
    engine._model = object()
    engine._tokenizer = object()
    engine._generation_config = {"do_sample": True, "temperature": 0.8, "top_p": 0.9}

    def fake_sampler(**kwargs):
        captured["sampler_kwargs"] = kwargs
        return "sampler"

    monkeypatch.setattr("gpt_oss_server.engine.make_sampler", fake_sampler)

    engine._generation_kwargs(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")], top_p=0.5
        )
    )

    assert captured["sampler_kwargs"] == {"temp": 0.8, "top_p": 0.5}

