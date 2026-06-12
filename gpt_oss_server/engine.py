import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from openai_harmony import Role, StreamableParser

from .config import ServerConfig
from .harmony import HarmonyAdapter
from .schemas import ChatCompletionRequest


@dataclass(frozen=True)
class CompletionResult:
    text: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class TokenDelta:
    text: str
    reasoning: str = ""
    finish_reason: str | None = None


class MLXEngine:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.harmony = HarmonyAdapter()
        self._model = None
        self._tokenizer = None
        self._model_config: dict = {}
        self._generation_config: dict = {}
        self._load_lock = Lock()
        self._generate_lock = Lock()

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is None or self._tokenizer is None:
                self._model, self._tokenizer, self._model_config = load(
                    self.config.model_id, return_config=True
                )
                self._generation_config = self._load_generation_config()

    def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        chunks = list(self.stream(request))
        text = "".join(chunk.text for chunk in chunks)
        reasoning = "".join(chunk.reasoning for chunk in chunks)
        finish_reason = chunks[-1].finish_reason if chunks else "stop"
        prompt_tokens = getattr(self, "_last_prompt_tokens", 0)
        completion_tokens = getattr(self, "_last_completion_tokens", 0)
        return CompletionResult(
            text=text,
            reasoning=reasoning,
            finish_reason=finish_reason or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stream(self, request: ChatCompletionRequest) -> Iterator[TokenDelta]:
        self.load_model()
        assert self._model is not None
        assert self._tokenizer is not None

        prompt_tokens = self.harmony.render_prompt(
            request.messages, request.reasoning_effort, request.tools
        )
        max_tokens = request.max_tokens or self.config.max_tokens
        generation_kwargs = self._generation_kwargs(request)

        generated_tokens: list[int] = []
        parser = StreamableParser(self.harmony.encoding, Role.ASSISTANT, strict=False)
        parser_failed = False
        self._last_prompt_tokens = len(prompt_tokens)
        self._last_completion_tokens = 0

        # MLX generation mutates model cache state; serialize requests for correctness.
        with self._generate_lock:
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt_tokens,
                max_tokens=max_tokens,
                **generation_kwargs,
            ):
                generated_tokens.append(int(response.token))
                self._last_completion_tokens = response.generation_tokens

                if response.text and parser_failed:
                    yield TokenDelta(text=response.text)
                elif response.text:
                    try:
                        parser.process(int(response.token))
                    except Exception:
                        parser_failed = True
                        yield TokenDelta(text=response.text)
                    else:
                        if (
                            parser.last_content_delta
                            and parser.current_channel in (None, "final")
                        ):
                            yield TokenDelta(text=parser.last_content_delta)
                        elif (
                            parser.last_content_delta
                            and parser.current_channel == "analysis"
                        ):
                            yield TokenDelta(
                                text="", reasoning=parser.last_content_delta
                            )

                if response.finish_reason:
                    if not parser_failed:
                        try:
                            parser.process_eos()
                        except Exception:
                            pass
                    yield TokenDelta(text="", finish_reason=response.finish_reason)

    def _generation_kwargs(self, request: ChatCompletionRequest) -> dict[str, object]:
        if request.temperature is not None:
            raise ValueError("temperature is not configurable")

        sampler_kwargs: dict[str, float | int] = {}
        if self._generation_config.get("do_sample") is True:
            sampler_kwargs["temp"] = float(
                self._generation_config.get("temperature", 1.0)
            )
        elif "temperature" in self._generation_config:
            sampler_kwargs["temp"] = float(self._generation_config["temperature"])

        for config_key, sampler_key in (
            ("top_p", "top_p"),
            ("min_p", "min_p"),
            ("top_k", "top_k"),
        ):
            if config_key in self._generation_config:
                value = self._generation_config[config_key]
                if sampler_key == "top_k":
                    sampler_kwargs[sampler_key] = int(value)
                else:
                    sampler_kwargs[sampler_key] = float(value)

        if request.top_p is not None:
            sampler_kwargs["top_p"] = request.top_p

        if not sampler_kwargs:
            return {}
        return {"sampler": make_sampler(**sampler_kwargs)}

    def _load_generation_config(self) -> dict:
        model_path = Path(self.config.model_id)
        if not model_path.exists():
            try:
                model_path = Path(snapshot_download(self.config.model_id))
            except Exception:
                return {}

        generation_config_path = model_path / "generation_config.json"
        if not generation_config_path.exists():
            return {}

        try:
            return json.loads(generation_config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
