# gpt-oss-server

FastAPI OpenAI-compatible server for GPT-OSS models on MLX. It renders GPT-OSS
Harmony prompts with `openai-harmony`, runs inference through `mlx-lm`, and
exposes only the minimal APIs this project needs.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
gpt-oss-server --model mlx-community/gpt-oss-20b-MXFP4-Q4 --host 127.0.0.1 --port 8000
```

Equivalent module form:

```bash
python -m gpt_oss_server --model mlx-community/gpt-oss-20b-MXFP4-Q4 --host 127.0.0.1 --port 8000
```

`--model` defaults to `mlx-community/gpt-oss-20b-MXFP4-Q4`.

## APIs

- `GET /health`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/responses`
- `POST /v1/chat/completions`

No API key check is installed by default.

`/v1/responses` is the primary text generation API. `/v1/chat/completions`
exists for OpenAI-compatible clients that still use chat messages. Batch APIs
and multimodal request parts are intentionally not implemented.
