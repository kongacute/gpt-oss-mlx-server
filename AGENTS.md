# Project Instructions

## Project Scope

- This project is a FastAPI OpenAI-compatible server for GPT-OSS models on MLX.
- Target model family: GPT-OSS only.
- Default model: `mlx-community/gpt-oss-20b-MXFP4-Q4`.
- Use `openai-harmony` for GPT-OSS prompt rendering and output parsing.
- Use `mlx-lm` for inference.
- Do not add API key authentication by default.
- Do not implement Batch APIs.
- Do not implement multimodal request support.

## Supported APIs

- `GET /health`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/responses`
- `POST /v1/chat/completions`

`/v1/responses` is primary. `/v1/chat/completions` exists for client compatibility.

## Python And Project Structure

- Follow standard Python packaging with `pyproject.toml`.
- Current dependency management is normal `pip`, not `uv`.
- Keep source under `gpt_oss_server/`.
- Keep tests under `tests/`.
- Prefer small modules with clear API boundaries:
  - `app.py` for FastAPI routes and request conversion.
  - `engine.py` for MLX model loading and generation.
  - `harmony.py` for Harmony rendering/parsing.
  - `schemas.py` for Pydantic models.
  - `cli.py` for command-line entrypoint.

## Runtime Behavior

- Keep model sampling defaults from model configuration.
- Only override generation settings when request setting is explicitly supported.
- Do not allow request or CLI temperature changes for GPT-OSS reasoning models.
- Preserve generated reasoning:
  - Responses API should include reasoning output items and `reasoning.content`.
  - Chat Completions should include `message.reasoning`.
  - Streaming should emit reasoning deltas before final text deltas when produced.
- Reject unsupported multimodal content with a clear `400` error.

## CLI

- CLI must support:
  - `--model`
  - `--host`
  - `--port`
  - `--max-tokens`
- CLI must not expose `--temperature`.

## Testing

- After adding behavior, add focused tests.
- Test comments should explain what each test is for.
- Run:

```bash
python3 -m pytest
```

- For syntax/import sanity, run when useful:

```bash
python3 -m compileall gpt_oss_server tests
```

## Working Rules

- Always inspect current project status before making changes.
- Decide whether user clarification is needed after status inspection.
- If working under an explicit goal/pursuit mode, do not stop to grill unless blocked.
- Highlight pros, cons, and recommendation when presenting multiple solution options.
- Keep changes scoped to requested behavior.
- Do not delete or rewrite unrelated files.

