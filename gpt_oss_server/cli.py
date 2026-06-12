import argparse

import uvicorn

from .app import create_app
from .config import DEFAULT_MODEL_ID, ServerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt-oss-server",
        description="Serve GPT-OSS MLX model through OpenAI-compatible APIs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="HF model id or local path.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Default max completion tokens.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ServerConfig(
        model_id=args.model,
        host=args.host,
        port=args.port,
        max_tokens=args.max_tokens,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
