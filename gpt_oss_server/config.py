from dataclasses import dataclass


DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q4"


@dataclass(frozen=True)
class ServerConfig:
    model_id: str = DEFAULT_MODEL_ID
    host: str = "127.0.0.1"
    port: int = 8000
    max_tokens: int = 1024
