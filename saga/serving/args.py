from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

from saga.config import Config


ENGINE_CLI_FIELD_NAMES = (
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "enforce_eager",
)


def _config_default(field_name: str) -> Any:
    return Config.__dataclass_fields__[field_name].default


@dataclass(frozen=True, slots=True)
class ServerArgs:
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    engine_kwargs: dict[str, Any] | None = None

    startup_timeout: float = 300.0
    request_timeout: float = 300.0

    @property
    def model_path(self) -> str:
        return os.path.abspath(os.path.expanduser(self.model))

    @property
    def effective_model_name(self) -> str:
        model_name = os.path.basename(self.model_path.rstrip("/"))
        return model_name or "saga-model"

    @property
    def llm_kwargs(self) -> dict[str, int | float | bool]:
        return dict(self.engine_kwargs or {})


def parse_args(argv: list[str] | None = None) -> ServerArgs:
    parser = argparse.ArgumentParser(description="saga OpenAI-compatible API server")
    parser.add_argument("--model", required=True, help="local model directory path")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--max-num-batched-tokens", type=int, default=_config_default("max_num_batched_tokens"))
    parser.add_argument("--max-num-seqs", type=int, default=_config_default("max_num_seqs"))
    parser.add_argument("--max-model-len", type=int, default=_config_default("max_model_len"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=_config_default("gpu_memory_utilization"))
    parser.add_argument("--tensor-parallel-size", type=int, default=_config_default("tensor_parallel_size"))
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)

    ns = parser.parse_args(argv)
    engine_kwargs = {name: getattr(ns, name) for name in ENGINE_CLI_FIELD_NAMES}
    return ServerArgs(
        model=ns.model,
        host=ns.host,
        port=ns.port,
        engine_kwargs=engine_kwargs,
        startup_timeout=ns.startup_timeout,
        request_timeout=ns.request_timeout,
    )
