import os
import json
from dataclasses import dataclass
import torch
from transformers import AutoConfig


class _DictConfig:

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = _DictConfig(value)
            elif isinstance(value, list):
                value = [_DictConfig(v) if isinstance(v, dict) else v for v in value]
            setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _resolve_torch_dtype(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        name = dtype_value.replace("torch.", "")
        mapped = getattr(torch, name, None)
        if isinstance(mapped, torch.dtype):
            return mapped
    return None


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    enable_continuous_batching: bool = True
    decode_steps_per_prefill: int = 4
    max_prefill_tokens_per_step: int = 2048
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    dist_init_addr: str = "127.0.0.1"
    dist_init_port: int = 0
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    hf_top_config: AutoConfig | None = None
    hf_architectures: list[str] | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.decode_steps_per_prefill >= 1
        assert self.max_prefill_tokens_per_step > 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert isinstance(self.dist_init_addr, str) and self.dist_init_addr
        assert self.dist_init_port >= 0
        if self.dist_init_port != 0:
            assert self.dist_init_port <= 65535
        try:
            self.hf_top_config = AutoConfig.from_pretrained(self.model)
        except Exception:
            config_path = os.path.join(self.model, "config.json")
            if not os.path.isfile(config_path):
                raise
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.hf_top_config = _DictConfig(raw)

        text_config = getattr(self.hf_top_config, "text_config", None)
        self.hf_config = text_config if text_config is not None else self.hf_top_config
        if not getattr(self.hf_config, "architectures", None):
            top_arch = getattr(self.hf_top_config, "architectures", None)
            if top_arch:
                self.hf_config.architectures = top_arch

        dtype = _resolve_torch_dtype(getattr(self.hf_config, "dtype", None))
        if dtype is None:
            dtype = _resolve_torch_dtype(getattr(self.hf_config, "torch_dtype", None))
        if dtype is None:
            dtype = _resolve_torch_dtype(getattr(self.hf_top_config, "dtype", None))
        if dtype is None:
            dtype = _resolve_torch_dtype(getattr(self.hf_top_config, "torch_dtype", None))
        if dtype is None:
            dtype = torch.bfloat16
        self.hf_config.dtype = dtype

        if getattr(self.hf_config, "num_key_value_heads", None) is None:
            self.hf_config.num_key_value_heads = self.hf_config.num_attention_heads

        top_arch = getattr(self.hf_top_config, "architectures", None) or []
        text_arch = getattr(self.hf_config, "architectures", None) or []
        self.hf_architectures = list(text_arch) if text_arch else list(top_arch)

        max_position_embeddings = getattr(self.hf_config, "max_position_embeddings", None)
        if max_position_embeddings is None:
            max_position_embeddings = getattr(self.hf_top_config, "max_position_embeddings", None)
        if max_position_embeddings is None:
            raise ValueError("Unable to resolve max_position_embeddings from model config")
        self.max_model_len = min(self.max_model_len, max_position_embeddings)
        if self.tensor_parallel_size > 1:
            assert torch.cuda.device_count() >= self.tensor_parallel_size, (
                f"tensor_parallel_size={self.tensor_parallel_size} exceeds visible GPU count "
                f"{torch.cuda.device_count()}"
            )
            num_heads = getattr(self.hf_config, "num_attention_heads", None)
            if isinstance(num_heads, int):
                assert num_heads % self.tensor_parallel_size == 0, (
                    f"num_attention_heads={num_heads} must be divisible by "
                    f"tensor_parallel_size={self.tensor_parallel_size}"
                )
            num_kv_heads = getattr(self.hf_config, "num_key_value_heads", None)
            if isinstance(num_kv_heads, int):
                assert num_kv_heads % self.tensor_parallel_size == 0, (
                    f"num_key_value_heads={num_kv_heads} must be divisible by "
                    f"tensor_parallel_size={self.tensor_parallel_size}"
                )

    @property
    def dist_init_method(self) -> str:
        assert self.dist_init_port > 0
        return f"tcp://{self.dist_init_addr}:{self.dist_init_port}"
