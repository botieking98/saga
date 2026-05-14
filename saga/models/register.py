from typing import Sequence

from saga.models.qwen3 import Qwen3ForCausalLM
from saga.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM


def get_model_class(config, architectures: Sequence[str] | None):
    arch = architectures[0] if architectures else ""
    model_type = str(getattr(config, "model_type", ""))

    if arch in {"Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"}:
        return Qwen3_5MoeForCausalLM
    if arch in {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"}:
        return Qwen3_5ForCausalLM

    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return Qwen3_5MoeForCausalLM
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return Qwen3_5ForCausalLM

    # fallback to original Qwen3 dense model
    return Qwen3ForCausalLM
