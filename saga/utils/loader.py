import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def iter_safetensor_weights(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                yield weight_name, f.get_tensor(weight_name)


def load_model(model: nn.Module, path: str):
    load_weights = getattr(model, "load_weights", None)
    if callable(load_weights):
        load_weights(iter_safetensor_weights(path))
        return

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for weight_name, weight in iter_safetensor_weights(path):
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, weight, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
