import re
from collections import defaultdict

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from saga.layers.activation import SiluAndMul
from saga.layers.attention import Attention
from saga.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from saga.layers.rotary_embedding import get_rope
from saga.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from saga.utils.context import get_context
from saga.utils.loader import default_weight_loader


class RMSNormZeroCentered(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm_and_scale(self, x: torch.Tensor, orig_dtype: torch.dtype) -> torch.Tensor:
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype)
        return x.mul(1.0 + self.weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm_and_scale(x, x.dtype)
        orig_dtype = residual.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        x = self._norm_and_scale(x, orig_dtype)
        return x, residual


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul(self.weight)
        x = x.float().mul(F.silu(gate.float()))
        return x.to(orig_dtype)


class Qwen3_5Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        rope_theta: float,
        partial_rotary_factor: float,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim * 2,
            bias=attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
        )

        self.q_norm = RMSNormZeroCentered(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNormZeroCentered(self.head_dim, eps=rms_norm_eps)
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rotary_dim = max(2, rotary_dim // 2 * 2)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # HF packs q/gate interleaved per head as [..., num_heads, head_dim * 2].
        # Splitting the flattened tensor in half mixes heads and diverges from reference.
        qg = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(-1, self.q_size)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v).flatten(1, -1)
        o = o * torch.sigmoid(gate)
        return self.o_proj(o)


class Qwen3_5MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.cat((gate, up), dim=-1)
        x = self.act_fn(x)
        return self.down_proj(x)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    # query/key/value: [L, H, D]
    # g/beta: [L, H]
    initial_dtype = query.dtype
    query = _l2norm(query.float(), dim=-1)
    key = _l2norm(key.float(), dim=-1)
    value = value.float()
    beta = beta.float()
    g = g.float()

    query = query.transpose(0, 1).unsqueeze(0).contiguous()
    key = key.transpose(0, 1).unsqueeze(0).contiguous()
    value = value.transpose(0, 1).unsqueeze(0).contiguous()
    beta = beta.transpose(0, 1).unsqueeze(0).contiguous()
    g = g.transpose(0, 1).unsqueeze(0).contiguous()

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(
        batch_size,
        num_heads,
        k_head_dim,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (
                k_i
                * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    return core_attn_out.squeeze(0).to(initial_dtype)


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        self.hidden_size = config.hidden_size
        self.total_num_v_heads = int(config.linear_num_value_heads)
        self.total_num_k_heads = int(config.linear_num_key_heads)
        assert self.total_num_v_heads % tp_size == 0
        assert self.total_num_k_heads % tp_size == 0
        self.num_v_heads = self.total_num_v_heads // tp_size
        self.num_k_heads = self.total_num_k_heads // tp_size

        self.head_k_dim = int(config.linear_key_head_dim)
        self.head_v_dim = int(config.linear_value_head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size,
            (self.total_num_k_heads * self.head_k_dim) * 2 + (self.total_num_v_heads * self.head_v_dim),
            bias=False,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_v_heads * self.head_v_dim,
            bias=False,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_v_heads,
            bias=False,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_v_heads,
            bias=False,
        )
        self.out_proj = RowParallelLinear(
            self.total_num_v_heads * self.head_v_dim,
            self.hidden_size,
            bias=False,
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.conv1d.weight.weight_loader = self._conv1d_weight_loader

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.dt_bias.weight_loader = self._shard_dim0_loader
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0.0, 16.0).log_())
        self.A_log.weight_loader = self._shard_dim0_loader
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def _shard_dim0_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            return
        local = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        param.data.copy_(local)

    def _conv1d_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            return
        local = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        param.data.copy_(local)

    def _forward_one_seq(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.size(0)

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(0, 1).unsqueeze(0)
        with torch.backends.cudnn.flags(enabled=False):
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).squeeze(0).transpose(0, 1)

        z = self.in_proj_z(hidden_states).view(seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states).view(seq_len, self.num_v_heads)
        a = self.in_proj_a(hidden_states).view(seq_len, self.num_v_heads)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.view(seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(seq_len, self.num_v_heads, self.head_v_dim)

        if self.num_v_heads > self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)

        beta = b.sigmoid()
        g = -self.A_log.float().exp().unsqueeze(0) * F.softplus(a.float() + self.dt_bias.float().unsqueeze(0))
        core = _chunk_gated_delta_rule(query, key, value, g, beta)
        core = self.norm(core, z)

        return self.out_proj(core.reshape(seq_len, -1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        cu_seqlens = context.cu_seqlens_q
        if cu_seqlens is None:
            raise RuntimeError("Qwen3.5 linear attention requires cu_seqlens_q context")

        outputs = []
        for i in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[i])
            end = int(cu_seqlens[i + 1])
            if end <= start:
                continue
            outputs.append(self._forward_one_seq(hidden_states[start:end]))
        return torch.cat(outputs, dim=0)


class Qwen3_5RoutedExperts(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.total_intermediate_size = intermediate_size
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert self.total_intermediate_size % self.tp_size == 0
        self.intermediate_size = self.total_intermediate_size // self.tp_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.gate_up_proj.weight_loader = self._gate_up_weight_loader
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.down_proj.weight_loader = self._down_weight_loader

        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def _gate_up_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            return
        gate, up = loaded_weight.chunk(2, dim=1)
        gate = gate.chunk(self.tp_size, dim=1)[self.tp_rank]
        up = up.chunk(self.tp_size, dim=1)[self.tp_rank]
        param.data.copy_(torch.cat((gate, up), dim=1))

    def _down_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.shape == param.data.shape:
            param.data.copy_(loaded_weight)
            return
        local = loaded_weight.chunk(self.tp_size, dim=2)[self.tp_rank]
        param.data.copy_(local)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in hit:
            eid = int(expert_idx[0])
            topk_pos, token_idx = torch.where(expert_mask[eid])
            current = hidden_states[token_idx]
            gate_up = F.linear(current, self.gate_up_proj[eid])
            current = self.act_fn(gate_up)
            current = F.linear(current, self.down_proj[eid])
            current = current * routing_weights[token_idx, topk_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))

        if self.tp_size > 1:
            dist.all_reduce(final_hidden_states)
        return final_hidden_states


class Qwen3_5TopKRouter(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        topk_values, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)
        topk_values = topk_values.to(router_logits.dtype)
        return router_logits, topk_values, topk_indices


class Qwen3_5MoeBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.gate = Qwen3_5TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
        )
        self.experts = Qwen3_5RoutedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            num_experts=int(config.num_experts),
            hidden_act=config.hidden_act,
        )
        self.shared_expert = Qwen3_5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=int(config.shared_expert_intermediate_size),
            hidden_act=config.hidden_act,
        )
        self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, routing_weights, selected_experts = self.gate(hidden_states)
        expert_output = self.experts(hidden_states, selected_experts, routing_weights)
        shared_output = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        return expert_output + shared_output * shared_gate


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int, use_moe: bool) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rope_scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
        rope_theta = float(getattr(config, "rope_theta", rope_scaling.get("rope_theta", 1000000.0)))
        partial_rotary_factor = float(rope_scaling.get("partial_rotary_factor", 1.0))

        if self.layer_type == "full_attention":
            self.token_mixer = Qwen3_5Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=config.max_position_embeddings,
                head_dim=head_dim,
                rms_norm_eps=config.rms_norm_eps,
                attention_bias=bool(getattr(config, "attention_bias", False)),
                rope_theta=rope_theta,
                partial_rotary_factor=partial_rotary_factor,
            )
        elif self.layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(config)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

        if use_moe:
            self.mlp = Qwen3_5MoeBlock(config)
        else:
            self.mlp = Qwen3_5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNormZeroCentered(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormZeroCentered(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
        else:
            residual = hidden_states.float().add_(residual.float()).to(hidden_states.dtype)
        hidden_states = self.input_layernorm(residual)
        if self.layer_type == "full_attention":
            hidden_states = self.token_mixer(positions, hidden_states)
        else:
            hidden_states = self.token_mixer(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states, None


class Qwen3_5Model(nn.Module):

    def __init__(self, config, use_moe: bool) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        layer_types = getattr(config, "layer_types", None)
        if not layer_types:
            layer_types = ["full_attention"] * int(config.num_hidden_layers)
            config.layer_types = layer_types
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i, use_moe=use_moe) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormZeroCentered(config.hidden_size, eps=config.rms_norm_eps)
        self.uses_linear_attention = any(t == "linear_attention" for t in layer_types)
        self.num_kv_cache_layers = sum(1 for t in layer_types if t == "full_attention")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if residual is not None:
            hidden_states = hidden_states.float().add_(residual.float()).to(hidden_states.dtype)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class _Qwen3_5BaseForCausalLM(nn.Module):
    _expert_pattern = re.compile(r"^(?P<prefix>.+\\.mlp\\.experts)\\.(?P<idx>\\d+)\\.(?P<name>gate_proj|up_proj|down_proj)\\.weight$")

    def __init__(self, config, use_moe: bool):
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        self.model = Qwen3_5Model(config, use_moe=use_moe)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.full_context_mode = self.model.uses_linear_attention
        self.num_kv_cache_layers = self.model.num_kv_cache_layers
        self.num_experts = int(getattr(config, "num_experts", 0)) if use_moe else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def _load_one_param(self, params: dict[str, nn.Parameter], name: str, tensor: torch.Tensor) -> bool:
        target_name = name
        if target_name not in params and target_name.endswith(".weight"):
            alt = target_name.removesuffix(".weight")
            if alt in params:
                target_name = alt
        if target_name not in params:
            return False
        param = params[target_name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)
        return True

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        expert_buf: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))

        for name, loaded_weight in weights:
            if name.startswith("vision_tower.") or name.startswith("multi_modal_projector."):
                continue
            name = name.removeprefix("model.language_model.")
            name = name.removeprefix("language_model.")
            name = name.removeprefix("model.")
            if not name.startswith("model.") and name.startswith(("layers.", "embed_tokens.", "norm.")):
                name = f"model.{name}"
            name = name.replace(".linear_attn.", ".token_mixer.")
            name = name.replace(".self_attn.", ".token_mixer.")
            if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            match = self._expert_pattern.match(name)
            if match is not None:
                prefix = match.group("prefix")
                idx = int(match.group("idx"))
                pname = match.group("name")
                expert_buf[prefix][idx][pname] = loaded_weight
                continue

            self._load_one_param(params, name, loaded_weight)

        if expert_buf:
            if not self.use_moe:
                raise RuntimeError("Encountered MoE expert weights when loading dense Qwen3.5 model")
            if self.num_experts <= 0:
                raise RuntimeError("Invalid num_experts for Qwen3.5-MoE")

        for prefix, per_expert in expert_buf.items():
            gate_up_list = []
            down_list = []
            for idx in range(self.num_experts):
                if idx not in per_expert:
                    raise RuntimeError(f"Missing expert {idx} under {prefix}")
                parts = per_expert[idx]
                for key in ("gate_proj", "up_proj", "down_proj"):
                    if key not in parts:
                        raise RuntimeError(f"Missing {key} for expert {idx} under {prefix}")
                gate_up_list.append(torch.cat((parts["gate_proj"], parts["up_proj"]), dim=0))
                down_list.append(parts["down_proj"])

            gate_up = torch.stack(gate_up_list, dim=0)
            down = torch.stack(down_list, dim=0)
            if not self._load_one_param(params, f"{prefix}.gate_up_proj", gate_up):
                raise RuntimeError(f"Parameter not found: {prefix}.gate_up_proj")
            if not self._load_one_param(params, f"{prefix}.down_proj", down):
                raise RuntimeError(f"Parameter not found: {prefix}.down_proj")


class Qwen3_5ForCausalLM(_Qwen3_5BaseForCausalLM):

    def __init__(self, config) -> None:
        super().__init__(config, use_moe=False)


class Qwen3_5MoeForCausalLM(_Qwen3_5BaseForCausalLM):

    def __init__(self, config) -> None:
        super().__init__(config, use_moe=True)
