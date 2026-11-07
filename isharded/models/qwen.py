"""Qwen family, dense and mixture-of-experts.

One adapter covers four generations, which differ by a handful of switches:

==================  =====  ========  ===========  ===========================
family              bias   qk norm   experts      token mixing
==================  =====  ========  ===========  ===========================
Qwen2, Qwen2.5      yes    no        no           attention
Qwen1.5/2 MoE       yes    no        + shared     attention
Qwen3               no     yes       no           attention
Qwen3 MoE           no     yes       routed       attention
Qwen3.5/3.6/3.8     no     yes       + shared     3:1 Gated DeltaNet, attention
==================  =====  ========  ===========  ===========================

The last row covers dense and MoE checkpoints of those releases. They are
multimodal, and only the language model is loaded: the vision tower and the
multi-token-prediction head are skipped, so their bytes are never fetched.
"""

from __future__ import annotations

import re
from typing import Any

from ..optional import functional as F
from ..optional import torch

from ..protocol import TensorRef
from ..weights import FP8_SCALE_SUFFIX, dequantize_fp8
from .base import Adapter, LayerState, UnsupportedModel, Weights, attend

HYBRID_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
BIASED_ATTENTION_TYPES = {"qwen2", "qwen2_moe"}
QK_NORM_TYPES = {"qwen3", "qwen3_moe"} | HYBRID_TYPES


def matrix(weights: Weights, name: str, like: torch.Tensor) -> torch.Tensor:
    """A weight matrix ready to multiply, expanding it if the pool kept it as FP8."""
    weight = weights[name]
    if weight.dtype == torch.float8_e4m3fn:
        return dequantize_fp8(weight, weights[name + FP8_SCALE_SUFFIX], like.dtype)
    return weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet recurrence, one token at a time, in float32.

    Inputs are ``[1, tokens, heads, dim]``. The state is a fixed-size matrix
    per head, so memory does not grow with context. Prompts run token by token
    here too, which is exact but slower than a chunked kernel.
    """
    out_dtype = query.dtype
    query, key, value, beta, decay = (
        item.transpose(1, 2).float().contiguous() for item in (query, key, value, beta, decay)
    )
    query = _l2_normalize(query) / (query.shape[-1] ** 0.5)
    key = _l2_normalize(key)
    if state is None:
        state = value.new_zeros(1, value.shape[1], key.shape[-1], value.shape[-1])
    else:
        state = state.float()
    output = torch.zeros_like(value)
    for step in range(query.shape[2]):
        k_t = key[:, :, step].unsqueeze(-1)
        state = state * decay[:, :, step].exp()[..., None, None]
        remembered = (state * k_t).sum(dim=-2)
        delta = (value[:, :, step] - remembered) * beta[:, :, step].unsqueeze(-1)
        state = state + k_t * delta.unsqueeze(-2)
        output[:, :, step] = (state * query[:, :, step].unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(out_dtype), state


class QwenAdapter(Adapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_type = str(config["model_type"])
        self.hybrid = self.model_type in HYBRID_TYPES
        if config.get("use_sliding_window"):
            raise UnsupportedModel("sliding-window attention is not implemented")
        self.num_layers = int(config["num_hidden_layers"])
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config.get("num_key_value_heads") or self.num_heads)
        self.head_dim = int(config.get("head_dim") or self.hidden_size // self.num_heads)
        self.epsilon = float(config.get("rms_norm_eps", 1e-6))
        self.attention_bias = self.model_type in BIASED_ATTENTION_TYPES or bool(config.get("attention_bias"))
        self.qk_norm = self.model_type in QK_NORM_TYPES
        self.output_gate = bool(config.get("attn_output_gate", self.hybrid))
        # Multimodal wrappers keep the language model one level deeper.
        self.prefix = "model.language_model." if config.get("_multimodal_wrapper", False) else "model."
        self._layer_pattern = re.compile(rf"^{re.escape(self.prefix)}layers\.(\d+)\.(.+)$")

        rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise UnsupportedModel(f"rope type {rope_type!r} is not implemented")
        self.rope_theta = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
        partial = float(rope.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0)))
        self.rotary_dim = int(self.head_dim * partial)
        self._inverse_frequencies: dict[torch.device, torch.Tensor] = {}

        self.layer_types = list(config.get("layer_types") or ["full_attention"] * self.num_layers)
        self.supports_kv_snapshots = "linear_attention" not in self.layer_types
        if not self.supports_kv_snapshots:
            self.linear_key_heads = int(config["linear_num_key_heads"])
            self.linear_value_heads = int(config["linear_num_value_heads"])
            self.linear_key_dim = int(config["linear_key_head_dim"])
            self.linear_value_dim = int(config["linear_value_head_dim"])
            self.conv_kernel = int(config["linear_conv_kernel_dim"])

        # Checkpoints re-saved by transformers 5 use the second spelling.
        self.num_experts = int(config.get("num_experts") or config.get("num_local_experts") or 0)
        self.top_k = int(config.get("num_experts_per_tok") or 0)
        # Hybrid releases always renormalize; earlier ones make it a setting.
        self.renormalize = self.hybrid or bool(config.get("norm_topk_prob"))
        only_dense = set(config.get("mlp_only_layers") or [])
        step = int(config.get("decoder_sparse_step") or 1)
        self.sparse = [
            self.num_experts > 0 and index not in only_dense and (index + 1) % step == 0
            for index in range(self.num_layers)
        ]
        if self.num_experts:
            self.expert_active_fraction = self.top_k / self.num_experts

    def classify(self, name: str) -> tuple[str, int | None] | None:
        match = self._layer_pattern.match(name)
        if match:
            part = "experts" if match.group(2).startswith("mlp.experts.") else "dense"
            return part, int(match.group(1))
        if name == f"{self.prefix}embed_tokens.weight":
            return "embed", None
        if name in {f"{self.prefix}norm.weight", "lm_head.weight"}:
            return "head", None
        return None

    def embed_tensor_name(self, tensors: dict[str, TensorRef]) -> str:
        return f"{self.prefix}embed_tokens.weight"

    def has_output_projection(self, tensors: dict[str, TensorRef]) -> bool:
        return "lm_head.weight" in tensors

    def local_name(self, name: str) -> str:
        match = self._layer_pattern.match(name)
        return match.group(2) if match else name.removeprefix(self.prefix)

    def keep_float32(self, name: str) -> bool:
        return name.endswith(("A_log", "dt_bias"))

    def embed(self, weights: Weights, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        table = weights["embed_tokens.weight"]
        return F.embedding(input_ids.to(table.device), table)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        """Scale to unit root mean square, in float32."""
        wide = x.float()
        return wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized = self._rms(x)
        if self.hybrid:
            # Hybrid releases store the norm scale as an offset from one.
            return (normalized * (1.0 + weight.float())).type_as(x)
        return weight * normalized.type_as(x)

    def _rotary(self, positions: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = like.device
        inverse = self._inverse_frequencies.get(device)
        if inverse is None:
            exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
            inverse = 1.0 / (self.rope_theta ** (exponents / self.rotary_dim))
            self._inverse_frequencies[device] = inverse
        angles = positions.to(device).float()[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(like.dtype), angles.sin().to(like.dtype)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rotated, untouched = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        rotated = rotated * cos + _rotate_half(rotated) * sin
        return torch.cat((rotated, untouched), dim=-1)

    def _attention(
        self,
        weights: Weights,
        x: torch.Tensor,
        state: LayerState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = x.shape[1]

        def project(name: str) -> torch.Tensor:
            bias = weights.get(f"self_attn.{name}.bias") if self.attention_bias else None
            return F.linear(x, matrix(weights, f"self_attn.{name}.weight", x), bias)

        gate = None
        query = project("q_proj")
        if self.output_gate:
            query, gate = query.view(1, tokens, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
            gate = gate.reshape(1, tokens, self.num_heads * self.head_dim)
        query = query.reshape(1, tokens, self.num_heads, self.head_dim)
        key = project("k_proj").view(1, tokens, self.num_kv_heads, self.head_dim)
        value = project("v_proj").view(1, tokens, self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            query = self._norm(query, weights["self_attn.q_norm.weight"])
            key = self._norm(key, weights["self_attn.k_norm.weight"])
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        cos, sin = self._rotary(positions, query)
        query = self._apply_rotary(query, cos, sin)
        key = self._apply_rotary(key, cos, sin)
        attended = attend(query, key, value, state)
        attended = attended.transpose(1, 2).reshape(1, tokens, self.num_heads * self.head_dim)
        if gate is not None:
            attended = attended * torch.sigmoid(gate)
        return F.linear(attended, matrix(weights, "self_attn.o_proj.weight", attended))

    def _linear_attention(self, weights: Weights, x: torch.Tensor, state: LayerState) -> torch.Tensor:
        tokens = x.shape[1]
        key_width = self.linear_key_heads * self.linear_key_dim
        value_width = self.linear_value_heads * self.linear_value_dim
        mixed = F.linear(x, matrix(weights, "linear_attn.in_proj_qkv.weight", x)).transpose(1, 2)
        gate = F.linear(x, matrix(weights, "linear_attn.in_proj_z.weight", x))
        beta = F.linear(x, matrix(weights, "linear_attn.in_proj_b.weight", x)).sigmoid()
        alpha = F.linear(x, matrix(weights, "linear_attn.in_proj_a.weight", x))

        # Causal depthwise convolution, carrying the last kernel-1 inputs over
        # so that decoding one token sees the same window as a full pass.
        history = state.get("conv")
        if history is None:
            history = mixed.new_zeros(1, mixed.shape[1], self.conv_kernel - 1)
        padded = torch.cat((history.to(mixed.dtype), mixed), dim=-1)
        state["conv"] = padded[:, :, -(self.conv_kernel - 1) :].contiguous()
        kernel = weights["linear_attn.conv1d.weight"]
        convolved = F.silu(F.conv1d(padded, kernel, groups=mixed.shape[1])).transpose(1, 2)

        query, key, value = convolved.split([key_width, key_width, value_width], dim=-1)
        query = query.reshape(1, tokens, self.linear_key_heads, self.linear_key_dim)
        key = key.reshape(1, tokens, self.linear_key_heads, self.linear_key_dim)
        value = value.reshape(1, tokens, self.linear_value_heads, self.linear_value_dim)
        repeat = self.linear_value_heads // self.linear_key_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        decay = -weights["linear_attn.A_log"].float().exp() * F.softplus(
            alpha.float() + weights["linear_attn.dt_bias"].float()
        )
        core, state["recurrent"] = gated_delta_rule(query, key, value, decay, beta, state.get("recurrent"))

        # Gated RMSNorm per value head, then the output projection.
        core = core.reshape(-1, self.linear_value_dim)
        gate = gate.reshape(-1, self.linear_value_dim)
        normalized = weights["linear_attn.norm.weight"] * self._rms(core).to(core.dtype)
        normalized = (normalized * F.silu(gate.float())).to(core.dtype)
        flattened = normalized.reshape(1, tokens, value_width)
        return F.linear(flattened, matrix(weights, "linear_attn.out_proj.weight", flattened))

    @staticmethod
    def _swiglu(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    def _dense_mlp(self, weights: Weights, prefix: str, x: torch.Tensor) -> torch.Tensor:
        return self._swiglu(
            matrix(weights, f"{prefix}.gate_proj.weight", x),
            matrix(weights, f"{prefix}.up_proj.weight", x),
            matrix(weights, f"{prefix}.down_proj.weight", x),
            x,
        )

    @staticmethod
    def _expert(bank: Weights, index: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert weights from either checkpoint layout, without copying."""
        fused = bank.get("mlp.experts.gate_up_proj")
        if fused is not None:
            gate, up = fused[index].chunk(2, dim=0)
            return gate, up, bank["mlp.experts.down_proj"][index]
        return (
            matrix(bank, f"mlp.experts.{index}.gate_proj.weight", like),
            matrix(bank, f"mlp.experts.{index}.up_proj.weight", like),
            matrix(bank, f"mlp.experts.{index}.down_proj.weight", like),
        )

    def _mixture(self, dense: Weights, bank: Weights, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.hidden_size)
        router = F.linear(flat, matrix(dense, "mlp.gate.weight", flat))
        probabilities = F.softmax(router, dim=-1, dtype=torch.float32)
        scores, chosen = torch.topk(probabilities, self.top_k, dim=-1)
        if self.renormalize:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        scores = scores.to(router.dtype)

        # Routed experts may live in another pool than attention. Only the
        # activations cross, a few kilobytes per token, never the weights.
        sample = next(iter(bank.values()))
        compute_dtype = flat.dtype if sample.dtype == torch.float8_e4m3fn else sample.dtype
        local = flat.to(sample.device, compute_dtype)
        chosen = chosen.to(sample.device)
        scores = scores.to(sample.device, compute_dtype)
        routed = torch.zeros_like(local)
        for expert in chosen.unique().tolist():
            token_index, slot = torch.where(chosen == expert)
            gate, up, down = self._expert(bank, expert, local)
            result = self._swiglu(gate, up, down, local[token_index])
            routed.index_add_(0, token_index, result * scores[token_index, slot, None])
        output = routed.to(flat.device, flat.dtype)

        if "mlp.shared_expert.gate_proj.weight" in dense:
            shared = self._dense_mlp(dense, "mlp.shared_expert", flat)
            routing = matrix(dense, "mlp.shared_expert_gate.weight", flat)
            output = output + torch.sigmoid(F.linear(flat, routing)) * shared
        return output.view_as(x)

    def run_layer(
        self,
        layer: int,
        dense: Weights,
        experts: Weights | None,
        hidden: torch.Tensor,
        state: LayerState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        reference = dense["input_layernorm.weight"]
        hidden = hidden.to(reference.device, reference.dtype)
        mixed_input = self._norm(hidden, reference)
        if self.layer_types[layer] == "linear_attention":
            hidden = hidden + self._linear_attention(dense, mixed_input, state)
        else:
            hidden = hidden + self._attention(dense, mixed_input, state, positions)
        mlp_input = self._norm(hidden, dense["post_attention_layernorm.weight"])
        if self.sparse[layer]:
            if experts is None:
                raise ValueError(f"layer {layer} has no expert weights")
            return hidden + self._mixture(dense, experts, mlp_input)
        return hidden + self._dense_mlp(dense, "mlp", mlp_input)

    def logits(self, weights: Weights, hidden: torch.Tensor) -> torch.Tensor:
        reference = weights["norm.weight"]
        hidden = hidden.to(reference.device, reference.dtype)
        projection = weights.get("lm_head.weight", weights.get("embed_tokens.weight"))
        return F.linear(self._norm(hidden, reference), projection)

    def kv_bytes_per_token(self, layer: int, itemsize: int) -> int:
        if self.layer_types[layer] == "linear_attention":
            return 0
        return 2 * self.num_kv_heads * self.head_dim * itemsize

    def recurrent_state_bytes(self, layer: int) -> int:
        if self.layer_types[layer] != "linear_attention":
            return 0
        recurrent = self.linear_value_heads * self.linear_key_dim * self.linear_value_dim * 4
        channels = 2 * self.linear_key_heads * self.linear_key_dim
        channels += self.linear_value_heads * self.linear_value_dim
        return recurrent + channels * (self.conv_kernel - 1) * 2
