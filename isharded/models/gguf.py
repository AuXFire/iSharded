"""Models that run from a GGUF file on llama.cpp's kernels.

This adapter only describes the model: the units a node fetches and what they
cost. The forward pass happens inside llama.cpp, one stage per worker.
"""

from __future__ import annotations

import re
from typing import Any

from ..optional import torch

from ..gguf import LAYER_NAME, GgufHeader
from ..protocol import TensorRef
from .base import Adapter, LayerState, UnsupportedModel, Weights

# Families whose graph ends each layer in an ``l_out`` tensor and that need no
# per-layer inputs beyond the hidden state. Stage splitting relies on both.
SUPPORTED_ARCHITECTURES = {"llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe"}
EXPERT_TENSOR = re.compile(r"^ffn_(gate|up|down|gate_up)_exps\.")


class GgufAdapter(Adapter):
    supports_kv_snapshots = False  # state lives inside llama.cpp: recovery uses statestore.py instead

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.header = GgufHeader.from_skeleton(config["gguf"])
        self.architecture = self.header.architecture
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise UnsupportedModel(f"GGUF architecture {self.architecture!r} cannot be split yet; supported: {supported}")
        self.num_layers = self.header.layer_count
        self.interval = int(self.header.arch_key("full_attention_interval") or 0)
        self.num_experts = int(self.header.arch_key("expert_count") or 0)
        used = int(self.header.arch_key("expert_used_count") or 0)
        self.expert_active_fraction = used / self.num_experts if self.num_experts else 1.0

    @property
    def output_width(self) -> int:
        """What an embedding model's vectors measure, which can differ from its hidden size."""
        return int(self.header.arch_key("embedding_length_out") or self.header.hidden_size)

    def classify(self, name: str) -> tuple[str, int | None] | None:
        if name == "token_embd.weight":
            return "embed", None
        if name in {"output.weight", "output_norm.weight"}:
            return "head", None
        match = LAYER_NAME.match(name)
        if not match:
            return None
        layer = int(match.group(1))
        if layer >= self.num_layers:
            return None  # multi-token-prediction blocks stay behind
        return ("experts" if EXPERT_TENSOR.match(match.group(2)) else "dense"), layer

    def embed_tensor_name(self, tensors: dict[str, TensorRef]) -> str:
        return "token_embd.weight"

    def has_output_projection(self, tensors: dict[str, TensorRef]) -> bool:
        return "output.weight" in tensors

    def local_name(self, name: str) -> str:
        return name

    def is_attention_layer(self, layer: int) -> bool:
        return not self.interval or (layer + 1) % self.interval == 0

    def kv_bytes_per_token(self, layer: int, itemsize: int) -> int:
        if not self.is_attention_layer(layer):
            return 0
        heads = int(self.header.arch_key("attention.head_count_kv") or self.header.arch_key("attention.head_count"))
        width = self.header.hidden_size // int(self.header.arch_key("attention.head_count"))
        key = int(self.header.arch_key("attention.key_length") or width)
        value = int(self.header.arch_key("attention.value_length") or width)
        return heads * (key + value) * itemsize

    def recurrent_state_bytes(self, layer: int) -> int:
        if self.is_attention_layer(layer):
            return 0
        state = int(self.header.arch_key("ssm.state_size") or 0)
        groups = int(self.header.arch_key("ssm.group_count") or 0)
        inner = int(self.header.arch_key("ssm.inner_size") or 0)
        heads = int(self.header.arch_key("ssm.time_step_rank") or 1)
        kernel = int(self.header.arch_key("ssm.conv_kernel") or 1)
        convolution = (kernel - 1) * (state * groups * 2 + inner)
        return (convolution + state * (inner // heads) * heads) * 4

    def embed(self, weights: Weights, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("llama.cpp runs this model")

    def run_layer(
        self,
        layer: int,
        dense: Weights,
        experts: Weights | None,
        hidden: torch.Tensor,
        state: LayerState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("llama.cpp runs this model")

    def logits(self, weights: Weights, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("llama.cpp runs this model")
