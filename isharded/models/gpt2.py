"""GPT-2 family: GPT-2 in every size, DistilGPT-2, and compatible fine-tunes."""

from __future__ import annotations

import re
from typing import Any

from ..optional import functional as F
from ..optional import torch

from ..protocol import TensorRef
from .base import Adapter, LayerState, UnsupportedModel, Weights, attend

_LAYER = re.compile(r"^(?:transformer\.)?h\.(\d+)\.(.+)$")


class Gpt2Adapter(Adapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if config.get("scale_attn_by_inverse_layer_idx") or config.get("reorder_and_upcast_attn"):
            raise UnsupportedModel("this GPT-2 variant uses non-standard attention scaling")
        self.num_layers = int(config["n_layer"])
        self.num_heads = int(config["n_head"])
        self.hidden_size = int(config["n_embd"])
        self.head_dim = self.hidden_size // self.num_heads
        self.epsilon = float(config.get("layer_norm_epsilon", 1e-5))

    def classify(self, name: str) -> tuple[str, int | None] | None:
        match = _LAYER.match(name)
        if match:
            # attn.bias and attn.masked_bias are causal-mask buffers, not weights.
            if match.group(2) in {"attn.bias", "attn.masked_bias"}:
                return None
            return "dense", int(match.group(1))
        local = self.local_name(name)
        if local in {"wte.weight", "wpe.weight"}:
            return "embed", None
        if local in {"ln_f.weight", "ln_f.bias", "lm_head.weight"}:
            return "head", None
        return None

    def embed_tensor_name(self, tensors: dict[str, TensorRef]) -> str:
        return "transformer.wte.weight" if "transformer.wte.weight" in tensors else "wte.weight"

    def has_output_projection(self, tensors: dict[str, TensorRef]) -> bool:
        return "lm_head.weight" in tensors

    def local_name(self, name: str) -> str:
        match = _LAYER.match(name)
        return match.group(2) if match else name.removeprefix("transformer.")

    def embed(self, weights: Weights, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        table = weights["wte.weight"]
        tokens = F.embedding(input_ids.to(table.device), table)
        return tokens + F.embedding(positions.to(table.device), weights["wpe.weight"])

    def _norm(self, weights: Weights, prefix: str, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.hidden_size,),
            weights[f"{prefix}.weight"],
            weights[f"{prefix}.bias"],
            self.epsilon,
        )

    @staticmethod
    def _project(weights: Weights, prefix: str, x: torch.Tensor) -> torch.Tensor:
        # GPT-2's Conv1D stores weights as [in, out], the transpose of Linear.
        return x @ weights[f"{prefix}.weight"] + weights[f"{prefix}.bias"]

    def run_layer(
        self,
        layer: int,
        dense: Weights,
        experts: Weights | None,
        hidden: torch.Tensor,
        state: LayerState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        reference = dense["ln_1.weight"]
        hidden = hidden.to(reference.device, reference.dtype)
        tokens = hidden.shape[1]
        query, key, value = self._project(dense, "attn.c_attn", self._norm(dense, "ln_1", hidden)).split(
            self.hidden_size, dim=-1
        )

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(1, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attended = attend(heads(query), heads(key), heads(value), state)
        attended = attended.transpose(1, 2).reshape(1, tokens, self.hidden_size)
        hidden = hidden + self._project(dense, "attn.c_proj", attended)
        inner = F.gelu(
            self._project(dense, "mlp.c_fc", self._norm(dense, "ln_2", hidden)),
            approximate="tanh",
        )
        return hidden + self._project(dense, "mlp.c_proj", inner)

    def logits(self, weights: Weights, hidden: torch.Tensor) -> torch.Tensor:
        reference = weights["ln_f.weight"]
        hidden = hidden.to(reference.device, reference.dtype)
        projection = weights.get("lm_head.weight", weights.get("wte.weight"))
        return F.linear(self._norm(weights, "ln_f", hidden), projection)

    def kv_bytes_per_token(self, layer: int, itemsize: int) -> int:
        return 2 * self.hidden_size * itemsize
