"""Shared pieces of the model adapters.

Adapters run the forward pass directly on tensors instead of through
``transformers`` model classes, for three reasons. Nodes can run different
library versions, since nothing about execution depends on the installed one.
Qwen releases work on a node whose ``transformers`` predates them. And each
unit of a layer can sit on its own device, which a stock module assumes away.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

from ..protocol import TensorRef, WeightUnit

Weights = dict[str, torch.Tensor]
# Per-session, per-layer state: KV tensors, or conv and recurrent state.
LayerState = dict[str, torch.Tensor]


def layer_unit(layer: int, part: str) -> str:
    return f"layer-{layer:05d}.{part}"


class UnsupportedModel(ValueError):
    """This checkpoint cannot run on ShardMesh, with the reason why."""


class Adapter(ABC):
    """One model family, described well enough to plan and execute it."""

    num_layers: int
    # Share of a layer's expert bytes one token reads. Lets the planner see
    # that experts tolerate slow memory far better than attention does.
    expert_active_fraction: float = 1.0
    # False when per-layer state cannot be checkpointed incrementally.
    supports_kv_snapshots: bool = True

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def classify(self, name: str) -> tuple[str, int | None] | None:
        """Map a tensor name to ``(kind, layer)``, or None to skip it."""

    @abstractmethod
    def embed_tensor_name(self, tensors: dict[str, TensorRef]) -> str:
        """Name of the token embedding, reused as the head when weights are tied."""

    @abstractmethod
    def local_name(self, name: str) -> str:
        """Tensor name relative to its unit."""

    @abstractmethod
    def has_output_projection(self, tensors: dict[str, TensorRef]) -> bool:
        """True when the checkpoint stores its own output head."""

    @abstractmethod
    def embed(self, weights: Weights, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Token ids to hidden states."""

    @abstractmethod
    def run_layer(
        self,
        layer: int,
        dense: Weights,
        experts: Weights | None,
        hidden: torch.Tensor,
        state: LayerState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run one transformer layer, updating ``state`` in place."""

    @abstractmethod
    def logits(self, weights: Weights, hidden: torch.Tensor) -> torch.Tensor:
        """Hidden states to vocabulary logits."""

    @abstractmethod
    def kv_bytes_per_token(self, layer: int, itemsize: int) -> int:
        """Memory one more token of context costs in this layer."""

    def recurrent_state_bytes(self, layer: int) -> int:
        """Fixed per-session state, for layers that have any."""
        return 0

    def keep_float32(self, name: str) -> bool:
        """True for tensors that lose accuracy below float32."""
        return False

    def partition(self, tensors: dict[str, TensorRef]) -> list[WeightUnit]:
        """Group a checkpoint's tensors into fetchable, placeable units."""
        groups: dict[str, tuple[str, int | None, list[TensorRef]]] = {}
        for name, ref in tensors.items():
            placed = self.classify(name)
            if placed is None:
                continue
            kind, layer = placed
            unit = kind if layer is None else layer_unit(layer, kind)
            groups.setdefault(unit, (kind, layer, []))[2].append(ref)
        if not self.has_output_projection(tensors):
            # Tied weights: the exit stage needs its own copy of the embedding.
            embedding = tensors[self.embed_tensor_name(tensors)]
            groups.setdefault("head", ("head", None, []))[2].append(embedding)
        missing = [
            unit
            for unit in ("embed", "head", *(layer_unit(i, "dense") for i in range(self.num_layers)))
            if unit not in groups
        ]
        if missing:
            raise UnsupportedModel(f"checkpoint is missing weights for {', '.join(missing[:4])}")
        return [
            WeightUnit(name=name, kind=kind, layer=layer, tensors=refs)
            for name, (kind, layer, refs) in groups.items()
        ]


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: LayerState) -> torch.Tensor:
    """Causal attention over cached and new positions.

    Shapes are ``[1, heads, tokens, head_dim]``. Grouped-query heads are
    expanded here so that the cached K and V stay at their compact size.
    """
    if "key" in state:
        k = torch.cat((state["key"], k), dim=2)
        v = torch.cat((state["value"], v), dim=2)
    state["key"], state["value"] = k, v
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    new_tokens, total = q.shape[2], k.shape[2]
    if new_tokens == 1:
        return F.scaled_dot_product_attention(q, k, v)
    if new_tokens == total:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    rows = torch.arange(total - new_tokens, total, device=q.device)[:, None]
    mask = torch.arange(total, device=q.device)[None, :] <= rows
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
