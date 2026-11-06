"""Model families iSharded can execute."""

from __future__ import annotations

from typing import Any

from .base import Adapter, UnsupportedModel
from .gguf import GgufAdapter
from .gpt2 import Gpt2Adapter
from .qwen import HYBRID_TYPES, QwenAdapter

QWEN_TYPES = {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"} | HYBRID_TYPES
SUPPORTED_TYPES = {"gpt2"} | QWEN_TYPES


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Lift a multimodal checkpoint's text settings to the top level."""
    text = config.get("text_config")
    if not text:
        return config
    return {**config, **text, "_multimodal_wrapper": True}


def _is_block_fp8(quantization: dict[str, Any]) -> bool:
    return (
        quantization.get("quant_method") == "fp8"
        and quantization.get("fmt", "e4m3") == "e4m3"
        and list(quantization.get("weight_block_size") or []) == [128, 128]
    )


def create_adapter(config: dict[str, Any]) -> Adapter:
    """Build the adapter for a raw ``config.json``, or explain why not."""
    flat = flatten_config(config)
    model_type = flat.get("model_type")
    if model_type == "gguf":
        return GgufAdapter(flat)
    quantization = flat.get("quantization_config")
    if quantization and not _is_block_fp8(quantization):
        method = quantization.get("quant_method", "unknown")
        raise UnsupportedModel(
            f"{method} checkpoints are not supported yet; use the BF16 or FP8 release of this model"
        )
    if model_type == "gpt2":
        return Gpt2Adapter(flat)
    if model_type in QWEN_TYPES:
        return QwenAdapter(flat)
    supported = ", ".join(sorted(SUPPORTED_TYPES))
    raise UnsupportedModel(f"model type {model_type!r} is not supported; iSharded runs: {supported}")


__all__ = ["Adapter", "GgufAdapter", "SUPPORTED_TYPES", "UnsupportedModel", "create_adapter", "flatten_config"]
