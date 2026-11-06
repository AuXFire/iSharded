"""Every adapter must match the reference ``transformers`` implementation.

Tiny random checkpoints are written with the official classes, then read back
through the real path: manifest, allowlist, selective fetch, unit cache, load.
Families newer than the installed ``transformers`` are skipped, so run this
under a current release as well to cover all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

from isharded.models import create_adapter
from isharded.weights import FetchProgress, UnitCache, build_manifest, load_unit, map_unit

COMMON = dict(
    vocab_size=96,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=96,
    max_position_embeddings=256,
    tie_word_embeddings=False,
)
MOE = dict(num_experts=6, num_experts_per_tok=2, moe_intermediate_size=32)
HYBRID = dict(
    head_dim=16,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
)

CASES = {
    "gpt2": ("GPT2Config", dict(vocab_size=96, n_embd=64, n_layer=4, n_head=4, n_positions=256)),
    "gpt2-tied": ("GPT2Config", dict(vocab_size=96, n_embd=64, n_layer=2, n_head=4, tie_word_embeddings=True)),
    "qwen2": ("Qwen2Config", COMMON),
    "qwen2-tied": ("Qwen2Config", {**COMMON, "tie_word_embeddings": True}),
    "qwen2_moe": (
        "Qwen2MoeConfig",
        {**COMMON, **MOE, "shared_expert_intermediate_size": 48, "norm_topk_prob": False},
    ),
    "qwen2_moe-partly-dense": (
        "Qwen2MoeConfig",
        {**COMMON, **MOE, "shared_expert_intermediate_size": 48, "decoder_sparse_step": 2},
    ),
    "qwen3": ("Qwen3Config", {**COMMON, "head_dim": 16}),
    "qwen3_moe": ("Qwen3MoeConfig", {**COMMON, **MOE, "head_dim": 16, "norm_topk_prob": True}),
    "qwen3_5": ("Qwen3_5TextConfig", {**COMMON, **HYBRID}),
    "qwen3_5_moe": (
        "Qwen3_5MoeTextConfig",
        {**COMMON, **MOE, **HYBRID, "shared_expert_intermediate_size": 32},
    ),
}


def build_reference(case: str, directory: Path) -> torch.nn.Module:
    config_name, options = CASES[case]
    config_class = getattr(transformers, config_name, None)
    if config_class is None:
        pytest.skip(f"transformers {transformers.__version__} has no {config_name}")
    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_config(config_class(**options), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # Norm offsets start at exactly zero, which would hide a mistake
            # in how they are applied.
            if "norm" in name and "linear_attn" not in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
    model.save_pretrained(directory, safe_serialization=True)
    return model


class Pipeline:
    """Runs a model from its cached units, the way a worker does."""

    def __init__(self, directory: Path, cache_root: Path, *, mapped: bool = False) -> None:
        manifest = build_manifest(str(directory))
        self.adapter = create_adapter(manifest.config)
        self.units = self.adapter.partition(manifest.tensors)
        cache = UnitCache(cache_root)
        cache.fetch(str(directory), "test", manifest.revision, self.units, FetchProgress("job"))
        self.weights: dict[str, dict[str, torch.Tensor]] = {}
        for unit in self.units:
            path = cache.path("test", manifest.revision, unit.name)
            loaded = map_unit(path) if mapped else load_unit(path, "cpu", None, self.adapter.keep_float32)
            self.weights[unit.name] = {
                self.adapter.local_name(name): tensor for name, tensor in loaded.tensors.items()
            }
        self.states = [{} for _ in range(self.adapter.num_layers)]
        self.length = 0

    def step(self, tokens: list[int]) -> torch.Tensor:
        ids = torch.tensor([tokens])
        positions = torch.arange(self.length, self.length + len(tokens))
        self.length += len(tokens)
        hidden = self.adapter.embed(self.weights["embed"], ids, positions[None, :])
        for layer in range(self.adapter.num_layers):
            hidden = self.adapter.run_layer(
                layer,
                self.weights[f"layer-{layer:05d}.dense"],
                self.weights.get(f"layer-{layer:05d}.experts"),
                hidden,
                self.states[layer],
                positions,
            )
        return self.adapter.logits(self.weights["head"], hidden)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mapped", [False, True], ids=["loaded", "disk-mapped"])
def test_adapter_matches_reference(case: str, mapped: bool, tmp_path: Path) -> None:
    reference = build_reference(case, tmp_path / "model")
    pipeline = Pipeline(tmp_path / "model", tmp_path / "cache", mapped=mapped)
    tokens = [5, 17, 3, 44, 9, 28, 61]

    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor([tokens]), use_cache=False).logits
        actual = pipeline.step(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)

    # Decode one token at a time from cached state, the path chat depends on.
    with torch.inference_mode():
        for token in (12, 70, 2, 33):
            tokens.append(token)
            expected = reference(input_ids=torch.tensor([tokens]), use_cache=False).logits[:, -1:]
            actual = pipeline.step([token])
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)


def test_experts_can_sit_in_a_different_pool_than_attention(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs an accelerator")
    reference = build_reference("qwen2_moe", tmp_path / "model")
    pipeline = Pipeline(tmp_path / "model", tmp_path / "cache")
    for name, weights in pipeline.weights.items():
        if not name.endswith(".experts"):
            pipeline.weights[name] = {key: value.cuda() for key, value in weights.items()}
    tokens = [5, 17, 3, 44]
    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor([tokens]), use_cache=False).logits
        actual = pipeline.step(tokens).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)


def test_vision_and_prediction_heads_are_never_fetched() -> None:
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "layer_types": ["full_attention"],
            "num_experts": 2,
            "num_experts_per_tok": 1,
        },
    }
    adapter = create_adapter(config)
    assert adapter.classify("model.visual.blocks.0.attn.qkv.weight") is None
    assert adapter.classify("mtp.layers.0.mlp.gate.weight") is None
    assert adapter.classify("model.language_model.layers.0.mlp.experts.down_proj") == ("experts", 0)
    assert adapter.classify("model.language_model.layers.0.mlp.gate.weight") == ("dense", 0)


def test_unsupported_quantization_is_refused_with_a_reason() -> None:
    for quantization in ({"quant_method": "awq"}, {"quant_method": "fp8", "weight_block_size": [64, 64]}):
        with pytest.raises(ValueError, match="BF16 or FP8 release"):
            create_adapter({"model_type": "qwen3", "quantization_config": quantization})
