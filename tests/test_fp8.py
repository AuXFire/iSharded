"""FP8 checkpoints: compact on the wire, expanded where memory allows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import transformers
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM

from isharded.models import create_adapter
from isharded.planner import Candidate, ModelProfile, plan_placement
from isharded.protocol import MemoryPool, TensorRef, WeightUnit
from isharded.weights import (
    FP8,
    FetchProgress,
    UnitCache,
    build_manifest,
    dequantize_fp8,
    load_unit,
    map_unit,
)
from test_adapters import CASES, build_reference

MIB = 1024**2
GIB = 1024**3
BLOCK = 128


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a matrix the way published FP8 checkpoints do."""
    rows, columns = weight.shape
    block_rows, block_columns = -(-rows // BLOCK), -(-columns // BLOCK)
    padded = torch.zeros(block_rows * BLOCK, block_columns * BLOCK)
    padded[:rows, :columns] = weight.float()
    blocks = padded.view(block_rows, BLOCK, block_columns, BLOCK)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12).to(torch.bfloat16)
    quantized = (blocks / scale.float()[:, None, :, None]).view_as(padded)[:rows, :columns]
    return quantized.to(FP8).contiguous(), scale


def test_expansion_applies_one_scale_per_block() -> None:
    weight = torch.zeros(256, 256, dtype=FP8).float() + 1
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    result = dequantize_fp8(weight.to(FP8), scale, torch.float32)
    assert result[0, 0] == 1 and result[0, 255] == 2 and result[255, 0] == 4 and result[255, 255] == 8


def test_expansion_handles_edges_that_do_not_fill_a_block() -> None:
    weight, scale = quantize(torch.randn(200, 150))
    expanded = dequantize_fp8(weight, scale, torch.float32)
    assert expanded.shape == (200, 150)
    assert torch.equal(expanded[:128, :128], weight[:128, :128].float() * scale[0, 0].float())
    assert torch.equal(expanded[128:, 128:], weight[128:, 128:].float() * scale[1, 1].float())


def test_quantization_error_stays_small() -> None:
    original = torch.randn(256, 512)
    weight, scale = quantize(original)
    error = (dequantize_fp8(weight, scale, torch.float32) - original).abs().mean() / original.abs().mean()
    assert error < 0.05


def quantize_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    """Rewrite a saved model as FP8. Returns the values that FP8 can represent."""
    path = directory / "model.safetensors"
    # Read from bytes: Windows cannot overwrite a file that is still mapped.
    tensors = load(path.read_bytes())
    rewritten: dict[str, torch.Tensor] = {}
    effective: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        convert = (
            tensor.dim() == 2
            and name.endswith(".weight")
            and "layers." in name
            and "norm" not in name
            and not name.endswith(("mlp.gate.weight", "shared_expert_gate.weight", "in_proj_a.weight", "in_proj_b.weight"))
        )
        if convert:
            weight, scale = quantize(tensor)
            rewritten[name] = weight
            rewritten[name + "_scale_inv"] = scale
            effective[name] = dequantize_fp8(weight, scale, torch.float32)
        else:
            rewritten[name] = tensor
    save_file(rewritten, path)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [BLOCK, BLOCK],
    }
    config_path.write_text(json.dumps(config))
    return effective


def run_fp8(directory: Path, cache_root: Path, mode: str, tokens: list[int]) -> torch.Tensor:
    manifest = build_manifest(str(directory))
    adapter = create_adapter(manifest.config)
    units = adapter.partition(manifest.tensors)
    cache = UnitCache(cache_root)
    cache.fetch(str(directory), "test", manifest.revision, units, FetchProgress("job"))
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for unit in units:
        path = cache.path("test", manifest.revision, unit.name)
        if mode == "disk":
            loaded = map_unit(path)
        else:
            loaded = load_unit(path, "cpu", torch.float32, adapter.keep_float32, compact=mode == "compact")
        weights[unit.name] = {adapter.local_name(name): tensor.clone() for name, tensor in loaded.tensors.items()}
    states = [{} for _ in range(adapter.num_layers)]
    ids = torch.tensor([tokens])
    positions = torch.arange(len(tokens))
    hidden = adapter.embed(weights["embed"], ids, positions[None, :])
    for layer in range(adapter.num_layers):
        hidden = adapter.run_layer(
            layer,
            weights[f"layer-{layer:05d}.dense"],
            weights.get(f"layer-{layer:05d}.experts"),
            hidden,
            states[layer],
            positions,
        )
    return adapter.logits(weights["head"], hidden)


@pytest.mark.parametrize("case", ["qwen2", "qwen3", "qwen3_5"])
@pytest.mark.parametrize("mode", ["expanded", "compact", "disk"])
def test_fp8_models_match_reference_in_every_form(case: str, mode: str, tmp_path: Path) -> None:
    if getattr(transformers, CASES[case][0], None) is None:
        pytest.skip(f"transformers {transformers.__version__} lacks {CASES[case][0]}")
    reference = build_reference(case, tmp_path / "model")
    effective = quantize_checkpoint(tmp_path / "model")
    state = reference.state_dict()
    with torch.no_grad():
        for name, value in effective.items():
            state[name].copy_(value)
    tokens = [5, 17, 3, 44, 9, 28]
    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor([tokens]), use_cache=False).logits
        actual = run_fp8(tmp_path / "model", tmp_path / "cache", mode, tokens)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


def test_routed_experts_run_from_compact_weights() -> None:
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
    }
    adapter = create_adapter(config)
    torch.manual_seed(0)
    compact: dict[str, torch.Tensor] = {}
    plain: dict[str, torch.Tensor] = {}
    for expert in range(3):
        for part, shape in (("gate_proj", (64, 128)), ("up_proj", (64, 128)), ("down_proj", (128, 64))):
            name = f"mlp.experts.{expert}.{part}.weight"
            weight, scale = quantize(torch.randn(shape) * 0.1)
            compact[name], compact[name + "_scale_inv"] = weight, scale
            plain[name] = dequantize_fp8(weight, scale, torch.bfloat16)
    dense = {"mlp.gate.weight": torch.randn(3, 128, dtype=torch.bfloat16)}
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    assert torch.equal(adapter._mixture(dense, compact, x), adapter._mixture(dense, plain, x))


def fp8_unit(name: str, kind: str, size: int, layer: int | None = None) -> WeightUnit:
    tensor = TensorRef(name=name, file="f", dtype="F8_E4M3", shape=[size], start=0, end=size)
    return WeightUnit(name=name, kind=kind, layer=layer, tensors=[tensor])


def bf16_unit(name: str, kind: str, size: int) -> WeightUnit:
    tensor = TensorRef(name=name, file="f", dtype="BF16", shape=[size // 2], start=0, end=size)
    return WeightUnit(name=name, kind=kind, tensors=[tensor])


def profile(layers: int = 8, size: int = 100 * MIB) -> ModelProfile:
    units = [bf16_unit("embed", "embed", MIB), bf16_unit("head", "head", MIB)]
    units += [fp8_unit(f"layer-{index:05d}.dense", "dense", size, index) for index in range(layers)]
    return ModelProfile(
        units=units,
        num_layers=layers,
        expert_active_fraction=1.0,
        kv_bytes_per_token=[0] * layers,
        state_bytes=[0] * layers,
    )


def pool(pool_id: str, kind: str, budget: int, speed: float) -> MemoryPool:
    return MemoryPool(
        pool_id=pool_id,
        kind=kind,
        device="cpu" if kind in {"ram", "disk"} else pool_id,
        capacity_group=f"host:{pool_id}",
        budget_bytes=budget,
        free_bytes=budget,
        bandwidth_gbps=speed,
        dtype="checkpoint" if kind == "disk" else "bfloat16",
    )


def plan(*pools: MemoryPool):
    worker = Candidate(worker_id="node", url="http://node", pools=list(pools))
    return plan_placement([worker], [], profile())[0].placements


def layers(placements) -> list:
    return [item for item in placements if item.unit.startswith("layer")]


def test_fp8_expands_at_load_when_memory_allows() -> None:
    placements = layers(plan(pool("cuda:0", "vram", 4 * GIB, 300)))
    assert not any(item.compact for item in placements)
    assert all(item.bytes == 200 * MIB for item in placements)


def test_fp8_that_only_fits_compact_stays_compact() -> None:
    placements = layers(plan(pool("cuda:0", "vram", 1200 * MIB, 300)))
    assert any(item.compact for item in placements)
    assert sum(item.bytes for item in placements) < 1100 * MIB


def test_fp8_prefers_compact_memory_over_disk() -> None:
    placements = layers(plan(pool("cuda:0", "vram", 1200 * MIB, 300), pool("disk", "disk", 8 * GIB, 1.5)))
    assert not any(item.pool_id == "disk" for item in placements)


def test_compact_in_vram_beats_overflowing_to_slow_memory() -> None:
    # Everything expanded needs 1.6 GiB. Compact units on the GPU avoid the CPU
    # path, where expanding them per use is by far the slowest option.
    placements = layers(plan(pool("cuda:0", "vram", 1100 * MIB, 300), pool("ram", "ram", 100 * MIB, 20)))
    compact = [item for item in placements if item.compact]
    assert compact
    assert all(item.pool_id == "cuda:0" for item in compact)


def test_disk_always_holds_fp8_as_stored() -> None:
    placements = layers(plan(pool("disk", "disk", 8 * GIB, 1.5)))
    assert all(item.compact and item.bytes == 100 * MIB for item in placements)


def test_expanding_into_slower_memory_beats_compact_in_fast_memory_on_cpu_cost() -> None:
    placements = layers(plan(pool("ram", "ram", 4 * GIB, 20)))
    assert not any(item.compact for item in placements)


def test_full_precision_pool_expands_fp8_to_four_bytes() -> None:
    fast = pool("cuda:0", "vram", 8 * GIB, 300).model_copy(update={"dtype": "float32"})
    placements = layers(plan(fast))
    assert all(item.bytes == 400 * MIB and not item.compact for item in placements)
