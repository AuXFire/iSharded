import asyncio
from dataclasses import replace
from itertools import combinations, permutations
import math
import random
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from isharded import cli, geometry, planner
from isharded import orchestrator as orchestrator_module
from isharded.gguf import holds_both_kinds
from isharded.orchestrator import Orchestrator
from isharded.planner import Candidate, ModelProfile, PlacementError, keep_layers, memory_report, plan_placement, stored_bytes
from isharded.protocol import LoadModelRequest, MemoryPool, TensorRef, WeightUnit

MIB = 1024**2
GIB = 1024**3


def unit(name: str, kind: str, size: int, layer: int | None = None, dtype: str = "BF16") -> WeightUnit:
    tensor = TensorRef(name=name, file="f", dtype=dtype, shape=[size // 2], start=0, end=size)
    return WeightUnit(name=name, kind=kind, layer=layer, tensors=[tensor])


def profile(
    layers: int = 8,
    dense: int = 100 * MIB,
    experts: int = 0,
    context_tokens: int = 128,
    tied: bool = False,
) -> ModelProfile:
    units = [unit("embed", "embed", 50 * MIB), unit("head", "head", 50 * MIB)]
    for index in range(layers):
        units.append(unit(f"layer-{index:05d}.dense", "dense", dense, index))
        if experts:
            units.append(unit(f"layer-{index:05d}.experts", "experts", experts, index))
    return ModelProfile(
        units=units,
        num_layers=layers,
        expert_active_fraction=0.03 if experts else 1.0,
        kv_bytes_per_token=[1024] * layers,
        state_bytes=[0] * layers,
        context_tokens=context_tokens,
        tied_embeddings=tied,
    )


def pool(
    pool_id: str,
    kind: str,
    budget: int,
    speed: float,
    group: str | None = None,
    dtype: str = "bfloat16",
) -> MemoryPool:
    return MemoryPool(
        pool_id=pool_id,
        kind=kind,
        device="cpu" if kind in {"ram", "disk"} else pool_id,
        capacity_group=group or f"host:{pool_id}",
        budget_bytes=budget,
        free_bytes=budget,
        bandwidth_gbps=speed,
        dtype="checkpoint" if kind == "disk" else dtype,
    )


def worker(name: str, *pools: MemoryPool) -> Candidate:
    return Candidate(worker_id=name, url=f"http://{name}", pools=list(pools))


def pools_used(stage) -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    for item in stage.placements:
        used.setdefault(item.pool_id, []).append(item.unit)
    return used


def test_every_layer_is_covered_once_and_contiguously() -> None:
    route = plan_placement(
        [worker("gpu", pool("cuda:0", "vram", 2 * GIB, 300)), worker("cpu", pool("ram", "ram", 2 * GIB, 30))],
        [],
        profile(),
    )
    assert route[0].start_layer == 0
    assert route[-1].end_layer == 8
    assert all(left.end_layer == right.start_layer for left, right in zip(route, route[1:]))
    assert all(stage.end_layer > stage.start_layer for stage in route)


def test_a_measured_zero_is_full_and_no_measurement_keeps_the_budget() -> None:
    def ram(**fields: int) -> MemoryPool:
        return MemoryPool(pool_id="ram", kind="ram", device="cpu", capacity_group="g", budget_bytes=GIB, **fields)

    assert planner._weight_room(ram(total_bytes=2 * GIB, free_bytes=0), 0) == 0
    assert planner._weight_room(ram(), 0) == GIB
    assert planner._weight_room(ram(total_bytes=2 * GIB, free_bytes=MIB), 0) == MIB
    assert planner._weight_room(ram(total_bytes=2 * GIB, free_bytes=4 * GIB), 0) == GIB


def test_a_card_reported_full_is_passed_over() -> None:
    card = pool("cuda:0", "vram", 2 * GIB, 300).model_copy(update={"total_bytes": 2 * GIB})
    hardware = SimpleNamespace(memory_pools=[card])
    record = SimpleNamespace(health={"memory_pools": [{"pool_id": "cuda:0", "free_bytes": 0}]}, registration=SimpleNamespace(hardware=hardware))
    full = Candidate(worker_id="full", url="http://full", pools=Orchestrator._live_pools(record))
    route = plan_placement([full, worker("spare", pool("cuda:1", "vram", 2 * GIB, 100))], [], profile())
    assert [stage.worker_id for stage in route] == ["spare"]


def test_a_stage_publishes_what_it_reads_per_token() -> None:
    route = plan_placement([worker("gpu", pool("cuda:0", "vram", 8 * GIB, 300))], [], profile(experts=200 * MIB))
    assert route[0].model_dump()["read_bytes"] == {"dense": 800 * MIB, "experts": round(1600 * MIB * 0.03), "embedding": 50 * MIB, "head": 50 * MIB}


def test_read_bytes_follow_the_stages_and_the_size_a_pool_holds() -> None:
    route = plan_placement([worker("a", pool("cuda:0", "vram", GIB, 300)), worker("b", pool("cuda:1", "vram", GIB, 300))], [], profile())
    first, last = route[0].read_bytes, route[-1].read_bytes
    assert len(route) == 2 and first.dense + last.dense == 800 * MIB
    assert (first.embedding, first.head, last.embedding, last.head) == (50 * MIB, 0, 0, 50 * MIB)

    wide = plan_placement([worker("gpu", pool("cuda:0", "vram", 8 * GIB, 300, dtype="float32"))], [], profile())
    assert wide[0].read_bytes.dense == 1600 * MIB


def test_a_tied_head_is_counted_once() -> None:
    route = plan_placement([worker("gpu", pool("cuda:0", "vram", 8 * GIB, 300))], [], profile(tied=True))
    assert (route[0].read_bytes.embedding, route[0].read_bytes.head) == (0, 50 * MIB)


def test_spread_gives_faster_workers_more_layers() -> None:
    route = plan_placement(
        [worker("gpu", pool("cuda:0", "vram", 2 * GIB, 300)), worker("cpu", pool("ram", "ram", 2 * GIB, 30))],
        [],
        profile(),
    )
    layers = {stage.worker_id: stage.end_layer - stage.start_layer for stage in route}
    assert layers["gpu"] > layers["cpu"]


def test_spread_minimizes_total_time_rather_than_the_slowest_stage() -> None:
    # A single stream crosses stages in sequence, so a stage that is slower than
    # its neighbor is fine if the total comes out lower.
    route = plan_placement(
        [worker("gpu", pool("cuda:0", "vram", 2 * GIB, 300)), worker("cpu", pool("ram", "ram", 2 * GIB, 30))],
        [],
        profile(),
    )
    layers = {stage.worker_id: stage.end_layer - stage.start_layer for stage in route}
    assert layers == {"gpu": 7, "cpu": 1}


def test_equal_workers_share_layers_evenly() -> None:
    workers = [worker(name, pool("cuda:0", "vram", GIB, 300, name)) for name in ("a", "b")]
    route = plan_placement(workers, [], profile())
    assert [stage.end_layer - stage.start_layer for stage in route] == [4, 4]


def big_head_profile() -> ModelProfile:
    model = profile(layers=4, dense=20 * MIB)
    head = unit("head", "head", 800 * MIB)
    model.units = [head if item.name == "head" else item for item in model.units]
    model.head = head
    return model


def test_the_expensive_output_head_lands_on_the_fast_machine() -> None:
    fast = worker("fast", pool("cuda:0", "vram", 2 * GIB, 300))
    slow = worker("slow", pool("ram", "ram", 2 * GIB, 10))
    route = plan_placement([fast, slow], [], big_head_profile())
    assert route[0].worker_id == "slow"
    assert route[-1].worker_id == "fast"


def test_direction_does_not_depend_on_the_order_workers_arrive() -> None:
    fast = worker("fast", pool("cuda:0", "vram", 2 * GIB, 300))
    slow = worker("slow", pool("ram", "ram", 2 * GIB, 10))
    forward = plan_placement([fast, slow], [], big_head_profile())
    backward = plan_placement([slow, fast], [], big_head_profile())
    assert [stage.worker_id for stage in forward] == [stage.worker_id for stage in backward]


def test_ends_can_be_pinned_against_the_planners_preference() -> None:
    fast = worker("fast", pool("cuda:0", "vram", 2 * GIB, 300))
    slow = worker("slow", pool("ram", "ram", 2 * GIB, 10))
    route = plan_placement([fast, slow], [], big_head_profile(), entry_worker="fast", exit_worker="slow")
    assert [stage.worker_id for stage in route] == ["fast", "slow"]


def test_pinning_one_end_leaves_the_other_to_the_planner() -> None:
    workers = [worker(name, pool("cuda:0", "vram", GIB, speed, name)) for name, speed in (("a", 300), ("b", 200), ("c", 100))]
    route = plan_placement(workers, [], profile(), exit_worker="c")
    assert route[-1].worker_id == "c"
    assert len(route) == 3


def test_one_worker_pinned_to_both_ends_runs_the_whole_model() -> None:
    workers = [worker(name, pool("cuda:0", "vram", 2 * GIB, 300, name)) for name in ("a", "b")]
    route = plan_placement(workers, [], profile(), entry_worker="b", exit_worker="b")
    assert [stage.worker_id for stage in route] == ["b"]


def test_compact_placement_keeps_pinned_workers() -> None:
    workers = [worker(name, pool("cuda:0", "vram", 4 * GIB, 300, name)) for name in ("a", "b")]
    route = plan_placement(workers, [], profile(), spread=False, entry_worker="a", exit_worker="b")
    assert [stage.worker_id for stage in route] == ["a", "b"]


def test_a_standby_worker_can_be_pinned() -> None:
    active = worker("active", pool("cuda:0", "vram", 2 * GIB, 300))
    spare = worker("spare", pool("ram", "ram", 2 * GIB, 30))
    route = plan_placement([active], [spare], profile(), exit_worker="spare")
    assert route[-1].worker_id == "spare"


def test_pinning_an_unknown_worker_names_it() -> None:
    with pytest.raises(PlacementError, match="ghost"):
        plan_placement([worker("a", pool("cuda:0", "vram", GIB, 300))], [], profile(), entry_worker="ghost")


def test_many_workers_still_plan_quickly_and_cover_every_layer() -> None:
    workers = [worker(str(index), pool("cuda:0", "vram", GIB, 100 + index, str(index))) for index in range(8)]
    route = plan_placement(workers, [], profile(layers=16))
    assert route[0].start_layer == 0
    assert route[-1].end_layer == 16
    assert len(route) == 8


def test_there_are_never_more_stages_than_layers() -> None:
    workers = [worker(str(index), pool("cuda:0", "vram", GIB, 300, f"h{index}")) for index in range(10)]
    assert len(plan_placement(workers, [], profile(layers=2))) == 2


def test_compact_uses_the_fewest_stages_that_fit() -> None:
    workers = [worker(name, pool("cuda:0", "vram", 4 * GIB, 300, name)) for name in ("a", "b")]
    assert len(plan_placement(workers, [], profile(), spread=False)) == 1


def test_one_node_splits_layers_across_its_own_memory_types() -> None:
    route = plan_placement(
        [
            worker(
                "laptop",
                pool("cuda:0", "vram", 300 * MIB, 300),
                pool("ram", "ram", 300 * MIB, 30),
                pool("disk", "disk", 4 * GIB, 1.5),
            )
        ],
        [],
        profile(),
    )
    assert len(route) == 1
    assert set(pools_used(route[0])) == {"cuda:0", "ram", "disk"}


def test_nodes_can_offer_entirely_different_memory() -> None:
    route = plan_placement(
        [
            worker("mac", pool("mps", "unified", 700 * MIB, 200)),
            worker("server", pool("ram", "ram", 700 * MIB, 40)),
            worker("box", pool("cuda:0", "vram", 300 * MIB, 500), pool("disk", "disk", GIB, 2)),
        ],
        [],
        profile(),
    )
    assert {stage.worker_id for stage in route} == {"mac", "server", "box"}


def test_experts_leave_fast_memory_before_attention_does() -> None:
    route = plan_placement(
        [worker("node", pool("cuda:0", "vram", 1400 * MIB, 300), pool("ram", "ram", 8 * GIB, 30))],
        [],
        profile(layers=8, dense=100 * MIB, experts=500 * MIB),
    )
    used = pools_used(route[0])
    assert all(name.endswith(".dense") for name in used["cuda:0"] if name.startswith("layer"))
    assert sum(name.endswith(".dense") for name in used["cuda:0"]) == 8
    assert any(name.endswith(".experts") for name in used["ram"])


def test_cold_embedding_is_first_to_leave_fast_memory() -> None:
    route = plan_placement(
        [worker("node", pool("cuda:0", "vram", 290 * MIB, 300), pool("disk", "disk", GIB, 1.5))],
        [],
        profile(layers=2),
    )
    used = pools_used(route[0])
    assert used["disk"] == ["embed"]
    assert "head" in used["cuda:0"]


def test_workers_on_one_host_cannot_spend_the_same_ram_twice() -> None:
    shared = "desktop:ram"
    workers = [
        worker("rocm", pool("ram", "ram", GIB, 30, shared)),
        worker("cpu", pool("ram", "ram", GIB, 30, shared)),
    ]
    with pytest.raises(PlacementError):
        plan_placement(workers, [], profile(layers=8, dense=200 * MIB))


def test_unified_memory_and_ram_draw_on_one_budget() -> None:
    shared = "mac:ram"
    candidate = worker("mac", pool("mps", "unified", GIB, 200, shared), pool("ram", "ram", GIB, 60, shared))
    with pytest.raises(PlacementError):
        plan_placement([candidate], [], profile(layers=8, dense=200 * MIB))


def test_full_precision_pool_holds_half_as_many_layers() -> None:
    bf16 = worker("bf16", pool("cuda:0", "vram", 900 * MIB, 300, "a"))
    fp32 = worker("fp32", pool("cuda:0", "vram", 900 * MIB, 300, "b", dtype="float32"))
    model = profile(layers=8)
    assert stored_bytes(model.dense[0], fp32.pools[0]) == 2 * stored_bytes(model.dense[0], bf16.pools[0])

    route = plan_placement([bf16, fp32], [], model)
    layers = {stage.worker_id: stage.end_layer - stage.start_layer for stage in route}
    assert layers["bf16"] > layers["fp32"]


def test_disk_tier_keeps_the_checkpoint_dtype_even_on_a_full_precision_node() -> None:
    model = profile()
    disk = pool("disk", "disk", GIB, 1.5)
    assert stored_bytes(model.dense[0], disk) == model.dense[0].nbytes


def test_float32_checkpoints_shrink_in_a_bf16_pool() -> None:
    weights = unit("layer-00000.dense", "dense", 400, 0, dtype="F32")
    assert stored_bytes(weights, pool("cuda:0", "vram", GIB, 300)) == 200


def test_standby_is_only_used_when_the_model_needs_it() -> None:
    small = worker("gpu", pool("cuda:0", "vram", 2 * GIB, 300))
    spare = worker("cpu", pool("ram", "ram", 8 * GIB, 30))
    assert {stage.worker_id for stage in plan_placement([small], [spare], profile())} == {"gpu"}

    big = profile(layers=8, dense=500 * MIB)
    assert "cpu" in {stage.worker_id for stage in plan_placement([small], [spare], big)}


def test_tiny_worker_does_not_block_a_plan_that_fits_elsewhere() -> None:
    route = plan_placement(
        [worker("big", pool("cuda:0", "vram", 4 * GIB, 300)), worker("tiny", pool("ram", "ram", MIB, 30))],
        [],
        profile(),
    )
    assert [stage.worker_id for stage in route] == ["big"]


def test_context_state_counts_against_memory() -> None:
    candidate = worker("gpu", pool("cuda:0", "vram", 1100 * MIB, 300))
    plan_placement([candidate], [], profile(context_tokens=128))
    with pytest.raises(PlacementError, match="context"):
        plan_placement([candidate], [], profile(context_tokens=200_000))


def test_tied_embedding_is_stored_once_when_one_worker_holds_both_ends() -> None:
    route = plan_placement([worker("gpu", pool("cuda:0", "vram", 2 * GIB, 300))], [], profile(tied=True))
    placements = {item.unit: item for item in route[0].placements}
    assert placements["embed"].bytes == 0
    assert placements["embed"].pool_id == placements["head"].pool_id


def test_a_model_that_cannot_fit_is_explained() -> None:
    with pytest.raises(PlacementError, match="GiB"):
        plan_placement([worker("gpu", pool("cuda:0", "vram", 100 * MIB, 300))], [], profile())


# A model that can only be cut at coarse boundaries. These are the real sizes of
# Qwen3.5-397B-A17B UD-Q3_K_XL: 60 layers of 2.72 to 3.73 GiB, 164.7 GiB in all,
# and 1.8 GiB of embedding and output, cut only at every fourth layer. How a
# layer divides between dense weights and experts is an estimate from its shape.
LAYER_GIB = [3.73, *[2.72] * 58, 3.21]
DENSE_GIB = 0.15
EXPERT_BYTES = int((2.72 - DENSE_GIB) * GIB)
# What the planner holds back on a card for a GGUF stage.
CARD_RESERVE_GIB = 0.75
STAGE_WORKSPACE = 768 * MIB


def raw_unit(name: str, kind: str, gib: float, layer: int | None = None) -> WeightUnit:
    size = int(gib * GIB)
    tensor = TensorRef(name=name, file="f", dtype="U8", shape=[size], start=0, end=size)
    return WeightUnit(name=name, kind=kind, layer=layer, tensors=[tensor])


def block_profile(sizes: list[float] | None = None) -> ModelProfile:
    sizes = LAYER_GIB if sizes is None else sizes
    units = [raw_unit("embed", "embed", 0.75), raw_unit("head", "head", 1.05)]
    for index, gib in enumerate(sizes):
        units.append(raw_unit(f"layer-{index:05d}.dense", "dense", DENSE_GIB, index))
        units.append(raw_unit(f"layer-{index:05d}.experts", "experts", gib - DENSE_GIB, index))
    attention = [(index + 1) % 4 == 0 for index in range(len(sizes))]
    return ModelProfile(
        units=units,
        num_layers=len(sizes),
        expert_active_fraction=10 / 512,
        kv_bytes_per_token=[2048 if layer else 0 for layer in attention],
        state_bytes=[0 if layer else 4 * MIB for layer in attention],
        layer_step=4,
        stage_workspace_bytes=STAGE_WORKSPACE,
    )


def card(name: str, weights_gib: float, speed: float, ram_gib: float = 0) -> Candidate:
    """A card with this much left for weights, once the planner has held its part back."""
    pools = [pool("cuda:0", "vram", int((weights_gib + CARD_RESERVE_GIB) * GIB), speed, f"{name}:vram")]
    if ram_gib:
        pools.append(pool("ram", "ram", int(ram_gib * GIB), 40, f"{name}:ram"))
    return worker(name, *pools)


def six_cards(ram_gib: float = 0) -> list[Candidate]:
    return [card(name, weights, 800, ram_gib) for name, weights in zip("abcdef", (44, 44, 29, 29, 21, 21))]


def held_in(stage, pool_id: str) -> int:
    return sum(item.bytes for item in stage.placements if item.pool_id == pool_id)


def spill_of(route) -> dict[str, int]:
    return {stage.worker_id: held_in(stage, "ram") for stage in route if held_in(stage, "ram")}


def on_block_boundaries(route, layers: int = 60) -> bool:
    joined = all(left.end_layer == right.start_layer for left, right in zip(route, route[1:]))
    aligned = all(stage.start_layer % 4 == 0 for stage in route)
    return joined and aligned and route[0].start_layer == 0 and route[-1].end_layer == layers


def test_cards_alone_cannot_hold_the_397b_model_and_the_error_says_why() -> None:
    # 188 GiB is offered for 166.5 GiB, but 4+4+2+2+1+1 is 14 of the 15 blocks. The
    # planner knows the rule and refuses, so plan_stage never sees a stage it rejects.
    with pytest.raises(PlacementError, match="start on a multiple of 4 layers.*also offers RAM"):
        plan_placement(six_cards(), [], block_profile())


def test_the_397b_model_plans_with_one_layers_experts_in_ram() -> None:
    route = plan_placement(six_cards(ram_gib=8), [], block_profile())
    assert len(route) == 6
    assert on_block_boundaries(route)
    # The excess is under 1 GiB, and one layer's experts is the least that can move.
    assert list(spill_of(route).values()) == [EXPERT_BYTES]


def test_a_spill_moves_only_experts_and_every_layers_dense_weights_stay_on_the_card() -> None:
    for stage in plan_placement(six_cards(ram_gib=8), [], block_profile()):
        assert all(item.unit.endswith(".experts") for item in stage.placements if item.pool_id == "ram")
        on_card = {item.unit for item in stage.placements if item.pool_id == "cuda:0"}
        assert {f"layer-{layer:05d}.dense" for layer in range(stage.start_layer, stage.end_layer)} <= on_card


def test_no_card_is_filled_past_its_budget() -> None:
    workers = {candidate.worker_id: candidate for candidate in six_cards(ram_gib=8)}
    for stage in plan_placement(list(workers.values()), [], block_profile()):
        assert held_in(stage, "cuda:0") <= workers[stage.worker_id].pools[0].budget_bytes - int(CARD_RESERVE_GIB * GIB)


def test_trying_every_order_halves_the_spill_of_sampling_them(monkeypatch) -> None:
    searched = spill_of(plan_placement(six_cards(ram_gib=8), [], block_profile()))
    monkeypatch.setattr(planner, "MAX_SEARCHED_WORKERS", 0)
    sampled = spill_of(plan_placement(six_cards(ram_gib=8), [], block_profile()))
    assert sum(sampled.values()) == 2 * sum(searched.values())


def test_a_spill_goes_only_to_a_node_that_offered_ram() -> None:
    workers = six_cards()
    workers[4] = card("e", 21, 800, ram_gib=4)
    route = plan_placement(workers, [], block_profile())
    assert on_block_boundaries(route)
    assert spill_of(route) == {"e": EXPERT_BYTES}


def test_a_spill_is_refused_when_the_only_ram_is_on_a_card_it_cannot_help() -> None:
    # Only this card offers RAM, so it must take a third block. That is two layers'
    # experts over, more than the 4 GiB it offers, even though the excess is not.
    workers = six_cards()
    workers[3] = card("d", 29, 800, ram_gib=4)
    with pytest.raises(PlacementError):
        plan_placement(workers, [], block_profile())


def test_each_nodes_ram_budget_is_a_limit() -> None:
    # A gibibyte less on every card makes three layers' experts spill. At 3 GiB of
    # RAM a node holds one of them, so three nodes must offer it.
    budgets = list(zip("abcdef", (43, 43, 28, 28, 20, 20)))
    route = plan_placement([card(name, weights, 800, 3) for name, weights in budgets], [], block_profile())
    assert sorted(spill_of(route).values()) == [EXPERT_BYTES] * 3
    with pytest.raises(PlacementError):
        plan_placement([card(name, weights, 800, 3 if name in "ef" else 0) for name, weights in budgets], [], block_profile())


def test_ram_too_small_for_one_layers_experts_cannot_take_a_spill() -> None:
    with pytest.raises(PlacementError):
        plan_placement(six_cards(ram_gib=2), [], block_profile())


def test_workers_on_one_host_cannot_spill_past_the_ram_they_share() -> None:
    def sharing(ram_gib: float) -> list[Candidate]:
        workers = six_cards()
        for candidate in workers:
            candidate.pools.append(pool("ram", "ram", int(ram_gib * GIB), 40, "host:ram"))
        return workers

    assert sum(spill_of(plan_placement(sharing(18), [], block_profile())).values()) == EXPERT_BYTES
    # Planning splits shared RAM evenly, so 1 GiB each is too little for one unit.
    with pytest.raises(PlacementError):
        plan_placement(sharing(6), [], block_profile())


def test_the_smallest_expert_unit_that_covers_the_excess_is_the_one_spilled() -> None:
    # Packing alone would spill the last unit, which here is also the largest.
    sizes = [3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.5]
    route = plan_placement([card("one", sum(sizes) + 0.8, 800, ram_gib=8)], [], block_profile(sizes))
    assert [item.unit for item in route[0].placements if item.pool_id == "ram"] == ["layer-00002.experts"]


def test_two_units_spill_when_no_single_unit_covers_the_excess() -> None:
    sizes = [2.0] * 8
    route = plan_placement([card("one", sum(sizes) - 1.2, 800, ram_gib=8)], [], block_profile(sizes))
    assert sum(item.pool_id == "ram" for item in route[0].placements) == 2


def test_a_model_that_fits_spills_nothing() -> None:
    route = plan_placement([card("a", 30, 800, 8), card("b", 30, 800, 8)], [], block_profile([2.72] * 16))
    assert spill_of(route) == {}


def mixed_cards() -> tuple[list[Candidate], ModelProfile]:
    workers = [card("a", 30, 700, 16), card("b", 22, 1100, 16), card("c", 16, 900, 16), card("d", 8, 1800, 16)]
    return workers, block_profile([3.2, *[2.72] * 22, 3.0])


def layouts_of(workers: list[Candidate], model: ModelProfile) -> list[tuple[int, int, float, float]]:
    """Every way through these workers, scored as the planner scores a route."""
    packer = planner._Packer(model, planner._capacities(workers, 1.0, model.stage_workspace_bytes))
    stage = planner._stage_scorer(model, packer)
    scored = []
    for order in permutations(workers):
        for inner in combinations(range(4, model.num_layers, 4), len(workers) - 1):
            cuts = (0, *inner, model.num_layers)
            found = [stage(candidate, left, right) for candidate, left, right in zip(order, cuts, cuts[1:])]
            if None not in found:
                total = sum(item[3] + planner.BALANCE_WEIGHT * item[3] ** 2 for item in found)
                scored.append(
                    (
                        sum(item[0] for item in found),
                        sum(item[1] for item in found),
                        max(item[2] for item in found),
                        total + planner.HOP_MS * (len(found) - 1),
                    )
                )
    return scored


def test_the_search_finds_what_trying_every_layout_finds() -> None:
    workers, model = mixed_cards()
    layouts = layouts_of(workers, model)
    fullest = min(layouts, key=lambda layout: (layout[0], layout[3]))
    allowed = [layout for layout in layouts if layout[2] <= fullest[2] * (1 + planner.EVEN_STAGE_ALLOWANCE)]
    best = min(allowed)
    # Less in RAM and a faster slowest stage, for a little more time in total.
    assert best[1] < fullest[1] and best[2] < fullest[2] and best[3] > fullest[3]

    route = plan_placement(workers, [], model)
    assert sum(spill_of(route).values()) == best[1]
    assert max(item.estimated_ms_per_token for item in route) == pytest.approx(best[2], abs=0.01)


def test_evening_out_stages_is_held_to_its_allowance(monkeypatch) -> None:
    workers, model = mixed_cards()
    even = plan_placement(workers, [], model)
    monkeypatch.setattr(planner, "EVEN_STAGE_ALLOWANCE", 0.0)
    plain = plan_placement(workers, [], model)
    assert sum(item.estimated_ms_per_token for item in plain) < sum(item.estimated_ms_per_token for item in even)


def test_a_worker_with_only_ram_does_not_draw_dense_weights_off_a_gpu() -> None:
    # Counting bytes of VRAM would give the GPU fewer layers, to gain a few hundred
    # MiB, and put their dense weights on the CPU. A card within a layer of full is
    # full, so speed decides, and keeping dense weights on the GPU is far faster.
    workers = [card("gpu", 10, 800, 64), worker("cpu", pool("ram", "ram", 64 * GIB, 40, "cpu:ram"))]
    route = plan_placement(workers, [], block_profile([2.72] * 24))
    stage = next(item for item in route if item.worker_id == "gpu")
    assert stage.end_layer - stage.start_layer == 20
    assert held_in(stage, "ram") > 0


def test_of_equal_spills_the_faster_slowest_stage_wins_over_a_faster_total() -> None:
    workers = [card("a", 30, 900, 16), card("b", 30, 700, 16), card("c", 8, 1800, 16), card("d", 30, 700, 16)]
    model = block_profile([3.2, *[2.72] * 22, 3.0])
    route = plan_placement(workers, [], model)
    assert sum(spill_of(route).values()) == 2 * EXPERT_BYTES
    # Ranking equal spills by total instead gives 3.17 here, for less total time.
    assert max(stage.estimated_ms_per_token for stage in route) == pytest.approx(3.11, abs=0.01)


def test_compact_plans_use_open_vram_before_ram() -> None:
    # Two of these cards and RAM would save a hop, but the third card has room.
    workers = [card(name, 70, 800, 60) for name in "abc"]
    route = plan_placement(workers, [], block_profile(), spread=False)
    assert len(route) == 3
    assert spill_of(route) == {}


def test_less_spill_is_not_worth_downloading_shards_already_in_place(monkeypatch) -> None:
    # A running model's shards sit where an earlier plan put them. Every card is
    # within a block of full, so nothing has to move, and moving would take far
    # longer than the spill it saves.
    monkeypatch.setattr(planner, "MAX_SEARCHED_WORKERS", 0)
    workers = six_cards(ram_gib=8)
    earlier = plan_placement(workers, [], block_profile())
    assert sum(spill_of(earlier).values()) == 2 * EXPERT_BYTES
    monkeypatch.undo()

    model = block_profile()
    model.amortize_tokens = planner.AMORTIZE_TOKENS
    held = {stage.worker_id: frozenset(item.unit for item in stage.placements) for stage in earlier}
    for candidate in workers:
        candidate.cached = held[candidate.worker_id]
    route = plan_placement(workers, [], model)
    assert [(stage.worker_id, stage.start_layer) for stage in route] == [(stage.worker_id, stage.start_layer) for stage in earlier]


def test_pins_hold_through_the_search() -> None:
    route = plan_placement(six_cards(ram_gib=8), [], block_profile(), entry_worker="e", exit_worker="b")
    assert route[0].worker_id == "e"
    assert route[-1].worker_id == "b"
    assert on_block_boundaries(route)


def test_one_worker_pinned_to_both_ends_runs_a_block_model_alone() -> None:
    workers = [card("a", 200, 800), card("b", 200, 800)]
    route = plan_placement(workers, [], block_profile(), entry_worker="b", exit_worker="b")
    assert [stage.worker_id for stage in route] == ["b"]


def test_compact_block_plans_may_skip_workers() -> None:
    workers = [card(name, 150, 800, 8) for name in "abc"]
    assert len(plan_placement(workers, [], block_profile(), spread=False)) == 2


def test_more_workers_than_blocks_still_plans() -> None:
    route = plan_placement([card(str(index), 30, 800, 8) for index in range(5)], [], block_profile([2.72] * 16))
    assert on_block_boundaries(route, layers=16)


def test_past_the_search_limit_block_models_fall_back_to_sampled_orders() -> None:
    workers = [card(str(index), 30, 800 + index, 8) for index in range(planner.MAX_SEARCHED_WORKERS + 1)]
    assert on_block_boundaries(plan_placement(workers, [], block_profile()))


def test_a_block_plan_does_not_depend_on_the_order_workers_arrive() -> None:
    forward = plan_placement(six_cards(ram_gib=8), [], block_profile())
    backward = plan_placement(six_cards(ram_gib=8)[::-1], [], block_profile())
    assert [(stage.worker_id, stage.start_layer) for stage in forward] == [(stage.worker_id, stage.start_layer) for stage in backward]


def no_block_rule_case(*sizes: tuple[str, int, int]) -> tuple[list[Candidate], ModelProfile]:
    layers = 12
    units = [unit("embed", "embed", 50 * MIB), unit("head", "head", 50 * MIB)]
    for index in range(layers):
        units.append(unit(f"layer-{index:05d}.dense", "dense", 40 * MIB, index))
        units.append(unit(f"layer-{index:05d}.experts", "experts", 300 * MIB, index))
    model = ModelProfile(
        units=units,
        num_layers=layers,
        expert_active_fraction=0.03,
        kv_bytes_per_token=[1024] * layers,
        state_bytes=[0] * layers,
        context_tokens=128,
    )
    workers = [
        worker(name, pool("cuda:0", "vram", size * MIB, speed, f"{name}:vram"), pool("ram", "ram", 2 * GIB, 40, f"{name}:ram"))
        for name, size, speed in sizes
    ]
    return workers, model


def test_a_plan_that_needs_no_ram_is_the_same_as_ever(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a plan that fits in VRAM should not get this far")

    for name in ("_search_blocks", "_search_order", "_spill_less"):
        monkeypatch.setattr(planner, name, refuse)
    # These numbers came from the planner as it was before any of this.
    sizes = (("a", 1500, 700), ("b", 1500, 700), ("c", 1000, 1800), ("d", 1000, 1800), ("e", 700, 1000), ("f", 700, 1000))
    route = plan_placement(*no_block_rule_case(*sizes)[:1], [], no_block_rule_case(*sizes)[1])
    assert [(stage.worker_id, stage.start_layer, stage.end_layer) for stage in route] == [
        ("b", 0, 3),
        ("a", 3, 6),
        ("f", 6, 7),
        ("e", 7, 8),
        ("d", 8, 10),
        ("c", 10, 12),
    ]
    assert [stage.estimated_ms_per_token for stage in route] == [0.22, 0.22, 0.05, 0.05, 0.06, 0.09]


def test_the_block_search_stays_out_of_models_with_no_block_rule(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("this is only for models with a block rule")

    monkeypatch.setattr(planner, "_search_blocks", refuse)
    monkeypatch.setattr(planner, "_spill_less", refuse)
    sizes = (("a", 900, 700), ("b", 900, 700), ("c", 600, 1800), ("d", 600, 1800), ("e", 400, 1000), ("f", 400, 1000))
    workers, model = no_block_rule_case(*sizes)
    plan_placement(workers, [], model)
    with pytest.raises(PlacementError) as refused:
        plan_placement([worker("tiny", pool("cuda:0", "vram", 100 * MIB, 300))], [], model)
    assert "multiple of" not in str(refused.value)


def row_of(report: dict, name: str, kind: str) -> dict:
    return next(row for row in report["pools"] if name in row["workers"] and row["kind"] == kind)


def test_the_report_states_the_spill_and_what_each_node_left_unused() -> None:
    workers = six_cards(ram_gib=8)
    model = block_profile()
    route = plan_placement(workers, [], model)
    report = memory_report(route, workers, model)
    assert report["spilled_bytes"] == EXPERT_BYTES
    assert report["unused_bytes"] == sum(row["unused_bytes"] for row in report["pools"])
    for stage in route:
        row = row_of(report, stage.worker_id, "vram")
        assert row["offered_bytes"] == row["reserved_bytes"] + row["used_bytes"] + row["unused_bytes"]
        assert row["reserved_bytes"] == int(CARD_RESERVE_GIB * GIB)
        ram = row_of(report, stage.worker_id, "ram")
        assert ram["used_bytes"] == ram["spilled_bytes"] == held_in(stage, "ram")
        assert ram["unused_bytes"] == 8 * GIB - ram["used_bytes"]


def test_the_report_counts_context_state_as_used() -> None:
    workers = [card("one", 40, 800)]
    model = block_profile([2.72] * 8)
    route = plan_placement(workers, [], model)
    context = sum(model.context_bytes(layer) for layer in range(8))
    assert memory_report(route, workers, model)["pools"][0]["used_bytes"] == held_in(route[0], "cuda:0") + context


def test_a_worker_with_no_stage_leaves_all_its_room_unused() -> None:
    workers = [card("big", 120, 800), card("spare", 30, 800)]
    model = block_profile([2.72] * 16)
    route = plan_placement(workers, [], model, spread=False)
    assert [stage.worker_id for stage in route] == ["big"]
    row = row_of(memory_report(route, workers, model), "spare", "vram")
    assert row["used_bytes"] == 0
    assert row["unused_bytes"] == 30 * GIB


def test_memory_two_workers_share_is_counted_once() -> None:
    workers = [worker(name, pool("ram", "ram", 4 * GIB, 30, "desktop:ram")) for name in ("rocm", "cpu")]
    model = profile()
    report = memory_report(plan_placement(workers, [], model), workers, model)
    assert len(report["pools"]) == 1
    assert report["pools"][0]["workers"] == ["rocm", "cpu"]
    assert report["pools"][0]["offered_bytes"] == 4 * GIB


def test_a_tied_embedding_held_with_the_head_adds_nothing_to_the_report() -> None:
    workers = [worker("gpu", pool("cuda:0", "vram", 2 * GIB, 300))]
    model = profile(tied=True)
    route = plan_placement(workers, [], model)
    weights = sum(item.bytes for item in route[0].placements)
    context = sum(model.context_bytes(layer) for layer in range(8))
    assert memory_report(route, workers, model)["pools"][0]["used_bytes"] == weights + context


def test_ram_on_a_worker_with_nothing_faster_is_not_spill() -> None:
    workers = [worker("cpu", pool("ram", "ram", 4 * GIB, 30))]
    model = profile()
    assert memory_report(plan_placement(workers, [], model), workers, model)["spilled_bytes"] == 0


def test_disk_under_ram_is_spill() -> None:
    workers = [worker("cpu", pool("ram", "ram", 300 * MIB, 30), pool("disk", "disk", 4 * GIB, 1.5))]
    model = profile()
    assert memory_report(plan_placement(workers, [], model), workers, model)["spilled_bytes"] > 0


def test_a_stage_that_left_its_pool_is_left_out_of_the_report() -> None:
    workers = six_cards(ram_gib=8)
    model = block_profile()
    route = plan_placement(workers, [], model)
    report = memory_report(route, workers[1:], model)
    assert "a" not in {stage["worker_id"] for stage in report["stages"]}


def test_a_route_gone_stale_does_not_break_the_report() -> None:
    # Status reads this on every poll, and a route can outlive what it was planned
    # on: a worker can lose a pool, or have no pools left at all.
    workers = six_cards(ram_gib=8)
    model = block_profile()
    route = plan_placement(workers, [], model)
    workers[0].pools = []
    workers[1].pools = workers[1].pools[1:]
    route[2].placements = []
    report = memory_report(route, workers, model)
    assert len(report["stages"]) == len(route) - 1
    assert report["spilled_bytes"] >= 0


# Qwen3-235B-A22B at Q6_K, which has no block rule: 94 layers of about 1.9 GiB, 180
# GiB in all. The budgets are what a pool of seven cards really offered, of which
# the first card is the smallest, and two cards offered RAM as well. It planned as
# stages of 1, 10, 17, 20, 20, 19, and 7 layers, with 16.6 GiB in RAM and about 20
# GiB of those budgets empty.
MOE_DENSE_GIB = 0.054
MOE_EXPERT_GIB = 1.845
MOE_LAYERS = 94
MOE_EXPERT_BYTES = int(MOE_EXPERT_GIB * GIB)
MOE_LAYER_BYTES = int((MOE_DENSE_GIB + MOE_EXPERT_GIB) * GIB)
MOE_BUDGETS = {"a": 4.0, "b": 20.0, "c": 20.0, "d": 41.0, "e": 41.0, "f": 41.0, "g": 16.0}
MOE_SPEEDS = {"a": 300, "b": 900, "c": 900, "d": 650, "e": 650, "f": 650, "g": 800}
MOE_SPILLED_LAYOUT = (("a", 0, 1), ("b", 1, 11), ("c", 11, 28), ("d", 28, 48), ("f", 48, 68), ("e", 68, 87), ("g", 87, 94))


def moe_units(start: int, end: int) -> frozenset[str]:
    names = {f"layer-{index:05d}.{kind}" for index in range(start, end) for kind in ("dense", "experts")}
    return frozenset(names | ({"embed"} if start == 0 else set()) | ({"head"} if end == MOE_LAYERS else set()))


def moe_profile(context: int = 4096, slots: int = 1, amortize: int = 0) -> ModelProfile:
    units = [raw_unit("embed", "embed", 0.475), raw_unit("head", "head", 0.48)]
    for index in range(MOE_LAYERS):
        units.append(raw_unit(f"layer-{index:05d}.dense", "dense", MOE_DENSE_GIB, index))
        units.append(raw_unit(f"layer-{index:05d}.experts", "experts", MOE_EXPERT_GIB, index))
    return ModelProfile(
        units=units,
        num_layers=MOE_LAYERS,
        expert_active_fraction=8 / 128,
        kv_bytes_per_token=[2048 * slots] * MOE_LAYERS,
        state_bytes=[0] * MOE_LAYERS,
        context_tokens=context,
        stage_workspace_bytes=STAGE_WORKSPACE,
        amortize_tokens=amortize,
    )


def moe_cards(budgets: dict[str, float] = MOE_BUDGETS, speeds: dict[str, float] = MOE_SPEEDS) -> list[Candidate]:
    cards = []
    for name, budget in budgets.items():
        pools = [pool("cuda:0", "vram", int(budget * GIB), speeds[name], f"{name}:vram")]
        if name in "bc":
            pools.append(pool("ram", "ram", 64 * GIB, 60, f"{name}:ram"))
        cards.append(worker(name, *pools))
    return cards


def holding(workers: list[Candidate], layout) -> list[Candidate]:
    held = {name: moe_units(start, end) for name, start, end in layout}
    for candidate in workers:
        candidate.cached = held.get(candidate.worker_id, frozenset())
    return workers


def experts_in_ram(route) -> int:
    return sum(item.bytes for stage in route for item in stage.placements if item.pool_id == "ram" and item.unit.endswith(".experts"))


def moved_bytes(route, workers: list[Candidate], model: ModelProfile) -> int:
    held = {candidate.worker_id: candidate.cached for candidate in workers}
    return sum(
        item.unit.nbytes
        for stage in route
        for item in planner._stage_items(model, stage.start_layer, stage.end_layer)
        if item.unit.name not in held[stage.worker_id]
    )


def layout_of(route) -> list[tuple[str, int, int]]:
    return [(stage.worker_id, stage.start_layer, stage.end_layer) for stage in route]


def test_the_235b_model_fills_every_card_before_it_uses_ram() -> None:
    workers = moe_cards()
    model = moe_profile()
    route = plan_placement(workers, [], model)
    # 91 of the 94 layers fit whole, so three layers' experts is the least there is.
    assert experts_in_ram(route) == 3 * MOE_EXPERT_BYTES
    rows = memory_report(route, workers, model)["pools"]
    assert all(row["unused_bytes"] < MOE_LAYER_BYTES for row in rows if row["kind"] == "vram")


def test_only_weights_that_fit_nowhere_in_vram_go_to_ram() -> None:
    route = plan_placement(moe_cards(), [], moe_profile())
    assert {name: 0 for name in "adefg"}.items() <= {stage.worker_id: experts_in_ram([stage]) for stage in route}.items()
    assert sum(stage.end_layer - stage.start_layer for stage in route if stage.worker_id in "def") == 63


def test_longer_context_leaves_less_room_and_the_plan_says_so() -> None:
    workers = moe_cards()
    model = moe_profile(context=8192, slots=2)
    route = plan_placement(workers, [], model)
    assert experts_in_ram(route) == 8 * MOE_EXPERT_BYTES
    assert memory_report(route, workers, model)["spilled_bytes"] == 8 * MOE_EXPERT_BYTES


def test_vram_comes_before_keeping_what_nodes_hold() -> None:
    # This layout leaves a card with room for a layer while 16.6 GiB sits in RAM.
    # Holding it does not save it: moving 17 GiB is the price of using that card.
    workers = holding(moe_cards(), MOE_SPILLED_LAYOUT)
    model = moe_profile(amortize=planner.AMORTIZE_TOKENS)
    route = plan_placement(workers, [], model)
    assert experts_in_ram(route) == 4 * MOE_EXPERT_BYTES
    assert moved_bytes(route, workers, model) < 20 * GIB
    rows = memory_report(route, workers, model)["pools"]
    assert all(row["unused_bytes"] < MOE_LAYER_BYTES for row in rows if row["kind"] == "vram")


def test_the_plan_says_how_much_each_stage_spills() -> None:
    workers = moe_cards()
    model = moe_profile()
    route = plan_placement(workers, [], model)
    report = memory_report(route, workers, model)
    assert [(item["worker_id"], item["start_layer"], item["end_layer"]) for item in report["stages"]] == layout_of(route)
    assert sum(item["spilled_bytes"] for item in report["stages"]) == report["spilled_bytes"] > 0
    assert all(item["spilled_bytes"] == 0 for item in report["stages"] if item["worker_id"] in "adefg")


def test_standby_vram_comes_before_ram() -> None:
    model = block_profile([2.72] * 16)
    busy = [card("a", 30, 800, 32)]
    spare = [card("spare", 30, 800)]
    route = plan_placement(busy, spare, model)
    assert {stage.worker_id for stage in route} == {"a", "spare"}
    assert spill_of(route) == {}


def test_standby_ram_is_not_worth_promoting() -> None:
    model = block_profile([2.72] * 16)
    route = plan_placement([card("a", 30, 800, 32)], [worker("spare", pool("ram", "ram", 64 * GIB, 40))], model)
    assert [stage.worker_id for stage in route] == ["a"]


def test_vram_is_used_first_even_if_a_probe_reads_ram_as_faster() -> None:
    route = plan_placement([worker("node", pool("cuda:0", "vram", 2 * GIB, 30), pool("ram", "ram", 2 * GIB, 300))], [], profile())
    assert set(pools_used(route[0])) == {"cuda:0"}


def test_a_gguf_stage_keeps_back_one_reserve_and_not_two() -> None:
    card_pool = pool("cuda:0", "vram", 41 * GIB, 650)
    assert planner._weight_room(card_pool, STAGE_WORKSPACE) == 41 * GIB - STAGE_WORKSPACE
    # With no workspace of its own, a stage keeps the general allowance as before.
    assert planner._weight_room(card_pool, 0) == 41 * GIB - planner.WORKSPACE_BYTES
    assert planner._weight_room(pool("ram", "ram", 41 * GIB, 60), STAGE_WORKSPACE) == 41 * GIB


def test_same_pool_same_model_all_held_keeps_the_layout_and_moves_nothing() -> None:
    model = moe_profile(amortize=planner.AMORTIZE_TOKENS)
    first = plan_placement(moe_cards(), [], model)
    workers = holding(moe_cards(), layout_of(first))
    route = plan_placement(workers, [], model)
    assert layout_of(route) == layout_of(first)
    assert moved_bytes(route, workers, model) == 0


def test_a_restart_that_measures_speeds_a_little_differently_moves_nothing() -> None:
    # Bigger budgets, so nothing is in RAM and the order is all that can change.
    budgets = {"a": 4, "b": 23, "c": 23, "d": 44, "e": 44, "f": 44, "g": 20}
    model = moe_profile(amortize=planner.AMORTIZE_TOKENS)
    first = plan_placement(moe_cards(budgets), [], model)
    # Enough to reorder these workers, and it moved 112 GiB before.
    speeds = {"a": 301, "b": 893, "c": 910, "d": 655, "e": 646, "f": 657, "g": 792}
    workers = holding(moe_cards(budgets, speeds), layout_of(first))
    route = plan_placement(workers, [], model)
    assert layout_of(route) == layout_of(first)
    assert moved_bytes(route, workers, model) == 0


def test_a_block_model_held_in_full_moves_nothing() -> None:
    model = block_profile()
    model.amortize_tokens = planner.AMORTIZE_TOKENS
    first = plan_placement(six_cards(ram_gib=8), [], model)
    workers = six_cards(ram_gib=8)
    held = {stage.worker_id: frozenset(item.unit for item in stage.placements) for stage in first}
    for candidate in workers:
        candidate.cached = held[candidate.worker_id]
    route = plan_placement(workers, [], model)
    assert layout_of(route) == layout_of(first)
    assert moved_bytes(route, workers, model) == 0


def test_a_worker_that_does_not_answer_keeps_what_it_was_known_to_hold() -> None:
    runtime = Orchestrator()

    async def silent(*args, **kwargs):
        raise httpx.ConnectError("no answer")

    try:
        runtime.client.get = silent
        record = SimpleNamespace(live=True, registration=SimpleNamespace(advertised_url="http://a:1"))
        runtime.workers = {"a": record}
        target = SimpleNamespace(
            model_id="m", manifest=SimpleNamespace(revision="r"), units={}, held={"a": {"unit"}}, serving={"a"}
        )
        asyncio.run(runtime.refresh_holdings(target))
        assert target.held == {"a": {"unit"}}
        assert target.serving == {"a"}
    finally:
        runtime.workers = {}
        asyncio.run(runtime.close())


def test_sharing_a_packing_between_equal_layers_changes_nothing(monkeypatch) -> None:
    def plan() -> list[dict]:
        model = block_profile([3.2, *[2.72] * 22, 3.0])
        model.layer_step = 1
        workers = [card("a", 30, 800, 8), card("b", 16, 700, 8), card("c", 10, 900, 8)]
        return [stage.model_dump() for stage in plan_placement(workers, [], model)]

    shared = plan()
    real = planner._Packer.__init__

    def unshared(self, model, capacities) -> None:
        real(self, model, capacities)
        self.shapes = None

    monkeypatch.setattr(planner._Packer, "__init__", unshared)
    assert plan() == shared


def test_a_stage_forced_onto_ram_stays_as_small_as_it_can() -> None:
    # Spread must give the worker with only RAM a stage, however much VRAM is open.
    # That RAM is not overflow, so it is one layer and one end of the model.
    workers = [card("gpu", 30, 300), worker("cpu", pool("ram", "ram", 16 * GIB, 30))]
    route = plan_placement(workers, [], profile(layers=16, dense=500 * MIB))
    stage = next(item for item in route if item.worker_id == "cpu")
    assert stage.end_layer - stage.start_layer == 1
    assert held_in(stage, "ram") == 550 * MIB


def random_case(rng: random.Random) -> tuple[list[Candidate], list[Candidate], ModelProfile, bool, dict[str, str]]:
    layers = rng.choice((8, 12, 16, 24))
    moe = rng.random() < 0.7
    units = [raw_unit("embed", "embed", rng.choice((0.01, 0.05, 0.5))), raw_unit("head", "head", rng.choice((0.01, 0.05, 0.8)))]
    for index in range(layers):
        units.append(raw_unit(f"layer-{index:05d}.dense", "dense", rng.choice((0.02, 0.05, 0.3)), index))
        if moe:
            units.append(raw_unit(f"layer-{index:05d}.experts", "experts", rng.choice((0.1, 0.2, 0.5, 1.0)), index))
    model = ModelProfile(
        units=units,
        num_layers=layers,
        expert_active_fraction=0.03 if moe else 1.0,
        kv_bytes_per_token=[rng.choice((0, 1024))] * layers,
        state_bytes=[0] * layers,
        context_tokens=rng.choice((128, 4096)),
        layer_step=rng.choice((1, 4, 4)),
        stage_workspace_bytes=rng.choice((0, 64 * MIB)),
    )
    preferred: list[Candidate] = []
    standby: list[Candidate] = []
    for index in range(rng.randint(1, 7)):
        kinds = rng.choice((("vram",), ("vram", "ram"), ("ram",), ("vram", "ram", "disk"), ("unified",), ("ram", "disk")))
        speed = rng.choice((200, 300, 500, 900))
        pools = []
        for kind in kinds:
            # Real memory: VRAM at least as fast as this node's RAM, and disk slowest.
            rate = {"vram": speed, "unified": speed, "ram": rng.choice((30, 40, 60)), "disk": 1.5}[kind]
            pools.append(pool(kind, kind, rng.choice((300, 800, 2000, 6000, 16000)) * MIB, rate, f"{index}:{kind}"))
        (standby if rng.random() < 0.15 else preferred).append(worker(f"w{index}", *pools))
    names = [item.worker_id for item in preferred + standby]
    pins = {role: rng.choice(names) for role in ("entry_worker", "exit_worker") if rng.random() < 0.2}
    return preferred, standby, model, rng.random() < 0.7, pins


def test_no_plan_is_much_slower_than_sampled_orders_alone(monkeypatch) -> None:
    # The planner's own earlier self, for this check: orders sampled, fastest kept.
    # Using open VRAM outranks speed, so the one exception is a plan with less in RAM.
    rng = random.Random(21)
    compared = 0
    for _ in range(120):
        preferred, standby, model, spread, pins = random_case(rng)
        everyone = preferred + standby
        with monkeypatch.context() as patch:
            patch.setattr(planner, "MAX_SEARCHED_WORKERS", 0)
            patch.setattr(planner, "_fill_fast_memory", lambda fastest, *args: fastest)
            try:
                earlier = plan_placement(preferred, standby, model, spread=spread, **pins)
            except PlacementError:
                continue
        route = plan_placement(preferred, standby, model, spread=spread, **pins)
        compared += 1
        slower = planner._route_cost(route, everyone, model) > 1.1 * planner._route_cost(earlier, everyone, model) + 0.05
        assert not slower or planner._slow_bytes(route, everyone) < planner._slow_bytes(earlier, everyone)
    assert compared > 60


def test_a_spread_no_sampled_order_reaches_still_skips_workers() -> None:
    # The embedding fits only the fourth card by speed, and sampled orders never
    # start there. Searching every order finds a spread that does, but it forces a
    # stage on every card, which made plans up to 12.7 times slower on random pools.
    model = block_profile([1.0] * 32)
    embedding = raw_unit("embed", "embed", 10)
    model.units = [embedding if item.name == "embed" else item for item in model.units]
    model.embed = embedding
    speeds = zip("abcdef", (900, 800, 700, 600, 500, 400))
    route = plan_placement([card(name, 24 if name == "d" else 9, speed) for name, speed in speeds], [], model)
    assert route[0].worker_id == "d"
    assert len(route) < 6


def test_less_ram_may_not_cost_the_slowest_stage_more_than_the_allowance() -> None:
    workers = [card("a", 22, 1800, 24), card("b", 16, 700, 24), card("c", 8, 500, 24)]
    route = plan_placement(workers, [], block_profile([3.2, *[2.72] * 22, 3.0]))
    # 2.6 GiB less in RAM is on offer, but its slowest stage would take 10.9 ms.
    assert max(stage.estimated_ms_per_token for stage in route) == pytest.approx(8.02, abs=0.01)


def test_a_plan_is_not_swapped_for_one_that_gains_nothing() -> None:
    # The small card must take a block and spill part of it. Another plan leaves
    # less VRAM open, but no less in RAM and no faster at its slowest, so it loses.
    dense = [0.05, 0.3, 0.05, 0.3, 0.02, 0.3, 0.05, 0.05, 0.05, 0.3, 0.05, 0.02, 0.3, 0.05, 0.05, 0.02]
    experts = [0.5, 0.5, 1.0, 0.5, 0.2, 0.5, 0.2, 0.1, 0.2, 1.0, 0.2, 0.5, 0.1, 0.1, 0.2, 0.1]
    units = [raw_unit("embed", "embed", 0.5), raw_unit("head", "head", 0.01)]
    for index, (attention, routed) in enumerate(zip(dense, experts)):
        units.append(raw_unit(f"layer-{index:05d}.dense", "dense", attention, index))
        units.append(raw_unit(f"layer-{index:05d}.experts", "experts", routed, index))
    model = ModelProfile(
        units=units,
        num_layers=16,
        expert_active_fraction=0.03,
        kv_bytes_per_token=[0] * 16,
        state_bytes=[0] * 16,
        layer_step=4,
    )
    workers = [
        worker("a", pool("cuda:0", "vram", 800 * MIB, 300, "a:vram"), pool("ram", "ram", 16000 * MIB, 60, "a:ram")),
        worker("b", pool("cuda:0", "vram", 16000 * MIB, 500, "b:vram")),
        worker("c", pool("cuda:0", "vram", 16000 * MIB, 900, "c:vram")),
    ]
    assert layout_of(plan_placement(workers, [], model)) == [("c", 0, 8), ("b", 8, 12), ("a", 12, 16)]
def ram_worker(name: str, budget: int = 2 * GIB, speed: float = 30) -> Candidate:
    return worker(name, pool(f"{name}-ram", "ram", budget, speed))


def layout(route) -> list[tuple[str, int, int]]:
    return [(stage.worker_id, stage.start_layer, stage.end_layer) for stage in route]


def running(*workers: Candidate):
    return plan_placement(list(workers), [], profile())


def test_a_spare_takes_exactly_what_was_lost() -> None:
    a, b, c = ram_worker("a"), ram_worker("b"), ram_worker("c")
    current = running(a, b)
    route = keep_layers(current, [a, c], [], profile())
    assert [item[1:] for item in layout(route)] == [item[1:] for item in layout(current)]
    assert [stage.worker_id for stage in route] == [stage.worker_id.replace("b", "c") for stage in current]


def test_stages_that_lost_nothing_are_left_exactly_as_they_were() -> None:
    a, b, c = ram_worker("a"), ram_worker("b"), ram_worker("c")
    current = running(a, b)
    route = keep_layers(current, [a, c], [], profile())
    kept = next(stage for stage in current if stage.worker_id == "a")
    assert next(stage for stage in route if stage.worker_id == "a").placements == kept.placements


def test_a_lost_stage_at_either_end_keeps_its_place() -> None:
    a, b, c, d = ram_worker("a"), ram_worker("b"), ram_worker("c"), ram_worker("d")
    current = running(a, b, c)
    for lost in (current[0].worker_id, current[-1].worker_id):
        left = [item for item in (a, b, c) if item.worker_id != lost]
        route = keep_layers(current, [*left, d], [], profile())
        assert [stage.worker_id for stage in route] == [stage.worker_id.replace(lost, "d") for stage in current]


def test_several_stages_can_be_replaced_at_once() -> None:
    a, b, c, d, e = (ram_worker(name) for name in "abcde")
    current = running(a, b, c)
    route = keep_layers(current, [current_worker for current_worker in (a, d, e) if current_worker], [], profile())
    assert [item[1:] for item in layout(route)] == [item[1:] for item in layout(current)]
    assert len({stage.worker_id for stage in route}) == 3


def test_with_no_spare_there_is_nothing_to_keep() -> None:
    a, b = ram_worker("a"), ram_worker("b")
    assert keep_layers(running(a, b), [a], [], profile()) is None


def test_a_spare_too_small_for_the_layers_is_passed_over() -> None:
    a, b = ram_worker("a"), ram_worker("b")
    current = running(a, b)
    small, big = ram_worker("small", 50 * MIB), ram_worker("big")
    route = keep_layers(current, [a, small, big], [], profile())
    assert {stage.worker_id for stage in route} == {"a", "big"}
    assert keep_layers(current, [a, small], [], profile()) is None


def test_a_worker_already_running_a_stage_is_not_a_spare() -> None:
    a, b, c = ram_worker("a"), ram_worker("b"), ram_worker("c")
    assert keep_layers(running(a, b, c), [a, c], [], profile()) is None


def test_keeping_layers_works_for_a_model_with_tied_embeddings() -> None:
    tied = profile(tied=True)
    a, b, c = ram_worker("a"), ram_worker("b"), ram_worker("c")
    current = plan_placement([a, b], [], tied)
    route = keep_layers(current, [a, c], [], tied)
    assert [item[1:] for item in layout(route)] == [item[1:] for item in layout(current)]


# Round trips in ms measured across a real mesh. The worker names stand for places,
# and "e1" and "e2" share a site. The link from "a" to "b" failed.
LINK_TABLE = {("a", "c"): 112, ("a", "e1"): 104, ("a", "e2"): 104, ("b", "c"): 111, ("b", "e1"): 96, ("b", "e2"): 96}
LINK_TABLE |= {("c", "e1"): 42, ("c", "e2"): 42, ("e1", "e2"): 1, ("a", "b"): planner.UNREACHABLE_LINK_MS}
LINKS = {**LINK_TABLE, **{(target, source): ms for (source, target), ms in LINK_TABLE.items()}}
# In speed order this would zigzag: a card's speed says nothing about where it is.
MESH_SPEEDS = {"a": 900, "b": 850, "c": 800, "e1": 750, "e2": 700}


def mesh_cards(room_gib: float = 40, ram_gib: float = 0) -> list[Candidate]:
    return [card(name, room_gib, speed, ram_gib) for name, speed in MESH_SPEEDS.items()]


def hops_of(route) -> float:
    return sum(LINKS[left.worker_id, right.worker_id] for left, right in zip(route, route[1:]))


def order_is_good(route) -> bool:
    names = [stage.worker_id for stage in route]
    neighbours = set(zip(names, names[1:])) | set(zip(names[1:], names))
    return ("a", "b") not in neighbours and ("e1", "e2") in neighbours


@pytest.mark.parametrize("model", [block_profile([2.72] * 32), moe_profile()], ids=["block rule", "no block rule"])
def test_a_route_follows_measured_links_and_not_card_speed(model) -> None:
    route = plan_placement(mesh_cards(), [], model, links=LINKS)
    assert len(route) == 5
    assert order_is_good(route)
    assert hops_of(route) == 251  # a, c, e, e, b, or the same backwards: the shortest there is


def test_a_link_nobody_measured_costs_the_constant() -> None:
    assert planner._hop_ms(worker("x"), worker("y")) == planner.HOP_MS
    assert planner.link_ms({}, "x", "y") == planner.HOP_MS
    assert planner.link_ms({("y", "x"): 7.0}, "x", "y") == 7.0


def test_no_links_plan_as_a_table_of_nothing_does() -> None:
    assert layout_of(plan_placement(mesh_cards(), [], moe_profile())) == layout_of(
        plan_placement(mesh_cards(), [], moe_profile(), links={})
    )


def test_a_route_costs_its_real_hops() -> None:
    model = block_profile([2.72] * 32)
    workers = [
        replace(item, links={target: ms for (source, target), ms in LINKS.items() if source == item.worker_id})
        for item in mesh_cards()
    ]
    route = plan_placement(workers, [], model)
    reads = sum(stage.estimated_ms_per_token for stage in route)
    assert planner._route_cost(route, workers, model) == pytest.approx(reads + 251)


def test_a_spread_gives_up_a_worker_rather_than_use_a_dead_link() -> None:
    links = {("a", "b"): planner.UNREACHABLE_LINK_MS, ("b", "a"): planner.UNREACHABLE_LINK_MS}
    route = plan_placement([card("a", 80, 900), card("b", 80, 850)], [], block_profile([2.72] * 16), links=links)
    assert len(route) == 1


def test_a_dead_link_outranks_open_vram() -> None:
    # The card with room is behind a dead link, so this stays on one card and RAM.
    links = {("a", "b"): planner.UNREACHABLE_LINK_MS, ("b", "a"): planner.UNREACHABLE_LINK_MS}
    workers = [card("a", 30, 900, 64), card("b", 20, 850)]
    route = plan_placement(workers, [], block_profile([2.72] * 16), links=links, spread=False)
    assert [stage.worker_id for stage in route] == ["a"]


def test_vram_still_comes_before_shorter_hops() -> None:
    # Every card is needed to keep this model out of RAM, however long the hops.
    route = plan_placement(mesh_cards(room_gib=36.5, ram_gib=64), [], moe_profile(), links=LINKS)
    assert len(route) == 5
    assert experts_in_ram(route) == 0
    assert order_is_good(route)


def test_equal_spill_goes_to_the_shortest_hops() -> None:
    route = plan_placement(mesh_cards(room_gib=30, ram_gib=64), [], moe_profile(), links=LINKS)
    assert order_is_good(route)
    assert hops_of(route) == 251


def test_a_held_layout_stays_unless_moving_it_earns_its_download() -> None:
    # Not the shortest order, but all of it is held, and moving most of 180 GiB takes
    # far longer than these hops save over the tokens it is spread across.
    model = moe_profile(amortize=planner.AMORTIZE_TOKENS)
    layout = [("a", 0, 19), ("e1", 19, 38), ("c", 38, 57), ("e2", 57, 76), ("b", 76, 94)]
    workers = holding(mesh_cards(), layout)
    route = plan_placement(workers, [], model, links=LINKS)
    assert layout_of(route) == layout
    assert moved_bytes(route, workers, model) == 0


def test_pins_hold_with_links() -> None:
    route = plan_placement(mesh_cards(), [], moe_profile(), links=LINKS, entry_worker="e1", exit_worker="c")
    assert route[0].worker_id == "e1"
    assert route[-1].worker_id == "c"


def test_a_hybrid_model_can_be_cut_off_its_blocks_but_each_stage_holds_both_kinds() -> None:
    from isharded.gguf import holds_both_kinds

    model = profile(layers=16)
    model.attention_interval = 4
    # 16 layers of 100 MiB. Blocks of four would strand a card with room for 7 layers.
    workers = [worker(name, pool(f"{name}-gpu", "vram", int(size * MIB), 500)) for name, size in (("a", 1000), ("b", 800), ("c", 300))]
    route = plan_placement(workers, [], model)
    assert [stage.start_layer for stage in route][0] == 0 and route[-1].end_layer == 16
    assert all(holds_both_kinds(stage.start_layer, stage.end_layer, 4) for stage in route)
    assert any(stage.start_layer % 4 for stage in route)


def stages_on(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(worker_id=name) for name in names]


def names_of(route) -> list[str]:
    return [stage.worker_id for stage in route]


def touching(route) -> set[tuple[str, str]]:
    """Which workers sit next to each other around a loop, in both directions."""
    names = names_of(route)
    pairs = set(zip(names, names[1:] + names[:1]))
    return pairs | {(right, left) for left, right in pairs}


def test_a_loop_costs_one_way_around_and_a_nested_route_every_round_trip() -> None:
    route = stages_on("a", "c", "e1")
    assert planner.route_network_ms(route, LINKS) == 112 + 42
    assert planner.route_network_ms(route, LINKS, ring=True) == (112 + 42 + 104) / 2 + 3 * planner.RING_HOP_MS


def test_one_stage_has_no_network_time_either_way() -> None:
    assert planner.route_network_ms(stages_on("a"), LINKS) == 0
    assert planner.route_network_ms(stages_on("a"), LINKS, ring=True) == 0


def test_a_loop_falls_back_on_the_constant_for_an_unmeasured_hop() -> None:
    assert planner.route_network_ms(stages_on("x", "y"), {}, ring=True) == planner.HOP_MS + 2 * planner.RING_HOP_MS


@pytest.mark.parametrize("model", [block_profile([2.72] * 32), moe_profile()], ids=["block rule", "no block rule"])
def test_a_ring_route_is_the_shortest_loop_through_the_real_table(model) -> None:
    route = plan_placement(mesh_cards(), [], model, links=LINKS, ring=True)
    assert len(route) == 5
    assert ("a", "b") not in touching(route)  # a loop can avoid that link, closing hop included
    assert planner.route_network_ms(route, LINKS, ring=True) == 424 / 2 + 5 * planner.RING_HOP_MS


def test_with_ring_off_the_same_table_still_costs_round_trips() -> None:
    route = plan_placement(mesh_cards(), [], moe_profile(), links=LINKS, ring=False)
    assert hops_of(route) == 251


def test_ring_comes_from_the_profile_unless_it_is_given() -> None:
    model = moe_profile()
    model.ring = True
    route = plan_placement(mesh_cards(), [], model, links=LINKS)
    assert ("a", "b") not in touching(route)


def test_a_dead_closing_link_is_survivable_and_ranks_below_a_dead_forward_link() -> None:
    # Both cards are needed. One way round has a dead forward link and the other only
    # a dead closing one, whose answer walks back along the route.
    links = {("a", "b"): 50, ("b", "a"): planner.UNREACHABLE_LINK_MS}
    workers = [card("a", 30, 850), card("b", 30, 900)]
    route = plan_placement(workers, [], block_profile([2.72] * 16), links=links, ring=True)
    assert names_of(route) == ["a", "b"]
    assert planner.route_network_ms(route, links, ring=True) == 2 * (25 + planner.RING_HOP_MS)


def test_a_dead_forward_link_still_outranks_everything_on_a_ring() -> None:
    links = {("a", "b"): planner.UNREACHABLE_LINK_MS, ("b", "a"): planner.UNREACHABLE_LINK_MS}
    workers = [card("a", 30, 900, 64), card("b", 20, 850)]
    assert names_of(plan_placement(workers, [], block_profile([2.72] * 16), links=links, ring=True, spread=False)) == ["a"]


def test_auto_gives_a_model_one_stage_if_one_node_holds_it() -> None:
    workers = [card("big", 300, 700), card("c", 41, 900), card("d", 41, 900)]
    route = plan_placement(workers, [], moe_profile(), spread=None, ring=True)
    assert names_of(route) == ["big"]


def test_auto_takes_the_fewest_close_workers_and_leaves_the_rest() -> None:
    workers = [card(name, 100, speed) for name, speed in MESH_SPEEDS.items()]
    for ring in (True, False):
        assert set(names_of(plan_placement(workers, [], moe_profile(), spread=None, links=LINKS, ring=ring))) == {"e1", "e2"}


def test_auto_holds_a_model_in_vram_before_it_takes_closer_workers() -> None:
    # The closest pair could hold this with RAM, but a third card keeps it all in VRAM.
    workers = [card("a", 70, 900), card("b", 70, 900), card("c", 70, 900), card("e1", 70, 900, 200), card("e2", 70, 900, 200)]
    route = plan_placement(workers, [], moe_profile(), spread=None, links=LINKS, ring=True)
    assert experts_in_ram(route) == 0
    assert set(names_of(route)) == {"c", "e1", "e2"}


def test_auto_uses_the_least_ram_when_nothing_holds_it_all() -> None:
    route = plan_placement(moe_cards(), [], moe_profile(), spread=None, ring=True)
    assert len(route) == 7
    assert experts_in_ram(route) == 3 * MOE_EXPERT_BYTES


def test_auto_on_the_seven_card_sizes_drops_what_it_does_not_need() -> None:
    sizes = {"a": 4.0, "b": 24.0, "c": 24.0, "d": 48.0, "e": 48.0, "f": 48.0, "g": 20.0}
    route = plan_placement(moe_cards(sizes), [], moe_profile(), spread=None, ring=True)
    assert experts_in_ram(route) == 0
    assert len(route) < 7


def test_auto_honours_pins() -> None:
    workers = [card(name, 100, speed) for name, speed in MESH_SPEEDS.items()]
    route = plan_placement(workers, [], moe_profile(), spread=None, links=LINKS, ring=True, entry_worker="a", exit_worker="c")
    assert names_of(route)[0] == "a" and names_of(route)[-1] == "c"


def test_auto_leaves_standby_alone_unless_vram_needs_it() -> None:
    model = block_profile([2.72] * 16)
    assert names_of(plan_placement([card("a", 60, 800)], [card("spare", 60, 900)], model, spread=None)) == ["a"]
    route = plan_placement([card("a", 30, 800, 32)], [card("spare", 30, 800)], model, spread=None)
    assert set(names_of(route)) == {"a", "spare"}
    assert spill_of(route) == {}


def test_auto_works_on_workers_with_no_vram() -> None:
    workers = [worker(name, pool("ram", "ram", 4 * GIB, 30, f"{name}:ram")) for name in "abc"]
    assert len(plan_placement(workers, [], profile(), spread=None)) == 1


def test_auto_keeps_a_held_layout_unless_moving_it_pays(monkeypatch) -> None:
    model = moe_profile(amortize=planner.AMORTIZE_TOKENS)
    layout_held = [("a", 0, 47), ("c", 47, 94)]

    def pool_of(rate: float) -> list[Candidate]:
        workers = [card(name, 100, speed) for name, speed in MESH_SPEEDS.items()]
        for item in workers:
            item.fetch_bytes_per_second = rate
        return holding(workers, layout_held)

    # Slow links for fetching: 180 GiB takes longer than the loop saves in 20,000 tokens.
    assert layout_of(plan_placement(pool_of(40e6), [], model, spread=None, links=LINKS, ring=True)) == layout_held
    # Fast ones: the closer pair earns its download.
    assert set(names_of(plan_placement(pool_of(4e9), [], model, spread=None, links=LINKS, ring=True))) == {"e1", "e2"}


@pytest.mark.parametrize("ring", [False, True], ids=["nested", "loop"])
def test_a_spare_that_adds_least_to_the_route_takes_over(ring) -> None:
    model = moe_profile()
    model.ring = ring
    current = plan_placement([card("e1", 100, 800), card("c", 100, 800)], [], model, entry_worker="e1", exit_worker="c")
    alive = planner.with_links([card(name, 100, 800) for name in ("e1", "a", "b", "e2")], LINKS)
    # "a" and "b" come first and both fit, but "e2" is next door to what stayed.
    assert names_of(keep_layers(current, alive, [], model)) == ["e1", "e2"]


def test_a_spare_behind_a_dead_link_is_passed_over() -> None:
    model = block_profile([2.72] * 16)
    current = plan_placement([card("a", 30, 800), card("c", 30, 800)], [], model)
    spare, near = card("b", 30, 900), card("e1", 30, 800)
    first = current[0].worker_id
    links = {(first, "b"): planner.UNREACHABLE_LINK_MS, ("b", first): planner.UNREACHABLE_LINK_MS, (first, "e1"): 200, ("e1", first): 200}
    lost = current[-1].worker_id
    alive = planner.with_links([card(first, 30, 800), spare, near], links)
    replaced = keep_layers(current, alive, [], model)
    assert names_of(replaced)[-1] == "e1" and lost not in names_of(replaced)


def scattered(count: int, sizes: tuple[float, ...], seed: int) -> tuple[list[Candidate], dict[tuple[str, str], float]]:
    rng = random.Random(seed)
    hubs = ((0, 0), (10, 3), (60, 12), (66, 9), (120, 40))
    points = [(hub[0] + rng.gauss(0, 3), hub[1] + rng.gauss(0, 3)) for hub in (rng.choice(hubs) for _ in range(count))]
    workers = [card(f"w{index}", rng.choice(sizes), rng.choice((300, 450, 650, 900))) for index in range(count)]
    links = {
        (f"w{left}", f"w{right}"): 1 + 1.6 * math.dist(points[left], points[right])
        for left in range(count)
        for right in range(count)
        if left != right
    }
    return workers, links


@pytest.mark.parametrize(
    "sizes",
    [(6, 12, 16, 24, 24, 48, 48, 80), (8,), (4, 6), (3.5, 4)],
    ids=["mixed", "small", "tiny", "more workers than a seed looks at"],
)
def test_two_hundred_workers_plan_in_time_and_hold_the_model_in_vram(sizes) -> None:
    workers, links = scattered(200, sizes, 1)
    started = time.perf_counter()
    route = plan_placement(workers, [], moe_profile(), spread=None, links=links, ring=True)
    seconds = time.perf_counter() - started
    assert sum(stage.end_layer - stage.start_layer for stage in route) == MOE_LAYERS
    assert experts_in_ram(route) == 0
    assert seconds < 4  # measured under 2 s on a busy laptop, with room for a slower one


def test_growing_a_loop_finds_what_trying_every_subset_finds(monkeypatch) -> None:
    workers = [card(name, 100, speed) for name, speed in MESH_SPEEDS.items()]
    monkeypatch.setattr(planner, "MAX_SUBSET_WORKERS", 0)
    assert set(names_of(plan_placement(workers, [], moe_profile(), spread=None, links=LINKS, ring=True))) == {"e1", "e2"}


def test_a_loop_that_grew_uncrosses_itself() -> None:
    hops = [[0, 1, 9, 1], [1, 0, 1, 9], [9, 1, 0, 1], [1, 9, 1, 0]]
    loop = planner._untangled([0, 2, 1, 3], hops)
    assert sum(hops[left][right] for left, right in zip(loop, loop[1:] + loop[:1])) == 4


def test_a_short_hybrid_stage_does_not_rule_out_a_longer_one() -> None:
    # Any card here could take all but one layer, and that one layer cannot stand
    # alone. That is no reason to stop looking at longer stages for the next card.
    model = block_profile([2.72] * 16)
    model.layer_step = 1
    model.attention_interval = 4
    route = plan_placement([card("a", 60, 900), card("b", 60, 850), card("c", 60, 800)], [], model)
    assert len(route) == 3
    assert all(holds_both_kinds(stage.start_layer, stage.end_layer, 4) for stage in route)


def test_a_compact_loop_counts_the_way_home_when_choosing_who_to_skip() -> None:
    # With the entry pinned every order starts at "a". The first hop to "b" is the
    # shortest there is, but the way back from it is far the longest.
    links = {("a", "b"): 2, ("b", "a"): 400, ("a", "c"): 20, ("c", "a"): 20, ("b", "c"): 50, ("c", "b"): 50}
    workers = [worker(name, pool("cuda:0", "vram", 3 * GIB, 500, f"{name}:vram")) for name in "abc"]
    route = plan_placement(workers, [], profile(layers=8, dense=500 * MIB), spread=False, links=links, ring=True, entry_worker="a")
    assert names_of(route) == ["a", "c"]


def test_a_loop_turns_so_that_the_right_worker_holds_the_head() -> None:
    # Only "b" can hold this head, and it is not next to where the loop was worked out
    # from, so the shortest loop has to be turned to end there.
    model = moe_profile()
    head = raw_unit("head", "head", 45)
    model.units = [head if item.name == "head" else item for item in model.units]
    model.head = head
    workers = [card(name, 80 if name == "b" else 40, speed) for name, speed in MESH_SPEEDS.items()]
    route = plan_placement(workers, [], model, links=LINKS, ring=True)
    assert names_of(route)[-1] == "b"
    assert planner.route_network_ms(route, LINKS, ring=True) == 424 / 2 + 5 * planner.RING_HOP_MS


def test_auto_moves_a_held_layout_out_of_ram_whatever_the_download() -> None:
    # VRAM outranks what nodes hold: this model sits partly in RAM on one worker, and
    # moving part of it costs far more than the estimate says it saves. Of the ways
    # to get it all into VRAM, the one that keeps what can stay moves least.
    model = moe_profile(amortize=planner.AMORTIZE_TOKENS)
    workers = [card("a", 100, 800, 120), card("e1", 100, 800), card("e2", 100, 800)]
    holding(workers, [("a", 0, MOE_LAYERS)])
    route = plan_placement(workers, [], model, spread=None, links=LINKS, ring=True)
    assert experts_in_ram(route) == 0
    assert "a" in names_of(route) and len(route) == 2


def test_a_long_loop_of_small_cards_plans_at_the_first_try(monkeypatch) -> None:
    # Trying again is what makes these slow, so a grown loop allows for both ends.
    real = planner._closest_subsets
    asked = []

    def counting(group, model, required, slack):
        asked.append(slack)
        return real(group, model, required, slack)

    monkeypatch.setattr(planner, "_closest_subsets", counting)
    workers, links = scattered(200, (4, 6), 1)
    plan_placement(workers, [], moe_profile(), spread=None, links=links, ring=True)
    assert asked == [0]


def test_auto_asks_for_spare_room_if_a_tight_fit_does_not_plan(monkeypatch) -> None:
    real = planner._closest_subsets
    asked = []

    def nothing_fits_at_first(group, model, required, slack):
        asked.append(slack)
        return real(group, model, required, slack) if slack else []

    monkeypatch.setattr(planner, "_closest_subsets", nothing_fits_at_first)
    workers = [card(name, 100, speed) for name, speed in MESH_SPEEDS.items()]
    route = plan_placement(workers, [], moe_profile(), spread=None, links=LINKS, ring=True)
    assert asked == [0, 2]
    assert set(names_of(route)) == {"e1", "e2"}


def test_less_ram_decides_between_plans_with_the_same_room_open() -> None:
    # Ring off, so that every hop is the constant: two spread plans differ only in RAM.
    route = plan_placement(moe_cards(), [], moe_profile(), ring=False)
    assert experts_in_ram(route) == 3 * MOE_EXPERT_BYTES


def test_auto_is_the_default_placement() -> None:
    assert LoadModelRequest(model_id="m").placement == "auto"
    assert cli.build_parser().parse_args(["load", "m"]).placement == "auto"


def with_mesh_links(workers: list[Candidate]) -> list[Candidate]:
    return planner.with_links(workers, LINKS)


def uneven_pool() -> list[Candidate]:
    """One card three times as fast as the other, so the quickest loop is the most lopsided."""
    return [card("fast", 300, 900), card("slow", 300, 300)]


def reads_of(route) -> list[float]:
    return [stage.estimated_ms_per_token for stage in route]


def test_the_law_gives_the_runs_it_was_fitted_to() -> None:
    assert 1000 / planner.ms_per_token(36, 12, 2.5, 9) == pytest.approx(39.7, abs=0.05)
    assert 1000 / planner.ms_per_token(86, 12, 2.5, 9) == pytest.approx(22.1, abs=0.05)


def test_with_no_draft_only_the_loop_counts() -> None:
    assert planner.ms_per_token(250, 90) == 250


@pytest.mark.parametrize(
    "model, before",
    [
        (block_profile([2.72] * 32), [("b", 0, 8), ("e1", 8, 12), ("e2", 12, 16), ("c", 16, 20), ("a", 20, 32)]),
        (moe_profile(), [("b", 0, 20), ("e2", 20, 34), ("e1", 34, 54), ("c", 54, 74), ("a", 74, 94)]),
    ],
    ids=["block rule", "no block rule"],
)
def test_a_route_with_nothing_measured_plans_as_it_did(model, before) -> None:
    # These layouts came from the planner as it was before the law.
    assert layout_of(plan_placement(mesh_cards(), [], model, links=LINKS)) == before
    assert layout_of(plan_placement(mesh_cards(), [], model, links=LINKS, tokens_per_miss=1.0)) == before


@pytest.mark.parametrize("model", [block_profile([2.72] * 32), moe_profile()], ids=["block rule", "no block rule"])
def test_a_measured_draft_trades_some_loop_for_an_evener_pace(model) -> None:
    quickest = plan_placement(uneven_pool(), [], model)
    paced = plan_placement(uneven_pool(), [], model, tokens_per_miss=6)
    assert sum(reads_of(paced)) > sum(reads_of(quickest))
    assert max(reads_of(paced)) < max(reads_of(quickest))
    law = [planner.ms_per_token(sum(reads_of(route)), max(reads_of(route)), 6) for route in (quickest, paced)]
    assert law[1] < law[0]


def test_a_route_cost_follows_the_law_and_still_charges_a_move_in_full() -> None:
    model = replace(moe_profile(amortize=20_000), tokens_per_miss=2.5)
    workers = with_mesh_links(mesh_cards())
    route = plan_placement(workers, [], model)
    reads = reads_of(route)
    loop = sum(reads) + planner._hops_ms(route, workers, model)
    moved = planner._moved_ms(route, workers, model)
    assert moved > 0
    assert planner._route_cost(route, workers, model) == pytest.approx((loop + 1.5 * max(reads)) / 2.5 + moved)


def test_a_tie_on_a_pool_of_equals_keeps_the_earlier_worker() -> None:
    route = plan_placement([ram_worker("a", 4 * GIB), ram_worker("b", 4 * GIB)], [], profile(), spread=False)
    assert names_of(route) == ["a"]


def law_runtime(telemetry: dict, model: str = "m") -> Orchestrator:
    runtime = Orchestrator()
    runtime.active = SimpleNamespace(model_id=model)
    runtime.route = plan_placement([ram_worker("a")], [], profile())
    runtime.workers = {"a": SimpleNamespace(health={"telemetry": telemetry})}
    return runtime


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"tokens": 1000, "misses": 400}, 2.5),
        ({"tokens": 300, "misses": 0}, 300),
        ({"tokens": 100, "misses": 40}, 1.0),  # too few to believe
        ({"tokens": 300, "misses": 400}, 1.0),  # a miss can come with no token
        ({}, 1.0),
    ],
)
def test_tokens_per_miss_comes_from_the_entry_stage(counts, expected) -> None:
    runtime = law_runtime({"pipeline": counts})
    assert runtime.tokens_per_miss(runtime.active) == expected


def test_a_target_that_is_not_running_has_measured_nothing() -> None:
    runtime = law_runtime({"pipeline": {"tokens": 1000, "misses": 400}})
    assert runtime.tokens_per_miss(SimpleNamespace(model_id="m")) == 1.0


def test_a_plan_shows_its_loop_and_pace(monkeypatch) -> None:
    model = moe_profile()
    runtime = law_runtime({"pipeline": {"tokens": 1000, "misses": 400}})
    runtime.links = dict(LINKS)
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    route = plan_placement(mesh_cards(), [], model, links=LINKS)
    adapter = SimpleNamespace(config={}, num_layers=model.num_layers)
    units = {item.name: item for item in model.units}
    target = SimpleNamespace(model_id="m", adapter=adapter, profile=model, units=units, held={}, draft=None, embedding=False, projector=None)
    runtime.active = target
    plan = orchestrator_module.describe_plan(target, route)
    loop, pace = planner.route_times(route, LINKS, model.ring)
    assert plan["loop_ms"] == pytest.approx(loop, abs=0.01)
    assert plan["pace_ms"] == pytest.approx(pace, abs=0.01)
    assert plan["tokens_per_miss"] == 2.5
    assert plan["ms_per_token"] == pytest.approx((loop + 1.5 * pace) / 2.5, abs=0.01)
    assert plan["estimated_tokens_per_second"] == pytest.approx(1000 / loop, abs=0.05)  # as it always was


ZIGZAG = ["a", "e1", "b", "e2", "c"]


def zigzag(model: ModelProfile, order: list[str] = ZIGZAG, workers: list[Candidate] | None = None, links=LINKS):
    """A route that runs in this order, its nodes holding what they run."""
    workers = planner.with_links(workers or mesh_cards(), links)
    by_name = {item.worker_id: item for item in workers}
    route = planner.plan_through([[by_name[name] for name in order]], [by_name[name] for name in order], model)
    assert names_of(route) == order
    held = {stage.worker_id: frozenset(item.unit.name for item in planner._stage_items(model, stage.start_layer, stage.end_layer)) for stage in route}
    return route, [replace(item, cached=held.get(item.worker_id, frozenset()), fetch_bytes_per_second=50e6) for item in workers]


def ring_model(**fields) -> ModelProfile:
    return replace(moe_profile(amortize=20_000), ring=True, **fields)


def test_nodes_group_by_their_shortest_links() -> None:
    groups = geometry.nearest_groups(list(MESH_SPEEDS), LINKS)
    assert sorted(map(sorted, groups)) == [["a"], ["b"], ["c"], ["e1", "e2"]]


def test_a_chain_of_short_links_is_one_group() -> None:
    assert geometry.nearest_groups(["x", "y", "z"], {("x", "y"): 3.0, ("y", "z"): 4.0}) == [["x", "y", "z"]]


def test_a_failed_or_unmeasured_link_groups_nothing() -> None:
    links = {("x", "y"): planner.UNREACHABLE_LINK_MS}
    assert geometry.nearest_groups(["x", "y", "z"], links) == [["x"], ["y"], ["z"]]


def loop_of(names: list[str]) -> tuple[int, float]:
    """Failed forward links, then the time around, the closing edge included."""
    forward = list(zip(names, names[1:]))
    dead = sum(LINKS[hop] >= planner.UNREACHABLE_LINK_MS for hop in forward)
    walked = sum(LINKS[hop] / 2 for hop in forward)
    closing = LINKS[names[-1], names[0]]
    return dead, walked + (walked if closing >= planner.UNREACHABLE_LINK_MS else closing / 2)


def test_a_tour_counts_the_edge_that_closes_it() -> None:
    order = geometry.tour(with_mesh_links(mesh_cards()), LINKS)
    best = min(loop_of(["a", *rest]) for rest in permutations(["b", "c", "e1", "e2"]))
    assert loop_of(names_of_workers(order)) == best


def names_of_workers(workers: list[Candidate]) -> list[str]:
    return [item.worker_id for item in workers]


def test_past_a_dozen_nodes_a_tour_goes_site_by_site() -> None:
    sites = {name: name[0] for name in (f"{site}{index}" for site in "xyz" for index in range(5))}
    links = {(left, right): (1.0 if sites[left] == sites[right] else 80.0) for left in sites for right in sites if left != right}
    workers = planner.with_links([ram_worker(name) for name in sorted(sites, key=lambda name: name[1])], links)
    assert len(workers) > planner.MAX_LINKED_WORKERS
    order = [sites[name] for name in names_of_workers(geometry.tour(workers, links))]
    assert sorted(order) == sorted(sites.values())
    assert sum(left != right for left, right in zip(order, order[1:] + order[:1])) == 3


def test_a_proposal_is_a_better_order_of_the_same_nodes() -> None:
    model = ring_model()
    route, workers = zigzag(model)
    proposal = geometry.propose(route, workers, LINKS, model)
    assert sorted(proposal["proposed"]["order"]) == sorted(ZIGZAG)
    assert proposal["current"]["order"] == ZIGZAG
    assert proposal["proposed"]["loop_ms"] < proposal["current"]["loop_ms"]
    assert proposal["saved_ms_per_token"] == pytest.approx(proposal["current"]["ms_per_token"] - proposal["proposed"]["ms_per_token"], abs=0.01)
    stages = proposal["proposed"]["stages"]
    assert [stage["worker_id"] for stage in stages] == proposal["proposed"]["order"]
    assert stages[0]["start_layer"] == 0 and stages[-1]["end_layer"] == model.num_layers
    assert ["e1", "e2"] in map(sorted, proposal["groups"])


def test_a_proposal_says_what_would_move_and_how_long_it_takes() -> None:
    model = ring_model()
    route, workers = zigzag(model)
    proposal = geometry.propose(route, workers, LINKS, model)
    held = {item.worker_id: item.cached for item in workers}
    by_worker = {
        stage["worker_id"]: sum(
            item.unit.nbytes
            for item in planner._stage_items(model, stage["start_layer"], stage["end_layer"])
            if item.unit.name not in held[stage["worker_id"]]
        )
        for stage in proposal["proposed"]["stages"]
    }
    assert proposal["moved_bytes"] == sum(by_worker.values()) > 0
    assert proposal["move_seconds"] == pytest.approx(max(by_worker.values()) / 50e6, abs=0.05)
    assert proposal["break_even_tokens"] == pytest.approx(proposal["move_seconds"] * 1000 / proposal["saved_ms_per_token"], rel=0.01)
    assert proposal["stages_changed"] > 0


def test_a_proposal_changes_nothing_it_was_given() -> None:
    model = ring_model()
    route, workers = zigzag(model)
    before = [stage.model_dump() for stage in route]
    geometry.propose(route, workers, LINKS, model)
    assert [stage.model_dump() for stage in route] == before


def test_the_shortest_loop_has_nothing_to_propose() -> None:
    model = ring_model()
    shortest = names_of_workers(geometry.tour(with_mesh_links(mesh_cards()), LINKS))
    route, workers = zigzag(model, shortest)
    assert geometry.propose(route, workers, LINKS, model) is None


def test_with_no_measured_link_there_is_nothing_to_propose() -> None:
    model = ring_model()
    route, workers = zigzag(model, links={})
    assert geometry.propose(route, workers, {}, model) is None


def test_a_node_that_left_ends_the_proposal() -> None:
    model = ring_model()
    route, workers = zigzag(model)
    assert geometry.propose(route, [item for item in workers if item.worker_id != "c"], LINKS, model) is None


def test_one_stage_has_no_loop_to_shorten() -> None:
    model = ring_model()
    route, workers = zigzag(model, ["a"], [card("a", 500, 900)])
    assert geometry.propose(route, workers, LINKS, model) is None


def test_a_saving_inside_a_links_swing_is_not_worth_showing() -> None:
    names = ["a", "b", "c", "d"]
    links = {(left, right): 20.0 for left in names for right in names if left != right}
    links["a", "b"] = links["b", "a"] = 20.3
    model = ring_model()
    route, workers = zigzag(model, names, [card(name, 60, 800) for name in names], links)
    assert geometry.propose(route, workers, links, model) is None


def test_a_proposal_never_adds_spill(monkeypatch) -> None:
    model = ring_model()
    route, workers = zigzag(model)
    monkeypatch.setattr(geometry, "route_faults", lambda found, _: (0, 0 if found is route else 1))
    assert geometry.propose(route, workers, LINKS, model) is None


def test_pins_stay_where_they_are() -> None:
    model = ring_model()
    route, workers = zigzag(model)
    proposal = geometry.propose(route, workers, LINKS, model, entry="a", exit_="c")
    assert proposal["proposed"]["order"][0] == "a"
    assert proposal["proposed"]["order"][-1] == "c"


def draft_cards(room: tuple[str, ...]) -> list[Candidate]:
    return [card(name, 70 if name in room else 48, speed) for name, speed in MESH_SPEEDS.items()]


def test_the_entry_stays_put_if_turning_the_loop_gains_nothing() -> None:
    model = ring_model(entry_extra_bytes=50 * GIB)
    route, workers = zigzag(replace(model, entry_extra_bytes=0), workers=draft_cards(("a", "c")))
    proposal = geometry.propose(route, workers, LINKS, model, entry_bytes=5 * GIB)
    assert proposal["proposed"]["order"][0] == "a"


def test_the_entry_goes_where_the_draft_fits() -> None:
    model = ring_model(entry_extra_bytes=50 * GIB)
    route, workers = zigzag(replace(model, entry_extra_bytes=0), workers=draft_cards(("e1",)))
    assert route[0].worker_id == "a"
    without = geometry.propose(route, workers, LINKS, replace(model, entry_extra_bytes=0))
    proposal = geometry.propose(route, workers, LINKS, model, entry_bytes=5 * GIB)
    assert proposal["proposed"]["order"][0] == "e1"
    assert proposal["moved_bytes"] > 5 * GIB
    assert without["proposed"]["order"][0] == "a"


def test_a_draft_that_follows_the_entry_counts_as_moved() -> None:
    model = moe_profile()
    route, workers = zigzag(model)
    assert planner.moved_weights(route, workers, model) == (0, 0.0)
    assert planner.moved_weights(route, workers, model, 5 * 10**8) == (5 * 10**8, 10.0)


def running_pool(monkeypatch):
    """A control plane with a zigzag route running, and what stops it moving anything."""
    model = ring_model()
    route, workers = zigzag(model)
    runtime = Orchestrator()
    active = SimpleNamespace(model_id="m", profile=model, draft=None, projector=None, entry_worker=None, exit_worker=None, held={}, serving=set(), reuse_cached=True)
    runtime.active, runtime.route, runtime.links, runtime.epoch = active, route, dict(LINKS), 7

    async def nothing(*args):
        return {}

    def refuse(*args, **kwargs):
        raise AssertionError("a plan must not touch a route")

    runtime.gguf_sources = runtime.refresh_worker_metrics = runtime.refresh_holdings = runtime.measure_links = nothing
    runtime.candidate_groups = lambda: (workers, [])
    runtime._with_holdings = lambda group, target: group
    runtime.plan_or_refuse = lambda target, **options: route
    runtime.plan_memory = lambda target, plan: {}
    runtime._activate = runtime._prefetch = runtime.deploy = refuse
    monkeypatch.setattr(orchestrator_module, "runtime", runtime)
    monkeypatch.setattr(orchestrator_module, "build_target", lambda *args: SimpleNamespace(model_id=args[0].model_id))
    monkeypatch.setattr(orchestrator_module, "row_bytes", lambda target: 0)
    monkeypatch.setattr(orchestrator_module, "describe_plan", lambda target, plan: {"model": target.model_id})
    return runtime


def test_the_plan_route_shows_a_proposal_and_moves_nothing(monkeypatch) -> None:
    runtime = running_pool(monkeypatch)
    before = [stage.model_dump() for stage in runtime.route]
    answer = TestClient(orchestrator_module.app).post("/admin/plan", json={"model_id": "m"})
    assert answer.status_code == 200
    assert answer.json()["proposal"]["current"]["order"] == ZIGZAG
    assert [stage.model_dump() for stage in runtime.route] == before
    assert runtime.epoch == 7


def test_a_model_that_is_not_running_has_no_proposal(monkeypatch) -> None:
    running_pool(monkeypatch)
    answer = TestClient(orchestrator_module.app).post("/admin/plan", json={"model_id": "other"})
    assert answer.status_code == 200
    assert answer.json()["proposal"] is None


def answers(plan: dict, status: dict | None = None):
    asked = []

    def get(url, **kwargs):
        asked.append(("GET", url, None))
        return SimpleNamespace(status_code=200, json=lambda: status or {})

    def post(url, json, **kwargs):
        asked.append(("POST", url, json))
        return SimpleNamespace(status_code=200, json=lambda: plan, text="")

    return SimpleNamespace(get=get, post=post, HTTPError=httpx.HTTPError), asked


def plan_args(*words: str):
    return cli.build_parser().parse_args(["plan", *words, "--url", "http://node:8209", "--token", "key"])


def test_the_plan_command_shows_times_and_bytes_and_says_nothing_moved(monkeypatch, capsys) -> None:
    model = ring_model()
    route, workers = zigzag(model)
    proposal = geometry.propose(route, workers, LINKS, model)
    fake, asked = answers({"route": [stage.model_dump() for stage in route], "loop_ms": 400.0, "proposal": proposal}, {"model": "m"})
    monkeypatch.setattr(cli, "httpx", fake)
    assert cli.run_plan(plan_args()) == 0
    printed = capsys.readouterr().out
    assert asked[-1] == ("POST", "http://node:8209/admin/plan", {"model_id": "m"})
    assert " > ".join(proposal["proposed"]["order"]) in printed
    assert f"{proposal['moved_bytes'] / GIB:.1f} GiB" in printed
    assert "Nothing has moved" in printed


def test_the_plan_command_says_when_there_is_nothing_to_propose(monkeypatch, capsys) -> None:
    fake, asked = answers({"route": [{"worker_id": "a"}], "loop_ms": 5.0, "proposal": None})
    monkeypatch.setattr(cli, "httpx", fake)
    assert cli.run_plan(plan_args("some-model")) == 0
    assert "No shorter order" in capsys.readouterr().out
    assert [item[0] for item in asked] == ["POST"]


def test_the_plan_command_needs_a_model_from_somewhere(monkeypatch, capsys) -> None:
    fake, _ = answers({}, {"model": None})
    monkeypatch.setattr(cli, "httpx", fake)
    assert cli.run_plan(plan_args()) == 1


def test_a_proposal_never_buys_a_working_link_with_spill(monkeypatch) -> None:
    model = ring_model()
    route, workers = zigzag(model)
    monkeypatch.setattr(geometry, "route_faults", lambda found, _: (1, 0) if found is route else (0, 1))
    assert geometry.propose(route, workers, LINKS, model) is None


def test_a_link_nobody_measured_is_no_reason_to_propose() -> None:
    model = ring_model()
    route, workers = zigzag(model)
    assert geometry.propose(route, workers, LINKS, model) is not None
    guessed = {hop: milliseconds for hop, milliseconds in LINKS.items() if set(hop) != {"a", "b"}}
    assert geometry.propose(route, workers, guessed, model) is None


def test_pins_stay_past_a_dozen_nodes_off_a_ring() -> None:
    names = [f"n{index:02d}" for index in range(planner.MAX_LINKED_WORKERS + 2)]
    links = {(left, right): 5.0 + abs(index - other) for index, left in enumerate(names) for other, right in enumerate(names) if left != right}
    members = [ram_worker(name) for name in reversed(names)]
    orders = geometry._orders(members, links, profile(), names[3], None)
    assert all(order[0].worker_id == names[3] for order in orders)


def test_a_load_that_lands_during_a_proposal_ends_it(monkeypatch) -> None:
    runtime = running_pool(monkeypatch)
    others = SimpleNamespace(model_id="m")

    async def swapped(target):
        runtime.active = others

    runtime.refresh_holdings = swapped
    assert asyncio.run(runtime.propose_order(SimpleNamespace(model_id="m"))) is None


def test_the_plan_command_says_when_it_was_refused(monkeypatch, capsys) -> None:
    refused = SimpleNamespace(status_code=401, text="bad key", json=lambda: {"detail": "bad key"})
    monkeypatch.setattr(cli, "httpx", SimpleNamespace(get=lambda url, **kwargs: refused))
    assert cli.run_plan(plan_args()) == 1
    said = capsys.readouterr().err
    assert "bad key" in said and "No model is running" not in said
