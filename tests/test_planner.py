import asyncio
from itertools import combinations, permutations
import random
from types import SimpleNamespace

import httpx
import pytest

from isharded import planner
from isharded.orchestrator import Orchestrator
from isharded.planner import Candidate, ModelProfile, PlacementError, memory_report, plan_placement, stored_bytes
from isharded.protocol import MemoryPool, TensorRef, WeightUnit

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
    stage = planner._stage_scorer(workers, model, packer)
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
