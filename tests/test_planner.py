from itertools import combinations, permutations

import pytest

from isharded import planner
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
# What the planner holds back on a card: its own workspace and a GGUF stage's.
CARD_RESERVE_GIB = 1.0
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


def test_the_search_finds_what_trying_every_layout_finds() -> None:
    workers, model = mixed_cards()
    capacities = planner._capacities(workers, 1.0, model.stage_workspace_bytes)

    def stage(candidate: Candidate, start: int, end: int) -> tuple[int, float] | None:
        items = planner._stage_items(model, start, end)
        filled = planner._fill(model, items, candidate.pools, capacities[candidate.worker_id])
        return None if filled is None else (planner._spilled_bytes(filled[0], candidate.pools), filled[1])

    layouts = []
    for order in permutations(workers):
        for inner in combinations(range(4, 24, 4), 3):
            cuts = (0, *inner, 24)
            found = [stage(candidate, left, right) for candidate, left, right in zip(order, cuts, cuts[1:])]
            if None not in found:
                layouts.append((sum(item[0] for item in found), max(item[1] for item in found), sum(item[1] for item in found)))
    fastest = min(layouts, key=lambda layout: layout[2])
    allowed = [layout for layout in layouts if layout[1] <= fastest[1] * (1 + planner.SPILL_ALLOWANCE)]
    leanest = min(allowed, key=lambda layout: layout[:2])
    # Less spill and a faster slowest stage, for a little more time in total.
    assert leanest[0] < fastest[0] and leanest[1] < fastest[1] and leanest[2] > fastest[2]

    route = plan_placement(workers, [], model)
    assert sum(spill_of(route).values()) == leanest[0]
    assert max(item.estimated_ms_per_token for item in route) == pytest.approx(leanest[1], abs=0.01)


def test_near_ties_go_to_the_plan_with_less_spill(monkeypatch) -> None:
    workers, model = mixed_cards()
    lean = sum(spill_of(plan_placement(workers, [], model)).values())
    monkeypatch.setattr(planner, "SPILL_ALLOWANCE", 0.0)
    assert sum(spill_of(plan_placement(workers, [], model)).values()) > lean


def test_least_spill_does_not_win_when_it_is_far_slower(monkeypatch) -> None:
    # A worker with only RAM has no spill, so least spill alone would take dense
    # weights off the GPU. Keeping them there and spilling experts is faster.
    workers = [card("gpu", 10, 800, 64), worker("cpu", pool("ram", "ram", 64 * GIB, 40, "cpu:ram"))]
    model = block_profile([2.72] * 24)

    def layers_on_gpu() -> int:
        stage = next(item for item in plan_placement(workers, [], model) if item.worker_id == "gpu")
        return stage.end_layer - stage.start_layer

    assert layers_on_gpu() == 20
    monkeypatch.setattr(planner, "SPILL_ALLOWANCE", 100.0)
    assert layers_on_gpu() == 4


def test_of_equal_spills_the_faster_slowest_stage_wins_over_a_faster_total() -> None:
    workers = [card("a", 30, 900, 16), card("b", 30, 700, 16), card("c", 8, 1800, 16), card("d", 30, 700, 16)]
    model = block_profile([3.2, *[2.72] * 22, 3.0])
    route = plan_placement(workers, [], model)
    assert sum(spill_of(route).values()) == 2 * EXPERT_BYTES
    # Ranking equal spills by total instead gives 3.17 here, for less total time.
    assert max(stage.estimated_ms_per_token for stage in route) == pytest.approx(3.11, abs=0.01)


def test_less_spill_may_not_cost_the_slowest_stage_more_than_the_allowance() -> None:
    workers = [card("a", 22, 1800, 24), card("b", 16, 700, 24), card("c", 8, 500, 24)]
    model = block_profile([3.2, *[2.72] * 22, 3.0])
    route = plan_placement(workers, [], model)
    # The very least spill is 2.6 GiB lower, but its slowest stage takes 10.9 ms.
    assert max(stage.estimated_ms_per_token for stage in route) == pytest.approx(8.02, abs=0.01)


def test_compact_plans_keep_a_hop_saved_over_spill_saved() -> None:
    workers = [card(name, 70, 800, 60) for name in "abc"]
    route = plan_placement(workers, [], block_profile(), spread=False)
    assert len(route) == 2
    assert sum(spill_of(route).values()) > 0


def test_less_spill_is_not_worth_downloading_shards_already_in_place(monkeypatch) -> None:
    # A running model's shards sit where an earlier plan put them. Less spill here
    # would move most of them, which outweighs what it saves for a long time.
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


def no_block_rule_case() -> tuple[list[Candidate], ModelProfile]:
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
    sizes = (("a", 900, 700), ("b", 900, 700), ("c", 600, 1800), ("d", 600, 1800), ("e", 400, 1000), ("f", 400, 1000))
    workers = [
        worker(name, pool("cuda:0", "vram", size * MIB, speed, f"{name}:vram"), pool("ram", "ram", 2 * GIB, 40, f"{name}:ram"))
        for name, size, speed in sizes
    ]
    return workers, model


def test_a_model_with_no_block_rule_plans_exactly_as_it_did() -> None:
    # These numbers came from the planner as it was before block models had a search.
    workers, model = no_block_rule_case()
    route = plan_placement(workers, [], model)
    assert [(stage.worker_id, stage.start_layer, stage.end_layer) for stage in route] == [
        ("b", 0, 2),
        ("a", 2, 4),
        ("f", 4, 5),
        ("e", 5, 6),
        ("d", 6, 9),
        ("c", 9, 12),
    ]
    assert [held_in(stage, "ram") // MIB for stage in route] == [0, 0, 0, 0, 600, 600]
    assert [stage.estimated_ms_per_token for stage in route] == [0.15, 0.15, 0.05, 0.05, 0.55, 0.58]


def test_a_model_with_no_block_rule_never_reaches_the_block_code(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("this is only for models with a block rule")

    monkeypatch.setattr(planner, "_plan_blocks", refuse)
    monkeypatch.setattr(planner, "_spill_less", refuse)
    workers, model = no_block_rule_case()
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
