import pytest

from shardmesh.planner import Candidate, ModelProfile, PlacementError, plan_placement, stored_bytes
from shardmesh.protocol import MemoryPool, TensorRef, WeightUnit

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
