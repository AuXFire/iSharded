"""How many replies a load runs at once, when the operator leaves that open."""

import asyncio
import math
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError
import pytest

from isharded import orchestrator as orchestrator_module
from isharded import worker as worker_module
from isharded.models import UnsupportedModel
from isharded.orchestrator import ModelTarget, Orchestrator, describe_plan, draft_sizes, exactness_of, pass_cost, pass_curve, settled_parallel, stage_devices, wanted_slots, with_slots
from isharded import planner
from isharded.planner import Candidate, ModelProfile
from isharded.protocol import MAX_PARALLEL, DraftModel, InferRequest, LoadModelRequest, MemoryPool, StageAssignment, TensorRef, UnitPlacement, WeightUnit

MIB = 1024**2
GIB = 1024**3
LAYERS = 8
KV_PER_TOKEN = 16 * 1024  # per layer, so 16 MiB a layer and a slot at 1024 tokens


def unit(name, kind, size, layer=None):
    tensor = TensorRef(name=name, file="f", dtype="BF16", shape=[size // 2], start=0, end=size)
    return WeightUnit(name=name, kind=kind, layer=layer, tensors=[tensor])


def target(parallel=None, draft=None, settled=None):
    request = LoadModelRequest(model_id="m.gguf", parallel=parallel, context_tokens=1024)
    units = [unit("embed", "embed", 50 * MIB), unit("head", "head", 50 * MIB)]
    units += [unit(f"layer-{index:05d}.dense", "dense", 100 * MIB, index) for index in range(LAYERS)]
    slots = parallel or 1
    profile = ModelProfile(
        units=units,
        num_layers=LAYERS,
        expert_active_fraction=1.0,
        kv_bytes_per_token=[KV_PER_TOKEN * slots] * LAYERS,
        state_bytes=[0] * LAYERS,
        context_tokens=1024,
        ring=True,
    )
    adapter = SimpleNamespace(num_layers=LAYERS, config={}, kv_bytes_per_token=lambda layer, itemsize: KV_PER_TOKEN, recurrent_state_bytes=lambda layer: 0)
    return ModelTarget(
        model_id="m.gguf",
        manifest=SimpleNamespace(revision="r"),
        adapter=adapter,
        units={item.name: item for item in units},
        profile=profile,
        spread=True,
        reuse_cached=False,
        request=request,
        draft=draft,
        slots=slots,
        parallel=settled or ({"slots": parallel, "reason": "set by the operator"} if parallel else {}),
    )


def worker(name, vram=8 * GIB, ram=8 * GIB):
    pools = [
        MemoryPool(pool_id=f"{name}-gpu", kind="vram", device="cuda:0", capacity_group=f"{name}:gpu", budget_bytes=vram, free_bytes=vram, bandwidth_gbps=100),
        MemoryPool(pool_id=f"{name}-ram", kind="ram", device="cpu", capacity_group=f"{name}:ram", budget_bytes=ram, free_bytes=ram, bandwidth_gbps=10),
    ]
    return Candidate(worker_id=name, url=f"http://{name}", pools=pools)


@pytest.fixture
def pool():
    runtime = Orchestrator()
    runtime.offered = [worker("a"), worker("b")]
    runtime.candidate_groups = lambda: (runtime.offered, [])
    runtime.links = {("a", "b"): 200.0, ("b", "a"): 200.0}
    yield runtime
    asyncio.run(runtime.close())


def choose(runtime, item):
    asyncio.run(runtime.choose_slots(item))


def stage(name, start, end, ms):
    return StageAssignment(worker_id=name, url=f"http://{name}", start_layer=start, end_layer=end, estimated_ms_per_token=ms, placements=[])


def test_a_route_wants_its_round_trip_over_its_slowest_row():
    route = [stage("a", 0, 4, 10.0), stage("b", 4, 8, 20.0)]
    # 10 + 20 of work, and 25 + 1 ms each way round the ring.
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 50.0}, True) == {"round_trip_ms": 82.0, "row_ms": 20.0, "devices": 2, "wanted": 4, "measured": False, "merging": False}


def test_measured_rows_stand_in_for_memory_speeds():
    route = [stage("a", 0, 4, 10.0), stage("b", 4, 8, 20.0)]
    # 30 + 5 of work as the stages measured it, and the same 52 ms of links. Two slots would leave
    # the 30 ms stage idle a third of each trip, and four replies would wait for a second round.
    measured = wanted_slots(route, ["a", "b"], {("a", "b"): 50.0}, True, [30.0, 5.0])
    assert measured == {"round_trip_ms": 87.0, "row_ms": 30.0, "devices": 2, "wanted": 4, "measured": True, "merging": False}


def test_replies_past_the_slots_wait_a_round_so_a_far_route_takes_enough_for_them():
    # 100 ms a hop round a CPU and a GPU stage: eleven 20 ms rows fit in a trip, and 16 replies on
    # 11 slots took two rounds. Measured, 11 slots made 32 tokens/s and 16 made 45.7.
    route = [stage("a", 0, 4, 3.2), stage("b", 4, 8, 20.0)]
    chosen = wanted_slots(route, ["a", "b"], {("a", "b"): 198.0}, True, [3.2, 20.0])
    assert math.floor(chosen["round_trip_ms"] / chosen["row_ms"]) == 11
    assert chosen["wanted"] == 16


def test_merged_passes_that_cost_less_let_a_route_take_more_replies_but_only_where_they_merge():
    route = [stage("a", 0, 4, 3.2), stage("b", 4, 8, 20.0)]
    links = {("a", "b"): 198.0}
    # As measured: 16 replies in one pass took 29.5 ms on the CPU stage, against 20 for one.
    passes = [{1: 3.2, 16: 4.7}, {1: 20.0, 16: 29.5}]
    merged = wanted_slots(route, ["a", "b"], links, True, [3.2, 20.0], passes, merge_most=64)
    assert merged["merging"] is True and merged["wanted"] == MAX_PARALLEL
    assert wanted_slots(route, ["a", "b"], links, True, [3.2, 20.0], passes, merge_most=1)["wanted"] == 16


def test_merging_counts_no_larger_a_pass_than_every_stage_measured():
    # Measured at 30 ms a hop: the GPU stage only ever ran two together, and the CPU stage took
    # 41 ms for three. Counting a line through those out to 64 chose 64 slots, whose caches put
    # 9 more layers on the CPU stage and halved what 8 replies made.
    route = [stage("a", 0, 12, 2.6), stage("b", 12, 28, 19.0)]
    passes = [{1: 2.6, 2: 8.06}, {1: 19.0, 3: 41.2}]
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 58.0}, True, [2.6, 19.0], passes, merge_most=64)["wanted"] == 8
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 198.0}, True, [2.6, 19.0], passes, merge_most=64)["wanted"] == 16


def test_a_trip_of_less_than_two_rows_stays_at_one():
    route = [stage("a", 0, 4, 2.0), stage("b", 4, 8, 12.0)]
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 6.0}, True)["wanted"] == 1


def test_a_lone_stage_has_nothing_to_overlap():
    assert wanted_slots([stage("a", 0, 8, 20.0)], ["a"], {}, True)["wanted"] == 1


def test_stages_on_one_device_have_nothing_to_overlap_however_far_their_links_look():
    route = [stage("a", 0, 4, 10.0), stage("b", 4, 8, 20.0)]
    assert wanted_slots(route, ["gpu", "gpu"], {("a", "b"): 50.0}, True)["wanted"] == 1


def test_stages_sharing_a_device_add_up_on_it():
    route = [stage("a", 0, 3, 10.0), stage("b", 3, 6, 20.0), stage("c", 6, 8, 25.0)]
    links = {("a", "b"): 0.0, ("b", "c"): 0.0, ("a", "c"): 0.0}
    # The first two take 30 ms between them, not 20 or 25.
    assert wanted_slots(route, ["one", "one", "two"], links, True)["row_ms"] == 30.0


def test_a_far_route_stops_at_the_cap():
    route = [stage("a", 0, 4, 2.0), stage("b", 4, 8, 2.0)]
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 5000.0}, True)["wanted"] == MAX_PARALLEL


def test_a_row_cannot_take_less_than_its_layers_do():
    route = [stage("a", 0, 4, 0.01), stage("b", 4, 8, 0.01)]
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 0.0}, True)["row_ms"] == 4 * planner.LAYER_FLOOR_MS


def test_an_unmeasured_link_says_nothing():
    route = [stage("a", 0, 4, 10.0), stage("b", 4, 8, 20.0)]
    assert wanted_slots(route, ["a", "b"], {}, True) is None
    assert wanted_slots(route, ["a", "b"], {("b", "a"): 1.0}, False) is not None


def test_a_link_whose_probe_failed_says_nothing_either():
    route = [stage("a", 0, 4, 10.0), stage("b", 4, 8, 20.0)]
    failed = orchestrator_module.UNREACHABLE_LINK_MS
    assert wanted_slots(route, ["a", "b"], {("a", "b"): failed, ("b", "a"): failed}, True) is None
    assert wanted_slots(route, ["a", "b"], {("a", "b"): failed, ("b", "a"): 1.0}, True) is None


def test_a_stage_runs_where_most_of_it_is_held():
    held = stage("a", 0, 4, 1.0).model_copy(update={"placements": [UnitPlacement(unit="u", pool_id="a-ram", bytes=1), UnitPlacement(unit="v", pool_id="a-gpu", bytes=5)]})
    assert stage_devices([held, stage("gone", 4, 8, 1.0)], [worker("a")]) == ["a:gpu", "gone"]


def test_two_workers_on_one_gpu_stay_at_one_slot_and_say_why(pool):
    shared = [item.model_copy(update={"capacity_group": "host:cuda:0"}) for item in worker("a").pools[:1]]
    pool.offered = [Candidate("a", "http://a", shared), Candidate("b", "http://b", [shared[0].model_copy(update={"pool_id": "b-gpu"})])]
    wanted = target()
    choose(pool, wanted)
    assert wanted.slots == 1
    assert "one device" in wanted.parallel["reason"]


def test_the_cap_is_64_and_nothing_set_leaves_it_to_the_load():
    assert LoadModelRequest(model_id="m").parallel is None
    assert LoadModelRequest(model_id="m", parallel=64).parallel == 64
    with pytest.raises(ValidationError):
        LoadModelRequest(model_id="m", parallel=65)


def test_only_a_model_with_slots_is_left_for_the_load_to_settle():
    assert settled_parallel(LoadModelRequest(model_id="m"), True) == {}
    assert settled_parallel(LoadModelRequest(model_id="m", parallel=3), True) == {"slots": 3, "reason": "set by the operator"}
    assert settled_parallel(LoadModelRequest(model_id="m", parallel=3), False)["slots"] == 1


def test_a_draft_cache_is_every_layer_of_it(monkeypatch):
    adapter = SimpleNamespace(num_layers=3, kv_bytes_per_token=lambda layer, itemsize: (layer + 1) * itemsize, recurrent_state_bytes=lambda layer: layer)
    monkeypatch.setattr(orchestrator_module, "create_adapter", lambda config: adapter)
    assert draft_sizes(SimpleNamespace(config={})) == {"kv_bytes_per_token": 12, "state_bytes": 3}


def test_a_draft_of_an_unknown_family_still_loads(monkeypatch):
    def refuse(config):
        raise UnsupportedModel("no")

    monkeypatch.setattr(orchestrator_module, "create_adapter", refuse)
    assert draft_sizes(SimpleNamespace(config={})) == {"kv_bytes_per_token": 0, "state_bytes": 0}


def test_a_draft_whose_header_cannot_size_its_cache_still_loads(monkeypatch):
    def unreadable(layer, itemsize):
        raise TypeError("head counts given per layer")

    monkeypatch.setattr(orchestrator_module, "create_adapter", lambda config: SimpleNamespace(num_layers=1, kv_bytes_per_token=unreadable))
    assert draft_sizes(SimpleNamespace(config={})) == {"kv_bytes_per_token": 0, "state_bytes": 0}


def test_a_load_with_nothing_measured_takes_half_of_what_memory_speeds_want(pool):
    wanted = target()
    choose(pool, wanted)
    route = pool.plan(wanted)
    # 44 rows fit in the trip, and 64 replies on 44 slots would take a second round.
    assert math.floor(wanted.parallel["round_trip_ms"] / wanted.parallel["row_ms"]) == 44
    assert wanted.parallel["wanted"] == MAX_PARALLEL
    assert wanted.slots == wanted.parallel["slots"] == wanted.parallel["wanted"] // 2 > 1
    assert wanted.parallel["measured"] is False
    assert wanted.profile.kv_bytes_per_token == [KV_PER_TOKEN * wanted.slots] * LAYERS
    assert f"from memory speeds, would want {wanted.parallel['wanted']}; with nothing measured yet a load takes {wanted.slots}" in wanted.parallel["reason"]
    assert {item.worker_id for item in route} == {"a", "b"}


def running(pool, item, rows, epoch=5, model=None):
    """A route as it runs, each stage's worker reporting the row it measured."""
    pool.active = item
    pool.route = pool.plan(item)
    pool.epoch = 5
    for placed, row in zip(pool.route, rows):
        telemetry = {"row_ms_ema": row}
        health = {"status": "ok", "model": model or item.model_id, "epoch": epoch, "assignment": placed.model_dump(), "telemetry": telemetry}
        pool.workers[placed.worker_id] = SimpleNamespace(health=health, live=True, quarantined=False)
    pool.note_rows()
    return pool.route


def test_a_later_load_chooses_by_what_the_model_measured(pool):
    first = target()
    choose(pool, first)
    route = running(pool, first, [40.0, 60.0])
    assert [item.end_layer - item.start_layer for item in route] == [4, 4]
    again = target()
    again.request = again.request.model_copy(update={"context_tokens": 2048})  # another load, so it chooses afresh
    choose(pool, again)
    choice = again.parallel
    assert choice["measured"] is True and choice["row_ms"] == 60.0
    # Five rows of 60 ms fit in the trip, and eight replies on five slots would take two rounds.
    assert math.floor(choice["round_trip_ms"] / 60.0) == 5
    assert again.slots == choice["wanted"] == 8
    assert "as this model measured on these devices" in choice["reason"]


def test_a_later_load_counts_merged_passes_only_where_its_replies_may_merge(pool):
    first = target()
    choose(pool, first)
    pool.active = first
    pool.route = pool.plan(first)
    pool.epoch = 5
    for placed, row in zip(pool.route, [40.0, 60.0]):
        telemetry = {"row_ms_ema": row, "batch_ms": {"1": row, "8": row * 1.5}, "exact": {"exact_merge_limit": 1, "merge_most": 64}}
        health = {"status": "ok", "model": first.model_id, "epoch": 5, "assignment": placed.model_dump(), "telemetry": telemetry}
        pool.workers[placed.worker_id] = SimpleNamespace(health=health, live=True, quarantined=False)
    pool.note_rows()
    devices = stage_devices(pool.route, pool.offered)
    assert pool.measured_passes(first, pool.route, devices) == [{1: 40.0, 8: 60.0}, {1: 60.0, 8: 90.0}]
    assert pool.planned_merge_most(first, pool.route) == MAX_PARALLEL
    first.exact = True
    assert pool.planned_merge_most(first, pool.route) == 2, "every stage saw one companion come out exact"
    first.merge_passes = False
    assert pool.planned_merge_most(first, pool.route) == 1
    again = target()
    again.request = again.request.model_copy(update={"context_tokens": 2048})
    choose(pool, again)
    assert again.parallel["merging"] is True and again.slots > 8
    assert "merged passes" in again.parallel["reason"]


def test_status_says_whether_replies_are_exact_and_why_not():
    probed = {"exact_merge_limit": 0, "verify_exact": False, "merge_most": 1, "guesses_in_one_pass": False}
    assert exactness_of(True, [("a", probed), ("b", probed)]) == {"exact": True, "setting": True, "reasons": []}
    merging = {**probed, "merge_most": 64}
    guessing = {**probed, "guesses_in_one_pass": True}
    found = exactness_of(False, [("a", merging), ("b", guessing), ("c", None)])
    assert found["exact"] is False and found["setting"] is False
    assert found["reasons"] == [
        "a merges up to 64 replies in a pass, and only 1 came out as one row alone",
        "a runs a draft's guesses several rows to a pass, checked by b, which rounds differently",
        "b runs a draft's guesses several rows to a pass, checked by it, which rounds differently",
        "c has not said how its passes round",
    ]
    unprobed = {"exact_merge_limit": None, "verify_exact": None, "merge_most": 64, "guesses_in_one_pass": False}
    assert exactness_of(False, [("a", unprobed)])["reasons"] == ["a merges up to 64 replies in a pass, and nothing was probed"]
    # A stage whose probe saw every merge come out exact keeps exact replies with merging on.
    assert exactness_of(False, [("a", {**probed, "exact_merge_limit": 63, "merge_most": 64})])["exact"] is True


def test_status_asks_every_stage_whether_a_checked_pass_rounds_as_one_row():
    """Before, only the entry's own probe was read, though every stage runs the checked pass."""
    entry = {"exact_merge_limit": 0, "verify_exact": True, "merge_most": 1, "guesses_in_one_pass": True}
    later = {"exact_merge_limit": 0, "verify_exact": False, "merge_most": 1, "guesses_in_one_pass": False}
    found = exactness_of(True, [("entry", entry), ("later", later)])
    assert found["exact"] is False
    assert found["reasons"] == ["later runs a draft's guesses several rows to a pass, checked by entry, which rounds differently"]
    assert exactness_of(True, [("entry", entry), ("later", {**later, "verify_exact": True})])["exact"] is True
    assert exactness_of(True, [("entry", {**entry, "guesses_in_one_pass": False}), ("later", later)])["exact"] is True, "nothing checks guesses"


def test_a_measured_row_is_kept_per_layer_for_a_route_planned_a_little_differently(pool):
    first = target()
    choose(pool, first)
    route = running(pool, first, [40.0, 60.0])
    shifted = [route[0].model_copy(update={"end_layer": 2}), route[1].model_copy(update={"start_layer": 2})]
    devices = stage_devices(shifted, pool.offered)
    assert pool.measured_rows(first, shifted, devices) == [20.0, 90.0]


def test_the_stage_with_the_head_is_measured_apart(pool):
    first = target()
    choose(pool, first)
    route = running(pool, first, [40.0, 60.0])
    turned = [route[1].model_copy(update={"start_layer": 0, "end_layer": 4}), route[0].model_copy(update={"start_layer": 4, "end_layer": 8})]
    assert pool.measured_rows(first, turned, stage_devices(turned, pool.offered)) is None


def test_only_a_report_for_the_running_stage_counts(pool):
    for stale in ({"epoch": 4}, {"model": "other.gguf"}):
        pool.row_readings.clear()
        item = target()
        choose(pool, item)
        running(pool, item, [40.0, 60.0], **stale)
        assert pool.row_readings == {}
    item = target()
    running(pool, item, [None, 60.0])
    assert len(pool.row_readings) == 1, "a stage that has run no rows yet says nothing"


def test_a_load_stops_where_memory_would_spill(pool):
    # A stage's weights leave room for a few of its 64 MiB slots, far from what 200 ms wants.
    pool.offered = [worker("a", 876 * MIB), worker("b", 876 * MIB)]
    wanted = target()
    choose(pool, wanted)
    assert 1 < wanted.slots < wanted.parallel["wanted"]
    assert pool.plan_memory(wanted, pool.plan(wanted))["spilled_bytes"] == 0
    assert "fit in memory" in wanted.parallel["reason"]
    with_slots(wanted, wanted.slots + 1)
    assert pool.plan_memory(wanted, pool.plan(wanted))["spilled_bytes"] > 0


def test_a_pool_that_already_spills_may_spill_no_more(pool):
    pool.offered = [worker("a", 300 * MIB), worker("b", 300 * MIB)]
    wanted = target()
    with_slots(wanted, 1)
    spilled = pool.plan_memory(wanted, pool.plan(wanted))["spilled_bytes"]
    assert spilled > 0
    choose(pool, wanted)
    assert pool.plan_memory(wanted, pool.plan(wanted))["spilled_bytes"] <= spilled


def test_slots_that_push_layers_onto_a_slower_device_are_left_out(pool):
    # The GPU worker has nothing slower of its own, so each slot's cache moves layers to the CPU worker.
    gpu, cpu = worker("a", 1200 * MIB), worker("b", 0)
    pool.offered = [Candidate("a", "http://a", gpu.pools[:1]), Candidate("b", "http://b", cpu.pools[1:])]
    wanted = target()

    def goal():
        route = pool.plan(wanted)
        return wanted_slots(route, stage_devices(route, pool.offered), pool.links, True)["wanted"] // 2

    with_slots(wanted, 1)
    alone = goal()
    choose(pool, wanted)
    assert 1 < wanted.slots < alone, "the route at one slot wants far more than the route it would get"
    assert wanted.slots <= goal()
    with_slots(wanted, wanted.slots + 1)
    assert wanted.slots > goal()
    assert "would put layers on slower devices" in wanted.parallel["reason"]


def test_slots_never_pull_in_another_worker(pool):
    pool.offered = [worker("a", 876 * MIB), worker("b", 876 * MIB), worker("c", 876 * MIB)]
    pool.links |= {("a", "c"): 200.0, ("b", "c"): 200.0}
    wanted = target()
    wanted.spread = None
    with_slots(wanted, 1)
    workers = {item.worker_id for item in pool.plan(wanted)}
    choose(pool, wanted)
    assert {item.worker_id for item in pool.plan(wanted)} == workers


def test_a_draft_cache_counts_against_the_entry(pool):
    # Little RAM, so the cache has nowhere to go but beside the layers in VRAM.
    pool.offered = [worker("a", 2 * GIB, GIB), worker("b", 2 * GIB, GIB)]
    plain = target()
    choose(pool, plain)
    draft = DraftModel(source="d", file="d.gguf", file_size=1, revision="r", kv_bytes_per_token=128 * 1024)
    drafted = target(draft=draft)
    choose(pool, drafted)
    assert drafted.slots < plain.slots
    assert drafted.profile.entry_extra_bytes >= 128 * 1024 * 1024 * (drafted.slots + 1)


def test_one_slot_still_holds_a_drafts_cache():
    draft = DraftModel(source="d", file="d.gguf", file_size=1, revision="r", kv_bytes_per_token=128 * 1024)
    drafted = target(draft=draft)
    with_slots(drafted, 1)
    cache = orchestrator_module.draft_cache_bytes(draft, drafted.request.context_tokens, 1)
    assert cache == 128 * 1024 * drafted.request.context_tokens * 2, "one context for the reply, and one for guesses"
    assert drafted.profile.entry_extra_bytes == draft.file_size + orchestrator_module.DRAFT_WORKSPACE_BYTES + cache


def test_an_explicit_parallel_always_wins(pool):
    pool.offered = [worker("a", 876 * MIB), worker("b", 876 * MIB)]
    set_by_hand = target(parallel=6)
    choose(pool, set_by_hand)
    assert set_by_hand.slots == 6
    assert set_by_hand.parallel == {"slots": 6, "reason": "set by the operator"}


def test_with_a_link_unmeasured_a_load_stays_at_one_and_says_why(pool):
    pool.links = {}
    wanted = target()
    choose(pool, wanted)
    assert wanted.slots == 1
    assert "not been measured" in wanted.parallel["reason"]


def test_heartbeats_get_a_turn_while_a_load_chooses(pool):
    turns = []

    async def both():
        async def beat():
            while True:
                turns.append(1)
                await asyncio.sleep(0)

        beating = asyncio.create_task(beat())
        await pool.choose_slots(target())
        beating.cancel()

    asyncio.run(both())
    assert turns


def test_a_choice_stands_when_the_links_change_later(pool):
    wanted = target()
    choose(pool, wanted)
    chosen = wanted.slots
    pool.links = {("a", "b"): 1.0, ("b", "a"): 1.0}
    choose(pool, wanted)  # a rebalance deploys the same target
    assert wanted.slots == chosen


def test_the_same_load_again_keeps_the_running_stages(pool):
    pool.active = target()
    choose(pool, pool.active)
    chosen = pool.active.slots
    pool.links = {("a", "b"): 1.0, ("b", "a"): 1.0}
    again = target()
    again.request = again.request.model_copy(update={"wait": False})
    choose(pool, again)
    assert again.slots == chosen


def test_the_same_load_on_a_pool_too_small_for_its_slots_chooses_again(pool):
    pool.active = target()
    choose(pool, pool.active)
    chosen = pool.active.slots
    assert chosen > 1
    pool.offered = [worker("a", 700 * MIB, 100 * MIB), worker("b", 700 * MIB, 100 * MIB)]
    again = target()
    choose(pool, again)
    assert again.slots < chosen
    assert pool.plan(again)


def test_another_load_chooses_for_itself(pool):
    pool.active = target()
    choose(pool, pool.active)
    pool.links = {("a", "b"): 1.0, ("b", "a"): 1.0}
    other = target()
    other.request = other.request.model_copy(update={"context_tokens": 2048})
    choose(pool, other)
    assert other.slots < pool.active.slots


def test_a_pool_that_shrank_past_the_choice_chooses_again(pool):
    wanted = target()
    choose(pool, wanted)
    assert wanted.slots > 1
    # What is left holds the model at one slot, and at the choice not even with its RAM.
    pool.offered = [worker("a", 700 * MIB, 100 * MIB), worker("b", 700 * MIB, 100 * MIB)]
    chosen = wanted.slots
    choose(pool, wanted)
    assert wanted.slots < chosen
    assert pool.plan(wanted)


def test_a_cordon_is_priced_at_the_slots_the_move_would_choose(pool):
    wanted = target()
    choose(pool, wanted)
    chosen, held = wanted.slots, wanted.profile
    assert chosen > 1
    # What is left holds the model at one slot, and not at the slots the load chose.
    pool.offered = [worker("a", 700 * MIB, 100 * MIB), worker("b", 700 * MIB, 100 * MIB)]
    assert pool._move_cost(wanted) is not None, "slots the load chose for itself must not refuse a cordon"
    assert (wanted.slots, wanted.profile) == (chosen, held), "pricing a move changes nothing on the running model"
    # Slots an operator set are theirs to keep, and a pool that cannot hold them says so.
    fixed = target(parallel=chosen)
    assert pool._move_cost(fixed) is None
    # And nothing holds the model at any count on a pool with no room at all.
    pool.offered = [worker("a", 10 * MIB, 10 * MIB)]
    assert pool._move_cost(wanted) is None


def test_a_pool_that_holds_nothing_for_now_leaves_the_running_choice_alone(pool):
    wanted = target()
    choose(pool, wanted)
    chosen, sizes = wanted.parallel, wanted.profile.kv_bytes_per_token
    assert wanted.slots > 1
    pool.offered = [worker("a", 10 * MIB, 10 * MIB)]
    choose(pool, wanted)
    assert wanted.parallel == chosen
    assert wanted.slots == chosen["slots"]
    assert wanted.profile.kv_bytes_per_token == sizes


def test_status_shows_the_choice_and_why(pool, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "runtime", pool)
    wanted = target()
    choose(pool, wanted)
    described = describe_plan(wanted, pool.plan(wanted))
    assert described["parallel"] == wanted.slots
    assert described["parallel_choice"]["reason"] == wanted.parallel["reason"]
    assert described["parallel_choice"]["round_trip_ms"] > described["parallel_choice"]["row_ms"]


def test_the_preview_shows_what_a_load_would_choose(pool, monkeypatch):
    async def nothing(*args):
        return {}

    pool.gguf_sources = pool.refresh_worker_metrics = pool.refresh_holdings = pool.measure_links = nothing
    wanted = target()
    monkeypatch.setattr(orchestrator_module, "runtime", pool)
    monkeypatch.setattr(orchestrator_module, "build_target", lambda *args: wanted)
    response = TestClient(orchestrator_module.app).post("/admin/plan", json={"model_id": "m.gguf"})
    assert response.status_code == 200, response.text
    assert response.json()["parallel"] == wanted.slots > 1
    assert not [event for event in pool.events if "at once" in event["title"]], "a preview chose nothing"


def test_a_load_says_what_it_chose(pool):
    wanted = target()
    choose(pool, wanted)
    assert [event["detail"] for event in pool.events if event["title"] == f"{wanted.slots} at once"] == [wanted.parallel["reason"]]


def test_a_follower_takes_the_choice_with_the_route(pool, monkeypatch):
    leader_target = target()
    choose(pool, leader_target)
    state = {
        "request": leader_target.request.model_dump(),
        "revision": "r",
        "epoch": 3,
        "route": [item.model_dump() for item in pool.plan(leader_target)],
        "parallel": leader_target.parallel,
    }
    fresh = target()

    async def resolved(request):
        return fresh

    pool._resolve = resolved
    asyncio.run(pool._install(state))
    assert pool.active is fresh
    assert fresh.slots == leader_target.slots
    assert fresh.profile.kv_bytes_per_token == leader_target.profile.kv_bytes_per_token
    assert fresh.parallel == leader_target.parallel
    choose(pool, fresh)
    assert fresh.slots == leader_target.slots


def test_saved_state_carries_the_choice(pool, tmp_path):
    wanted = target()
    choose(pool, wanted)
    written = {}
    pool._write_state = written.update
    pool.save_state(wanted)
    assert written["parallel"]["slots"] == wanted.slots
    assert written["request"]["parallel"] is None


class FakeStage:
    last = False

    def forward(self, rows, position, slot, places, cuts, saves):
        return rows

    def forward_many(self, rows, positions, slots):
        return rows


def stage_runtime(monkeypatch, clock):
    """A stage whose clock moves only as the test says."""
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.adapter = runtime.assignment = object()
    runtime.gguf = FakeStage()
    runtime.compute_ms_ema = runtime.row_ms_ema = runtime.ring_counted = runtime.ring_busy = runtime.busy_share = None
    runtime.batch_ms = {}
    runtime.inference_calls = runtime.tokens_processed = runtime.prompt_tokens_computed = runtime.merged_passes = runtime.merged_rows = 0
    runtime._pass_entry = lambda request: SimpleNamespace(length=0)
    monkeypatch.setattr(worker_module.time, "perf_counter", lambda: clock[0])
    return runtime


def test_a_stage_counts_a_replys_pass_as_a_row_and_a_prompt_as_none(monkeypatch):
    clock = [0.0]
    runtime = stage_runtime(monkeypatch, clock)
    original = FakeStage.forward

    def timed(self, rows, *rest):
        clock[0] += 0.012 * len(rows)
        return original(self, rows, *rest)

    monkeypatch.setattr(FakeStage, "forward", timed)
    prompt = InferRequest(session_id="s", epoch=1, input_ids=[1, 2, 3], prompt_rows=True)
    runtime._run_local(np.ones((1, 3, 4), dtype=np.float32), prompt)
    assert runtime.row_ms_ema is None
    runtime._run_local(np.ones((1, 1, 4), dtype=np.float32), InferRequest(session_id="s", epoch=1, input_ids=[7]))
    assert runtime.row_ms_ema == pytest.approx(12.0)


def test_replies_that_share_a_pass_share_its_time(monkeypatch):
    clock = [0.0]
    runtime = stage_runtime(monkeypatch, clock)

    def timed(rows, positions, slots):
        clock[0] += 0.016
        return rows

    runtime.gguf.forward_many = timed
    steps = [SimpleNamespace(hidden=np.ones((1, 1, 4), dtype=np.float32), request=InferRequest(session_id="s", epoch=1, input_ids=[1], slot=index)) for index in range(4)]
    runtime._run_merged(steps)
    assert runtime.row_ms_ema == pytest.approx(4.0)


def test_the_engines_passes_count_as_rows_since_the_last_look():
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.row_ms_ema = runtime.ring_counted = runtime.ring_busy = runtime.busy_share = None
    runtime.batch_ms = {}
    runtime.note_ring({"passes": 10, "merged_passes": 0, "merged_rows": 0, "compute_us": 9000})
    assert runtime.row_ms_ema is None, "a first look has nothing to compare"
    # Four passes ran as one batch, averaged at 8 ms: 2 ms for each reply.
    runtime.note_ring({"passes": 14, "merged_passes": 1, "merged_rows": 4, "compute_us": 8000})
    assert runtime.row_ms_ema == 2.0
    runtime.note_ring({"passes": 14, "merged_passes": 1, "merged_rows": 4, "compute_us": 3000})
    assert runtime.row_ms_ema == 2.0, "with no passes since, the average is from before"
    runtime.note_ring({"passes": 3, "merged_passes": 0, "merged_rows": 0, "compute_us": 3000})
    assert runtime.row_ms_ema == 2.0, "an engine that started again is only looked at"
    runtime.note_ring(None)
    assert runtime.ring_counted == (3, 0, 0)


def test_the_engines_totals_give_exact_pass_times_and_how_busy_the_stage_was(monkeypatch):
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.row_ms_ema = runtime.ring_counted = runtime.ring_busy = runtime.busy_share = None
    runtime.batch_ms = {}
    clock = [100.0]
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: clock[0])
    runtime.note_ring({"passes": 10, "merged_passes": 0, "merged_rows": 0, "compute_us": 1, "batches_total": 10, "compute_us_total": 50_000})
    clock[0] += 1.0
    # 16 passes since, in 4 batches of 4 that took 120 ms between them, in one second.
    runtime.note_ring({"passes": 26, "merged_passes": 4, "merged_rows": 16, "compute_us": 1, "batches_total": 14, "compute_us_total": 170_000})
    assert runtime.batch_ms == {4: 30.0}
    assert runtime.row_ms_ema == 7.5
    assert runtime.busy_share == 0.12
    # An engine without the totals is read as before.
    runtime.ring_busy = None
    runtime.note_ring({"passes": 30, "merged_passes": 5, "merged_rows": 20, "compute_us": 8000})
    assert runtime.batch_ms[4] == pytest.approx(30.0 * 0.8 + 8.0 * 0.2)


def exact_runtime(exact=True, limit=0, verify=False, stages=2, merge=True, route_verifies=None):
    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.exact, runtime.merge_passes = exact, merge
    runtime.exactness = None if limit is None else {"exact_merge_limit": limit, "verify_exact": verify}
    runtime.route = [object()] * stages
    runtime.epoch, runtime.route_verifies = 5, route_verifies
    return runtime


def test_exact_replies_merge_no_more_than_the_probe_allowed():
    assert exact_runtime(exact=False).merge_most == MAX_PARALLEL
    assert exact_runtime(limit=0).merge_most == 1
    assert exact_runtime(limit=3).merge_most == 4
    assert exact_runtime(limit=None).merge_most == 1, "a stage that probed nothing merges nothing"
    assert exact_runtime(exact=False, merge=False).merge_most == 1


def test_exact_replies_check_guesses_in_one_pass_only_where_every_stage_rounds_the_same():
    sampled = 0.7
    assert exact_runtime(exact=False).guesses_in_one_pass(sampled)
    assert not exact_runtime(stages=1, verify=False).guesses_in_one_pass(sampled)
    assert exact_runtime(stages=1, verify=True).guesses_in_one_pass(sampled)
    assert not exact_runtime(stages=2, verify=True).guesses_in_one_pass(sampled), "the other stages have not been heard from"
    assert not exact_runtime(stages=2, verify=True, route_verifies=4).guesses_in_one_pass(sampled), "that was another route"
    assert exact_runtime(stages=2, verify=True, route_verifies=5).guesses_in_one_pass(sampled)
    assert not exact_runtime(stages=1, limit=None).guesses_in_one_pass(sampled)


def test_no_seeded_sampled_reply_checks_guesses_in_one_pass():
    """A checked pass accepts or rejects each guess with its own draws, so a seeded reply at a
    temperature comes out other than plain decoding even where the rounding is the same."""
    assert not exact_runtime(stages=1, verify=True).guesses_in_one_pass(0.8, seeded=True)
    assert not exact_runtime(exact=False, stages=1, verify=True).guesses_in_one_pass(0.8, seeded=True), "nor without --exact"
    assert exact_runtime(exact=False).guesses_in_one_pass(0.8, seeded=False), "an unseeded sampled reply may"


@pytest.mark.parametrize("exact", [False, True])
def test_a_reply_that_must_repeat_checks_guesses_in_one_pass_only_where_the_probe_proved_it(exact):
    """Temperature 0, or a seed at temperature 0, without --exact too: the way is chosen by timing,
    so a checked pass that rounds differently gave two texts for one request."""
    for seeded in (False, True):
        assert not exact_runtime(exact=exact, limit=None, stages=1).guesses_in_one_pass(0.0, seeded), "nothing was probed"
        assert not exact_runtime(exact=exact, stages=1, verify=False).guesses_in_one_pass(0.0, seeded)
        assert exact_runtime(exact=exact, stages=1, verify=True).guesses_in_one_pass(0.0, seeded)
        assert not exact_runtime(exact=exact, stages=3, verify=True).guesses_in_one_pass(0.0, seeded)
        assert exact_runtime(exact=exact, stages=3, verify=True, route_verifies=5).guesses_in_one_pass(0.0, seeded)


def test_a_merge_queue_hands_out_no_more_than_one_pass_may_hold():
    runtime = exact_runtime(limit=1)
    runtime.epoch, runtime.gguf = 1, object()
    loop = asyncio.new_event_loop()
    try:
        steps = [SimpleNamespace(request=InferRequest(session_id=str(slot), epoch=1, input_ids=[1], slot=slot), outcome=loop.create_future()) for slot in range(5)]
        runtime.merge_queue = list(steps)
        assert runtime._take_merge_queue() == steps[:2]
        assert runtime.merge_queue == steps[2:]
        runtime.exact = False
        assert runtime._take_merge_queue() == steps[2:]
    finally:
        loop.close()


def test_the_entry_keeps_what_it_hears_about_the_route_only_for_its_own_epoch(monkeypatch):
    runtime = worker_module.runtime
    monkeypatch.setattr(worker_module, "verify_token", lambda authorization: None)
    monkeypatch.setattr(runtime, "epoch", 7)
    monkeypatch.setattr(runtime, "route_verifies", None)
    monkeypatch.setattr(runtime, "route", [object(), object()])
    monkeypatch.setattr(runtime, "exactness", {"exact_merge_limit": 0, "verify_exact": True})
    client = TestClient(worker_module.app)
    assert client.post("/internal/route/exact", json={"epoch": 6, "verify_exact": True}).status_code == 409
    assert runtime.route_verifies is None and not runtime.route_verifies_exactly()
    assert client.post("/internal/route/exact", json={"epoch": 7, "verify_exact": True}).status_code == 200
    assert runtime.route_verifies == 7 and runtime.route_verifies_exactly()
    monkeypatch.setattr(runtime, "exactness", {"exact_merge_limit": 0, "verify_exact": False})
    assert not runtime.route_verifies_exactly(), "its own stage must have proved it too"
    assert client.post("/internal/route/exact", json={"epoch": 7, "verify_exact": False}).status_code == 200
    assert runtime.route_verifies is None


def test_a_model_says_whether_it_has_a_thinking_switch(pool):
    # Qwen3's template reads enable_thinking. The 235B Instruct release never reasons and has no such line.
    pool.tokenizer = SimpleNamespace(tool_template="{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}")
    assert pool.thinking_switch() and pool.model_kind()["thinking"] is True
    pool.tokenizer = SimpleNamespace(tool_template="{{- '<|im_start|>assistant' }}")
    assert not pool.thinking_switch() and pool.model_kind() == {"kind": "chat", "vision": False, "thinking": False}
    pool.tokenizer = None
    assert not pool.thinking_switch()


def test_a_pass_costs_what_was_measured_with_straight_lines_between():
    cost = pass_cost(10.0, {1: 4.0, 2: 5.0, 8: 11.0})
    assert [cost(replies) for replies in (1, 2, 5, 8)] == [4.0, 5.0, 8.0, 11.0]
    # Past the largest pass measured, a reply adds what one added on average up to there.
    assert cost(10) == 13.0


def test_a_pass_with_nothing_measured_past_one_reply_costs_a_row_for_each_more():
    assert pass_cost(10.0, None)(3) == 30.0
    assert pass_cost(10.0, {1: 4.0})(3) == 24.0


def test_a_noisy_reading_never_makes_a_larger_pass_the_cheaper_one():
    cost = pass_cost(10.0, {1: 4.0, 2: 6.0, 4: 5.0})
    assert (cost(2), cost(3), cost(4)) == (6.0, 6.0, 6.0)


def test_what_a_load_timed_carries_a_stage_past_the_passes_its_replies_measured():
    timed = {1: 4.0, 2: 5.0, 4: 6.0, 8: 8.0, 16: 12.0}
    # Two slots ran, and with the whole route working a pass took half as long again.
    assert pass_curve({1: 5.0, 2: 7.5}, timed) == {1: 5.0, 2: 7.5, 4: 9.0, 8: 12.0, 16: 18.0}
    assert pass_curve(None, timed) == timed
    assert pass_curve({1: 5.0, 2: 7.5}, None) == {1: 5.0, 2: 7.5}
    assert pass_curve({2: 7.5}, {2: 5.0, 4: 6.0}) == {2: 7.5}, "a load that timed no pass of one row has no shape to give"
    # Replies that never ran alone still get a pass of one row, and it is no dearer than theirs.
    assert pass_curve({4: 9.0}, timed) == {1: 6.0, 4: 9.0, 8: 12.0, 16: 18.0}
    assert pass_curve({4: 5.0}, {1: 4.0, 4: 4.0, 8: 8.0})[1] == 5.0


def test_a_stage_no_reply_has_reached_still_says_what_its_load_timed(pool):
    item = target()
    choose(pool, item)
    route = running(pool, item, [None, None])
    devices = stage_devices(route, pool.offered)
    assert pool.measured_rows(item, route, devices) is None
    for placed in route:
        pool.workers[placed.worker_id].health["telemetry"]["pass_probe"] = {"1": 8.0, "2": 10.0, "4": 12.0}
    pool.note_rows()
    assert pool.row_readings == {}
    assert pool.measured_rows(item, route, devices) == [8.0, 8.0]
    assert pool.measured_passes(item, route, devices) == [{1: 8.0, 2: 10.0, 4: 12.0}] * 2


def test_what_another_route_timed_at_load_does_not_count(pool):
    item = target()
    choose(pool, item)
    route = running(pool, item, [None, None], epoch=4)
    for placed in route:
        pool.workers[placed.worker_id].health["telemetry"]["pass_probe"] = {"1": 8.0}
    pool.note_rows()
    assert pool.pass_probes == {}


def test_replies_that_share_passes_open_slots_on_one_device_too():
    # One GPU stage as measured with passes gathered: 16 replies took 8.6 ms a round, one alone 3.6.
    lone = [stage("a", 0, 8, 3.6)]
    passes = [{1: 3.6, 2: 4.2, 4: 5.3, 8: 6.9, 16: 8.6}]
    chosen = wanted_slots(lone, ["gpu"], {}, True, [3.6], passes, merge_most=64)
    assert chosen["merging"] is True and chosen["devices"] == 1 and chosen["wanted"] == 16
    # A route that merges nothing has its replies take turns, as before.
    assert wanted_slots(lone, ["gpu"], {}, True, [3.6], passes, merge_most=1)["wanted"] == 1
    # Two stages on one GPU share it, and their merged passes still add up to less than turns.
    both = [stage("a", 0, 4, 2.0), stage("b", 4, 8, 12.0)]
    merged = [{1: 2.0, 8: 3.0}, {1: 12.0, 8: 20.0}]
    assert wanted_slots(both, ["gpu", "gpu"], {("a", "b"): 6.0}, True, [2.0, 12.0], merged, merge_most=64)["wanted"] == 16


def test_a_short_trip_opens_slots_only_where_a_merged_pass_costs_less_than_its_rows():
    route = [stage("a", 0, 4, 2.0), stage("b", 4, 8, 12.0)]
    links = {("a", "b"): 6.0}
    assert wanted_slots(route, ["a", "b"], links, True, [2.0, 12.0], [{1: 2.0, 8: 3.0}, {1: 12.0, 8: 20.0}], merge_most=64)["wanted"] == 16
    # A stage whose pass of eight costs eight rows gains nothing from running them together.
    assert wanted_slots(route, ["a", "b"], links, True, [2.0, 12.0], [{1: 2.0, 8: 3.0}, {1: 12.0, 8: 184.0}], merge_most=64)["wanted"] == 1


def test_a_pool_that_has_run_replies_is_sized_for_twice_the_most_it_ran_at_once(pool):
    fresh = target()
    choose(pool, fresh)
    assert fresh.slots == 32, "a pool that has run nothing has no say"
    pool.replies_peak = 6
    busy = target()
    choose(pool, busy)
    assert busy.slots == 12 and "12 is twice the most replies this pool has run at once" in busy.parallel["reason"]
    pool.replies_peak = 1
    quiet = target()
    choose(pool, quiet)
    assert quiet.slots == 8 and "has run no more than 1 at once, so it takes 8" in quiet.parallel["reason"]
    pool.replies_peak = 40
    crowd = target()
    choose(pool, crowd)
    assert crowd.slots == 32, "more demand than the route wants changes nothing"
