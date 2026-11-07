"""How many replies a load runs at once, when the operator leaves that open."""

import asyncio
import math
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from isharded import orchestrator as orchestrator_module
from isharded.models import UnsupportedModel
from isharded.orchestrator import ModelTarget, Orchestrator, describe_plan, draft_cache_bytes, settled_parallel, stage_devices, wanted_slots, with_slots
from isharded.planner import Candidate, ModelProfile
from isharded.protocol import MAX_PARALLEL, DraftModel, LoadModelRequest, MemoryPool, StageAssignment, TensorRef, UnitPlacement, WeightUnit

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
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 50.0}, True) == {"round_trip_ms": 82.0, "row_ms": 20.0, "devices": 2, "wanted": 4}


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
    assert wanted_slots(route, ["a", "b"], {("a", "b"): 0.0}, True)["row_ms"] == 4 * orchestrator_module.LAYER_FLOOR_MS


def test_an_unmeasured_link_says_nothing():
    route = [stage("a", 0, 4, 10.0), stage("b", 4, 8, 20.0)]
    assert wanted_slots(route, ["a", "b"], {}, True) is None
    assert wanted_slots(route, ["a", "b"], {("b", "a"): 1.0}, False) is not None


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
    adapter = SimpleNamespace(num_layers=3, kv_bytes_per_token=lambda layer, itemsize: (layer + 1) * itemsize)
    monkeypatch.setattr(orchestrator_module, "create_adapter", lambda config: adapter)
    assert draft_cache_bytes(SimpleNamespace(config={})) == 12


def test_a_draft_of_an_unknown_family_still_loads(monkeypatch):
    def refuse(config):
        raise UnsupportedModel("no")

    monkeypatch.setattr(orchestrator_module, "create_adapter", refuse)
    assert draft_cache_bytes(SimpleNamespace(config={})) == 0


def test_a_load_with_room_takes_what_the_route_wants(pool):
    wanted = target()
    choose(pool, wanted)
    route = pool.plan(wanted)
    assert wanted.slots == wanted.parallel["slots"] == wanted.parallel["wanted"] > 1
    assert wanted.slots == math.floor(wanted.parallel["round_trip_ms"] / wanted.parallel["row_ms"])
    assert wanted.profile.kv_bytes_per_token == [KV_PER_TOKEN * wanted.slots] * LAYERS
    assert str(wanted.slots) in wanted.parallel["reason"]
    assert {item.worker_id for item in route} == {"a", "b"}


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
    pool.offered = [worker("a", 2 * GIB), worker("b", 2 * GIB)]
    plain = target()
    choose(pool, plain)
    draft = DraftModel(source="d", file="d.gguf", file_size=1, revision="r", kv_bytes_per_token=128 * 1024)
    drafted = target(draft=draft)
    choose(pool, drafted)
    assert drafted.slots < plain.slots
    assert drafted.profile.entry_extra_bytes >= 128 * 1024 * 1024 * (drafted.slots + 1)


def test_one_slot_plans_exactly_as_it_always_has():
    draft = DraftModel(source="d", file="d.gguf", file_size=1, revision="r", kv_bytes_per_token=128 * 1024)
    drafted = target(draft=draft)
    with_slots(drafted, 1)
    assert drafted.profile.entry_extra_bytes == orchestrator_module.entry_bytes(draft, None)


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
