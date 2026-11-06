import asyncio
import json
from pathlib import Path
import time

import httpx
import pytest

from isharded import orchestrator as orchestrator_module
from isharded.node import Service
from isharded.orchestrator import Orchestrator, WorkerRecord
from isharded.planner import ModelProfile, _stage_items
from isharded.protocol import (
    PROTOCOL_VERSION,
    HardwareInfo,
    LoadModelRequest,
    SessionState,
    StageAssignment,
    TensorRef,
    WeightUnit,
    WorkerRegistration,
)


class FakeProcess:
    pid = 1

    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def service_with(clock, processes):
    return Service("test", lambda: processes.pop(0), clock)


def test_a_dead_service_comes_back_after_a_delay():
    clock = Clock()
    processes = [FakeProcess(code=1), FakeProcess()]
    service = service_with(clock, processes)
    service.start()
    service.supervise()
    assert service.restart_at == clock.now + 2
    service.supervise()
    assert processes, "must not restart before the delay"
    clock.now += 2
    service.supervise()
    assert not processes and service.process.poll() is None


def test_a_crash_loop_backs_off():
    clock = Clock()
    service = service_with(clock, [FakeProcess(code=1) for _ in range(10)])
    service.start()
    delays = []
    for _ in range(9):
        service.supervise()
        delays.append(service.restart_at - clock.now)
        clock.now = service.restart_at
        service.supervise()
    assert delays[:4] == [2, 4, 8, 16]
    assert max(delays) == Service.LONGEST_DELAY_SECONDS


def test_a_service_that_stayed_up_starts_its_backoff_over():
    clock = Clock()
    service = service_with(clock, [FakeProcess(code=1), FakeProcess(code=1), FakeProcess(code=1)])
    service.start()
    service.supervise()
    clock.now = service.restart_at
    service.supervise()
    clock.now += Service.HEALTHY_AFTER_SECONDS + 1
    service.supervise()
    assert service.restart_at == clock.now + 2


def registration(worker_id="w1"):
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=4, ram_total_bytes=1)
    return WorkerRegistration(worker_id=worker_id, advertised_url=f"http://{worker_id}:1", hardware=hardware, protocol_version=PROTOCOL_VERSION)


def record(worker_id="w1", **fields):
    return WorkerRecord(registration=registration(worker_id), last_seen=time.monotonic(), **fields)


def pool(**records):
    orchestrator = Orchestrator()
    orchestrator.workers = records
    return orchestrator


def done(orchestrator):
    asyncio.run(orchestrator.close())


def test_a_quarantined_worker_returns_after_staying_healthy():
    worker = record(quarantined=True, quarantine_count=1, health={"status": "ok"})
    orchestrator = pool(w1=worker)
    for _ in range(orchestrator_module.PROBATION_PASSES - 1):
        orchestrator._review_quarantine()
        assert worker.quarantined
    orchestrator._review_quarantine()
    assert not worker.quarantined
    done(orchestrator)


def test_one_bad_check_restarts_probation():
    worker = record(quarantined=True, quarantine_count=1, health={"status": "ok"})
    orchestrator = pool(w1=worker)
    orchestrator._review_quarantine()
    worker.health = {"status": "unreachable"}
    orchestrator._review_quarantine()
    assert worker.probation_passes == 0
    done(orchestrator)


def test_repeat_offenders_wait_longer():
    worker = record(quarantined=True, quarantine_count=3, health={"status": "ok"})
    orchestrator = pool(w1=worker)
    for _ in range(orchestrator_module.PROBATION_PASSES * 4 - 1):
        orchestrator._review_quarantine()
    assert worker.quarantined
    orchestrator._review_quarantine()
    assert not worker.quarantined
    done(orchestrator)


def stage(worker_id, start=0, end=1):
    return StageAssignment(worker_id=worker_id, url=f"http://{worker_id}:1", start_layer=start, end_layer=end)


def repairing_pool(**records):
    orchestrator = pool(**records)
    orchestrator.active = object()
    orchestrator.route = [stage(name) for name in records]
    # These workers have been gone long enough. Grace has its own test.
    orchestrator.absent_since = {name: -1e9 for name in records}
    calls = []

    async def rebalance(**kwargs):
        calls.append(kwargs)
        orchestrator.route = [item for item in orchestrator.route if orchestrator.workers[item.worker_id].live]
        return orchestrator.route

    orchestrator.rebalance = rebalance
    return orchestrator, calls


def test_a_lost_stage_is_replaced_while_idle():
    dead = record("w2")
    dead.last_seen -= 60
    orchestrator, calls = repairing_pool(w1=record("w1"), w2=dead)
    asyncio.run(orchestrator._repair_route())
    assert calls
    assert "w2" in orchestrator.lost_from_route
    done(orchestrator)


def test_a_short_absence_moves_no_weights():
    dead = record("w2")
    dead.last_seen -= 60
    orchestrator, calls = repairing_pool(w1=record("w1"), w2=dead)
    orchestrator.absent_since = {}
    asyncio.run(orchestrator._repair_route())
    assert not calls and not orchestrator.lost_from_route
    dead.last_seen = time.monotonic()
    asyncio.run(orchestrator._repair_route())
    assert orchestrator.absent_since == {}, "back in time: the clock starts over"
    done(orchestrator)


def test_a_worker_that_restarted_empty_gets_its_stage_back():
    worker = record("w1", health={"status": "ok", "epoch": 0, "model": None})
    orchestrator, calls = repairing_pool(w1=worker)
    orchestrator.absent_since = {}
    orchestrator.epoch = 5
    asyncio.run(orchestrator._repair_route())
    assert calls, "live is not enough: it no longer holds its stage"
    done(orchestrator)


def test_a_worker_on_the_current_route_needs_nothing():
    worker = record("w1", health={"status": "ok", "epoch": 5, "model": "m"})
    orchestrator, calls = repairing_pool(w1=worker)
    orchestrator.absent_since = {}
    orchestrator.epoch = 5
    asyncio.run(orchestrator._repair_route())
    assert not calls
    done(orchestrator)


def stream_lines(*events):
    return httpx.Response(200, content="".join(json.dumps(event) + "\n" for event in events).encode())


def reply_through(orchestrator, handler, tokens=(1, 2), limit=6):
    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = orchestrator.route or [stage("w1")]

        class Stops:
            stop_token_ids = {99}

        orchestrator.tokenizer = Stops()
        session = SessionState(tokens=list(tokens), epoch=orchestrator.epoch, needs_reset=True)
        got = []
        async for token in orchestrator.token_stream(session, limit, 0.0, 1.0, 0, None):
            got.append(token)
            session.tokens.append(token)
            if token == 99:
                break
        await orchestrator.close()
        return got

    return asyncio.run(scenario())


def test_a_restarted_stage_resumes_through_a_new_stream():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream_lines({"token_id": 5}, {"error": {"retry": True, "message": "stage died"}})
        return stream_lines({"token_id": 6}, {"token_id": 99})

    orchestrator = pool(w1=record())
    assert reply_through(orchestrator, handler) == [5, 6, 99]
    assert requests[1]["input_ids"] == [1, 2, 5], "the new stream carries what was already said"
    assert requests[1]["max_tokens"] == 5
    assert requests[1]["checkpoint_at"] is None
    assert not orchestrator.workers["w1"].quarantined


def test_a_dead_stage_is_repaired_and_the_reply_resumes():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream_lines({"error": {"failed_worker": "w2", "message": "gone"}})
        return stream_lines({"token_id": 99})

    orchestrator = pool(w1=record(), w2=record("w2"))
    repairs = []

    async def rebalance():
        repairs.append(True)

    orchestrator.rebalance = rebalance
    assert reply_through(orchestrator, handler) == [99]
    assert repairs and orchestrator.workers["w2"].quarantined
    assert not orchestrator.workers["w1"].quarantined


class FailingTarget:
    """Just enough of a target for the deploy loop."""

    profile = type("Profile", (), {"total_bytes": 10})()


def deploy_pool(monkeypatch, outcomes, moved_bytes):
    orchestrator = pool(w1=record("w1", health={"status": "ok"}), w2=record("w2", health={"status": "ok"}))
    monkeypatch.setattr(orchestrator_module, "ACTIVATE_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(orchestrator_module, "describe_plan", lambda target, route: {"download_bytes": moved_bytes})
    attempts = []

    async def nothing(*args):
        return None

    async def prefetch(*args):
        return {}

    async def activate(target, route):
        attempts.append(len(attempts))
        return outcomes.pop(0)

    orchestrator.refresh_worker_metrics = nothing
    orchestrator.refresh_holdings = nothing
    orchestrator._prefetch = prefetch
    orchestrator._activate = activate
    orchestrator.plan = lambda target: [stage("w1")]
    return orchestrator, attempts


def run_deploy(orchestrator):
    job = orchestrator_module.LoadJob(job_id="j", model_id="m")
    return asyncio.run(orchestrator.deploy(FailingTarget(), job, generation_lock_held=True))


def test_a_passing_load_failure_is_retried_without_blame(monkeypatch):
    orchestrator, attempts = deploy_pool(monkeypatch, [{"w1": "out of memory"}, {}], 0)
    assert run_deploy(orchestrator)
    assert len(attempts) == 2
    assert not orchestrator.workers["w1"].quarantined


def test_a_live_node_failing_never_moves_the_whole_model(monkeypatch):
    failure = {"w1": "out of memory"}
    orchestrator, attempts = deploy_pool(monkeypatch, [failure] * 3, 15 * 1024**3)
    with pytest.raises(orchestrator_module.HTTPException) as caught:
        run_deploy(orchestrator)
    assert caught.value.status_code == 503
    assert "15.0 GiB" in caught.value.detail and "out of memory" in caught.value.detail
    assert not orchestrator.workers["w1"].quarantined


def test_a_small_replan_around_a_failing_node_goes_ahead(monkeypatch):
    failure = {"w1": "out of memory"}
    orchestrator, attempts = deploy_pool(monkeypatch, [failure, failure, failure, {}], 1024)
    assert run_deploy(orchestrator)
    assert orchestrator.workers["w1"].quarantined


def test_a_dead_node_is_replanned_however_much_moves(monkeypatch):
    failure = {"w1": "gone"}
    orchestrator, attempts = deploy_pool(monkeypatch, [failure, failure, failure, {}], 15 * 1024**3)
    orchestrator.workers["w1"].health = {"status": "unreachable"}
    assert run_deploy(orchestrator)
    assert orchestrator.workers["w1"].quarantined


def test_failures_reach_the_log(capsys):
    orchestrator = pool()
    orchestrator.add_event("failure", "Load failed", "the reason")
    orchestrator.add_event("join", "w joined", "noise")
    output = capsys.readouterr().out
    assert "Load failed: the reason" in output and "noise" not in output
    done(orchestrator)


def test_a_route_change_blames_nobody():
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            orchestrator.epoch += 1
            return httpx.Response(409, json={"detail": "pipeline epoch mismatch"})
        return httpx.Response(200, json={"token_id": 3})

    orchestrator = pool(w1=record())

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = [stage("w1")]
        session = SessionState(tokens=[1, 2], epoch=orchestrator.epoch, needs_reset=False)
        token = await orchestrator.next_token(session, 0.0, 1.0, 0, None)
        await orchestrator.close()
        return token

    assert asyncio.run(scenario()) == 3
    assert not orchestrator.workers["w1"].quarantined
    assert attempts[1]["reset"] is True


def test_several_route_changes_in_a_row_do_not_fail_a_reply():
    answers = iter([409, 409, 409, 409, 200])

    def handler(request):
        code = next(answers)
        if code == 409:
            orchestrator.epoch += 1
            return httpx.Response(409, json={"detail": "pipeline epoch mismatch"})
        return httpx.Response(200, json={"token_id": 3})

    orchestrator = pool(w1=record())

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = [stage("w1")]
        session = SessionState(tokens=[1], epoch=orchestrator.epoch, needs_reset=False)
        token = await orchestrator.next_token(session, 0.0, 1.0, 0, None)
        await orchestrator.close()
        return token

    assert asyncio.run(scenario()) == 3


def test_a_new_route_for_the_same_model_reads_nothing(monkeypatch):
    orchestrator = pool()
    request = LoadModelRequest(model_id="m")

    class Held:
        manifest = type("Manifest", (), {"revision": "r"})()

    Held.request = request
    orchestrator.active = Held()

    async def resolve(_):
        raise AssertionError("the model has not changed")

    orchestrator._resolve = resolve
    state = {"request": request.model_dump(), "revision": "r", "epoch": 9, "route": [stage("w9").model_dump()]}
    asyncio.run(orchestrator._install(state))
    assert orchestrator.epoch == 9 and orchestrator.route[0].worker_id == "w9"
    done(orchestrator)


def test_a_follower_reports_a_failure_and_waits_for_the_repair(monkeypatch):
    from isharded.mesh import ControlMesh

    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/internal/report":
            orchestrator.epoch += 1
            return httpx.Response(200, json={"status": "repairing"})
        if len(seen) == 1:
            return httpx.Response(502, json={"detail": {"failed_worker": "w2", "message": "gone"}})
        return httpx.Response(200, json={"token_id": 8})

    orchestrator = pool(w1=record())
    orchestrator.mesh = ControlMesh("me", "http://me", ())
    orchestrator.mesh.leader_id = "boss"
    orchestrator.mesh.observe({"node_id": "boss", "url": "http://boss", "term": 1, "leader_id": "boss"})

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = [stage("w1")]
        session = SessionState(tokens=[1], epoch=orchestrator.epoch, needs_reset=False)
        token = await orchestrator.next_token(session, 0.0, 1.0, 0, None)
        await orchestrator.close()
        return token

    assert asyncio.run(scenario()) == 8
    assert seen == ["/v1/infer", "/internal/report", "/v1/infer"]
    assert not orchestrator.workers["w1"].quarantined, "only the coordinator judges workers"


def test_heartbeat_answers_with_the_control_planes():
    from fastapi.testclient import TestClient

    orchestrator_module.runtime.register(registration())
    client = TestClient(orchestrator_module.app)
    response = client.post("/internal/heartbeat", json={"worker_id": "w1"})
    assert response.status_code == 200
    assert response.json()["controls"] == []


def test_an_unchanged_stage_keeps_its_weights():
    from isharded import worker
    from isharded.models import GgufAdapter
    from isharded.protocol import ConfigureRequest, UnitPlacement

    assignment = stage("w").model_copy(update={"placements": [UnitPlacement(unit="u", pool_id="p", bytes=1)]})
    request = ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=2, assignment=assignment, route=[assignment])
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf = object()
    runtime.gguf_signature = runtime._stage_signature(request)
    runtime.stages_kept = 0
    adapter = GgufAdapter.__new__(GgufAdapter)
    runtime._materialize(request.model_copy(update={"epoch": 3}), adapter)
    assert runtime.stages_kept == 1
    assert runtime.gguf is not None


def test_a_moved_stage_reloads():
    from isharded import worker
    from isharded.protocol import ConfigureRequest

    def request(end):
        assignment = stage("w", 0, end)
        return ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=2, assignment=assignment, route=[assignment])

    signature = worker.WorkerRuntime._stage_signature
    assert signature(request(4)) != signature(request(8))
    assert signature(request(4)) == signature(request(4))


def test_a_drain_waits_for_the_reply_and_gives_up_on_time():
    from isharded.slots import SlotGate

    async def scenario():
        gate = SlotGate(1)
        slot = await gate.acquire()
        assert await gate.hold(0.01) is False, "a reply is still running"
        assert not gate.held, "a failed drain must not block new replies"
        await gate.release(slot)
        assert await gate.hold(0.01) is True
        waiter = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0.01)
        assert not waiter.done(), "no new reply may start"
        await gate.resume()
        assert await waiter == 0

    asyncio.run(scenario())


def test_slots_are_handed_out_and_returned():
    from isharded.slots import SlotGate

    async def scenario():
        gate = SlotGate(2)
        first, second = await gate.acquire(), await gate.acquire()
        assert {first, second} == {0, 1}
        third = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0.01)
        assert not third.done(), "the third reply queues"
        await gate.release(first)
        assert await third == first

    asyncio.run(scenario())


def test_shrinking_slots_forgets_the_ones_gone():
    from isharded.slots import SlotGate

    async def scenario():
        gate = SlotGate(3)
        slot = await gate.acquire(max)
        gate.resize(1)
        await gate.release(slot)
        assert gate.free == {0}

    asyncio.run(scenario())


def test_a_conversation_returns_to_the_sequence_that_holds_it():
    from isharded import worker

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.checkpoints = {0: ("a", [1, 2]), 1: ("b", [7, 8, 9])}
    runtime.slot_used_at = {0: 5.0, 1: 9.0, 2: 1.0}
    assert runtime.choose_slot({0, 1, 2}, [7, 8, 9, 10]) == 1
    assert runtime.choose_slot({0, 1, 2}, [5, 5]) == 2, "otherwise the one idle longest"


def test_repairs_are_rate_limited():
    dead = record("w2")
    dead.last_seen -= 60
    orchestrator, calls = repairing_pool(w1=record("w1"), w2=dead)
    asyncio.run(orchestrator._repair_route())
    asyncio.run(orchestrator._repair_route())
    assert len(calls) == 1
    done(orchestrator)


def test_a_returned_worker_triggers_a_rebalance():
    orchestrator, calls = repairing_pool(w1=record("w1"))
    orchestrator.workers["w2"] = record("w2")
    orchestrator.lost_from_route = {"w2"}
    asyncio.run(orchestrator._repair_route())
    assert calls
    assert "w2" not in orchestrator.lost_from_route
    done(orchestrator)


def test_a_restarted_stage_is_replayed_and_nobody_is_blamed():
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(503, json={"detail": {"retry": True, "message": "stage died"}})
        return httpx.Response(200, json={"token_id": 7})

    async def scenario():
        orchestrator = pool(w1=record())
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = [stage("w1")]
        session = SessionState(tokens=[1, 2, 3], epoch=orchestrator.epoch, needs_reset=False)
        token = await orchestrator.next_token(session, 0.0, 1.0, 0, None)
        await orchestrator.close()
        return orchestrator, token

    orchestrator, token = asyncio.run(scenario())
    assert token == 7
    assert not orchestrator.workers["w1"].quarantined
    assert attempts[1]["reset"] is True and attempts[1]["input_ids"] == [1, 2, 3]


def test_other_errors_are_not_mistaken_for_a_restart():
    assert not Orchestrator._asks_for_replay(httpx.Response(503, json={"detail": "busy"}))
    assert not Orchestrator._asks_for_replay(httpx.Response(500, json={"detail": {"retry": True}}))
    assert not Orchestrator._asks_for_replay(httpx.Response(503, content=b"not json"))


def saved(tmp_path, monkeypatch, epoch=5):
    path = tmp_path / "state.json"
    monkeypatch.setenv("ISHARDED_STATE_FILE", str(path))
    state = {
        "request": LoadModelRequest(model_id="m").model_dump(),
        "revision": "abc",
        "epoch": epoch,
        "route": [stage("w1").model_dump()],
    }
    path.write_text(json.dumps(state), encoding="utf-8")
    orchestrator = Orchestrator()
    orchestrator.load_saved_state()
    return orchestrator


def test_saved_state_restores_the_epoch(tmp_path, monkeypatch):
    orchestrator = saved(tmp_path, monkeypatch)
    assert orchestrator.epoch == 5
    assert orchestrator.saved_state["revision"] == "abc"
    done(orchestrator)


def test_corrupt_saved_state_is_survived(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("ISHARDED_STATE_FILE", str(path))
    orchestrator = Orchestrator()
    orchestrator.load_saved_state()
    assert orchestrator.saved_state is None
    done(orchestrator)


class Target:
    model_id = "m"
    tokenizer = "tokenizer"
    projector = None


def adopting(monkeypatch, orchestrator):
    monkeypatch.setattr(orchestrator_module, "build_target", lambda *args: Target())
    monkeypatch.setattr(orchestrator_module, "load_tokenizer", lambda *args: "tokenizer")

    async def nobody():
        return {}

    orchestrator.gguf_sources = nobody


def test_workers_still_serving_are_adopted_without_a_reload(tmp_path, monkeypatch):
    orchestrator = saved(tmp_path, monkeypatch)
    adopting(monkeypatch, orchestrator)
    orchestrator.workers = {"w1": record(health={"status": "ok", "model": "m", "epoch": 5})}
    asyncio.run(orchestrator._adopt_saved_route())
    assert orchestrator.route and orchestrator.route[0].worker_id == "w1"
    assert orchestrator.model_id == "m"
    assert orchestrator.saved_state is None
    done(orchestrator)


def test_a_route_is_not_adopted_while_a_worker_is_missing(tmp_path, monkeypatch):
    orchestrator = saved(tmp_path, monkeypatch)
    adopting(monkeypatch, orchestrator)
    orchestrator.workers = {"w1": record(health={"status": "ok", "model": "m", "epoch": 4})}
    asyncio.run(orchestrator._adopt_saved_route())
    assert not orchestrator.route
    assert orchestrator.saved_state is not None
    done(orchestrator)


def test_the_model_reloads_once_workers_have_had_their_chance(tmp_path, monkeypatch):
    orchestrator = saved(tmp_path, monkeypatch)
    adopting(monkeypatch, orchestrator)
    orchestrator.workers = {"w1": record(health={"status": "ok", "model": None, "epoch": 0})}
    orchestrator.saved_state_seen_at -= orchestrator_module.ADOPTION_GRACE_SECONDS + 1
    orchestrator.candidates = lambda: ["someone"]
    loads = []

    async def load(request, job):
        loads.append(request.model_id)
        return []

    orchestrator.load = load
    asyncio.run(orchestrator._adopt_saved_route())
    assert loads == ["m"]
    assert orchestrator.saved_state is None
    done(orchestrator)


def test_saving_state_is_atomic_and_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "deep" / "state.json"
    monkeypatch.setenv("ISHARDED_STATE_FILE", str(path))
    orchestrator = Orchestrator()
    orchestrator.epoch = 3
    orchestrator.route = [stage("w1")]

    class Saved:
        request = LoadModelRequest(model_id="m")

        class manifest:
            revision = "r"

    orchestrator.save_state(Saved())
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 3
    assert not list(path.parent.glob("*.part"))
    done(orchestrator)


def test_a_node_on_another_protocol_is_refused():
    from fastapi.testclient import TestClient

    client = TestClient(orchestrator_module.app)
    body = json.loads(registration().model_dump_json())
    body["protocol_version"] = PROTOCOL_VERSION - 1
    response = client.post("/internal/register", json=body)
    assert response.status_code == 426
    assert "update the older one" in response.json()["detail"]


def test_draft_memory_is_planned_on_the_entry_stage():
    def unit(name, kind, layer=None):
        return WeightUnit(name=name, kind=kind, layer=layer, tensors=[TensorRef(name=name, file="f", dtype="U8", shape=[10], start=0, end=10)])

    profile = ModelProfile(
        units=[unit("embed", "embed"), unit("head", "head"), unit("layer-00000.dense", "dense", 0), unit("layer-00001.dense", "dense", 1)],
        num_layers=2,
        expert_active_fraction=1.0,
        kv_bytes_per_token=[0, 0],
        state_bytes=[0, 0],
        entry_extra_bytes=500,
    )
    entry = {item.unit.name: item.extra_bytes for item in _stage_items(profile, 0, 1)}
    later = {item.unit.name: item.extra_bytes for item in _stage_items(profile, 1, 2)}
    assert entry["layer-00000.dense"] == 500
    assert later["layer-00001.dense"] == 0


def test_a_mismatched_llama_build_is_refused():
    from isharded.llamacpp import ContextParams, LlamaError, LlamaLibrary, ModelParams

    class Native:
        def __init__(self, context_size):
            self.size = context_size

        def llama_model_default_params(self):
            return ModelParams(n_gpu_layers=-1, main_gpu=0, use_extra_bufts=True, load_mtp=False)

        def llama_context_default_params(self):
            return ContextParams(n_ctx=self.size, n_batch=2048, n_ubatch=512, n_seq_max=1, yarn_ext_factor=-1.0, offload_kqv=True)

    library = LlamaLibrary.__new__(LlamaLibrary)
    library.llama = Native(512)
    library._check_binding()
    library.llama = Native(99)
    with pytest.raises(LlamaError, match="does not match"):
        library._check_binding()


def test_cordoning_a_node_moves_it_off_while_idle():
    orchestrator, calls = repairing_pool(**{"a-gpu": record("a-gpu"), "b-gpu": record("b-gpu")})
    orchestrator.access.cordon("a")
    assert orchestrator._absent_from_route() == {"a-gpu"}
    assert not orchestrator._cordoned("a-b-gpu")
    assert orchestrator._cordoned("a-cpu") and orchestrator._cordoned("a-memory")
    orchestrator.route = [stage("b-gpu")]
    orchestrator.lost_from_route = {"a-gpu"}
    orchestrator.absent_since = {}
    asyncio.run(orchestrator._repair_route())
    assert not calls  # still cordoned, so it has not "returned"
    done(orchestrator)


def test_a_cordon_says_what_it_would_cost(monkeypatch):
    orchestrator = pool(w1=record("w1"))
    orchestrator.active = object()
    orchestrator.route = [stage("w1")]
    monkeypatch.setattr(orchestrator_module, "describe_plan", lambda target, route: {"download_bytes": 7})

    def plan(target):
        assert "w1" in orchestrator.access.cordoned
        return []

    orchestrator.plan = plan
    assert orchestrator.cost_of_losing("w1") == 7
    assert not orchestrator.access.cordoned

    def no_room(target):
        raise orchestrator_module.PlacementError("no room")

    orchestrator.plan = no_room
    assert orchestrator.cost_of_losing("w1") is None
    done(orchestrator)


def test_a_cordon_that_moves_the_whole_model_needs_force(monkeypatch):
    from fastapi.testclient import TestClient

    orchestrator = pool(w1=record("w1"))
    orchestrator.active = None
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    monkeypatch.setattr(orchestrator, "cost_of_losing", lambda name: 3 * orchestrator_module.REPLAN_MOVE_LIMIT_BYTES)
    client = TestClient(orchestrator_module.app)
    refused = client.post("/admin/nodes/w1/cordon")
    assert refused.status_code == 409 and "GiB" in refused.json()["detail"]
    assert not orchestrator.access.cordoned
    assert client.post("/admin/nodes/w1/cordon", params={"force": "true"}).status_code == 200
    assert "w1" in orchestrator.access.cordoned


def test_putting_a_node_back_stops_a_move_still_under_way(monkeypatch):
    from fastapi.testclient import TestClient

    orchestrator = pool(w1=record("w1"))
    orchestrator.access.cordon("w1")
    orchestrator.job = orchestrator_module.LoadJob(job_id="j", model_id="m")
    orchestrator.job.phase = "fetching"
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    assert TestClient(orchestrator_module.app).post("/admin/nodes/w1/uncordon").status_code == 200
    assert orchestrator.job.cancelled


def test_an_event_that_cannot_be_printed_fails_nothing(monkeypatch):
    import io
    import sys

    orchestrator = pool()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    orchestrator.add_event("failure", "Route", "a → b")
    assert orchestrator.events[0]["detail"] == "a → b"
    monkeypatch.undo()
    done(orchestrator)


def test_a_silent_stage_is_blamed_by_the_nearest_hop():
    from isharded.protocol import pass_seconds

    assert pass_seconds(1, 1) < pass_seconds(1, 2) < pass_seconds(1, 3)
    assert pass_seconds(1000, 1) > pass_seconds(1, 1) + 900
    assert pass_seconds(1, 0) == pass_seconds(1, 1)


def test_repair_does_not_wait_for_a_reply_that_needs_it():
    dead = record("w2")
    dead.last_seen -= 60
    orchestrator, calls = repairing_pool(w1=record("w1"), w2=dead)
    orchestrator.replies_in_flight = 1
    asyncio.run(orchestrator._repair_route())
    assert calls
    done(orchestrator)


def test_repair_does_not_wait_for_a_reply_stuck_on_a_restarted_stage():
    # Found on a one-stage lane: the reply waited for the repair, and the repair for the reply.
    orchestrator, calls = repairing_pool(w1=record("w1", health={"status": "ok", "epoch": 99}))
    orchestrator.absent_since = {}
    orchestrator.replies_in_flight = 1
    asyncio.run(orchestrator._repair_route())
    assert calls
    done(orchestrator)


def test_a_repair_with_nowhere_to_go_is_tried_again_soon():
    from fastapi import HTTPException

    from isharded.orchestrator import REPAIR_COOLDOWN_SECONDS, REPAIR_RETRY_SECONDS

    dead = record("w1")
    dead.last_seen -= 60
    orchestrator, _ = repairing_pool(w1=dead)

    async def nowhere(**_):
        raise HTTPException(status_code=409, detail="no worker offers any memory for weights")

    orchestrator.rebalance = nowhere
    with pytest.raises(HTTPException):
        asyncio.run(orchestrator._repair_route())
    waited = time.monotonic() - orchestrator.last_repair_at
    assert REPAIR_COOLDOWN_SECONDS - REPAIR_RETRY_SECONDS - 1 < waited <= REPAIR_COOLDOWN_SECONDS - REPAIR_RETRY_SECONDS + 1
    done(orchestrator)


def test_a_node_is_only_planned_for_models_its_owner_agreed_to_host():
    choosy, willing = record("w1"), record("w2")
    choosy.registration.hardware.contribution.host_models = ["*llama*"]
    orchestrator = pool(w1=choosy, w2=willing)
    orchestrator.planning_model = "Qwen3-0.6B-Q8_0.gguf"
    assert [item.worker_id for item in orchestrator.candidates()] == ["w2"]
    orchestrator.planning_model = "tiny-llama.gguf"
    assert sorted(item.worker_id for item in orchestrator.candidates()) == ["w1", "w2"]
    done(orchestrator)


def test_a_vulkan_gpu_is_only_planned_for_gguf_models():
    gpu, other = record("w1"), record("w2")
    gpu.registration.hardware.backend = "vulkan"
    orchestrator = pool(w1=gpu, w2=other)
    orchestrator.planning_model = "Qwen/Qwen3-0.6B"
    assert [item.worker_id for item in orchestrator.candidates()] == ["w2"]
    orchestrator.planning_model = "owner/repo:model.gguf"
    assert sorted(item.worker_id for item in orchestrator.candidates()) == ["w1", "w2"]
    done(orchestrator)


def test_a_returned_node_still_waits_for_replies_to_finish():
    orchestrator, calls = repairing_pool(w1=record("w1"), w2=record("w2"))
    orchestrator.lost_from_route = {"w2"}
    orchestrator.absent_since = {}
    orchestrator.replies_in_flight = 1
    asyncio.run(orchestrator._repair_route())
    assert not calls
    done(orchestrator)


def test_no_drain_is_asked_of_a_worker_nobody_can_reach():
    asked = []

    def handler(request):
        asked.append(str(request.url))
        return httpx.Response(200, json={"drained": True})

    gone = record("w2")
    gone.last_seen -= 60
    orchestrator = pool(w1=record("w1"), w2=gone)

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await orchestrator._drain_entries([stage("w2")])
        orchestrator.route = [stage("w1")]
        await orchestrator._drain_entries([stage("w2")])
        await orchestrator.close()

    asyncio.run(scenario())
    assert len(asked) == 1


def test_a_stream_from_a_silent_worker_ends():
    gone = record("w1")
    gone.last_seen -= 60
    orchestrator = pool(w1=gone)

    class Quiet:
        async def aiter_lines(self):
            yield "first"
            await asyncio.sleep(30)

    async def scenario(monkeypatch_seconds=0.05):
        orchestrator_module.STREAM_CHECK_SECONDS, held = monkeypatch_seconds, orchestrator_module.STREAM_CHECK_SECONDS
        seen = []
        try:
            with pytest.raises(orchestrator_module.WorkerSilent):
                async for line in orchestrator._lines_from(Quiet(), "w1"):
                    seen.append(line)
        finally:
            orchestrator_module.STREAM_CHECK_SECONDS = held
        await orchestrator.close()
        return seen

    assert asyncio.run(scenario()) == ["first"]


def test_a_slow_first_token_from_a_live_worker_is_not_silence():
    orchestrator = pool(w1=record("w1"))

    class Slow:
        async def aiter_lines(self):
            await asyncio.sleep(0.2)
            yield "late"

    async def scenario():
        orchestrator_module.STREAM_CHECK_SECONDS, held = 0.02, orchestrator_module.STREAM_CHECK_SECONDS
        try:
            return [line async for line in orchestrator._lines_from(Slow(), "w1")]
        finally:
            orchestrator_module.STREAM_CHECK_SECONDS = held
            await orchestrator.close()

    assert asyncio.run(scenario()) == ["late"]


def test_a_node_holding_nothing_costs_nothing_to_cordon():
    orchestrator = pool(w1=record("w1"))
    orchestrator.active = object()
    orchestrator.route = [stage("other")]
    assert orchestrator.cost_of_losing("w1") == 0
    done(orchestrator)


def test_uncordoning_a_node_in_service_moves_nothing(monkeypatch):
    from fastapi.testclient import TestClient

    orchestrator = pool(w1=record("w1"))
    orchestrator.active = object()
    orchestrator.spawn = lambda work: pytest.fail("nothing should move")
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    assert TestClient(orchestrator_module.app).post("/admin/nodes/w1/uncordon").status_code == 200


def test_merged_passes_stay_off_unless_asked_for():
    from types import SimpleNamespace

    import torch

    from isharded.worker import WorkerRuntime

    step = SimpleNamespace(verify=False, reset=False, prefill_only=False, save_checkpoint=None, restore_length=None, next_position=None)
    hidden = torch.zeros(1, 1, 4)
    held = SimpleNamespace(merge_passes=False, gguf=object())
    assert not WorkerRuntime._can_merge(held, hidden, step)
    held.merge_passes = True
    assert WorkerRuntime._can_merge(held, hidden, step)
    assert not WorkerRuntime._can_merge(held, torch.zeros(1, 2, 4), step)
    step.reset = True
    assert not WorkerRuntime._can_merge(held, hidden, step)


def test_a_worker_leaving_the_route_is_released_from_its_drain():
    asked = []

    def handler(request):
        asked.append(request.url.path)
        return httpx.Response(200, json={"drained": True})

    orchestrator = pool(w1=record("w1"), w2=record("w2"))

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = [stage("w1")]
        await orchestrator._drain_entries([stage("w2")])
        await orchestrator._release_entries()
        await orchestrator._release_entries()
        await orchestrator.close()

    asyncio.run(scenario())
    assert asked.count("/internal/drain") == 2
    assert asked.count("/internal/release") == 2


def test_a_reply_arriving_during_a_move_waits_for_it():
    orchestrator = pool(w1=record("w1"))
    orchestrator.active = object()

    async def scenario():
        async def finish_move():
            await asyncio.sleep(0.05)
            orchestrator.route = [stage("w1")]

        moved = asyncio.ensure_future(finish_move())
        await orchestrator.wait_for_route()
        await moved
        await orchestrator.close()

    asyncio.run(scenario())
    assert orchestrator.route


def test_with_no_model_loaded_a_reply_does_not_wait():
    orchestrator = pool()
    started = time.monotonic()
    asyncio.run(orchestrator.wait_for_route())
    assert time.monotonic() - started < 0.5
    done(orchestrator)


def test_a_dead_stage_is_only_restarted_once():
    from types import SimpleNamespace

    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    started = []
    current = object()

    class Stage:
        host = current

        def respawn(self):
            started.append(1)
            self.host = object()

    async def scenario():
        worker = SimpleNamespace(replacing=asyncio.Lock(), gguf=Stage(), last_error=None, caches={}, stage_restarts=0)
        lost = [StageLost("gone", current), StageLost("gone", current)]
        return await asyncio.gather(*(WorkerRuntime.replace_stage(worker, item) for item in lost))

    answers = asyncio.run(scenario())
    assert len(started) == 1
    assert all(answer.status_code == 503 for answer in answers)


def test_diagnostics_show_what_is_waiting():
    from isharded.serving import where_everything_is

    async def scenario():
        return where_everything_is()

    found = asyncio.run(scenario())
    assert any("scenario" in line for line in found["tasks"])
    assert found["threads"]


def test_a_slow_control_plane_does_not_delay_check_ins_with_the_others(monkeypatch) -> None:
    import asyncio

    from isharded import worker as worker_module

    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.control_urls = {"http://slow": worker_module.ControlLink(), "http://fast": worker_module.ControlLink()}
    calls = {"http://slow": 0, "http://fast": 0}

    async def check_in(url, link):
        calls[url] += 1
        if url == "http://slow":
            await asyncio.sleep(30)  # a link that hangs, far past the liveness window

    runtime._check_in = check_in
    monkeypatch.setattr(worker_module, "CHECK_IN_SECONDS", 0.01)

    async def run():
        task = asyncio.create_task(runtime.registration_loop())
        await asyncio.sleep(0.2)
        task.cancel()

    asyncio.run(run())
    assert calls["http://slow"] == 1  # still in flight, so not stacked up
    assert calls["http://fast"] > 5


def test_links_are_measured_from_each_end_and_a_dead_one_stays_on_record() -> None:
    from isharded import orchestrator as module

    orchestrator = pool(one=record("one"), two=record("two"))
    asked = []

    def handler(request):
        asked.append(request.url.host)
        if request.url.host == "two":
            return httpx.Response(500)
        return httpx.Response(200, json={"100": {"median_ms": 42.5}})

    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await orchestrator.measure_links()
        await orchestrator.measure_links()

    asyncio.run(run())
    assert orchestrator.links == {("one", "two"): 42.5, ("two", "one"): module.UNREACHABLE_LINK_MS}
    assert sorted(asked) == ["one", "two"]  # still fresh the second time, so nothing is sent again
    done(orchestrator)
