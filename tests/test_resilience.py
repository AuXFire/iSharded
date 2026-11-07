import asyncio
import json
import re
import time
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx
import pytest

from isharded import orchestrator as orchestrator_module
from isharded.models import GgufAdapter
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
    parallel = {"slots": 1, "reason": "set by the operator"}


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


def test_a_failed_load_leaves_the_running_routes_link_costs_as_they_were(monkeypatch):
    orchestrator, _ = deploy_pool(monkeypatch, [{"w1": "out of memory"}] * 3, 15 * 1024**3)
    orchestrator.link_row_bytes = 4096
    orchestrator.links[("w1", "w2")] = 5.0
    orchestrator.link_readings[("w1", "w2")] = {"cost_ms": 5.0}
    orchestrator.links_checked[("w1", "w2")] = 1.0

    async def measure_links(row):
        orchestrator.link_row_bytes = row
        orchestrator.links[("w1", "w2")] = orchestrator.links[("w2", "w1")] = 80.0
        orchestrator.link_readings[("w1", "w2")] = {"cost_ms": 80.0}
        orchestrator.links_checked[("w1", "w2")] = 2.0

    orchestrator.measure_links = measure_links
    monkeypatch.setattr(orchestrator_module, "row_bytes", lambda target: 65536)
    with pytest.raises(orchestrator_module.HTTPException):
        run_deploy(orchestrator)
    assert orchestrator.link_row_bytes == 4096
    assert orchestrator.links == {("w1", "w2"): 5.0}
    assert orchestrator.link_readings == {("w1", "w2"): {"cost_ms": 5.0}}
    assert orchestrator.links_checked == {("w1", "w2"): 1.0}


def test_a_probe_that_outlasts_its_time_shows_the_worker_unreachable(monkeypatch):
    """The client tries a refused connection again, so its own timeout does not bound a probe."""
    orchestrator = pool(w1=record("w1", health={"status": "ok"}))
    monkeypatch.setattr(orchestrator_module, "PROBE_SECONDS", 0.1)

    async def retrying(*args, **kwargs):
        await asyncio.sleep(5)

    orchestrator.client = SimpleNamespace(get=retrying)
    began = time.monotonic()
    asyncio.run(orchestrator.refresh_worker_metrics())
    assert time.monotonic() - began < 2
    view = orchestrator.status()["workers"]["w1"]
    assert view["state"] == "unreachable" and view["live"]
    assert view["runtime"] == {"status": "unreachable", "error": "no answer in 0 s"}


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


def test_a_reply_takes_the_sequence_idle_longest():
    """A saved state can go on any sequence, so nothing ties a conversation to one."""
    from isharded import worker

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.slot_used_at = {0: 5.0, 1: 9.0, 2: 1.0}
    assert runtime.choose_slot({0, 1, 2}) == 2
    assert runtime.choose_slot({0, 1}) == 0


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
        parallel = {}

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
    entry = {item.unit.name: item.entry_bytes for item in _stage_items(profile, 0, 1)}
    later = {item.unit.name: item.entry_bytes for item in _stage_items(profile, 1, 2)}
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

    step = SimpleNamespace(verify=False, reset=False, prompt_rows=False, restore_length=None, next_position=None)
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
        worker = SimpleNamespace(replacing=asyncio.Lock(), gguf=Stage(), last_error=None, caches={}, stage_restarts=0, draft=None, draft_request=None, ring_stop=lambda: None)
        lost = [StageLost("gone", current), StageLost("gone", current)]
        return await asyncio.gather(*(WorkerRuntime.replace_stage(worker, item) for item in lost))

    answers = asyncio.run(scenario())
    assert len(started) == 1
    assert all(answer.status_code == 503 for answer in answers)


class FakeDraft:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class RestartingStage:
    host = object()

    def respawn(self):
        self.host = object()


def draft_worker(gguf=None):
    """A worker that can load a draft, and counts how often it does."""
    from types import SimpleNamespace

    from isharded.worker import WorkerRuntime

    worker = SimpleNamespace(replacing=asyncio.Lock(), gguf=gguf, last_error=None, caches={}, stage_restarts=0, draft=None, draft_request=("request",), loads=[], ring_stop=lambda: None)

    def load(_):
        worker.loads.append(FakeDraft())
        worker.draft = worker.loads[-1]

    worker._load_draft = load
    worker.replace_draft = lambda *args: WorkerRuntime.replace_draft(worker, *args)
    return worker


def test_a_stage_that_restarts_brings_back_the_draft_that_lived_in_it(monkeypatch):
    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    monkeypatch.delenv("ISHARDED_DRAFT_APART", raising=False)

    async def scenario():
        worker = draft_worker(RestartingStage())
        worker.draft = dead = FakeDraft()
        await WorkerRuntime.replace_stage(worker, StageLost("gone", worker.gguf.host))
        return worker, dead

    worker, dead = asyncio.run(scenario())
    assert dead.closed == 1
    assert worker.draft is worker.loads[0]


def test_a_draft_in_its_own_process_outlives_a_stage_restart(monkeypatch):
    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    monkeypatch.setenv("ISHARDED_DRAFT_APART", "1")

    async def scenario():
        worker = draft_worker(RestartingStage())
        worker.draft = alive = FakeDraft()
        await WorkerRuntime.replace_stage(worker, StageLost("gone", worker.gguf.host))
        return worker, alive

    worker, alive = asyncio.run(scenario())
    assert alive.closed == 0
    assert worker.draft is alive
    assert not worker.loads


def test_a_new_model_without_a_draft_forgets_the_last_ones():
    """A stage restart reloads what this holds, on a stage that may have no draft now."""
    from types import SimpleNamespace

    from isharded.worker import WorkerRuntime

    worker = SimpleNamespace(speculate_tokens=4, draft_request=("the last model",))
    for request in (
        SimpleNamespace(speculate_tokens=0, draft=None, assignment=SimpleNamespace(start_layer=0)),
        SimpleNamespace(speculate_tokens=0, draft=object(), assignment=SimpleNamespace(start_layer=12)),
    ):
        worker.draft_request = ("the last model",)
        WorkerRuntime._load_draft(worker, request, "folder", 2)
        assert worker.draft_request is None


def test_a_stage_restart_keeps_the_draft_a_reply_already_replaced(monkeypatch):
    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    monkeypatch.delenv("ISHARDED_DRAFT_APART", raising=False)

    async def scenario():
        worker = draft_worker(RestartingStage())
        worker.draft = dead = FakeDraft()
        respawn = worker.gguf.respawn

        def reply_gets_there_first():
            respawn()
            worker.draft = FakeDraft()

        worker.gguf.respawn = reply_gets_there_first
        await WorkerRuntime.replace_stage(worker, StageLost("gone", worker.gguf.host))
        return worker, dead

    worker, dead = asyncio.run(scenario())
    assert not worker.loads
    assert worker.draft is not dead and worker.draft.closed == 0


def shared_draft():
    """A worker whose draft lives in its stage's process."""
    from types import SimpleNamespace

    process = object()
    worker = draft_worker(SimpleNamespace(host=SimpleNamespace(process=process)))
    worker.draft = FakeDraft()
    worker.draft.host = SimpleNamespace(process=process)
    return worker


def test_a_draft_that_hears_first_reports_the_stage_as_lost():
    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    worker = shared_draft()
    answer = asyncio.run(WorkerRuntime.draft_failed(worker, StageLost("gone", object()), worker.draft))
    assert isinstance(answer, StageLost) and answer.host is worker.gguf.host
    assert not worker.loads, "the restart brings the draft back"


@pytest.mark.parametrize("case", ["replaced", "own process", "not lost"])
def test_a_draft_failure_alone_leaves_the_stage_running(case):
    from types import SimpleNamespace

    from isharded.llamacpp import LlamaError, StageLost
    from isharded.worker import WorkerRuntime

    worker = shared_draft()
    draft = worker.draft
    reason = LlamaError("decode failed") if case == "not lost" else StageLost("closed", object())
    if case == "replaced":
        worker.draft = FakeDraft()  # a closed connection reads as lost, and this process is fine
    if case == "own process":
        draft.host = SimpleNamespace(process=object())
    answer = asyncio.run(WorkerRuntime.draft_failed(worker, reason, draft))
    assert not isinstance(answer, StageLost)


def test_replies_on_one_dead_draft_load_one_new_draft():
    """A draft sharing a stage's process would stay in its memory if a second load replaced the first."""
    from isharded.worker import WorkerRuntime

    async def scenario():
        worker = draft_worker()
        worker.draft = dead = FakeDraft()
        answers = await asyncio.gather(*(WorkerRuntime.replace_draft(worker, RuntimeError("gone"), dead) for _ in range(3)))
        return worker, dead, answers

    worker, dead, answers = asyncio.run(scenario())
    assert dead.closed == 1
    assert len(worker.loads) == 1
    assert worker.draft is worker.loads[0]
    assert worker.loads[0].closed == 0
    assert all(isinstance(answer, RuntimeError) for answer in answers)


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


def measure(orchestrator, handler, *rows):
    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        for row in rows:
            await orchestrator.measure_links(row)

    asyncio.run(run())


def test_a_link_is_charged_what_a_row_adds_and_never_less_than_its_base_cost() -> None:
    orchestrator = pool(one=record("one"), two=record("two"))
    asked = []

    def handler(request):
        asked.append((json.loads(request.content)["sizes"], request.extensions["timeout"]["read"]))
        row = {"median_ms": 14.0 if request.url.host == "one" else 9.0, "max_ms": 99.0}
        return httpx.Response(200, json={"100": {"median_ms": 10.0, "max_ms": 12.0}, "16384": row})

    measure(orchestrator, handler, 16384)
    assert orchestrator.links == {("one", "two"): 15.0, ("two", "one"): 11.0}
    assert asked == [([100, 16384], 20.0)] * 2  # twice the work must not make a slow link look dead
    done(orchestrator)


def test_a_row_no_bigger_than_the_probe_is_not_probed() -> None:
    orchestrator = pool(one=record("one"), two=record("two"))
    asked = []

    def handler(request):
        asked.append(json.loads(request.content)["sizes"])
        return httpx.Response(200, json={"100": {"median_ms": 10.0}})

    measure(orchestrator, handler, 64)
    assert asked == [[100], [100]]
    assert orchestrator.links == {("one", "two"): 10.0, ("two", "one"): 10.0}
    done(orchestrator)


def test_a_model_with_rows_of_another_size_measures_links_again() -> None:
    orchestrator = pool(one=record("one"), two=record("two"))
    asked = []

    def handler(request):
        sizes = json.loads(request.content)["sizes"]
        asked.append(sizes[-1])
        return httpx.Response(200, json={str(size): {"median_ms": size / 1000} for size in sizes})

    measure(orchestrator, handler, 4096, 4096, 8192)
    assert asked == [4096, 4096, 8192, 8192]
    assert orchestrator.links[("one", "two")] == 8.19
    done(orchestrator)


def test_status_publishes_measured_links_by_worker_id_alone() -> None:
    orchestrator = pool(one=record("one"), two=record("two"))

    def handler(request):
        if request.url.host == "two":
            return httpx.Response(500)
        return httpx.Response(200, json={"100": {"median_ms": 10.0, "max_ms": 12.0}, "4096": {"median_ms": 13.0}})

    before = int(time.time())
    measure(orchestrator, handler, 4096)
    links = orchestrator.status()["links"]
    assert before <= links[0]["checked_at"] <= time.time()
    for link in links:
        del link["checked_at"]
    assert links == [
        {"source": "one", "target": "two", "median_ms": 10.0, "max_ms": 12.0, "cost_ms": 14.0},
        {"source": "two", "target": "one", "median_ms": None, "max_ms": None, "cost_ms": orchestrator_module.UNREACHABLE_LINK_MS},
    ]
    assert "http" not in json.dumps(links) and ":1" not in json.dumps(links)
    done(orchestrator)


def test_a_failed_link_drops_its_old_reading() -> None:
    orchestrator = pool(one=record("one"), two=record("two"))
    measure(orchestrator, lambda request: httpx.Response(200, json={"100": {"median_ms": 10.0}}), 0)
    orchestrator.links_checked.clear()
    measure(orchestrator, lambda request: httpx.Response(500), 0)
    assert [link["median_ms"] for link in orchestrator.status()["links"]] == [None, None]
    done(orchestrator)


def test_a_row_is_a_float32_per_hidden_unit_in_every_model_family() -> None:
    from types import SimpleNamespace

    assert orchestrator_module.row_bytes(SimpleNamespace(adapter=SimpleNamespace(hidden_size=1024))) == 4096
    assert orchestrator_module.row_bytes(SimpleNamespace(adapter=SimpleNamespace(header=SimpleNamespace(hidden_size=5120)))) == 20480
    assert orchestrator_module.row_bytes(SimpleNamespace()) == 0


def stage_answering(status, body, failure=None):
    """A middle stage whose next stage gives this answer, or whose request ends in this failure."""
    import torch
    from types import SimpleNamespace

    from isharded import worker as worker_module

    class Client:
        async def post(self, url, content, headers, timeout):
            if failure is not None:
                raise failure
            return httpx.Response(status, request=httpx.Request("POST", url), **body)

    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.assignment, runtime.model_id, runtime.epoch = object(), "m", 1
    runtime.gguf, runtime.dtype_name, runtime.pool_token = None, "float32", None
    runtime.merge_passes = False
    runtime.activation_bytes_sent = 0
    runtime.downstream_ms_ema = None
    runtime.last_error = None
    runtime.client = Client()
    runtime._next_stage = lambda: SimpleNamespace(url="http://next:1", worker_id="next")
    runtime._stages_after = lambda: 1
    runtime._record = lambda *args: None
    runtime.schedule_checkpoint = lambda request: None

    async def alone(hidden, request, places=None):
        return hidden

    runtime._step_alone = alone
    request = worker_module.InferRequest(session_id="s", epoch=1, input_ids=[1], reset=False, temperature=0.0)

    async def go():
        runtime.lock = asyncio.Lock()
        await runtime.execute(torch.zeros(1, 1, 4), request)

    return go


@pytest.mark.parametrize(
    ("status", "body", "expected", "detail"),
    [
        (502, {"json": {"detail": {"failed_worker": "far", "message": "gone"}}}, 502, {"failed_worker": "far", "message": "gone"}),
        (503, {"json": {"detail": {"retry": True, "message": "restarted"}}}, 503, {"retry": True, "message": "restarted"}),
        (502, {"text": "bad gateway"}, 502, {"failed_worker": "next"}),
        (504, {"json": {"detail": "slow"}}, 502, {"failed_worker": "next"}),
    ],
)
def test_a_failure_further_down_the_route_reaches_the_entry_as_itself(status, body, expected, detail) -> None:
    """Three stages or more: calling it a bad request would drop the reply instead of recovering."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(stage_answering(status, body)())
    assert caught.value.status_code == expected
    assert detail.items() <= caught.value.detail.items()


def test_a_stage_that_refuses_a_request_still_blames_the_request() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(stage_answering(500, {"json": {"detail": "boom"}})())
    assert caught.value.status_code == 500
    assert "stage_error" in caught.value.detail


def waiting_pool(monkeypatch, moved):
    """A route with one worker lost, where replacing it would move this many bytes."""
    from types import SimpleNamespace

    from isharded import orchestrator as module

    orchestrator = pool(one=record("one"), two=record("two", quarantined=True))
    orchestrator.route = [SimpleNamespace(worker_id="one"), SimpleNamespace(worker_id="two")]
    orchestrator.active = object()
    monkeypatch.setattr(module, "WORKER_RETURN_CHECK_SECONDS", 0.01)
    monkeypatch.setattr(module, "describe_plan", lambda target, route: {"download_bytes": moved})
    monkeypatch.setattr(orchestrator, "plan", lambda target: [])

    async def nothing(*args):
        return None

    monkeypatch.setattr(orchestrator, "refresh_holdings", nothing)
    monkeypatch.setattr(orchestrator, "refresh_worker_metrics", nothing)
    return orchestrator, module


def test_a_big_move_waits_for_a_lost_worker_to_return(monkeypatch) -> None:
    orchestrator, module = waiting_pool(monkeypatch, 40 * 1024**3)
    checks = []

    def review():
        checks.append(1)
        if len(checks) == 3:
            orchestrator.workers["two"].quarantined = False

    monkeypatch.setattr(orchestrator, "_review_quarantine", review)
    asyncio.run(orchestrator._await_lost_workers())
    assert len(checks) == 3
    done(orchestrator)


def test_a_worker_that_never_returns_is_replaced_after_a_while(monkeypatch) -> None:
    orchestrator, module = waiting_pool(monkeypatch, 40 * 1024**3)
    monkeypatch.setattr(module, "WORKER_RETURN_SECONDS", 0.05)
    asyncio.run(orchestrator._await_lost_workers())
    assert orchestrator.workers["two"].quarantined
    done(orchestrator)


def test_a_small_move_does_not_wait(monkeypatch) -> None:
    orchestrator, module = waiting_pool(monkeypatch, 1024)
    monkeypatch.setattr(orchestrator, "_review_quarantine", lambda: pytest.fail("it should not have waited"))
    asyncio.run(orchestrator._await_lost_workers())
    done(orchestrator)


def test_a_coordinator_waits_for_the_repair_of_a_restarted_stage(monkeypatch):
    """A stream can restore a saved state and the slow way cannot, so a short wait saves a reread."""
    monkeypatch.setattr(orchestrator_module, "RESTART_WAIT_SECONDS", 2.0)
    orchestrator = pool()
    orchestrator.epoch = 4

    async def scenario():
        async def repair():
            await asyncio.sleep(0.3)
            orchestrator.epoch = 5

        asyncio.ensure_future(repair())
        return await orchestrator._recover_stream("epoch", 4)

    assert asyncio.run(scenario()), "the reply goes on through a new stream"
    done(orchestrator)


def test_a_coordinator_stops_waiting_when_no_repair_comes(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "RESTART_WAIT_SECONDS", 0.05)
    orchestrator = pool()
    orchestrator.epoch = 4
    assert not asyncio.run(orchestrator._recover_stream("epoch", 4)), "the slow way still gets it there"
    done(orchestrator)


# A repair with a spare keeps the layers where they are.


def planning_pool(monkeypatch, *, preferred, standby=(), fits_without_standby=True):
    from types import SimpleNamespace

    orchestrator = Orchestrator()
    target = SimpleNamespace(profile=object(), spread=True, embedding=False, entry_worker=None, exit_worker=None)
    orchestrator.active = target
    orchestrator.route = [stage("a", 0, 1), stage("b", 1, 2)]
    candidates = lambda names: [SimpleNamespace(worker_id=name) for name in names]  # noqa: E731
    monkeypatch.setattr(orchestrator, "candidate_groups", lambda: (candidates(preferred), candidates(standby)))
    monkeypatch.setattr(orchestrator, "_with_holdings", lambda group, target: group)
    asked = []

    def keeping(route, chosen, spare, profile):
        asked.append(sorted(item.worker_id for item in [*chosen, *spare]))
        wanted = {"a", "c"} <= {item.worker_id for item in [*chosen, *spare]}
        return ["kept"] if wanted else None

    def replanning(chosen, spare, profile, **options):
        if not spare and not fits_without_standby:
            raise orchestrator_module.PlacementError("does not fit")
        return ["replanned"]

    monkeypatch.setattr(orchestrator_module, "keep_layers", keeping)
    monkeypatch.setattr(orchestrator_module, "plan_placement", replanning)
    return orchestrator, target, asked


def test_a_lost_stage_goes_to_a_spare_and_nothing_else_moves(monkeypatch):
    orchestrator, target, _ = planning_pool(monkeypatch, preferred=["a", "c"])
    assert orchestrator.plan(target) == ["kept"]
    done(orchestrator)


def test_a_standby_worker_stays_out_when_the_rest_can_hold_the_model(monkeypatch):
    """--standby is a promise to its owner, and a repair does not get to break it."""
    orchestrator, target, _ = planning_pool(monkeypatch, preferred=["a"], standby=["c"], fits_without_standby=True)
    assert orchestrator.plan(target) == ["replanned"]
    done(orchestrator)


def test_a_standby_worker_takes_only_the_lost_layers_when_it_is_needed(monkeypatch):
    orchestrator, target, _ = planning_pool(monkeypatch, preferred=["a"], standby=["c"], fits_without_standby=False)
    assert orchestrator.plan(target) == ["kept"]
    done(orchestrator)


def test_with_nothing_lost_a_plan_starts_fresh(monkeypatch):
    orchestrator, target, asked = planning_pool(monkeypatch, preferred=["a", "b", "c"])
    assert orchestrator.plan(target) == ["replanned"]
    assert not asked
    done(orchestrator)


def test_a_new_model_is_planned_from_scratch(monkeypatch):
    from types import SimpleNamespace

    orchestrator, _, asked = planning_pool(monkeypatch, preferred=["a", "c"])
    other = SimpleNamespace(profile=object(), spread=True, embedding=False, entry_worker=None, exit_worker=None)
    assert orchestrator.plan(other) == ["replanned"]
    assert not asked
    done(orchestrator)


def test_with_no_way_to_keep_the_layers_it_replans(monkeypatch):
    orchestrator, target, _ = planning_pool(monkeypatch, preferred=["a"])
    assert orchestrator.plan(target) == ["replanned"]
    done(orchestrator)


# Saved states on a link that has gone silent. A real socket that takes the connection
# and never answers, because a lost link gives no error, and a stub cannot show a timeout firing.


async def silent_peer():
    held = []

    async def swallow(reader, writer):
        held.append(writer)
        while await reader.read(65536):
            pass

    for port in range(9690, 9700):
        try:
            server = await asyncio.start_server(swallow, "127.0.0.1", port)
            return server, f"http://127.0.0.1:{port}", held
        except OSError:
            continue
    raise RuntimeError("no free port for a silent peer")


def stage_on_a_silent_link(tmp_path, url, host=None):
    from types import SimpleNamespace

    from isharded import worker

    class Host:
        def call(self, operation, **meta):
            return ({"states": []} if operation == "state_list" else {"held": False}), b""

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf = SimpleNamespace(host=host or Host())
    runtime.namespace, runtime.state_peers, runtime.fetched, runtime.copies_fetched = "space", [url], set(), 0
    runtime.pool_token, runtime.cache_dir, runtime.worker_id = None, tmp_path, "w1"
    (runtime.state_root / "space").mkdir(parents=True)
    runtime.client = httpx.AsyncClient(trust_env=False)
    return runtime


def timed_against_a_silent_peer(work):
    async def scenario():
        server, url, held = await silent_peer()
        began = time.monotonic()
        try:
            result = await asyncio.wait_for(work(url), 10)
            return result, time.monotonic() - began
        finally:
            for writer in held:
                writer.close()
            server.close()

    return asyncio.run(scenario())


def test_a_silent_holder_costs_a_listing_only_its_timeout(tmp_path, monkeypatch):
    from isharded import worker

    monkeypatch.setattr(worker, "PEER_SECONDS", 0.2)

    async def work(url):
        return await stage_on_a_silent_link(tmp_path, url).restorable()

    found, spent = timed_against_a_silent_peer(work)
    assert found == {} and spent < 3


def test_fetching_from_a_silent_holder_gives_up_and_leaves_nothing(tmp_path, monkeypatch):
    from isharded import statecopy

    monkeypatch.setattr(statecopy, "FETCH_SILENCE_SECONDS", 0.2)

    async def work(url):
        return await stage_on_a_silent_link(tmp_path, url).ensure_state("abc")

    ready, spent = timed_against_a_silent_peer(work)
    assert not ready and spent < 3
    assert not list(tmp_path.rglob("*.part"))


def test_a_copy_to_a_silent_holder_gives_up_and_costs_nothing_else(tmp_path, monkeypatch):
    from isharded import statecopy
    from isharded.protocol import SavePoint
    from isharded.statecopy import CopySender
    from isharded.statestore import HISTORY, SUFFIX

    monkeypatch.setattr(statecopy, "SILENCE_SECONDS", 0.2)
    (tmp_path / ("abc" + SUFFIX)).write_bytes(b"x" * 1000)

    async def work(url):
        async with httpx.AsyncClient(trust_env=False) as client:
            sender = CopySender(client, dict, 10**9)
            sender.aim(url, tmp_path, "space")
            handed_over = time.perf_counter()
            sender.add([SavePoint(at=300, length=300, key="abc", kind=HISTORY)])
            handed_over = time.perf_counter() - handed_over
            await sender.task
            return handed_over, sender.report()

    (handed_over, report), spent = timed_against_a_silent_peer(work)
    assert handed_over < 0.01, "the reply that saved this state does not wait for the link"
    assert report["copies_failed"] == 1 and spent < 3


def test_a_silent_stage_only_costs_the_index_its_timeout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from isharded import worker
    from isharded.statestore import PrefixIndex

    monkeypatch.setattr(worker, "INDEX_SECONDS", 0.2)

    async def work(url):
        runtime = stage_on_a_silent_link(tmp_path, url)
        runtime.state_peers = []
        runtime.worker_id, runtime.epoch, runtime.index_epoch = "w1", 3, 1
        runtime.prompt_cache, runtime.prefix, runtime.index_lock = True, PrefixIndex(), asyncio.Lock()
        runtime.route = [SimpleNamespace(worker_id="w1", url="http://unused:1"), SimpleNamespace(worker_id="w2", url=url)]
        await runtime.sync_index()
        return runtime.prefix.entries, runtime.index_epoch

    (entries, seen), spent = timed_against_a_silent_peer(work)
    assert not entries and seen == 3 and spent < 3, "and a reply goes on to read the prompt"


def test_a_stage_gone_silent_is_not_waited_on_for_a_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from isharded import worker

    monkeypatch.setattr(worker, "FETCH_SECONDS", 0.3)

    async def work(url):
        runtime = stage_on_a_silent_link(tmp_path, url)
        runtime.state_peers = []
        runtime.worker_id = "w1"
        runtime.route = [SimpleNamespace(worker_id="w2", url=url)]
        return await runtime.prepare_state("abc")

    ready, spent = timed_against_a_silent_peer(work)
    assert not ready and spent < 3


def test_a_step_from_an_old_route_does_not_stop_the_steps_beside_it() -> None:
    import torch

    from isharded import worker as worker_module
    from isharded.protocol import InferRequest

    runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    runtime.epoch, runtime.gguf, runtime.lock, runtime.merge_queue = 2, object(), asyncio.Lock(), []

    async def alone(hidden, request):
        return {"token_id": 7}

    runtime._step_alone = alone

    async def run():
        steps = [
            worker_module.MergedStep(
                torch.zeros(1, 1, 4),
                InferRequest(session_id=name, epoch=epoch, input_ids=[1]),
                asyncio.get_running_loop().create_future(),
            )
            for name, epoch in (("old", 1), ("new", 2))
        ]
        runtime.merge_queue = list(steps)
        await asyncio.wait_for(runtime._run_merge_queue(), 5)
        return steps

    old, new = asyncio.run(run())
    assert old.outcome.exception().status_code == 409
    assert new.outcome.result() == {"token_id": 7}


# A route and a tokenizer say a reply could run. Only the workers say it can.


def serving_pool(**fields):
    fields.setdefault("health", {"status": "ok", "epoch": 3})
    orchestrator = pool(w1=record("w1", **fields))
    orchestrator.epoch = 3
    orchestrator.route = [stage("w1")]
    orchestrator.tokenizer = object()
    return orchestrator


def test_a_route_whose_workers_hold_their_stages_is_serving():
    orchestrator = serving_pool()
    assert orchestrator.serving() and orchestrator.status()["status"] == "ready"
    done(orchestrator)


def lose_the_route(orchestrator):
    orchestrator.route = []


def lose_the_tokenizer(orchestrator):
    orchestrator.tokenizer = None


def lose_the_worker(orchestrator):
    del orchestrator.workers["w1"]


def go_silent(orchestrator):
    orchestrator.workers["w1"].last_seen -= 60


def quarantine(orchestrator):
    orchestrator.workers["w1"].quarantined = True


def cordon(orchestrator):
    orchestrator.access.cordoned = {"w1"}


def restart_empty(orchestrator):
    orchestrator.workers["w1"].health = {"status": "ok", "epoch": 0}


@pytest.mark.parametrize("change", [lose_the_route, lose_the_tokenizer, lose_the_worker, go_silent, quarantine, cordon, restart_empty])
def test_a_route_is_not_serving_unless_every_stage_is_held(change):
    orchestrator = serving_pool()
    change(orchestrator)
    assert not orchestrator.serving()
    done(orchestrator)


def test_asking_whether_a_route_is_serving_changes_nothing():
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.serving()
    assert not orchestrator.need_reconfigure, "that belongs to the repair loop"
    done(orchestrator)


def test_a_worker_that_cannot_be_reached_right_now_keeps_its_place():
    """One missed health check must not turn a load balancer away. Heartbeats decide this."""
    orchestrator = serving_pool(health={"status": "unreachable"})
    assert orchestrator.serving()
    done(orchestrator)


def test_a_route_saved_from_before_a_restart_says_repairing_and_not_ready(monkeypatch):
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    assert orchestrator.status()["status"] == "repairing"
    answer = TestClient(orchestrator_module.app).get("/ready")
    assert answer.status_code == 503 and answer.json() == {"ready": False}
    done(orchestrator)


def test_a_broken_route_says_loading_during_a_load():
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.job = orchestrator_module.LoadJob(job_id="j", model_id="m")
    assert orchestrator.status()["status"] == "loading"
    done(orchestrator)


def test_a_new_layout_being_prepared_leaves_a_whole_route_ready():
    orchestrator = serving_pool()
    orchestrator.job = orchestrator_module.LoadJob(job_id="j", model_id="m")
    assert orchestrator.status()["status"] == "ready"
    done(orchestrator)


def test_a_lane_is_only_ready_when_it_is_serving(monkeypatch):
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    monkeypatch.setattr(orchestrator_module, "verify_token", lambda authorization: None)
    assert asyncio.run(orchestrator_module.lane_status(None))["ready"] is False
    orchestrator.workers["w1"].health = {"status": "ok", "epoch": 3}
    assert asyncio.run(orchestrator_module.lane_status(None))["ready"] is True
    done(orchestrator)


def test_metrics_say_how_many_guesses_came_from_the_text():
    pipeline = {"tokens": 9, "misses": 1, "depth": 2, "lookups": 5}
    orchestrator = pool(w1=record("w1", health={"status": "ok", "telemetry": {"pipeline": pipeline}}))
    assert re.search(r"^isharded_worker_pipeline_lookups\S* 5", orchestrator.metrics_text(), re.M)
    done(orchestrator)


def lookup_target(guesses: int) -> MagicMock:
    target = MagicMock()
    target.adapter.supports_kv_snapshots = False
    target.lookup_guesses = guesses
    target.speculate_tokens = guesses
    return target


@pytest.mark.parametrize(
    "guesses, stages, expected",
    [(16, 2, 16), (16, 1, 0), (0, 1, 7)],
    ids=["two stages can overlap", "one stage keeps its memory", "a draft guesses on one stage too"],
)
def test_lookup_alone_asks_for_guess_room_only_where_a_pipeline_can_run(monkeypatch, guesses, stages, expected):
    orchestrator = pool(w1=record("w1"), w2=record("w2"))
    orchestrator.tokenizer = object()
    orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    made = MagicMock(return_value=MagicMock(model_dump=dict))
    monkeypatch.setattr(orchestrator_module, "ConfigureRequest", made)
    orchestrator.save_state = lambda target: None
    target = lookup_target(guesses)
    if not guesses:
        target.speculate_tokens = expected

    async def scenario():
        await orchestrator._activate(target, [stage(f"w{index + 1}") for index in range(stages)])
        await asyncio.gather(*orchestrator.background, return_exceptions=True)
        await orchestrator.client.aclose()

    asyncio.run(scenario())
    assert target.speculate_tokens == expected
    assert made.call_args.kwargs["speculate_tokens"] == expected
    done(orchestrator)


def test_a_new_route_is_ready_at_once_and_not_after_the_next_health_check(monkeypatch):
    """Each worker's last report is about the old epoch until the next check, ten seconds off."""
    orchestrator = pool(w1=record("w1", health={"status": "ok", "epoch": 3}))
    orchestrator.epoch = 3
    orchestrator.tokenizer = object()
    orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    monkeypatch.setattr(orchestrator_module, "ConfigureRequest", MagicMock(return_value=MagicMock(model_dump=dict)))
    orchestrator.save_state = lambda target: None
    target = MagicMock()
    target.adapter.supports_kv_snapshots = False

    async def scenario():
        assert await orchestrator._activate(target, [stage("w1")]) == {}
        assert orchestrator.serving()
        await asyncio.gather(*orchestrator.background, return_exceptions=True)
        await orchestrator.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("known", [True, False])
def test_a_stage_that_registers_is_checked_at_once(monkeypatch, known):
    """It may have restarted empty, and until the next check it would pass for one that holds its stage."""
    orchestrator = serving_pool()
    if not known:
        del orchestrator.workers["w1"]
    orchestrator.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok", "epoch": 0}))
    )
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    monkeypatch.setattr(orchestrator_module, "verify_token", lambda authorization: None)
    monkeypatch.setattr(orchestrator_module, "verify_worker_name", lambda worker_id: None)

    async def scenario():
        await orchestrator_module.register(registration("w1"), None)
        await asyncio.gather(*orchestrator.background)
        assert not orchestrator.serving()
        await orchestrator.close()

    asyncio.run(scenario())


def test_a_reply_arriving_during_a_repair_waits_for_it():
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()

    async def scenario():
        async def repair():
            await asyncio.sleep(0.05)
            orchestrator.workers["w1"].health = {"status": "ok", "epoch": 3}

        repaired = asyncio.ensure_future(repair())
        await orchestrator.wait_for_route()
        assert repaired.done()
        await orchestrator.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["saved_state", "job"])
def test_a_reply_arriving_before_a_route_is_back_waits_for_it(field):
    """With nothing installed yet there is no active model to go by."""
    orchestrator = pool(w1=record("w1"))
    setattr(orchestrator, field, {"route": []} if field == "saved_state" else orchestrator_module.LoadJob(job_id="j", model_id="m"))

    async def scenario():
        async def back():
            await asyncio.sleep(0.05)
            orchestrator.route = [stage("w1")]

        returned = asyncio.ensure_future(back())
        await orchestrator.wait_for_route()
        assert returned.done()
        await orchestrator.close()

    asyncio.run(scenario())


def test_a_reply_does_not_wait_forever_for_a_repair(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "MOVE_WAIT_SECONDS", 0.05)
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()
    asyncio.run(orchestrator.wait_for_route())
    assert not orchestrator.serving()
    done(orchestrator)


def test_a_follower_no_worker_can_reach_does_not_hold_a_reply_back(monkeypatch):
    """It has the route and no records, so the whole wait would pass before every reply."""
    monkeypatch.setattr(orchestrator_module, "MOVE_WAIT_SECONDS", 30.0)
    orchestrator = pool()
    orchestrator.route = [stage("w1")]
    orchestrator.mesh = SimpleNamespace(replica={"epoch": 3}, is_leader=False)
    started = time.monotonic()
    asyncio.run(orchestrator.wait_for_route())
    assert time.monotonic() - started < 5
    assert not orchestrator.serving()
    orchestrator.mesh = None
    done(orchestrator)


def test_a_leader_back_from_a_restart_still_waits_for_its_workers(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "MOVE_WAIT_SECONDS", 0.2)
    orchestrator = pool()
    orchestrator.route = [stage("w1")]
    orchestrator.mesh = SimpleNamespace(replica={"epoch": 3}, is_leader=True)
    started = time.monotonic()
    asyncio.run(orchestrator.wait_for_route())
    assert time.monotonic() - started >= 0.2
    orchestrator.mesh = None
    done(orchestrator)


def test_a_load_gets_longer_than_a_repair(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "MOVE_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(orchestrator_module, "LOAD_WAIT_SECONDS", 0.5)
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()
    orchestrator.job = orchestrator_module.LoadJob(job_id="j", model_id="m")
    started = time.monotonic()
    asyncio.run(orchestrator.wait_for_route())
    assert time.monotonic() - started >= 0.5
    done(orchestrator)


def test_a_reply_that_arrives_during_a_repair_goes_ahead_when_it_is_done():
    """The real failure: a reply reached stages that were still empty, and came back a 502."""
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()
    asked = []

    def handler(request):
        asked.append(orchestrator.workers["w1"].health["epoch"])
        return stream_lines({"token_id": 7}, {"token_id": 99})

    async def repair():
        await asyncio.sleep(0.05)
        orchestrator.workers["w1"].health = {"status": "ok", "epoch": 3}

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        repaired = asyncio.ensure_future(repair())
        session = SessionState(tokens=[1, 2], epoch=3, needs_reset=True)
        got = [token async for token in orchestrator.token_stream(session, 5, 0.0, 1.0, 0, None)]
        await repaired
        await orchestrator.close()
        return got

    orchestrator.tokenizer = type("Stops", (), {"stop_token_ids": {99}})()
    assert asyncio.run(scenario()) == [7, 99]
    assert asked == [3], "it reached the stages only once they held their layers"


def test_a_repair_slower_than_the_restart_wait_is_still_waited_for(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "RESTART_WAIT_SECONDS", 0.05)
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()

    async def scenario():
        async def repair():
            await asyncio.sleep(0.3)
            orchestrator.epoch = 4
            orchestrator.workers["w1"].health = {"status": "ok", "epoch": 4}

        repaired = asyncio.ensure_future(repair())
        goes_on = await orchestrator._recover_stream("epoch", 3)
        await repaired
        await orchestrator.close()
        return goes_on

    assert asyncio.run(scenario()), "a new stream carries on exactly, and the slow way does not"


# A GGUF reply resumes from what it wrote, one row at a time, on the entry worker.


def gguf_pool(**records):
    orchestrator = pool(**(records or {"w1": record()}))
    orchestrator.active = SimpleNamespace(adapter=GgufAdapter.__new__(GgufAdapter))
    return orchestrator


def test_a_gguf_reply_resumes_with_what_it_wrote_and_not_as_a_prompt():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream_lines({"token_id": 5}, {"token_id": 6}, {"error": {"retry": True, "message": "stage died"}})
        return stream_lines({"token_id": 7}, {"token_id": 99})

    assert reply_through(gguf_pool(), handler) == [5, 6, 7, 99]
    assert requests[0]["input_ids"] == [1, 2] and requests[0]["written"] == []
    assert requests[1]["input_ids"] == [1, 2], "the prompt is read as it always is"
    assert requests[1]["written"] == [5, 6]
    assert requests[1]["max_tokens"] == 4, "only what comes after counts"


def test_every_new_stream_carries_all_that_was_written():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) < 3:
            return stream_lines({"token_id": 4 + len(requests)}, {"error": {"retry": True, "message": "again"}})
        return stream_lines({"token_id": 99})

    assert reply_through(gguf_pool(), handler) == [5, 6, 99]
    assert [item["written"] for item in requests] == [[], [5], [5, 6]]
    assert {tuple(item["input_ids"]) for item in requests} == {(1, 2)}


def test_a_prompt_with_images_still_replays_as_rows():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream_lines({"token_id": 5}, {"error": {"retry": True, "message": "stage died"}})
        return stream_lines({"token_id": 99})

    async def scenario():
        orchestrator = gguf_pool()
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orchestrator.route = [stage("w1")]
        orchestrator.tokenizer = type("Stops", (), {"stop_token_ids": {99}})()
        session = SessionState(tokens=[1, 2], epoch=0, needs_reset=True)
        session.images = ["aW1hZ2U="]
        async for token in orchestrator.token_stream(session, 5, 0.0, 1.0, 0, None):
            session.tokens.append(token)
        await orchestrator.close()

    asyncio.run(scenario())
    assert requests[1]["input_ids"] == [1, 2, 5] and requests[1]["written"] == []


def test_a_pytorch_reply_still_replays_as_rows():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream_lines({"token_id": 5}, {"error": {"retry": True, "message": "stage died"}})
        return stream_lines({"token_id": 99})

    orchestrator = pool(w1=record())
    orchestrator.active = SimpleNamespace(adapter=object())
    assert reply_through(orchestrator, handler) == [5, 99]
    assert requests[1]["input_ids"] == [1, 2, 5] and requests[1]["written"] == []


def lost_stream_pool(answers):
    """A stream whose connection drops part way, on a worker that does or does not still answer."""
    requests = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200 if answers else 503, json={})
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            raise httpx.ReadError("connection lost")
        return stream_lines({"token_id": 99})

    return requests, handler


def test_a_lost_connection_to_a_worker_that_answers_opens_a_new_stream():
    requests, handler = lost_stream_pool(answers=True)
    orchestrator = gguf_pool(w1=record())
    assert reply_through(orchestrator, handler) == [99]
    assert len(requests) == 2 and not orchestrator.workers["w1"].quarantined


def test_an_entry_worker_that_died_is_repaired_and_the_reply_resumes():
    """The slow way would read the reply back as prompt rows."""
    requests, handler = lost_stream_pool(answers=False)
    orchestrator = gguf_pool(w1=record())
    repairs = []

    async def rebalance():
        repairs.append(True)
        orchestrator.workers["w1"].quarantined = False

    orchestrator.rebalance = rebalance
    assert reply_through(orchestrator, handler) == [99]
    assert repairs and len(requests) == 2


# A route that stays broken costs a reply one wait, and not one for each step.


def test_a_request_on_a_broken_route_leaves_the_waiting_to_the_reply(monkeypatch):
    """token_stream waits too, and waiting in both would double what the guide promises."""
    monkeypatch.setattr(orchestrator_module, "MOVE_WAIT_SECONDS", 30.0)
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = SimpleNamespace(embedding=False)
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    asyncio.run(asyncio.wait_for(orchestrator_module.check_request(None), 2))
    done(orchestrator)


def test_a_request_with_no_route_yet_still_waits_for_one(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator := pool(w1=record("w1")))
    orchestrator.tokenizer = object()
    orchestrator.job = orchestrator_module.LoadJob(job_id="j", model_id="m")

    async def scenario():
        async def back():
            await asyncio.sleep(0.05)
            orchestrator.route = [stage("w1")]

        returned = asyncio.ensure_future(back())
        await orchestrator_module.check_request(None)
        assert returned.done()

    asyncio.run(scenario())
    done(orchestrator)


def test_the_slow_way_does_not_wait_again_for_every_token(monkeypatch):
    """A token that went through says the route works, whatever the records say."""
    monkeypatch.setattr(orchestrator_module, "MOVE_WAIT_SECONDS", 30.0)
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"token_id": 7})))
        session = SessionState(tokens=[1, 2], epoch=3, needs_reset=False)
        got = await asyncio.wait_for(orchestrator.next_token(session, 0.0, 1.0, 0, None), 2)
        await orchestrator.close()
        return got

    assert asyncio.run(scenario()) == 7


def test_a_replay_on_the_slow_way_still_waits_for_the_repair():
    orchestrator = serving_pool()
    restart_empty(orchestrator)
    orchestrator.active = object()
    asked = []

    def handler(request):
        asked.append(orchestrator.workers["w1"].health["epoch"])
        return httpx.Response(200, json={"token_id": 7})

    async def repair():
        await asyncio.sleep(0.05)
        orchestrator.workers["w1"].health = {"status": "ok", "epoch": 3}

    async def scenario():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        repaired = asyncio.ensure_future(repair())
        await orchestrator.next_token(SessionState(tokens=[1, 2], epoch=3, needs_reset=True), 0.0, 1.0, 0, None)
        await repaired
        await orchestrator.close()

    asyncio.run(scenario())
    assert asked == [3]


def stuck_stage(reason):
    from types import SimpleNamespace

    def fails():
        raise reason

    return SimpleNamespace(host=object(), respawn=fails)


def failed_restart(reason):
    from types import SimpleNamespace

    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    async def scenario():
        worker = SimpleNamespace(
            replacing=asyncio.Lock(), gguf=stuck_stage(reason), last_error=None, caches={}, stage_restarts=0, draft=None, draft_request=None, worker_id="w2", ring_live=True, ring_epoch=None, ring_door=None, ring_client=None
        )
        worker.ring_stop = MethodType(WorkerRuntime.ring_stop, worker)
        return worker, await WorkerRuntime.replace_stage(worker, StageLost("gone", worker.gguf.host))

    return asyncio.run(scenario())


def test_a_stage_that_will_not_restart_names_its_worker():
    from isharded.llamacpp import LlamaError

    worker, answer = failed_restart(LlamaError("the file is gone"))
    assert answer.status_code == 502
    assert answer.detail["failed_worker"] == "w2"
    assert "would not restart" in answer.detail["message"] and "the file is gone" in answer.detail["message"]
    assert "would not restart" in worker.last_error
    assert worker.stage_restarts == 0


def test_the_stage_before_passes_on_who_would_not_restart():
    """Blame stays with the stage that saw it. Its neighbour must not take it, or lose it."""
    from isharded.llamacpp import LlamaError
    from isharded.worker import failure_report

    _, answer = failed_restart(LlamaError("no"))
    assert failure_report(httpx.Response(answer.status_code, json={"detail": answer.detail})) == answer.detail


def test_a_failed_restart_is_no_refused_request_and_the_reply_goes_on(monkeypatch):
    from isharded.llamacpp import LlamaError
    from isharded.orchestrator import Orchestrator

    _, answer = failed_restart(LlamaError("no"))
    orchestrator = Orchestrator()
    response = httpx.Response(answer.status_code, json={"detail": answer.detail})
    assert not orchestrator._refused_request(response), "that drops the reply and blames nobody"
    assert Orchestrator._failed_worker(response, "w1") == "w2"

    quarantined, repairs = [], []
    monkeypatch.setattr(orchestrator, "_quarantine", lambda worker, reason: quarantined.append(worker))

    async def rebalance():
        repairs.append(1)

    monkeypatch.setattr(orchestrator, "rebalance", rebalance)
    assert asyncio.run(orchestrator._recover_stream(answer.detail, 3)), "a new stream follows on the new route"
    assert quarantined == ["w2"] and repairs == [1]


def test_a_stage_that_will_not_restart_fails_the_check_a_report_gets(monkeypatch):
    """A follower reports the node to the coordinator, which only believes what /health says."""
    from isharded import worker as worker_module
    from isharded.llamacpp import LlamaError

    worker, _ = failed_restart(LlamaError("no"))
    assert worker.stage_down is worker.gguf

    stage = SimpleNamespace(refresh_states=lambda: None, ring_stats=lambda: None, states={})
    monkeypatch.setattr(worker_module, "authorized", lambda _: True)
    monkeypatch.setattr(worker_module.runtime, "gguf", stage)
    monkeypatch.setattr(worker_module.runtime, "stage_down", stage)
    client = TestClient(worker_module.app)
    assert client.get("/health").status_code == 503
    monkeypatch.setattr(worker_module, "authorized", lambda _: False)
    assert client.get("/health").json() == {"status": "ok"}, "a stranger hears nothing of it"

    monkeypatch.setattr(worker_module, "authorized", lambda _: True)
    monkeypatch.setattr(worker_module.runtime, "stage_down", object())  # the stage of an older route
    assert client.get("/health").status_code == 200


def test_a_coordinator_believes_a_report_of_a_stage_that_will_not_restart(monkeypatch):
    orchestrator = gguf_pool(w1=record())
    orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503, json={})))
    quarantined = []
    monkeypatch.setattr(orchestrator, "_quarantine", lambda worker, reason: quarantined.append(worker))

    async def rebalance(**_):
        pass

    monkeypatch.setattr(orchestrator, "rebalance", rebalance)
    asyncio.run(orchestrator.handle_report("w1", "reply failed"))
    assert quarantined == ["w1"]


def test_a_stage_that_restarts_takes_back_its_good_name():
    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    async def scenario():
        stage = RestartingStage()
        worker = SimpleNamespace(replacing=asyncio.Lock(), gguf=stage, last_error=None, caches={}, stage_restarts=0, draft=None, draft_request=None, ring_live=True, stage_down=stage, ring_epoch=None, ring_door=None, ring_client=None)
        worker.ring_stop = MethodType(WorkerRuntime.ring_stop, worker)
        await WorkerRuntime.replace_stage(worker, StageLost("gone", stage.host))
        return worker

    assert asyncio.run(scenario()).stage_down is None


def test_a_lost_stage_takes_its_ring_with_it(monkeypatch):
    from types import SimpleNamespace

    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    async def scenario():
        worker = SimpleNamespace(replacing=asyncio.Lock(), gguf=RestartingStage(), last_error=None, caches={}, stage_restarts=0, draft=None, draft_request=None, ring_live=True, ring_epoch=None, ring_door=None, ring_client=None)
        worker.ring_stop = MethodType(WorkerRuntime.ring_stop, worker)
        await WorkerRuntime.replace_stage(worker, StageLost("gone", worker.gguf.host))
        return worker

    assert asyncio.run(scenario()).ring_live is False


def test_a_connection_that_broke_is_not_blamed_on_the_stage_behind_it() -> None:
    """A kept connection can break while the next stage is fine. Someone should ask it first."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(stage_answering(200, {}, httpx.RemoteProtocolError("Server disconnected without sending a response."))())
    assert caught.value.status_code == 502
    assert caught.value.detail["lost"] == caught.value.detail["failed_worker"] == "next"


def test_a_stage_that_does_not_answer_is_still_blamed_at_once() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(stage_answering(200, {}, httpx.ReadTimeout("nothing came back"))())
    assert caught.value.detail["failed_worker"] == "next"
    assert "lost" not in caught.value.detail


def test_a_stale_connection_to_the_next_stage_replays_the_reply_and_moves_nothing() -> None:
    """The failure a worker reports for one, as it reaches the coordinator."""
    from isharded.worker import failure_report

    orchestrator = pool(one=record("one"), two=record("two"))
    detail = {"failed_worker": "two", "lost": "two", "message": "Server disconnected"}
    assert failure_report(httpx.Response(502, json={"detail": detail})) == detail
    rebalanced = []

    async def rebalance(*args, **options):
        rebalanced.append(1)

    orchestrator.rebalance = rebalance

    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        return await orchestrator._recover_stream(detail, 1)

    assert asyncio.run(run())
    assert not orchestrator.workers["two"].quarantined and not rebalanced
    done(orchestrator)


def test_a_broken_connection_to_a_stage_that_also_stopped_answering_is_a_failure() -> None:
    orchestrator = pool(one=record("one"), two=record("two"))
    rebalanced = []

    async def rebalance(*args, **options):
        rebalanced.append(1)

    orchestrator.rebalance = rebalance

    def gone(request):
        raise httpx.ConnectError("refused")

    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(gone))
        return await orchestrator._recover_stream({"failed_worker": "two", "lost": "two", "message": "gone"}, 1)

    assert asyncio.run(run())
    assert orchestrator.workers["two"].quarantined and rebalanced
    done(orchestrator)


def test_a_real_kept_connection_that_went_stale_reaches_recovery() -> None:
    """A real socket, kept by a node's own pool, closed by a peer after the request went out."""
    from isharded.tls import node_transport

    async def scenario():
        seen = []

        async def peer(reader, writer):
            while await reader.readline():
                while (await reader.readline()).strip():
                    pass
                seen.append(1)
                if len(seen) > 1:
                    writer.close()
                    return
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()

        server = await asyncio.start_server(peer, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server, httpx.AsyncClient(transport=node_transport(), trust_env=False) as client:
            assert (await client.get(f"http://127.0.0.1:{port}/")).text == "ok"
            with pytest.raises(httpx.HTTPError) as caught:
                await client.get(f"http://127.0.0.1:{port}/")
            return caught.value

    from isharded.worker import BROKEN_CONNECTION

    assert isinstance(asyncio.run(scenario()), BROKEN_CONNECTION)


def test_a_restarted_stage_leaves_the_ring_so_that_the_ring_can_come_back() -> None:
    """Its links died with the process. Saying the ring is live would keep it down for good."""
    from isharded.llamacpp import StageLost
    from isharded.worker import WorkerRuntime

    class Stage:
        host = object()

        def respawn(self):
            self.host = object()

    class Client:
        alive, closed = False, False

        def close(self):
            self.closed = True

    async def scenario():
        worker = WorkerRuntime.__new__(WorkerRuntime)
        worker.replacing, worker.gguf, worker.last_error, worker.caches = asyncio.Lock(), Stage(), None, {}
        worker.stage_restarts, worker.draft, worker.draft_request = 0, None, None
        worker.ring_live, worker.ring_epoch, worker.ring_door, worker.ring_client = True, 3, {}, Client()
        client = worker.ring_client
        await worker.replace_stage(StageLost("gone", worker.gguf.host))
        return worker, client

    worker, client = asyncio.run(scenario())
    assert not worker.ring_live and worker.ring_epoch is None and client.closed
    assert worker.stage_restarts == 1


def test_a_coordinator_reopens_the_ring_of_an_entry_whose_stage_restarted() -> None:
    orchestrator = pool(one=record("one", health={"telemetry": {"ring_live": False}}), two=record("two", health={"telemetry": {}}))
    orchestrator.route = [stage("one", 0, 1), stage("two", 1, 2)]
    orchestrator.epoch = orchestrator.ring_epoch = 4
    orchestrator._check_ring()
    assert orchestrator.ring_epoch is None
    done(orchestrator)


def links_pool():
    orchestrator = pool(one=record("one"), two=record("two"))

    def handler(request):
        return httpx.Response(200, json={"100": {"median_ms": 5.0, "max_ms": 6.0}, "4096": {"median_ms": 50.0, "max_ms": 60.0}})

    return orchestrator, handler


def test_a_preview_leaves_the_running_models_link_costs_alone() -> None:
    orchestrator, handler = links_pool()
    measure(orchestrator, handler, 0)
    before = dict(orchestrator.links), dict(orchestrator.links_checked), dict(orchestrator.link_readings), orchestrator.link_row_bytes

    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await orchestrator.preview_links(4096)

    costs = asyncio.run(run())
    assert costs[("one", "two")] > before[0][("one", "two")]
    assert (orchestrator.links, orchestrator.links_checked, orchestrator.link_readings, orchestrator.link_row_bytes) == before
    done(orchestrator)


def test_a_preview_of_the_same_rows_reuses_what_is_fresh() -> None:
    orchestrator, handler = links_pool()
    measure(orchestrator, handler, 4096)
    asked = []

    def counting(request):
        asked.append(1)
        return handler(request)

    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(counting))
        return await orchestrator.preview_links(4096)

    assert asyncio.run(run()) == orchestrator.links and not asked
    done(orchestrator)


def test_the_plan_preview_route_plans_by_its_own_link_costs(monkeypatch) -> None:
    from fastapi.testclient import TestClient as Client

    orchestrator, handler = links_pool()
    measure(orchestrator, handler, 0)
    running = dict(orchestrator.links)
    seen = {}

    async def nothing(*args, **kwargs):
        return {}

    async def preview(row):
        return {("one", "two"): 99.0}

    def plan(target, **options):
        seen["links"] = options["links"]
        return []

    orchestrator.gguf_sources = orchestrator.refresh_worker_metrics = orchestrator.refresh_holdings = orchestrator.choose_slots = nothing
    orchestrator.preview_links = preview
    orchestrator.plan_or_refuse = plan
    orchestrator.plan_memory = lambda target, route: {}
    monkeypatch.setattr(orchestrator_module, "runtime", orchestrator)
    monkeypatch.setattr(orchestrator_module, "build_target", lambda *args: SimpleNamespace())
    monkeypatch.setattr(orchestrator_module, "row_bytes", lambda target: 4096)
    monkeypatch.setattr(orchestrator_module, "describe_plan", lambda target, route, links: {"links": len(links)})
    answer = Client(orchestrator_module.app).post("/admin/plan", json={"model_id": "m"})
    assert answer.status_code == 200 and answer.json()["links"] == 1
    assert seen["links"] == {("one", "two"): 99.0}
    assert orchestrator.links == running
    done(orchestrator)


def test_a_plan_can_be_given_link_costs_of_its_own() -> None:
    from isharded.orchestrator import describe_plan

    route = [stage("one", 0, 1), stage("two", 1, 2)]
    target = SimpleNamespace(
        profile=SimpleNamespace(ring=False, total_bytes=1, context_tokens=1), adapter=SimpleNamespace(config={}, num_layers=2), units={}, held={}, model_id="m",
        draft=None, embedding=False, projector=None, slots=1, parallel={},
    )  # fmt: skip
    near = describe_plan(target, route, {("one", "two"): 1.0})
    far = describe_plan(target, route, {("one", "two"): 500.0})
    assert near["estimated_tokens_per_second"] > far["estimated_tokens_per_second"]


def test_readings_for_workers_that_left_are_forgotten() -> None:
    """A pool with members coming and going would otherwise hold a reading for every pair ever seen."""
    orchestrator, handler = links_pool()
    orchestrator.workers["three"] = record("three")
    measure(orchestrator, handler, 0)
    assert len(orchestrator.link_readings) == 6
    del orchestrator.workers["three"]
    orchestrator.workers["two"].quarantined = True
    measure(orchestrator, handler, 0)
    for table in (orchestrator.links, orchestrator.links_checked, orchestrator.link_readings):
        assert not table
    done(orchestrator)


def test_a_worker_that_comes_back_is_measured_afresh() -> None:
    orchestrator, handler = links_pool()
    measure(orchestrator, handler, 0)
    orchestrator.workers["two"].quarantined = True
    measure(orchestrator, handler, 0)
    orchestrator.workers["two"].quarantined = False
    measure(orchestrator, handler, 0)
    assert set(orchestrator.links) == {("one", "two"), ("two", "one")}
    done(orchestrator)


def warming_pool(handler):
    orchestrator = pool(one=record("one"))
    orchestrator.route = [stage("one", 0, 1)]
    orchestrator.epoch = 5
    orchestrator.active = SimpleNamespace(embedding=False)
    orchestrator.tokenizer = SimpleNamespace(encode=lambda text: [7, 8])
    orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return orchestrator


def test_a_new_route_runs_one_reply_of_64_rows_and_clears_it_away() -> None:
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        return httpx.Response(200, text='{"token_id": 1}\n')

    async def run():
        orchestrator = warming_pool(handler)
        await orchestrator._warm_route(5)
        await orchestrator.close()

    asyncio.run(run())
    (method, path, body), (cleared, gone, _) = seen
    assert (method, path) == ("POST", "/v1/generate")
    assert (body["input_ids"], body["max_tokens"], body["epoch"]) == ([7, 8] * 32, 1, 5)
    assert (cleared, gone) == ("DELETE", f"/internal/sessions/{body['session_id']}")


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_a_warm_pass_that_fails_blames_nobody(failure) -> None:
    def handler(request):
        raise failure

    async def run():
        orchestrator = warming_pool(handler)
        await orchestrator._warm_route(5)
        await orchestrator.close()
        return orchestrator.workers["one"]

    assert not asyncio.run(run()).quarantined


@pytest.mark.parametrize(("epoch", "embedding", "replying"), [(4, False, 0), (5, True, 0), (5, False, 1)])
def test_there_is_nothing_to_warm_on_some_routes(epoch, embedding, replying) -> None:
    """One that moved on, one that makes no replies, and one a real reply is already warming."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    async def run():
        orchestrator = warming_pool(handler)
        orchestrator.active.embedding = embedding
        orchestrator.replies_in_flight = replying
        await orchestrator._warm_route(epoch)
        await orchestrator.close()

    asyncio.run(run())
    assert not seen


def test_an_answer_lost_to_a_broken_connection_asks_before_it_blames() -> None:
    """A kept connection can break under a healthy entry. Its model should not move for that."""
    orchestrator = pool(one=record("one"))
    orchestrator.route, orchestrator.epoch = [stage("one", 0, 1)], 3
    posts = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={})
        posts.append(request)
        if len(posts) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, json={"done": True})

    async def run():
        orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await orchestrator.ask_entry("/v1/embed", {}, 5.0)

    assert asyncio.run(run()) == {"done": True}
    assert len(posts) == 2 and not orchestrator.workers["one"].quarantined
    done(orchestrator)
