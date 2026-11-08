"""Handover step one: a route change with the old route serving until the new one is proven.

The engine's side, two routes on one Node, is tested in engine/tests/ring.rs on real links.
These cover the worker's and the coordinator's side with the network faked, and the spare
stage process. A real pool's cordon is measured in docs/RESULTS.md.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from isharded import gguf_stage
from isharded import llama_host
from isharded import orchestrator as orchestrator_module
from isharded import worker
from isharded.models import GgufAdapter
from isharded.orchestrator import Orchestrator, WorkerRecord
from isharded.protocol import PROTOCOL_VERSION, ConfigureRequest, HardwareInfo, StageAssignment, WorkerRegistration
from isharded.slots import SlotGate


def stage(worker_id, start=0, end=1):
    return StageAssignment(worker_id=worker_id, url=f"http://{worker_id}:1", start_layer=start, end_layer=end, placements=[])


def registration(worker_id):
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=4, ram_total_bytes=1, host_id=worker_id)
    return WorkerRegistration(worker_id=worker_id, advertised_url=f"http://{worker_id}:1", hardware=hardware, protocol_version=PROTOCOL_VERSION)


def record(worker_id, staged=True):
    health = {"status": "ok", "staged": 1} if staged else {"status": "ok"}
    return WorkerRecord(registration=registration(worker_id), last_seen=time.monotonic(), health=health)


class Tokenizer:
    def encode(self, text):
        return [11, 12, 13]


def target(slots=2):
    adapter = GgufAdapter.__new__(GgufAdapter)
    return SimpleNamespace(
        model_id="m", adapter=adapter, embedding=False, lookup_guesses=0, speculate_tokens=0, slots=slots, tokenizer=Tokenizer(),
        manifest=SimpleNamespace(revision="r", config={}), profile=SimpleNamespace(context_tokens=512, total_bytes=1), rewinds=False,
        prompt_lookup=False, draft=None, merge_passes=True, exact=False, request=None, projector=None,
    )  # fmt: skip


class Answer:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = body if body is not None else {}
        self.text = str(self.body)

    def json(self):
        return self.body


class Network:
    """Stands in for every worker. Records each call with the route the coordinator had then."""

    def __init__(self, orchestrator, fail=None):
        self.orchestrator = orchestrator
        self.calls = []
        self.fail = fail or {}

    async def post(self, url, json=None, headers=None, timeout=None):
        worker_id, path = url.split("//")[1].split(":")[0], "/" + url.split("/", 3)[3]
        routed = [item.worker_id for item in self.orchestrator.route]
        self.calls.append((worker_id, path, json, routed))
        if (worker_id, path) in self.fail:
            return Answer(self.fail[(worker_id, path)], {"detail": "no"})
        if path in ("/internal/ring/open", "/internal/ring/stage"):
            return Answer(body={"engine": True, "address": f"{worker_id}:7", "port": 7, "fingerprint": "f"})
        if path == "/internal/ring/warm":
            return Answer(body={"ms": 3.5, "token": 1})
        if path == "/internal/configure":
            return Answer(body={"status": "configured", "flip_quiet": True if json.get("staged") else None})
        return Answer(body={})

    def paths(self, worker_id=None):
        return [(who, path) for who, path, _, _ in self.calls if worker_id in (None, who)]

    async def aclose(self):
        pass


def serving(monkeypatch, old, staged=True, **records):
    """A coordinator serving ``old`` on a ring, with fakes for the network and the side work."""
    orchestrator = Orchestrator()
    orchestrator.workers = records or {name: record(name, staged) for name in ("a", "b", "c")}
    orchestrator.active = target()
    orchestrator.model_id = "m"
    orchestrator.tokenizer = Tokenizer()
    orchestrator.route = old
    orchestrator.epoch = orchestrator.ring_epoch = 4

    async def nothing(*args):
        return None

    orchestrator._clear_old_copies = nothing
    orchestrator._gossip_round = nothing
    network = Network(orchestrator)
    orchestrator.client = network
    return orchestrator, network


def run(coroutine):
    return asyncio.run(coroutine)


# ---- The coordinator ----


def test_staged_switching_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("ISHARDED_STAGED", raising=False)
    assert not orchestrator_module.staged_transitions()
    monkeypatch.setenv("ISHARDED_STAGED", "1")
    assert orchestrator_module.staged_transitions()


def deploying(monkeypatch, staged_result=None):
    orchestrator, network = serving(monkeypatch, [stage("a"), stage("b", 1, 2)])
    monkeypatch.setattr(orchestrator_module, "row_bytes", lambda target: 0)
    calls = []

    async def nothing(*args):
        return None

    async def prefetch(*args):
        return {}

    async def activate(target, route):
        calls.append("activate")
        return {}

    async def activate_staged(target, route):
        calls.append("staged")
        return staged_result

    async def lost():
        calls.append("waited")

    orchestrator.refresh_worker_metrics = nothing
    orchestrator.refresh_holdings = nothing
    orchestrator.measure_links = nothing
    orchestrator.choose_slots = nothing
    orchestrator._prefetch = prefetch
    orchestrator._activate = activate
    orchestrator._activate_staged = activate_staged
    orchestrator._await_lost_workers = lost
    orchestrator.keep_proposal = nothing
    orchestrator.plan_or_refuse = lambda target: [stage("a"), stage("c", 1, 2)]
    return orchestrator, calls


def test_with_staging_off_a_cordon_waits_and_switches_the_usual_way(monkeypatch):
    monkeypatch.delenv("ISHARDED_STAGED", raising=False)
    orchestrator, calls = deploying(monkeypatch)
    run(orchestrator.move_off())
    assert calls == ["waited", "activate"]


def test_a_staged_cordon_waits_for_nobody_and_switches_staged(monkeypatch):
    monkeypatch.setenv("ISHARDED_STAGED", "1")
    orchestrator, calls = deploying(monkeypatch, staged_result=True)
    run(orchestrator.move_off())
    assert calls == ["staged"]


def test_a_staged_switch_that_is_called_off_falls_back_to_the_usual_way(monkeypatch):
    monkeypatch.setenv("ISHARDED_STAGED", "1")
    orchestrator, calls = deploying(monkeypatch, staged_result=False)
    run(orchestrator.move_off())
    assert calls == ["staged", "activate"]


def test_only_a_cordon_is_staged(monkeypatch):
    monkeypatch.setenv("ISHARDED_STAGED", "1")
    orchestrator, calls = deploying(monkeypatch, staged_result=True)
    run(orchestrator.rebalance())
    assert calls == ["waited", "activate"]


def test_what_cannot_be_staged_yet_is_named(monkeypatch):
    old = [stage("a", 0, 1), stage("b", 1, 2)]
    orchestrator, _ = serving(monkeypatch, old)
    refusal = orchestrator._staged_refusal
    assert refusal(target(), [stage("a", 0, 1), stage("c", 1, 2)], old) is None
    assert "other layers" in refusal(target(), [stage("a", 0, 2)], old)
    assert refusal(target(), old, old) == "nothing changes"
    orchestrator.workers["c"] = record("c", staged=False)
    assert refusal(target(), [stage("a", 0, 1), stage("c", 1, 2)], old) == "c cannot stage a route"
    orchestrator.workers["c"] = record("c")
    orchestrator.ring_epoch = None
    assert refusal(target(), [stage("a", 0, 1), stage("c", 1, 2)], old) == "the serving route has no native links up"
    orchestrator.ring_epoch = orchestrator.epoch
    monkeypatch.setenv("ISHARDED_RING", "0")
    assert refusal(target(), [stage("a", 0, 1), stage("c", 1, 2)], old) == "the native path is off"


def test_a_staged_switch_keeps_the_old_route_up_until_the_new_one_ran_its_warm_pass(monkeypatch):
    old, new = [stage("a", 0, 1), stage("b", 1, 2)], [stage("a", 0, 1), stage("c", 1, 2)]
    orchestrator, network = serving(monkeypatch, old)
    assert run(orchestrator._activate_staged(orchestrator.active, new))
    paths = network.paths()
    # The new node loads at the new epoch, the ring opens beside the old one, a slot of the old
    # route is lent, the new route proves itself on it, and only then does anything flip.
    assert paths == [
        ("c", "/internal/configure"),
        ("a", "/internal/ring/stage"),
        ("c", "/internal/ring/open"),
        ("a", "/internal/ring/stage/connect"),
        ("c", "/internal/ring/connect"),
        ("a", "/internal/ring/lend"),
        ("a", "/internal/ring/slots"),
        ("a", "/internal/ring/warm"),
        ("a", "/internal/configure"),
    ]
    for worker_id, path, body, routed in network.calls:
        assert routed == ["a", "b"], f"the route was not the old one during {path}"
        assert path not in ("/internal/drain", "/internal/release")
    configures = [body for _, path, body, _ in network.calls if path == "/internal/configure"]
    assert [(body["assignment"]["worker_id"], body["epoch"], body["staged"]) for body in configures] == [("c", 5, False), ("a", 5, True)]
    lend = next(body for _, path, body, _ in network.calls if path == "/internal/ring/lend")
    warm = next(body for _, path, body, _ in network.calls if path == "/internal/ring/warm")
    assert lend["slot"] == warm["slot"] == 1 and warm["epoch"] == 5 and len(warm["tokens"]) == orchestrator_module.WARM_ROWS
    stage_order = next(body for _, path, body, _ in network.calls if path == "/internal/ring/stage")
    assert (stage_order["current_epoch"], stage_order["layers"], stage_order["index"]) == (4, [0, 1], 0)
    assert [item.worker_id for item in orchestrator.route] == ["a", "c"]
    assert (orchestrator.epoch, orchestrator.ring_epoch) == (5, 5)
    assert orchestrator.last_staged["epoch"] == 5 and orchestrator.last_staged["warm_ms"] == 3.5


def test_a_new_entry_goes_live_on_its_ring_and_the_old_one_gets_its_slot_back(monkeypatch):
    old, new = [stage("a", 0, 1), stage("b", 1, 2)], [stage("c", 0, 1), stage("b", 1, 2)]
    orchestrator, network = serving(monkeypatch, old)
    assert run(orchestrator._activate_staged(orchestrator.active, new))
    run(asyncio.sleep(0))
    paths = network.paths()
    assert ("a", "/internal/ring/lend") in paths and ("c", "/internal/ring/warm") in paths
    assert ("c", "/internal/ring/go") in paths
    assert ("a", "/internal/ring/unlend") in paths
    assert ("b", "/internal/ring/stage") in paths and ("b", "/internal/ring/slots") in paths


def test_a_warm_pass_that_fails_calls_the_switch_off_and_leaves_the_old_route(monkeypatch):
    old, new = [stage("a", 0, 1), stage("b", 1, 2)], [stage("a", 0, 1), stage("c", 1, 2)]
    orchestrator, network = serving(monkeypatch, old)
    network.fail[("a", "/internal/ring/warm")] = 502
    assert not run(orchestrator._activate_staged(orchestrator.active, new))
    paths = network.paths()
    assert ("a", "/internal/ring/unstage") in paths and ("a", "/internal/ring/unlend") in paths
    assert not any(body.get("staged") for _, path, body, _ in network.calls if path == "/internal/configure")
    assert [item.worker_id for item in orchestrator.route] == ["a", "b"] and orchestrator.epoch == 4
    assert any(event["title"] == "Staged switch called off" for event in orchestrator.events)


def test_a_new_stage_that_does_not_load_calls_the_switch_off_before_any_ring(monkeypatch):
    old, new = [stage("a", 0, 1), stage("b", 1, 2)], [stage("a", 0, 1), stage("c", 1, 2)]
    orchestrator, network = serving(monkeypatch, old)
    network.fail[("c", "/internal/configure")] = 400
    assert not run(orchestrator._activate_staged(orchestrator.active, new))
    assert not any("ring" in path and path != "/internal/ring/unstage" for _, path in network.paths())
    assert orchestrator.epoch == 4


def test_a_failed_flip_hands_over_to_the_usual_way(monkeypatch):
    old, new = [stage("a", 0, 1), stage("b", 1, 2)], [stage("a", 0, 1), stage("c", 1, 2)]
    orchestrator, network = serving(monkeypatch, old)
    original = network.post

    async def post(url, json=None, headers=None, timeout=None):
        if url.endswith("/internal/configure") and json.get("staged"):
            network.calls.append(("a", "/internal/configure", json, []))
            return Answer(400, {"detail": "a staged flip keeps the stage as it is"})
        return await original(url, json=json, headers=headers, timeout=timeout)

    network.post = post
    assert not run(orchestrator._activate_staged(orchestrator.active, new))
    assert orchestrator.epoch == 4
    assert any(event["title"] == "Staged flip failed" for event in orchestrator.events)


def test_epoch_zero_mode_builds_the_same_configure_order(monkeypatch):
    """The usual path's configure is the same request as before, staged false."""
    orchestrator, _ = serving(monkeypatch, [stage("a")])
    request = orchestrator._configure_request(target(), stage("a"), [stage("a")], 5, None, {}, True)
    assert request.staged is False and request.epoch == 5 and request.slots == 2


# ---- The worker ----


class FakeStage:
    def __init__(self):
        self.calls = []

    def ring_call(self, operation, **meta):
        self.calls.append((operation, meta))
        if operation == "ring_stage":
            return {"engine": True}
        if operation == "ring_flip":
            return {"quiet": True}
        if operation == "ring_close_epoch":
            return {"closed": True}
        return {}

    def ring_stats(self):
        return {"staged_connect": "up"}


def runtime_serving():
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf = FakeStage()
    runtime.term = 3
    runtime.staged_term = 0
    runtime.staged_epoch = None
    runtime.epoch = runtime.ring_epoch = 4
    runtime.ring_term = 3
    runtime.ring_door = {"port": 9001, "fingerprint": "f"}
    runtime.assignment = stage("w", 0, 2)
    runtime.model_id = "m"
    runtime.route = [runtime.assignment]
    runtime.revoked_nodes = set()
    runtime.merge_passes = True
    runtime.exact = False
    runtime.exactness = None
    runtime.advertised_url = "http://127.0.0.1:8100"
    runtime.gate = SlotGate(2)
    runtime.lent_slot = None
    runtime.lend_timer = None
    runtime.flip_quiet = None
    return runtime


def stage_order(**changes):
    body = {"term": 3, "epoch": 5, "index": 0, "count": 2, "keys": {"back": None, "next": "aa", "closing": "bb"}, "current_epoch": 4, "layers": [0, 2], "model_id": "m"}
    return {**body, **changes}


def test_a_kept_stage_stages_the_next_ring_on_its_open_door():
    runtime = runtime_serving()
    door = run(runtime.ring_stage(stage_order()))
    assert door == {"engine": True, "address": "127.0.0.1:9001", "port": 9001, "fingerprint": "f"}
    assert runtime.staged_epoch == 5 and runtime.staged_term == 3
    operation, meta = runtime.gguf.calls[0]
    assert operation == "ring_stage" and (meta["epoch"], meta["index"], meta["count"]) == (5, 0, 2)
    # Serving goes on as it was: the ring this stage serves keeps its epoch and term.
    assert (runtime.epoch, runtime.ring_epoch, runtime.ring_term) == (4, 4, 3)


@pytest.mark.parametrize(
    "changes, why",
    [
        ({"layers": [0, 1]}, "the stage this worker already serves"),
        ({"current_epoch": 3}, "not serving that route"),
        ({"epoch": 4}, "must come after"),
        ({"term": 2}, "term 2"),
        ({"keys": {"sideways": "aa"}}, "a key for each"),
    ],
)
def test_a_staged_ring_is_refused_for_anything_but_the_stage_it_serves(changes, why):
    runtime = runtime_serving()
    with pytest.raises(HTTPException) as refused:
        run(runtime.ring_stage(stage_order(**changes)))
    assert why in str(refused.value.detail)
    assert runtime.staged_epoch is None and not runtime.gguf.calls


def test_an_order_for_the_staged_ring_must_come_from_the_term_that_staged_it():
    runtime = runtime_serving()
    run(runtime.ring_stage(stage_order()))
    runtime.term = 4  # a newer coordinator, which has not staged anything here
    with pytest.raises(HTTPException):
        run(runtime.ring_connect_staged({"term": 4, "epoch": 5, "next": None, "entry": None}))
    with pytest.raises(HTTPException):
        run(runtime.ring_slots({"term": 4, "epoch": 5, "slots": [1]}))


def test_a_flip_hands_every_slot_to_the_staged_ring_and_returns_the_lent_slot():
    runtime = runtime_serving()
    run(runtime.ring_stage(stage_order()))
    assert run(runtime.lend_slot({"term": 3, "slot": 1})) == {"slot": 1}
    assert runtime.gate.free == {0}
    request = ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=5, assignment=runtime.assignment, route=[runtime.assignment], slots=2, term=3, staged=True)
    run(runtime.flip_staged(request))
    assert ("ring_flip", {"epoch": 5, "slots": [0, 1], "wait_ms": worker.FLIP_WAIT_MS}) in runtime.gguf.calls
    assert (runtime.ring_epoch, runtime.staged_epoch, runtime.flip_quiet) == (5, None, True)
    assert runtime.gate.free == {0, 1} and runtime.lent_slot is None


def configuring_runtime():
    runtime = runtime_serving()
    runtime.roles = {"inference"}
    runtime.host_models = []
    runtime.worker_id = "w"
    runtime.ring_live = True
    runtime.ring_client = None
    return runtime


def test_a_staged_flip_is_refused_for_a_stage_that_would_change():
    runtime = configuring_runtime()
    request = ConfigureRequest(model_id="m", revision="r", architecture={}, epoch=5, assignment=runtime.assignment, route=[runtime.assignment], term=3, staged=True)
    runtime.gguf_signature = runtime._stage_signature(request.model_copy(update={"slots": 4}))
    runtime.staged_epoch = 5
    with pytest.raises(ValueError, match="would change"):
        run(runtime.configure(request))
    runtime.gguf_signature = runtime._stage_signature(request)
    runtime.staged_epoch = None
    with pytest.raises(ValueError, match="not staged here"):
        run(runtime.configure(request))
    assert runtime.ring_live, "a refused flip left the ring it serves"


def test_a_lent_slot_waits_for_its_reply_and_is_given_back():
    runtime = runtime_serving()

    async def scenario():
        held = await runtime.gate.acquire(lambda free: 1)
        lending = asyncio.ensure_future(runtime.lend_slot({"term": 3, "slot": 1, "seconds": 5}))
        await asyncio.sleep(0.05)
        assert not lending.done(), "lent while a reply held the slot"
        await runtime.gate.release(held)
        assert await lending == {"slot": 1}
        # While lent, replies get the other slot only.
        assert await runtime.gate.acquire() == 0
        with pytest.raises(HTTPException):
            await runtime.lend_slot({"term": 3, "slot": 0, "seconds": 0.05})
        await runtime.unlend_slot()
        assert 1 in runtime.gate.free and runtime.lend_timer is None

    run(scenario())


def test_a_lend_gives_up_in_time_and_takes_nothing():
    runtime = runtime_serving()

    async def scenario():
        await runtime.gate.acquire(lambda free: 1)
        with pytest.raises(HTTPException) as refused:
            await runtime.lend_slot({"term": 3, "slot": 1, "seconds": 0.05})
        assert refused.value.status_code == 409 and runtime.lent_slot is None

    run(scenario())


def test_unstaging_closes_only_the_staged_ring():
    runtime = runtime_serving()
    run(runtime.ring_stage(stage_order()))
    assert run(runtime.ring_unstage({"term": 3, "epoch": 5})) == {"closed": True}
    assert ("ring_close_epoch", {"epoch": 5}) in runtime.gguf.calls and runtime.staged_epoch is None
    # The route it serves is never closed this way.
    runtime.gguf.calls.clear()
    assert run(runtime.ring_unstage({"term": 3, "epoch": 4})) == {"closed": False}
    assert not runtime.gguf.calls


def test_a_slot_can_be_taken_by_number():
    async def scenario():
        gate = SlotGate(2)
        assert await gate.take(1, 0.01)
        assert not await gate.take(1, 0.01)
        assert not await gate.take(5, 0.01)
        assert await gate.acquire() == 0
        await gate.release(1)
        assert gate.free == {1}

    run(scenario())


# ---- The spare stage process ----


class FakeHost:
    def __init__(self, alive=True):
        self.process = SimpleNamespace(poll=lambda: None if alive else 1)
        self.closed = self.normal = False

    def serve_normally(self):
        self.normal = True

    def close(self):
        self.closed = True


def test_a_spare_is_taken_only_alive_and_for_its_own_library(tmp_path):
    spare = llama_host.Spare()
    spare.host, spare.folder = FakeHost(), str(tmp_path)
    assert spare.take(str(tmp_path / "other")) is None and spare.host is None
    host = FakeHost()
    spare.host, spare.folder = host, str(tmp_path)
    assert spare.take(str(tmp_path)) is host and host.normal and spare.taken == 1
    dead = FakeHost(alive=False)
    spare.host, spare.folder = dead, str(tmp_path)
    assert spare.take(str(tmp_path)) is None and dead.closed


def test_a_spare_comes_up_once_and_then_is_left_alone(monkeypatch, tmp_path):
    started = []

    class Primed(FakeHost):
        def __init__(self, low_priority=False):
            super().__init__()
            started.append(low_priority)

        def call(self, operation, **meta):
            assert operation == "prime"
            return {"folder": meta["folder"], "devices": 1}, b""

    monkeypatch.setattr(llama_host, "StageHost", Primed)
    spare = llama_host.Spare()
    assert spare.prime(str(tmp_path), [])
    assert not spare.prime(str(tmp_path), []), "a second spare started beside the first"
    assert started == [True] and spare.report()["ready"]


def test_only_a_stage_load_takes_the_spare(monkeypatch, tmp_path):
    taken = []

    class Host(FakeHost):
        def call(self, operation, **meta):
            return {}, b""

        def watch(self):
            pass

    monkeypatch.setattr(gguf_stage.SPARE, "take", lambda folder: taken.append(folder) or Host())
    monkeypatch.setattr(gguf_stage, "StageHost", Host)
    gguf_stage.start_host("load_projector", {"folder": str(tmp_path)})
    assert taken == []
    gguf_stage.start_host("load", {"folder": str(tmp_path)})
    assert taken == [str(tmp_path)]
