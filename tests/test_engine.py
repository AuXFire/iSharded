"""The native path: what it refuses, how failures map to recovery, and how a ring gets opened."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import pytest

from isharded import engine as native
from isharded import llama_host
from isharded import orchestrator as orchestrator_module
from isharded import worker as worker_module
from isharded.engine import EngineClient, EngineRefusal, find_library, library_name, pass_frame
from isharded.gguf_stage import DRAFT_TRIAL_MOST, DRAFT_TRIAL_TOKENS, GUESSED, PIPELINE_STRETCH, PIPELINED, PLAIN, TRIAL_GLANCE, DraftChoice
from isharded.orchestrator import Orchestrator
from isharded.protocol import InferRequest, StageAssignment


def request(**fields) -> InferRequest:
    return InferRequest(**{"session_id": "s", "epoch": 3, "input_ids": [1], **fields})


def runtime(**fields):
    item = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    item.ring_live = True
    item.ring_epoch = 3
    item.ring_client = SimpleNamespace(alive=True)
    item.epoch = 3
    item.route = [SimpleNamespace(worker_id="a"), SimpleNamespace(worker_id="b")]
    for key, value in fields.items():
        setattr(item, key, value)
    return item


def test_the_library_is_optional_and_can_be_turned_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ISHARDED_ENGINE_LIB", raising=False)
    monkeypatch.setattr(native, "__file__", str(tmp_path / "engine.py"))
    assert find_library() is None
    made = tmp_path / library_name()
    made.write_bytes(b"x")
    monkeypatch.setenv("ISHARDED_ENGINE_LIB", str(made))
    assert find_library() == made
    monkeypatch.setenv("ISHARDED_ENGINE", "0")
    assert find_library() is None


def test_a_pass_frame_has_the_layout_the_engine_reads() -> None:
    frame = pass_frame(9, 2**64 - 3, 3, 1, 40, 38, [5, 6, 7], [6, 7], True, 0.5, 0.9, 40, 123)
    magic, version, kind, length = native.HEADER.unpack_from(frame)
    assert (magic, version, kind, length) == (native.MAGIC, native.WIRE_VERSION, native.KIND_PASS, len(frame) - 8)
    fields = struct.unpack_from("<QQQIIIBffIBQ", frame, 8)
    assert fields[:7] == (9, 2**64 - 3, 3, 1, 40, 38, native.FLAG_TOKENS | native.FLAG_VERIFY)
    assert fields[9:] == (40, 1, 123)
    at = 8 + struct.calcsize("<QQQIIIBffIBQ")
    assert struct.unpack_from("<I2I", frame, at) == (2, 6, 7)
    assert struct.unpack_from("<I3I", frame, at + 12) == (3, 5, 6, 7)
    assert frame[-8:] == bytes(8)


def test_no_rollback_and_no_seed_have_their_own_encoding() -> None:
    frame = pass_frame(1, 5, 1, 0, 0, None, [1], [], False, 0.0, 1.0, 0, None)
    fields = struct.unpack_from("<QQQIIIBffIBQ", frame, 8)
    assert fields[5] == native.NO_ROLLBACK
    assert fields[10:] == (0, 0)


def test_only_a_pass_that_asks_carries_the_keep_flag() -> None:
    arguments = (1, 5, 1, 0, 0, None, [1], [], False, 0.0, 1.0, 0, None)
    flags = lambda frame: struct.unpack_from("<QQQIIIB", frame, 8)[6]  # noqa: E731
    assert flags(pass_frame(*arguments)) == native.FLAG_TOKENS
    assert flags(pass_frame(*arguments, keep=True)) == native.FLAG_TOKENS | native.FLAG_KEEP
    assert len(pass_frame(*arguments, keep=True)) == len(pass_frame(*arguments))


@pytest.mark.parametrize(("rewinds", "keep"), [(True, False), (False, True)])
def test_only_a_model_that_cannot_rewind_pays_for_snapshots(rewinds, keep) -> None:
    sent = {}

    class Client:
        alive = True

        def send(self, **fields):
            sent.update(fields)
            return "answer"

    item = runtime(ring_client=Client(), rewinds=rewinds, caches={"s": worker_module.CacheEntry(epoch=3, length=4)}, inference_calls=0, tokens_processed=0)
    assert item.ring_send(request(), 7, 4, 4) == "answer"
    assert sent["keep"] is keep
    assert (sent["position"], sent["rollback_to"], sent["tokens"]) == (4, 4, [7])


@pytest.mark.parametrize("rewinds", [True, False])
def test_a_ring_of_two_stages_pipelines_with_or_without_rewinding(rewinds) -> None:
    assert runtime(rewinds=rewinds).ring_pipelines(request())
    assert not runtime(rewinds=rewinds, route=[SimpleNamespace(worker_id="a")]).ring_pipelines(request())
    assert not runtime(rewinds=rewinds, ring_live=False).ring_pipelines(request())


@pytest.mark.parametrize(
    ("operation", "meta", "forgotten"),
    [
        ("clear", {"slot": 2}, [2]),
        ("clear", {}, [None]),
        ("restore", {"slot": 1, "key": "k"}, [1]),
        ("truncate", {"slot": 3, "keep": 9}, [3]),
        ("forward", {"slot": 1}, []),
        ("save", {"slot": 1}, []),
    ],
)
def test_a_stage_forgets_snapshots_only_when_a_slot_changes_history(monkeypatch, operation, meta, forgotten) -> None:
    calls = []

    class Engine:
        def lock(self) -> None:
            calls.append("lock")

        def unlock(self) -> None:
            calls.append("unlock")

        def forget(self, slot) -> None:
            calls.append(slot)

    class Host:
        engine = Engine()

        def __getattr__(self, name):
            return lambda meta, payload: ({}, b"")

    class Connection:
        def __init__(self) -> None:
            self.incoming = [json.dumps({"op": operation, **meta}).encode(), b""]
            self.sent = []

        def recv_bytes(self) -> bytes:
            if not self.incoming:
                raise EOFError
            return self.incoming.pop(0)

        def send_bytes(self, data: bytes) -> None:
            self.sent.append(data)

    connection = Connection()
    monkeypatch.setattr(llama_host, "Client", lambda address, authkey: connection)
    monkeypatch.setattr(llama_host, "_Host", Host)
    llama_host.run(None, b"")
    assert json.loads(connection.sent[0])["ok"]
    assert calls == ["lock", *forgotten, "unlock"]  # under the lock, so no pass can land between


def test_forgetting_every_slot_reaches_the_library_as_a_negative_slot() -> None:
    told = []
    item = native.Engine.__new__(native.Engine)
    item.handle = 5
    item.lib = SimpleNamespace(isharded_engine_forget=lambda handle, slot: told.append((handle, slot)))
    item.forget(None)
    item.forget(3)
    assert told == [(5, -1), (5, 3)]


@pytest.mark.parametrize(
    "fields",
    [
        {"reset": True},
        {"prompt_rows": True},
        {"cuts": [5]},
        {"images": ["x"]},
        {"restore_length": 4},
        {"next_position": 9},
    ],
)
def test_anything_that_changes_what_a_stage_holds_stays_on_http(fields) -> None:
    assert runtime().ring_ready(request()) is True
    assert runtime().ring_ready(request(**fields)) is False


@pytest.mark.parametrize(
    "fields",
    [{"ring_live": False}, {"ring_client": None}, {"ring_client": SimpleNamespace(alive=False)}, {"ring_epoch": 2}],
)
def test_a_ring_that_is_not_up_is_not_used(fields) -> None:
    assert not runtime(**fields).ring_ready(request())


def test_every_refusal_becomes_an_error_that_recovery_already_knows() -> None:
    item = runtime()
    assert item.ring_refused(EngineRefusal(native.REFUSE_EPOCH, 0, "moved")).status_code == 409
    error = item.ring_refused(EngineRefusal(native.REFUSE_STAGE_ERROR, 1, "bad"))
    assert (error.status_code, error.detail) == (500, {"stage_error": "bad"})
    assert item.ring_live  # a refused request says nothing against the ring


def test_a_dead_link_blames_nobody_so_a_live_worker_keeps_its_weights() -> None:
    item = runtime()
    error = item.ring_refused(EngineRefusal(native.REFUSE_FAILED_STAGE, 1, "gone"))
    assert error.status_code == 503 and error.detail["retry"] is True
    assert "failed_worker" not in error.detail
    assert not item.ring_live


def ring_step(client, entry) -> dict:
    item = runtime(ring_client=client, caches={"s": entry})
    item.inference_calls = 0
    item.tokens_processed = 0
    return item, asyncio.run(item.ring_step(request(verify=True, rollback_to=entry.length - 2, input_ids=[1, 2, 3])))


def test_a_verify_pass_rewinds_and_counts_what_it_ran() -> None:
    sent = {}

    class Client:
        alive = True

        async def run(self, deadline, **fields):
            sent.update(fields)
            return [2, 3, 9]

    entry = worker_module.CacheEntry(epoch=3, length=12)
    _, answer = ring_step(Client(), entry)
    assert answer == {"token_ids": [2, 3, 9]}
    assert (sent["position"], sent["rollback_to"], sent["guesses"], sent["verify"]) == (10, 10, [2, 3], True)
    assert entry.length == 13
    assert entry.ring is True


def test_a_session_this_worker_does_not_hold_asks_for_a_replay() -> None:
    item = runtime(caches={})
    with pytest.raises(HTTPException) as caught:
        asyncio.run(item.ring_step(request()))
    assert caught.value.status_code == 503 and caught.value.detail["retry"]


def test_a_pass_that_never_comes_back_asks_for_a_replay() -> None:
    class Client:
        alive = True

        async def run(self, deadline, **fields):
            raise asyncio.TimeoutError

    item = runtime(ring_client=Client(), caches={"s": worker_module.CacheEntry(epoch=3, length=4)})
    with pytest.raises(HTTPException) as caught:
        asyncio.run(item.ring_step(request()))
    assert caught.value.status_code == 503 and caught.value.detail["retry"]
    assert not item.ring_live


def stage(name: str) -> StageAssignment:
    return StageAssignment(worker_id=name, url=f"http://{name}:1", start_layer=0, end_layer=1, placements=[])


def pool(handler, names=("a", "b", "c")) -> Orchestrator:
    orchestrator = Orchestrator()
    orchestrator.route = [stage(name) for name in names]
    orchestrator.epoch = 4
    orchestrator.mesh = None
    orchestrator.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return orchestrator


def run_ring(orchestrator: Orchestrator) -> None:
    asyncio.run(orchestrator._start_ring())


def test_a_pass_out_when_the_stage_is_stopped_fails_though_its_door_stays_open() -> None:
    async def scenario() -> None:
        async def held(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.read()  # a process no kill ended: nothing ever comes back
            writer.close()

        server = await asyncio.start_server(held, "127.0.0.1", 0)
        client = EngineClient(*await asyncio.open_connection(*server.sockets[0].getsockname()[:2]))
        client.task = asyncio.create_task(client._read_loop())
        out = asyncio.get_running_loop().create_future()
        client.waiting[1] = native.Flight(out, 0, 1, 0.0)
        stopped = threading.Event()
        guard = asyncio.create_task(runtime()._close_with_stage(client, stopped))
        stopped.set()
        with pytest.raises(EngineRefusal):
            await asyncio.wait_for(out, 2)
        await guard
        assert not client.alive
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())


def test_a_ring_opens_every_stage_before_any_dials_and_the_last_dials_the_entry() -> None:
    calls = []

    def handler(item):
        body = json.loads(item.content)
        calls.append((item.url.host, item.url.path, body))
        if item.url.path.endswith("/open"):
            return httpx.Response(200, json={"engine": True, "address": f"{item.url.host}:9", "fingerprint": item.url.host * 8})
        return httpx.Response(200, json={})

    orchestrator = pool(handler)
    orchestrator.ring_closing_down = 1.0  # an open ring starts its closing link's clock again
    run_ring(orchestrator)
    assert orchestrator.ring_closing_down is None
    opens = [call for call in calls if call[1].endswith("/open")]
    connects = {call[0]: call[2] for call in calls if call[1].endswith("/connect")}
    assert [call[0] for call in calls[:3]] == ["a", "b", "c"]
    keys = [call[2]["keys"] for call in opens]
    # Each link has its own key, and each stage holds only the keys of its own links.
    assert keys[0]["back"] is None and keys[0]["next"] == keys[1]["back"] and keys[1]["next"] == keys[2]["back"] and keys[2]["next"] is None
    assert keys[0]["closing"] == keys[2]["closing"] and keys[1]["closing"] is None
    assert len({keys[0]["next"], keys[1]["next"], keys[0]["closing"]}) == 3 and all(len(key) == 64 for key in (keys[0]["next"], keys[0]["closing"]))
    assert all(call[2]["term"] == 0 for call in calls)
    assert [call[2]["index"] for call in opens] == [0, 1, 2]
    assert connects["a"]["next"]["address"] == "b:9" and connects["a"]["entry"] is None
    assert connects["c"]["next"] is None and connects["c"]["entry"]["address"] == "a:9"
    assert calls[-1][:2] == ("a", "/internal/ring/go")
    assert orchestrator.ring_epoch == 4


def ring_worker(**fields):
    """A worker with a stage process that records what it is asked."""
    asked = []

    def ring_call(operation, **meta):
        asked.append(operation)
        return {"engine": False}

    closed = []
    item = runtime(term=5, ring_term=5, ring_door={}, revoked_nodes=set(), merge_passes=True, exact=False, exactness=None, gguf=SimpleNamespace(ring_call=ring_call), **fields)
    item.ring_client = SimpleNamespace(alive=True, close=lambda: closed.append(True))
    return item, asked, closed


RING_KEYS = {"back": None, "next": "01" * 32, "closing": "02" * 32}


def test_a_replaced_coordinator_cannot_open_dial_or_start_a_ring() -> None:
    item, asked, _ = ring_worker()
    opening = {"epoch": 3, "index": 0, "count": 2, "keys": RING_KEYS}

    def refused(call) -> int:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(call) if asyncio.iscoroutine(call) else call()
        return caught.value.status_code

    for term in (4, None, "5"):
        assert refused(item.ring_open({**opening, "term": term})) == 409
        assert refused(item.ring_connect({"epoch": 3, "term": term})) == 409
        assert refused(lambda: item.ring_go({"epoch": 3, "term": term})) == 409
    # Only the term that opened a ring may dial it or start it, so a newer one is refused too.
    assert refused(item.ring_connect({"epoch": 3, "term": 6})) == 409
    assert asked == []
    item.ring_go({"epoch": 3, "term": 5})
    asyncio.run(item.ring_open({**opening, "term": 6}))
    assert (item.term, item.ring_term, asked) == (6, 6, ["ring_open"])
    assert refused(item.ring_open({**opening, "term": 5})) == 409


def test_a_reopened_ring_starts_the_stage_count_afresh() -> None:
    """Opening a ring zeroes the engine's totals but not its pass counts, so a window
    across the reopen would share one ring's time out over another's passes."""
    item, asked, _ = ring_worker(ring_counted=(100, 4, 16), ring_busy=(40, 400_000, 1.0), row_ms_ema=None, batch_ms={}, busy_share=None)
    asyncio.run(item.ring_open({"epoch": 3, "term": 5, "index": 0, "count": 2, "keys": RING_KEYS}))
    assert asked == ["ring_open"]
    assert item.ring_counted is None and item.ring_busy is None
    # The new ring has done 60 batches since; the old one's last 10 are not counted with them.
    item.note_ring({"passes": 170, "merged_passes": 4, "merged_rows": 16, "compute_us": 1, "batches_total": 60, "compute_us_total": 600_000})
    assert item.row_ms_ema is None and item.batch_ms == {}, "a first look after a reopen has nothing to compare"


def test_a_ring_needs_keys_for_its_own_links_only() -> None:
    item, asked, _ = ring_worker()
    for keys in (None, "00", {"route": "00"}, {"next": 7}):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(item.ring_open({"epoch": 3, "term": 5, "index": 0, "count": 2, "keys": keys}))
        assert caught.value.status_code == 400
    assert asked == []


def test_a_revoked_node_gets_no_ring_and_its_revocation_closes_one_at_once() -> None:
    item, asked, closed = ring_worker(worker_id="a", pool_token=None, authority=None, last_error=None)
    item.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"revoked_nodes": ["b"]})))
    link = worker_module.ControlLink()
    link.registered = True
    asyncio.run(item._check_in("http://control", link))
    assert closed and asked == ["ring_close"] and not item.ring_live
    with pytest.raises(HTTPException) as caught:
        asyncio.run(item.ring_open({"epoch": 3, "term": 5, "index": 0, "count": 2, "keys": RING_KEYS}))
    assert caught.value.status_code == 409 and asked == ["ring_close"]
    asyncio.run(item._check_in("http://control", link))  # nothing new, so nothing more to close
    assert asked == ["ring_close"]


def test_revoking_a_node_on_the_route_closes_the_ring_at_once_and_keeps_it_shut() -> None:
    calls = []
    orchestrator = pool(lambda item: calls.append((item.url.host, item.url.path)) or httpx.Response(200, json={}))
    orchestrator.ring_epoch = 4

    async def scenario() -> None:
        orchestrator.access.revoked.add("b")
        orchestrator.apply_revocations()
        assert orchestrator.ring_epoch is None
        await asyncio.gather(*orchestrator.background)
        await orchestrator._start_ring()

    asyncio.run(scenario())
    assert sorted(calls) == [("a", "/internal/ring/close"), ("c", "/internal/ring/close")]


def test_a_stage_with_no_engine_leaves_every_reply_on_http() -> None:
    def handler(item):
        engine = item.url.host != "b"
        return httpx.Response(200, json={"engine": engine, "address": "x:1", "fingerprint": "f"})

    orchestrator = pool(handler)
    run_ring(orchestrator)
    assert orchestrator.ring_epoch is None


def test_a_stage_with_no_engine_is_named_once_for_each_route() -> None:
    def handler(item):
        return httpx.Response(200, json={"engine": item.url.host != "b", "address": "x:1", "fingerprint": "f"})

    orchestrator = pool(handler)
    run_ring(orchestrator)
    orchestrator.ring_tried = None  # the retry a later check would make
    run_ring(orchestrator)
    said = [event for event in orchestrator.events if event["title"] == "Native path off"]
    assert len(said) == 1 and "on b," in said[0]["detail"]
    orchestrator.epoch = 5
    run_ring(orchestrator)
    assert sum(event["title"] == "Native path off" for event in orchestrator.events) == 2


def test_a_node_without_the_engine_says_so_when_it_starts(monkeypatch) -> None:
    monkeypatch.setenv("ISHARDED_ENGINE", "0")
    assert any(event["title"] == "No native engine" for event in Orchestrator().events)


def test_a_node_with_the_engine_has_nothing_to_say(tmp_path: Path, monkeypatch) -> None:
    made = tmp_path / library_name()
    made.write_bytes(b"x")
    monkeypatch.setenv("ISHARDED_ENGINE_LIB", str(made))
    monkeypatch.delenv("ISHARDED_ENGINE", raising=False)
    assert not any(event["title"] == "No native engine" for event in Orchestrator().events)


def test_a_failed_ring_is_reported_and_tried_again_later() -> None:
    def handler(item):
        return httpx.Response(502, json={"detail": "no route to host"})

    orchestrator = pool(handler)
    run_ring(orchestrator)
    assert orchestrator.ring_epoch is None
    assert any(event["title"] == "Direct links not up" for event in orchestrator.events)
    tries = orchestrator.ring_tried_at
    run_ring(orchestrator)
    assert orchestrator.ring_tried_at == tries  # not again straight away


def test_the_ring_can_be_turned_off(monkeypatch) -> None:
    monkeypatch.setenv("ISHARDED_RING", "0")
    orchestrator = pool(lambda item: pytest.fail("nothing should be asked"))
    run_ring(orchestrator)


def healthy(worker: dict) -> SimpleNamespace:
    return SimpleNamespace(health={"telemetry": worker})


@pytest.mark.parametrize(
    ("entry", "middle", "stays"),
    [
        ({"ring_live": True, "ring": {"epoch": 4, "next_up": True}}, {"ring": {"epoch": 4, "next_up": True}}, True),
        ({"ring_live": False, "ring": {"epoch": 4, "next_up": True}}, {"ring": {"epoch": 4, "next_up": True}}, False),
        ({"ring_live": True, "ring": {"epoch": 4, "next_up": True}}, {"ring": {"epoch": 4, "next_up": False}}, False),
        ({"ring_live": True, "ring": {"epoch": 4, "next_up": True}}, {"ring": {"epoch": None, "next_up": False}}, False),
        # A stage too busy to say is not a stage in trouble.
        ({"ring_live": True, "ring": None}, {"ring": None}, True),
    ],
)
def test_a_ring_that_lost_a_link_is_noticed(entry, middle, stays) -> None:
    orchestrator = pool(lambda item: httpx.Response(200, json={}), ("a", "b"))
    orchestrator.ring_epoch = 4
    orchestrator.route = [stage("a"), stage("b"), stage("c")]
    orchestrator.workers = {"a": healthy(entry), "b": healthy(middle), "c": healthy({"ring": {"epoch": 4, "next_up": False}})}
    orchestrator._check_ring()
    assert (orchestrator.ring_epoch == 4) is stays


HEARD: dict[int, int] = {}  # each pass id's generation, as the door last heard it


def closing_pool(closing_up: bool | None) -> Orchestrator:
    orchestrator = pool(lambda item: httpx.Response(200, json={}), ("a", "b"))
    orchestrator.ring_epoch = 4
    entry = {"ring_live": True, "ring": {"epoch": 4, "next_up": True, "closing_up": bool(closing_up)}}
    last = {"ring": None if closing_up is None else {"epoch": 4, "next_up": False, "closing_up": closing_up}}
    orchestrator.workers = {"a": healthy(entry), "b": healthy(last)}
    return orchestrator


def test_a_closing_link_that_stays_down_reopens_the_ring_and_a_short_dip_does_not(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    orchestrator = closing_pool(False)
    orchestrator._check_ring()
    now[0] += orchestrator_module.RING_CLOSING_SECONDS - 1
    orchestrator._check_ring()
    assert orchestrator.ring_epoch == 4
    orchestrator.workers["b"] = closing_pool(True).workers["b"]
    orchestrator._check_ring()
    orchestrator.workers["b"] = closing_pool(False).workers["b"]
    orchestrator._check_ring()
    now[0] += orchestrator_module.RING_CLOSING_SECONDS - 1
    orchestrator._check_ring()
    assert orchestrator.ring_epoch == 4
    now[0] += 2
    orchestrator._check_ring()
    assert orchestrator.ring_epoch is None
    assert any("losing link" in str(event) for event in orchestrator.events)


def test_a_closing_link_that_cannot_exist_does_not_reopen_the_ring_every_time(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    orchestrator = closing_pool(False)
    reopened = []
    for _ in range(600):
        orchestrator._check_ring()
        if orchestrator.ring_epoch is None:
            reopened.append(now[0])
            orchestrator.ring_epoch = 4
        now[0] += 5
    assert reopened[1] - reopened[0] > 1.9 * orchestrator_module.RING_CLOSING_SECONDS
    assert len(reopened) < 8


def test_a_stage_too_busy_to_say_is_not_a_closing_link_down(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    orchestrator = closing_pool(None)
    for _ in range(3):
        orchestrator._check_ring()
        now[0] += orchestrator_module.RING_CLOSING_SECONDS
    assert orchestrator.ring_epoch == 4


def test_one_stage_has_no_closing_link(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    orchestrator = closing_pool(False)
    orchestrator.route = orchestrator.route[1:]
    for _ in range(3):
        orchestrator._check_ring()
        now[0] += orchestrator_module.RING_CLOSING_SECONDS
    assert orchestrator.ring_epoch == 4


class FakeDoor:
    """The engine's local door, spoken from the other end."""

    def __init__(self, key: bytes, answers) -> None:
        self.key = key
        self.answers = answers

    async def serve(self, reader, writer) -> None:
        kind, body = await EngineClient._read_frame(reader)
        role, epoch = body[0], struct.unpack_from("<Q", body, 1)[0]
        theirs = body[9:]
        nonce = bytes(32)
        mac = native._proof(self.key, native.SERVER_LABEL, role, epoch, theirs, nonce)
        writer.write(native._frame(native.KIND_HELLO_OK, nonce + mac))
        await writer.drain()
        kind, body = await EngineClient._read_frame(reader)
        assert kind == native.KIND_AUTH
        assert hmac.compare_digest(body, native._proof(self.key, native.CLIENT_LABEL, role, epoch, theirs, nonce))
        while True:
            try:
                kind, body = await EngineClient._read_frame(reader)
            except asyncio.IncompleteReadError:
                return
            identity, HEARD[identity] = struct.unpack_from("<QQ", body)
            writer.write(self.answers(identity))
            await writer.drain()


async def with_door(answers, use, key=b"k" * 32, wrong_key=None):
    server = await asyncio.start_server(FakeDoor(key, answers).serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        client = await EngineClient.open(port, wrong_key or key)
        try:
            return await use(client)
        finally:
            client.close()
    finally:
        server.close()


FIELDS = {
    "epoch": 3,
    "slot": 0,
    "position": 4,
    "rollback_to": None,
    "tokens": [1],
    "guesses": [],
    "verify": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": None,
}


def answer(identity: int, tokens: list[int], generation: int | None = None) -> bytes:
    generation = HEARD[identity] if generation is None else generation
    return native._frame(native.KIND_RESULT, native.RESULT_HEAD.pack(identity, generation, 3, 0, len(tokens)) + struct.pack(f"<{len(tokens)}I", *tokens))


def refusal(identity: int, kind: int, text: str = "no") -> bytes:
    data = text.encode()
    return native._frame(native.KIND_NACK, native.NACK_HEAD.pack(identity, HEARD.get(identity, 0), 3, kind, 2, len(data)) + data)


def test_answers_find_their_own_pass_when_several_are_out() -> None:
    async def use(client):
        return await asyncio.gather(*(client.run(5, **FIELDS) for _ in range(5)))

    got = asyncio.run(with_door(lambda identity: answer(identity, [identity * 10]), use))
    assert got == [[10], [20], [30], [40], [50]]


def test_a_reply_keeps_one_generation_and_the_next_on_its_slot_gets_another() -> None:
    async def use(client):
        for _ in range(2):
            await client.run(5, **FIELDS)
        await client.run(5, **{**FIELDS, "slot": 1})
        assert await client.settled(0, 5)
        await client.run(5, **FIELDS)

    asyncio.run(with_door(lambda identity: answer(identity, [1]), use))
    got = [HEARD[identity] for identity in range(1, 5)]
    assert got[0] == got[1]
    assert len(set(got[1:])) == 3
    assert all(0 < item < 2**64 for item in got)


def test_a_result_from_another_generation_is_dropped_and_the_real_one_still_lands() -> None:
    def answers(identity):
        return answer(identity, [1], HEARD[identity] ^ 2) + answer(identity, [2])

    async def use(client):
        return await client.run(5, **FIELDS)

    assert asyncio.run(with_door(answers, use)) == [2]


def test_a_refusal_from_another_generation_is_dropped() -> None:
    def answers(identity):
        data = native.NACK_HEAD.pack(identity, HEARD[identity] ^ 2, 3, native.REFUSE_STAGE_ERROR, 2, 0)
        return native._frame(native.KIND_NACK, data) + answer(identity, [2])

    async def use(client):
        return await client.run(5, **FIELDS)

    assert asyncio.run(with_door(answers, use)) == [2]


def test_a_repeated_result_changes_nothing() -> None:
    async def use(client):
        got = await client.run(5, **FIELDS)
        return got, await client.run(5, **FIELDS), client.alive

    assert asyncio.run(with_door(lambda identity: answer(identity, [identity]) * 2, use)) == ([1], [2], True)


def test_a_slot_that_will_not_settle_drops_what_is_left_and_starts_again() -> None:
    def answers(identity):
        return b"" if identity == 1 else answer(identity, [identity]) + answer(1, [9])

    async def use(client):
        lost = client.send(**FIELDS)
        settled = await client.settled(0, 0.01)
        return settled, await client.run(5, **FIELDS), lost.cancelled(), client.waiting

    assert asyncio.run(with_door(answers, use)) == (False, [2], True, {})
    assert HEARD[1] != HEARD[2]


def test_a_round_trip_is_measured_once_something_comes_back() -> None:
    async def use(client):
        before = client.round_trip
        await client.run(5, **FIELDS)
        return before, client.round_trip

    before, after = asyncio.run(with_door(lambda identity: answer(identity, [1]), use))
    assert before is None and 0 < after < 5


@pytest.mark.parametrize(
    ("round_trip", "seconds"),
    [
        (None, worker_module.pass_seconds(1, 2)),
        (0.01, worker_module.RING_ANSWER_FLOOR_SECONDS),
        (4.0, worker_module.RING_ACK_MS / 1000 + 8.0),
    ],
)
def test_an_answer_is_waited_for_by_what_the_ring_measured(monkeypatch, round_trip, seconds) -> None:
    waits = []

    async def wait_for(answer, timeout):
        waits.append(timeout)
        raise asyncio.TimeoutError

    monkeypatch.setattr(worker_module.asyncio, "wait_for", wait_for)
    item = runtime(ring_client=SimpleNamespace(alive=True, round_trip=round_trip))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(item.ring_answer(None))
    assert caught.value.detail["retry"] and not item.ring_live
    assert waits == [pytest.approx(seconds)]


def test_a_ring_that_has_left_still_waits_the_old_way(monkeypatch) -> None:
    waits = []

    async def wait_for(answer, timeout):
        waits.append(timeout)
        return [7]

    monkeypatch.setattr(worker_module.asyncio, "wait_for", wait_for)
    assert asyncio.run(runtime(ring_client=None).ring_answer(None)) == 7
    assert waits == [worker_module.pass_seconds(1, 2)]


def test_a_refused_pass_raises_and_the_link_stays_up() -> None:
    async def use(client):
        with pytest.raises(EngineRefusal) as caught:
            await client.run(5, **FIELDS)
        assert (caught.value.refusal, caught.value.stage, caught.value.message) == (native.REFUSE_STAGE_ERROR, 2, "no")
        return client.alive

    assert asyncio.run(with_door(lambda identity: refusal(identity, native.REFUSE_STAGE_ERROR), use))


def test_a_slot_has_settled_when_its_newest_pass_is_back() -> None:
    def answers(identity):
        return answer(identity, [identity])

    async def use(client):
        first = client.send(**FIELDS)
        second = client.send(**FIELDS)
        first.cancel()  # a stage may drop this one, so nothing can wait for it
        settled = await client.settled(0, 5)
        return settled, second.done(), client.waiting

    assert asyncio.run(with_door(answers, use)) == (True, True, {})


def test_a_pass_let_go_leaves_nothing_behind() -> None:
    async def use(client):
        future = client.send(**FIELDS)
        future.cancel()
        await asyncio.sleep(0)
        return client.waiting

    assert asyncio.run(with_door(lambda identity: b"", use)) == {}


def test_settling_says_so_when_a_pass_never_lands() -> None:
    async def use(client):
        client.send(**FIELDS)
        return await client.settled(0, 0.05), await client.settled(1, 0.05)

    assert asyncio.run(with_door(lambda identity: b"", use)) == (False, True)


def test_a_lost_link_fails_every_pass_in_flight_at_once() -> None:
    async def use(client):
        got = await asyncio.gather(*(client.run(5, **FIELDS) for _ in range(3)), return_exceptions=True)
        assert all(isinstance(item, EngineRefusal) and item.refusal == native.REFUSE_FAILED_STAGE for item in got)
        return client.alive

    assert asyncio.run(with_door(lambda identity: refusal(0, native.REFUSE_FAILED_STAGE), use)) is False


def test_the_door_must_prove_it_holds_the_key() -> None:
    async def use(client):
        pytest.fail("this door should have been refused")

    with pytest.raises(OSError, match="does not hold"):
        asyncio.run(with_door(lambda identity: b"", use, wrong_key=b"w" * 32))


def test_the_proof_covers_what_the_rust_side_signs() -> None:
    made = native._proof(b"key", b"label", 3, 7, b"a" * 32, b"b" * 32)
    wanted = hmac.new(b"key", b"label" + bytes([3]) + struct.pack("<Q", 7) + b"a" * 32 + b"b" * 32, hashlib.sha256).digest()
    assert made == wanted


needs_engine = pytest.mark.skipif(find_library() is None, reason="needs the built engine")


@needs_engine
def test_a_real_engine_refuses_a_stale_library_version_check(monkeypatch) -> None:
    import ctypes

    library = ctypes.CDLL(str(find_library()))
    library.isharded_engine_abi.restype = ctypes.c_uint32
    assert library.isharded_engine_abi() == native.ABI_VERSION


def drive(choice: DraftChoice, ways: list[str], speeds: dict[str, float], tokens: int, per_pass: dict[str, int] | None = None) -> list[tuple[str, int]]:
    """Runs ``tokens`` tokens through the choice as the worker does, each way at its speed in
    tokens a second, and gives the runs of each way in order, as (way, tokens)."""
    runs: list[tuple[str, int]] = []
    made = 0
    while made < tokens:
        way = choice.pick(ways)
        left = choice.trial_left(way)
        step = (per_pass or {}).get(way, 1)
        if way == PIPELINED:
            step = PIPELINE_STRETCH if left is None else min(left, PIPELINE_STRETCH)
        choice.note(way, step, step / speeds[way])
        made += step
        if runs and runs[-1][0] == way:
            runs[-1] = (way, runs[-1][1] + step)
        else:
            runs.append((way, step))
    return runs


def test_each_way_gets_a_whole_trial_before_the_choice_settles_plain_first() -> None:
    """Plain decoding costs nothing against a load with no draft, so it is measured first."""
    choice = DraftChoice()
    runs = drive(choice, [GUESSED, PLAIN], {GUESSED: 24.0, PLAIN: 28.0}, 60, {GUESSED: 3})
    assert runs[0] == (PLAIN, PIPELINE_STRETCH), "a first trial is not cut, since nothing is there to compare"
    assert runs[1] == (GUESSED, PIPELINE_STRETCH), "close enough behind for the whole trial"
    assert choice.best([GUESSED, PLAIN]) == PLAIN
    assert choice.trials_won == 1 and choice.trials_lost == 1 and choice.gap(GUESSED) == 2 * DRAFT_TRIAL_TOKENS


def test_a_slow_guessing_way_costs_a_glance_from_the_start() -> None:
    choice = DraftChoice()
    runs = drive(choice, [PIPELINED, PLAIN], {PIPELINED: 14.4, PLAIN: 27.6}, 60)
    assert runs[:2] == [(PLAIN, PIPELINE_STRETCH), (PIPELINED, TRIAL_GLANCE)]


def test_a_way_that_keeps_losing_waits_twice_as_long_for_each_trial() -> None:
    """On 3 A40s the pipelined way lost every trial and still cost 6 of 44 ms a token."""
    choice = DraftChoice()
    runs = drive(choice, [PIPELINED, PLAIN], {PIPELINED: 14.4, PLAIN: 27.6}, 20000)
    gaps = [tokens for way, tokens in runs if way == PLAIN][1:]
    assert gaps[:6] == [192, 384, 768, 1536, 3072, 4096], "the first is the trial of plain itself, then each gap doubles"
    assert set(gaps[6:-1]) == {DRAFT_TRIAL_MOST}, "it stops doubling at the most"
    trials = [tokens for way, tokens in runs if way == PIPELINED][1:]
    assert set(trials) == {TRIAL_GLANCE}, "each trial that far behind stops after a glance"
    trial_tokens = sum(tokens for way, tokens in runs[2:] if way == PIPELINED)
    assert trial_tokens / sum(tokens for _, tokens in runs[2:]) < 0.006


def test_a_trial_close_behind_runs_its_whole_length() -> None:
    choice = DraftChoice()
    runs = drive(choice, [PIPELINED, PLAIN], {PIPELINED: 25.0, PLAIN: 28.0}, 400)
    assert [tokens for way, tokens in runs if way == PIPELINED][1:2] == [PIPELINE_STRETCH]
    assert choice.trials_cut == 0


def test_a_trial_far_behind_is_cut_after_its_first_few_tokens() -> None:
    choice = DraftChoice()
    drive(choice, [PIPELINED, PLAIN], {PIPELINED: 30.0, PLAIN: 30.0}, 48)  # both measured, plain a hair ahead
    choice.note(PLAIN, 10, 10 / 30.0)
    choice.since_trial = choice.gap(PIPELINED)
    assert choice.pick([PIPELINED, PLAIN]) == PIPELINED
    assert choice.trial_left(PIPELINED) == TRIAL_GLANCE
    choice.note(PIPELINED, TRIAL_GLANCE, TRIAL_GLANCE / 19.0)  # more than 1.5 times slower
    assert choice.trial is None and choice.trials_cut == 1
    assert choice.pick([PIPELINED, PLAIN]) == PLAIN


def test_a_trial_that_wins_takes_over_and_the_old_way_is_tried_after_its_own_gap() -> None:
    choice = DraftChoice()
    drive(choice, [PIPELINED, PLAIN], {PIPELINED: 14.0, PLAIN: 28.0}, 3000)
    assert choice.gap(PIPELINED) > 4 * DRAFT_TRIAL_TOKENS
    runs = drive(choice, [PIPELINED, PLAIN], {PIPELINED: 40.0, PLAIN: 28.0}, 5000)  # the route changed under it
    assert runs[0][0] == PLAIN and runs[1] == (PIPELINED, PIPELINE_STRETCH + DRAFT_TRIAL_TOKENS), "plain is tried again after the first gap"
    assert runs[-1][0] == PIPELINED


def test_a_lucky_win_halves_a_long_gap_and_does_not_start_it_again() -> None:
    """On the laptop GPU a lucky stretch of text won a slow pipeline a trial now and then, and a
    gap set back to 96 each time put 505 of 906 tokens on the slow way."""
    choice = DraftChoice()
    drive(choice, [PIPELINED, PLAIN], {PIPELINED: 14.0, PLAIN: 28.0}, 4000)
    long = choice.gap(PIPELINED)
    assert long >= 8 * DRAFT_TRIAL_TOKENS
    choice.since_trial = long
    drive(choice, [PIPELINED, PLAIN], {PIPELINED: 30.0, PLAIN: 28.0}, PIPELINE_STRETCH)  # one lucky trial
    assert choice.trials_won >= 2 and choice.gap(PIPELINED) == long // 2


def test_a_trial_still_behind_halfway_stops_there() -> None:
    choice = DraftChoice()
    runs = drive(choice, [PIPELINED, PLAIN], {PIPELINED: 22.0, PLAIN: 28.0}, 60)  # 1.27 times slower
    assert runs[:2] == [(PLAIN, PIPELINE_STRETCH), (PIPELINED, PIPELINE_STRETCH // 2)]
    assert choice.trials_cut == 1


def test_a_trial_is_judged_on_what_it_measures_now() -> None:
    """A way measured as slow long ago is not held back by those old figures once it is fast."""
    choice = DraftChoice()
    for _ in range(30):
        choice.note(PIPELINED, 24, 24 / 10.0)
    choice.note(PLAIN, 24, 24 / 28.0)
    choice.since_trial = DRAFT_TRIAL_MOST
    drive(choice, [PIPELINED, PLAIN], {PIPELINED: 32.0, PLAIN: 28.0}, PIPELINE_STRETCH)
    assert choice.best([PIPELINED, PLAIN]) == PIPELINED
    assert choice.rate(PIPELINED) == pytest.approx(32.0)


def test_a_trial_waits_while_its_way_cannot_run() -> None:
    choice = DraftChoice()
    drive(choice, [GUESSED, PLAIN], {GUESSED: 20.0, PLAIN: 28.0}, 200, {GUESSED: 3})
    choice.since_trial = DRAFT_TRIAL_MOST
    assert choice.pick([GUESSED, PLAIN]) == GUESSED
    assert choice.pick([PLAIN]) == PLAIN, "a temperature-0 reply may not check guesses in one pass"
    choice.note(PLAIN, 1, 1 / 28.0)
    assert choice.pick([GUESSED, PLAIN]) == GUESSED, "the trial goes on when it can"


def test_a_draft_that_wins_is_kept_and_a_change_in_the_route_can_turn_it_around() -> None:
    choice = DraftChoice()
    ways = [GUESSED, PLAIN]
    choice.note(GUESSED, 3, 0.24)  # 12.5 tokens/s across countries
    choice.note(PLAIN, 1, 0.102)  # 9.8 tokens/s
    assert choice.pick(ways) == GUESSED
    for _ in range(20):
        choice.note(PLAIN, 1, 0.02)  # the slow link went away
    assert choice.best(ways) == PLAIN


def test_lucky_passes_do_not_flatter_a_draft() -> None:
    # Measured on one site: 2.3 tokens per 110 ms pass on average, against 36 ms per token.
    choice = DraftChoice()
    for tokens in [5, 1, 1, 5, 1, 1, 5, 1, 1, 2] * 4:
        choice.note(GUESSED, tokens, 0.11)
    for _ in range(40):
        choice.note(PLAIN, 1, 0.036)
    assert choice.best([GUESSED, PLAIN]) == PLAIN


def test_trials_go_round_every_way_that_is_losing() -> None:
    choice = DraftChoice()
    ways = [PIPELINED, GUESSED, PLAIN]
    choice.note(PIPELINED, 24, 0.5)
    choice.note(GUESSED, 3, 0.15)
    choice.note(PLAIN, 1, 0.036)
    tried = []
    for _ in range(2):
        while (way := choice.pick(ways)) == PIPELINED:
            choice.note(PIPELINED, 24, 0.5)
        tried.append(way)
        while choice.trial == way:
            choice.note(way, 1, 0.1)
    assert sorted(tried) == sorted([GUESSED, PLAIN])


def test_a_way_that_cannot_run_is_never_picked() -> None:
    choice = DraftChoice()
    choice.note(PIPELINED, 24, 0.1)
    choice.note(GUESSED, 3, 0.15)
    choice.note(PLAIN, 1, 0.036)
    assert choice.pick([GUESSED, PLAIN]) == PLAIN  # the ring went down


def test_a_steady_link_costs_less_than_a_jittery_one_of_the_same_median() -> None:
    from isharded.orchestrator import link_cost

    assert link_cost({"median_ms": 40.0, "max_ms": 41.0}) < link_cost({"median_ms": 40.0, "max_ms": 120.0})
    assert link_cost({"median_ms": 40.0}) == 40.0


BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
needs_real_stage = pytest.mark.skipif(not (BUILD and MODEL and find_library()), reason="needs a real model and the built engine")
PROMPT = [151644, 872, 198, 9707, 1879, 11, 3838, 374, 279, 6722, 315, 9625, 30]


def real_stage(slots=2, threads=4, exact=False):
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    host = StageHost()
    host.call(
        "load", folder=BUILD, extra_paths=[], path=MODEL, last=True, context_tokens=256, gpu_layers=0, threads=threads,
        table=embed, cpu_tensors=[], rollback_tokens=4, slots=slots, exact=exact,
    )  # fmt: skip
    return host, bool(header.arch_key("ssm.state_size"))


async def real_ring(host, **limits) -> EngineClient:
    keys = {"back": None, "next": None, "closing": None}  # one stage has no links
    door = host.call("ring_open", host="127.0.0.1", port=0, epoch=1, index=0, count=1, keys=keys, ack_timeout_ms=30000, silence_timeout_ms=30000, **limits)[0]
    host.call("ring_connect", links={"next": None, "entry": None, "timeout_ms": 5000})
    while (state := host.call("ring_stats")[0].get("connect")) == "dialling":
        await asyncio.sleep(0.02)
    assert state == "up"
    return await EngineClient.open(door["local_port"], bytes(door["local_key"]))


def real_pass(client, tokens, position, rollback_to=None, keep=False, temperature=0.0, seed=None, top_k=0, slot=0):
    return client.send(
        epoch=1, slot=slot, position=position, rollback_to=rollback_to, tokens=tokens, guesses=[], verify=False,
        temperature=temperature, top_p=1.0, top_k=top_k, seed=seed, keep=keep,
    )  # fmt: skip


@needs_real_stage
def test_a_real_stage_probes_at_load_and_leaves_every_slot_as_it_found_it() -> None:
    """The probe runs passes on every slot. A reply after it must not see any of them."""

    async def replies(host) -> list[list[int]]:
        client = await real_ring(host)
        try:
            made = []
            for slot in (0, 3):
                tokens = [(await real_pass(client, PROMPT, 0, slot=slot))[0]]
                for position in range(len(PROMPT), len(PROMPT) + 8):
                    tokens += await real_pass(client, tokens[-1:], position, slot=slot)
                made.append(tokens)
            return made
        finally:
            client.close()

    host, _ = real_stage(slots=4, exact=True)
    try:
        found = host.call("exactness")[0]["found"]
        assert set(found) == {"exact_merge_limit", "verify_exact"}
        assert 0 <= found["exact_merge_limit"] <= 3 and isinstance(found["verify_exact"], bool)
        after_probe = asyncio.run(replies(host))
        host.call("ring_close")
        host.call("clear")
        assert asyncio.run(replies(host)) == after_probe
    finally:
        host.close()


@needs_real_stage
def test_a_stage_probes_nothing_unless_exact_replies_are_asked_for() -> None:
    """The probe can add seconds to a load, and a load without --exact never reads it."""
    host, _ = real_stage(slots=4)
    try:
        assert host.call("exactness")[0]["found"] is None
    finally:
        host.close()


@needs_real_stage
def test_a_seeded_top_k_reply_is_the_same_in_the_engine_and_on_the_python_path() -> None:
    """A reply leaves the ring when a stage fails, and must not change course when it does."""
    sampling = {"temperature": 1.0, "top_k": 40, "seed": 7}

    async def engine_reply(host) -> list[int]:
        client = await real_ring(host)
        try:
            await real_pass(client, PROMPT[:-1], 0, **sampling)
            made = PROMPT[-1:]
            for position in range(len(PROMPT) - 1, len(PROMPT) + 24):
                made += (await real_pass(client, made[-1:], position, **sampling))[:1]
            return made
        finally:
            client.close()

    def python_reply(host) -> list[int]:
        host.call("commit", tokens=PROMPT[:-1], position=0)
        made = PROMPT[-1:]
        for position in range(len(PROMPT) - 1, len(PROMPT) + 24):
            rows = host.call("embed", tokens=made[-1:])[1]
            made += host.call("verify", rows, position=position, guesses=[], top_p=1.0, **sampling)[0]["tokens"]
        return made

    host, _ = real_stage()
    try:
        ringed = asyncio.run(engine_reply(host))
        host.call("clear", slot=0)
        assert python_reply(host) == ringed
    finally:
        host.close()


@needs_real_stage
@pytest.mark.parametrize(("temperature", "seed"), [(0.0, None), (0.8, 7)])
def test_a_real_stage_undoes_finished_passes_and_gives_the_reply_it_would_have_given(temperature, seed) -> None:
    sampling = {"temperature": temperature, "seed": seed}

    async def reply(client, guessing: bool) -> list[int]:
        token = (await real_pass(client, PROMPT, 0, keep=guessing, **sampling))[0]
        made, position = [token], len(PROMPT)
        for _ in range(8):
            answer = real_pass(client, [token], position, position, keep=guessing, **sampling)
            if guessing:
                # Waited for, so these have really run and only a rewind can undo them.
                await asyncio.gather(*(real_pass(client, [1000 + ahead], position + ahead, keep=True, **sampling) for ahead in (1, 2)))
            token = (await answer)[0]
            made.append(token)
            position += 1
        return made

    async def both(host) -> tuple[list[int], list[int]]:
        client = await real_ring(host)
        try:
            plain = await reply(client, False)
            host.call("clear", slot=0)
            return plain, await reply(client, True)
        finally:
            client.close()

    host, _ = real_stage()
    try:
        plain, guessed = asyncio.run(both(host))
    finally:
        host.close()
    assert guessed == plain


@needs_real_stage
def test_a_real_stage_never_goes_back_to_a_snapshot_from_before_a_clear() -> None:
    async def run(host) -> None:
        client = await real_ring(host)
        try:
            await real_pass(client, PROMPT, 0, keep=True)
            await real_pass(client, [1], len(PROMPT), keep=True)
            await real_pass(client, [2], len(PROMPT) + 1, keep=True)
            host.call("clear", slot=0)
            await real_pass(client, PROMPT[::-1], 0)
            await real_pass(client, [3], len(PROMPT))
            await real_pass(client, [4], len(PROMPT) + 1)
            # Without the clear reaching the engine, this restores the old reply's state.
            with pytest.raises(EngineRefusal) as caught:
                await real_pass(client, [5], len(PROMPT) + 1, len(PROMPT) + 1)
            assert caught.value.refusal == native.REFUSE_RETRY
        finally:
            client.close()

    host, recurrent = real_stage()
    try:
        if not recurrent:
            pytest.skip("attention rewinds by itself, so this model never needs a snapshot")
        asyncio.run(run(host))
    finally:
        host.close()


@needs_real_stage
def test_a_real_pass_past_its_limit_refuses_and_ends_the_process() -> None:
    from isharded.llamacpp import StageLost

    async def run(host) -> EngineRefusal:
        client = await real_ring(host, pass_ms=1, row_ms=0)
        try:
            with pytest.raises(EngineRefusal) as caught:
                # Enough rows to outlast the watchdog's first look on any machine. llama.cpp runs a
                # pass of 32 rows or more on a GPU if there is one, which is fast once it is warm.
                await client.run(60, epoch=1, slot=0, position=0, rollback_to=None, tokens=(PROMPT * 3)[:31], guesses=[], verify=False, temperature=0.0, top_p=1.0, top_k=0, seed=None)
            return caught.value
        finally:
            client.close()

    host, _ = real_stage(threads=1)
    try:
        refused = asyncio.run(run(host))
        assert refused.refusal == native.REFUSE_FAILED_STAGE and "hung" in refused.message
        assert host.process.wait(10) != 0
        with pytest.raises(StageLost):
            host.call("clear", slot=0)
    finally:
        host.close()


@needs_real_stage
def test_a_real_pass_inside_its_limit_is_left_alone() -> None:
    async def run(host) -> list[int]:
        client = await real_ring(host)
        try:
            return await real_pass(client, PROMPT, 0)
        finally:
            client.close()

    host, _ = real_stage()
    try:
        assert asyncio.run(run(host))
        assert host.process.poll() is None
    finally:
        host.close()


def test_a_passs_deadline_reaches_the_engine_in_the_plan_it_already_reads() -> None:
    from isharded.deadlines import pass_seconds

    plans = []

    def opened(_, plan):
        plans.append(json.loads(plan))
        return 0

    engine = native.Engine.__new__(native.Engine)
    engine.lib, engine.handle = SimpleNamespace(isharded_engine_open=opened), None
    host = llama_host._Host()
    host.stage = SimpleNamespace(engine=engine)
    engine.listen = lambda *_: {}
    keys = {"back": None, "next": "01" * 32, "closing": "02" * 32}
    host.ring_open({"host": "h", "port": 0, "epoch": 1, "index": 0, "count": 2, "keys": keys, "ack_timeout_ms": 5, "silence_timeout_ms": 5}, b"")
    assert plans[0]["pass_ms"] + plans[0]["row_ms"] == pass_seconds(1, 1) * 1000
    assert plans[0]["pass_ms"] + 9 * plans[0]["row_ms"] == pass_seconds(9, 1) * 1000
    # Each link's key reaches the engine as it was given, and a link the stage is not on has none.
    assert plans[0]["keys"] == {"next": [1] * 32, "closing": [2] * 32}


def test_a_load_that_merges_nothing_tells_the_engine_in_its_plan() -> None:
    plans, calls = [], []

    def opened(_, plan):
        plans.append(json.loads(plan))
        return 0

    engine = native.Engine.__new__(native.Engine)
    engine.lib, engine.handle = SimpleNamespace(isharded_engine_open=opened), None
    engine.listen = lambda *_: {}
    host = llama_host._Host()
    host.stage = SimpleNamespace(engine=engine)
    meta = {"host": "h", "port": 0, "epoch": 1, "index": 0, "count": 2, "keys": RING_KEYS, "ack_timeout_ms": 5, "silence_timeout_ms": 5}
    host.ring_open({**meta, "merge_passes": False}, b"")
    host.ring_open(meta, b"")  # a worker from before the plan carried it
    assert [plan["merge_passes"] for plan in plans] == [False, True]

    item = runtime(ring_client=None, merge_passes=False, advertised_url="http://w:1", term=5, ring_term=5, revoked_nodes=set())
    item.gguf = SimpleNamespace(ring_call=lambda kind, **fields: calls.append(fields) or {"engine": False})
    asyncio.run(item.ring_open({"epoch": 3, "term": 5, "index": 0, "count": 2, "keys": RING_KEYS}))
    assert calls[0]["merge_passes"] is False


@needs_real_stage
def test_a_real_route_that_merges_nothing_never_merges_passes() -> None:
    async def run(host) -> dict:
        client = await real_ring(host, merge_passes=False)
        send = lambda slot, tokens, position: client.send(  # noqa: E731
            epoch=1, slot=slot, position=position, rollback_to=None, tokens=tokens, guesses=[], verify=False,
            temperature=0.0, top_p=1.0, top_k=0, seed=None,
        )  # fmt: skip
        try:
            await send(1, PROMPT, 0)
            long = PROMPT * 8
            # The two one-row passes from different slots wait behind the long one together.
            await asyncio.gather(send(0, long, 0), send(1, [PROMPT[0]], len(PROMPT)), send(0, [PROMPT[0]], len(long)))
            return host.call("ring_stats")[0]
        finally:
            client.close()

    host, _ = real_stage()
    try:
        stats = asyncio.run(run(host))
        assert stats["passes"] >= 4 and stats["merged_passes"] == 0
    finally:
        host.close()


@needs_real_stage
def test_a_real_stage_reports_its_compute_totals_and_hop_trace() -> None:
    async def run(host) -> dict:
        client = await real_ring(host)
        try:
            made = [(await real_pass(client, PROMPT, 0))[0]]
            for position in range(len(PROMPT), len(PROMPT) + 4):
                made += (await real_pass(client, made[-1:], position))[:1]
            return host.call("ring_stats")[0]
        finally:
            client.close()

    host, _ = real_stage()
    try:
        stats = asyncio.run(run(host))
    finally:
        host.close()
    assert stats["batches_total"] == 5 and stats["compute_us_total"] >= stats["batches_total"]
    # One stage is the entry and the last: it picks tokens and waits on its worker.
    assert stats["sample_us"] > 0 and stats["turnaround_us"] > 0
    assert {"queue_us", "write_us", "enqueue_us", "wait_us"} <= stats.keys()


def test_a_load_of_another_model_starts_its_pipeline_counts_again(monkeypatch) -> None:
    """A stage that restarts reloads the same model, and its counts still describe it."""
    from isharded.protocol import ConfigureRequest

    def stop(*_):
        raise ValueError("stop here")

    monkeypatch.setattr(worker_module, "create_adapter", lambda _: SimpleNamespace(num_layers=2))
    monkeypatch.setattr(worker_module, "has_torch", lambda: True)
    item = runtime(pipeline_tokens=900, pipeline_misses=40, term=5, roles={"inference"}, host_models=[], worker_id="w", ring_client=None, model_id="m")
    item._materialize = lambda *_: None
    item.caches, item.prefix, item.gguf = {}, SimpleNamespace(clear=lambda: None), None
    item.copies = SimpleNamespace(aim=stop)  # the load stops here, once the model is in place
    stage = StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)

    def load(model_id):
        order = ConfigureRequest(model_id=model_id, revision="r", architecture={}, epoch=9, assignment=stage, route=[stage], term=5)
        with pytest.raises(ValueError, match="stop here"):
            asyncio.run(item.configure(order))
        return item.pipeline_tokens, item.pipeline_misses

    assert load("m") == (900, 40)
    assert load("other") == (0, 0)


def waiting_ring(client, **fields):
    """A worker whose ring has a pass out that will never come back."""
    client.send(**FIELDS)
    return runtime(ring_client=client, **fields)


def test_settling_does_not_wait_on_a_ring_known_to_be_lost() -> None:
    async def use(client):
        await asyncio.wait_for(waiting_ring(client, ring_live=False).ring_settle(0), 1)
        return True

    assert asyncio.run(with_door(lambda identity: b"", use))


def test_a_lost_ring_still_ends_the_replys_generation() -> None:
    async def use(client):
        await waiting_ring(client, ring_live=False).ring_settle(0)
        return 0 in client.generations

    assert asyncio.run(with_door(lambda identity: b"", use)) is False


def test_a_pipe_that_fails_to_close_still_ends_the_replys_generation(monkeypatch) -> None:
    released = []

    async def broken():
        raise RuntimeError("the pipe broke")

    async def release(slot):
        released.append(slot)

    async def use(client):
        item = waiting_ring(client, pipes={0: SimpleNamespace(close=broken)}, slot_used_at={}, gate=SimpleNamespace(release=release))
        monkeypatch.setattr(worker_module, "runtime", item)
        with pytest.raises(RuntimeError, match="the pipe broke"):
            await worker_module.finish_reply(0)
        return 0 in client.generations

    monkeypatch.setattr(worker_module, "pass_seconds", lambda tokens, hops=1: 0.05)
    assert asyncio.run(with_door(lambda identity: b"", use)) is False
    assert released == [0]


def test_settling_stops_when_the_ring_is_lost_during_the_wait() -> None:
    """Another reply can find the stage lost while this one waits on the same ring."""

    async def use(client):
        item = waiting_ring(client)
        asyncio.get_running_loop().call_later(0.05, lambda: setattr(item, "ring_live", False))
        await asyncio.wait_for(item.ring_settle(0), 2)
        return True

    assert asyncio.run(with_door(lambda identity: b"", use))


def test_a_pass_that_never_settles_still_turns_the_ring_off(monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "pass_seconds", lambda tokens, hops=1: 0.05)

    async def use(client):
        item = waiting_ring(client)
        await item.ring_settle(0)
        return item.ring_live

    assert asyncio.run(with_door(lambda identity: b"", use)) is False


def test_a_slot_that_settles_leaves_the_ring_up() -> None:
    async def use(client):
        item = waiting_ring(client)
        await item.ring_settle(0)
        return item.ring_live

    assert asyncio.run(with_door(lambda identity: answer(identity, [identity]), use)) is True
