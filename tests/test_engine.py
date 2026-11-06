"""The native path: what it refuses, how failures map to recovery, and how a ring gets opened."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import pytest

from isharded import engine as native
from isharded import worker as worker_module
from isharded.engine import EngineClient, EngineRefusal, find_library, library_name, pass_frame
from isharded.gguf_stage import DRAFT_TRIAL_TOKENS, DraftChoice
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
    frame = pass_frame(9, 3, 1, 40, 38, [5, 6, 7], [6, 7], True, 0.5, 0.9, 40, 123)
    magic, version, kind, length = native.HEADER.unpack_from(frame)
    assert (magic, version, kind, length) == (native.MAGIC, native.WIRE_VERSION, native.KIND_PASS, len(frame) - 8)
    fields = struct.unpack_from("<QQIIIBffIBQ", frame, 8)
    assert fields[:6] == (9, 3, 1, 40, 38, native.FLAG_TOKENS | native.FLAG_VERIFY)
    assert fields[8:] == (40, 1, 123)
    at = 8 + struct.calcsize("<QQIIIBffIBQ")
    assert struct.unpack_from("<I2I", frame, at) == (2, 6, 7)
    assert struct.unpack_from("<I3I", frame, at + 12) == (3, 5, 6, 7)
    assert frame[-8:] == bytes(8)


def test_no_rollback_and_no_seed_have_their_own_encoding() -> None:
    frame = pass_frame(1, 1, 0, 0, None, [1], [], False, 0.0, 1.0, 0, None)
    fields = struct.unpack_from("<QQIIIBffIBQ", frame, 8)
    assert fields[4] == native.NO_ROLLBACK
    assert fields[9:] == (0, 0)


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


def test_a_ring_opens_every_stage_before_any_dials_and_the_last_dials_the_entry() -> None:
    calls = []

    def handler(item):
        body = json.loads(item.content)
        calls.append((item.url.host, item.url.path, body))
        if item.url.path.endswith("/open"):
            return httpx.Response(200, json={"engine": True, "address": f"{item.url.host}:9", "fingerprint": item.url.host * 8})
        return httpx.Response(200, json={})

    orchestrator = pool(handler)
    run_ring(orchestrator)
    opens = [call for call in calls if call[1].endswith("/open")]
    connects = {call[0]: call[2] for call in calls if call[1].endswith("/connect")}
    assert [call[0] for call in calls[:3]] == ["a", "b", "c"]
    assert len({call[2]["key"] for call in opens}) == 1 and len(opens[0][2]["key"]) == 64
    assert [call[2]["index"] for call in opens] == [0, 1, 2]
    assert connects["a"]["next"]["address"] == "b:9" and connects["a"]["entry"] is None
    assert connects["c"]["next"] is None and connects["c"]["entry"]["address"] == "a:9"
    assert calls[-1][:2] == ("a", "/internal/ring/go")
    assert orchestrator.ring_epoch == 4


def test_a_stage_with_no_engine_leaves_every_reply_on_http() -> None:
    def handler(item):
        engine = item.url.host != "b"
        return httpx.Response(200, json={"engine": engine, "address": "x:1", "fingerprint": "f"})

    orchestrator = pool(handler)
    run_ring(orchestrator)
    assert orchestrator.ring_epoch is None


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
            identity = struct.unpack_from("<Q", body)[0]
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


def answer(identity: int, tokens: list[int]) -> bytes:
    return native._frame(native.KIND_RESULT, struct.pack("<QQI", identity, 3, len(tokens)) + struct.pack(f"<{len(tokens)}I", *tokens))


def refusal(identity: int, kind: int, text: str = "no") -> bytes:
    data = text.encode()
    return native._frame(native.KIND_NACK, struct.pack("<QQBHI", identity, 3, kind, 2, len(data)) + data)


def test_answers_find_their_own_pass_when_several_are_out() -> None:
    async def use(client):
        return await asyncio.gather(*(client.run(5, **FIELDS) for _ in range(5)))

    got = asyncio.run(with_door(lambda identity: answer(identity, [identity * 10]), use))
    assert got == [[10], [20], [30], [40], [50]]


def test_a_refused_pass_raises_and_the_link_stays_up() -> None:
    async def use(client):
        with pytest.raises(EngineRefusal) as caught:
            await client.run(5, **FIELDS)
        assert (caught.value.refusal, caught.value.stage, caught.value.message) == (native.REFUSE_STAGE_ERROR, 2, "no")
        return client.alive

    assert asyncio.run(with_door(lambda identity: refusal(identity, native.REFUSE_STAGE_ERROR), use))


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


def test_a_draft_that_loses_is_set_aside_and_tried_again_later() -> None:
    choice = DraftChoice()
    assert choice.use_draft() is True  # each way gets a turn first
    choice.note(True, 3, 0.15)  # 20 tokens/s
    assert choice.use_draft() is False
    choice.note(False, 1, 0.036)  # 28 tokens/s
    tokens = 0
    while not choice.use_draft():
        choice.note(False, 1, 0.036)
        tokens += 1
    assert DRAFT_TRIAL_TOKENS - 8 <= tokens <= DRAFT_TRIAL_TOKENS  # time for a trial
    choice.note(True, 3, 0.15)
    assert choice.use_draft() is False  # it lost again, so back to the faster way


def test_a_draft_that_wins_is_kept_and_a_change_in_the_route_can_turn_it_around() -> None:
    choice = DraftChoice()
    choice.note(True, 3, 0.24)  # 12.5 tokens/s across countries
    choice.note(False, 1, 0.102)  # 9.8 tokens/s
    assert choice.use_draft() is True
    for _ in range(20):
        choice.note(False, 1, 0.02)  # the slow link went away
    assert choice.best() is False


def test_lucky_passes_do_not_flatter_a_draft() -> None:
    # Measured on one site: 2.3 tokens per 110 ms pass on average, against 36 ms per token.
    choice = DraftChoice()
    for tokens in [5, 1, 1, 5, 1, 1, 5, 1, 1, 2] * 4:
        choice.note(True, tokens, 0.11)
    for _ in range(40):
        choice.note(False, 1, 0.036)
    assert choice.best() is False


def test_a_steady_link_costs_less_than_a_jittery_one_of_the_same_median() -> None:
    from isharded.orchestrator import link_cost

    assert link_cost({"median_ms": 40.0, "max_ms": 41.0}) < link_cost({"median_ms": 40.0, "max_ms": 120.0})
    assert link_cost({"median_ms": 40.0}) == 40.0
