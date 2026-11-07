"""Other addresses for a node: what it registers, what proves one, and what a failed one costs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import pytest

from isharded import addresses as module
from isharded import worker as worker_module
from isharded.addresses import NodeProof, claim, host_port, prove, reach_options, reach_urls
from isharded.node import _worker_command
from isharded.orchestrator import Orchestrator, WorkerRecord
from isharded.pool import add_node, init_pool
from isharded.protocol import PROTOCOL_VERSION, HardwareInfo, StageAssignment, WorkerRegistration
from isharded.tickets import AUDIENCE, NodeTickets, TicketMaker

TOKEN = "pool-secret"
REUSED = "http://10.0.0.5:8100"


def nic(address: str, family: int = socket.AF_INET) -> SimpleNamespace:
    return SimpleNamespace(family=family, address=address)


def test_a_worker_finds_the_addresses_other_machines_can_use(monkeypatch) -> None:
    found = {"lan": [nic("192.168.1.5"), nic("fe80::1", socket.AF_INET6)], "lo": [nic("127.0.0.1")], "auto": [nic("169.254.1.1")], "v6": [nic("2606:4700::1", socket.AF_INET6)]}
    monkeypatch.setattr(module.psutil, "net_if_addrs", lambda: found)
    urls, left_out = reach_urls("auto", "http://relay.example:9000", "0.0.0.0", 8100, True)
    assert urls == ["http://192.168.1.5:8100", "http://[2606:4700::1]:8100"]
    assert left_out == 0


def test_a_worker_listening_on_one_address_offers_no_other(monkeypatch) -> None:
    monkeypatch.setattr(module.psutil, "net_if_addrs", lambda: {"lan": [nic("192.168.1.5")]})
    assert reach_urls("auto", "http://relay.example:9000", "127.0.0.1", 8100, True)[0] == []
    assert reach_urls("auto", "http://relay.example:9000", "192.168.1.9", 8100, True)[0] == ["http://192.168.1.9:8100"]


def test_a_shorter_path_never_loses_the_encryption_the_old_one_had() -> None:
    assert reach_urls("192.168.1.5,100.64.0.9", "http://100.64.0.1:1", "0.0.0.0", 8100, False) == (["http://100.64.0.9:8100"], 1)
    assert reach_urls("192.168.1.5", "http://100.64.0.1:1", "0.0.0.0", 8100, True) == (["http://192.168.1.5:8100"], 0)
    assert reach_urls("192.168.1.5", "https://relay.example:1", "0.0.0.0", 8100, False) == (["https://192.168.1.5:8100"], 0)


def test_none_registers_nothing_and_the_advertised_address_is_not_listed_twice() -> None:
    assert reach_urls("10.0.0.1,none", "http://x:1", "0.0.0.0", 8100, True) == ([], 0)
    assert reach_urls("10.0.0.1", "http://10.0.0.1:8100/", "0.0.0.0", 8100, True) == ([], 0)


def test_a_node_passes_its_hosts_to_its_workers() -> None:
    args = SimpleNamespace(bind_host="0.0.0.0", advertise_scheme="http", advertise_host="relay", reach_hosts="10.0.0.1", control_port=1, base_port=1)
    assert _worker_command(args, "a-gpu", 8100, "cpu", "inference", [])[-2:] == ["--reach-hosts", "10.0.0.1"]
    assert "--reach-hosts" not in _worker_command(SimpleNamespace(**{**vars(args), "reach_hosts": None}), "a-gpu", 8100, "cpu", "inference", [])


def test_host_port_brackets_ipv6() -> None:
    assert host_port("::1", 5) == "[::1]:5" and host_port("10.0.0.1", 5) == "10.0.0.1:5"


def test_a_request_speaks_to_the_advertised_node_whatever_address_it_goes_to() -> None:
    assert reach_options("https://relay.example:9000") == {"sni_hostname": "relay.example", AUDIENCE: "relay.example:9000"}


@pytest.fixture
def members(tmp_path: Path, monkeypatch) -> dict[str, NodeProof]:
    folder = init_pool(str(tmp_path / "pool"))
    monkeypatch.setenv("ISHARDED_CA_FILE", str(folder / "ca.pem"))
    monkeypatch.delenv("ISHARDED_OWNERLESS", raising=False)
    made = {}
    for name in ("alpha", "beta"):
        certificate, key = add_node(str(folder), name, ["10.0.0.5"])
        made[name] = NodeProof(None, TicketMaker(str(certificate), str(key)))
    return made


@pytest.fixture
def no_authority(monkeypatch) -> None:
    monkeypatch.delenv("ISHARDED_CA_FILE", raising=False)
    monkeypatch.delenv("ISHARDED_OWNERLESS", raising=False)


def test_a_ticket_names_a_node_only_for_the_claim_it_signed(members) -> None:
    text = claim("answer", "n", REUSED, "alpha-gpu")
    assert members["beta"].signer(members["alpha"].sign(text), text) == "alpha"
    assert members["beta"].signer(members["alpha"].sign(text), claim("ask", "n", REUSED, "alpha-gpu")) is None
    assert members["beta"].signer("not a ticket", text) is None


def test_the_pool_token_alone_proves_nothing_in_a_pool_with_an_authority(members) -> None:
    text = claim("answer", "n", REUSED, "alpha-gpu")
    assert NodeProof(TOKEN, None).sign(text) is None
    assert members["alpha"].signer("0" * 64, text) is None


def test_a_mac_shows_the_pool_token_without_giving_it_away(no_authority) -> None:
    text = claim("answer", "n", REUSED, "alpha-gpu")
    proof = NodeProof(TOKEN, None).sign(text)
    assert TOKEN not in proof
    assert NodeProof(TOKEN, None).signer(proof, text) == ""
    assert NodeProof("another pool", None).signer(proof, text) is None
    assert NodeProof(TOKEN, None).signer(proof, claim("answer", "n", "http://10.0.0.6:8100", "alpha-gpu")) is None


def test_a_pool_with_no_identity_can_prove_nothing(no_authority) -> None:
    assert NodeProof(None, None).sign("anything") is None
    assert NodeProof(None, None).signer("anything", "anything") is None


def far_end(proof: NodeProof, worker_id: str, seen: list[httpx.Request] | None = None) -> httpx.AsyncClient:
    """A client whose every request reaches a node that answers for ``worker_id``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"proof": proof.sign(claim("answer", body["nonce"], body["url"], worker_id))})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def proved(client: httpx.AsyncClient, proof: NodeProof, worker_id: str = "alpha-gpu", revoked: set[str] | None = None) -> bool:
    return asyncio.run(prove(client, proof, REUSED, "http://relay:1", worker_id, revoked or set()))


def test_the_node_that_registered_proves_its_address(members) -> None:
    assert proved(far_end(members["alpha"], "alpha-gpu"), members["beta"])


def test_another_site_on_the_same_private_address_is_refused(members) -> None:
    assert not proved(far_end(members["beta"], "alpha-gpu"), members["alpha"])  # a good certificate, on the wrong node


def test_a_revoked_node_proves_nothing(members) -> None:
    assert not proved(far_end(members["alpha"], "alpha-gpu"), members["beta"], revoked={"alpha"})


def test_something_that_merely_answers_is_refused(members) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"status": "ok"})))
    assert not proved(client, members["beta"])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401)))
    assert not proved(client, members["beta"])


def test_a_stranger_at_an_address_is_never_shown_a_credential(no_authority) -> None:
    seen: list[httpx.Request] = []
    assert proved(far_end(NodeProof(TOKEN, None), "alpha-gpu", seen), NodeProof(TOKEN, None))
    assert TOKEN not in str(seen[0].headers) and TOKEN not in seen[0].content.decode()


def test_a_ticket_sent_to_a_shorter_address_names_the_advertised_one(members) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    maker = members["alpha"].maker
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=NodeTickets(maker))
    asyncio.run(client.post(REUSED + "/x", extensions=reach_options("http://relay:1")))
    asyncio.run(client.post(REUSED + "/x"))
    assert seen[0].headers["authorization"] == f"Bearer {maker.ticket('relay:1')}"
    assert seen[1].headers["authorization"] == f"Bearer {maker.ticket('10.0.0.5:8100')}"


def worker_at(monkeypatch, proof: NodeProof, **fields) -> SimpleNamespace:
    item = SimpleNamespace(proof=proof, revoked_nodes=set(), worker_id="alpha-gpu", advertised_url="http://relay:1", reach_urls=[REUSED], **fields)
    monkeypatch.setattr(worker_module, "runtime", item)
    return item


def ask(proof: NodeProof, url: str = REUSED, worker_id: str = "alpha-gpu") -> dict[str, str | None]:
    body = {"nonce": "n", "url": url, "worker_id": worker_id}
    asked = proof.sign(claim("ask", "n", url, worker_id))
    return asyncio.run(worker_module.diag_prove(body, f"Bearer {asked}"))


def test_a_worker_signs_only_for_its_own_addresses(no_authority, monkeypatch) -> None:
    proof = NodeProof(TOKEN, None)
    worker_at(monkeypatch, proof)
    assert proof.signer(ask(proof)["proof"], claim("answer", "n", REUSED, "alpha-gpu")) == ""
    for bad in ({"url": "http://10.9.9.9:1"}, {"worker_id": "beta-gpu"}):
        with pytest.raises(HTTPException) as caught:
            ask(proof, **bad)
        assert caught.value.status_code == 404


def test_the_prove_route_turns_away_whoever_is_not_in_the_pool(no_authority, monkeypatch) -> None:
    worker_at(monkeypatch, NodeProof(TOKEN, None))
    for outsider in (NodeProof("another pool", None), SimpleNamespace(sign=lambda _: "")):
        with pytest.raises(HTTPException) as caught:
            ask(outsider)
        assert caught.value.status_code == 401


def test_the_prove_route_turns_away_a_revoked_node(members, monkeypatch) -> None:
    item = worker_at(monkeypatch, members["alpha"])
    item.revoked_nodes.add("beta")
    with pytest.raises(HTTPException) as caught:
        ask(members["beta"])
    assert caught.value.status_code == 401


def registration(worker_id: str, addresses: tuple[str, ...] = ()) -> WorkerRegistration:
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=4, ram_total_bytes=1)
    return WorkerRegistration(worker_id=worker_id, advertised_url=f"http://{worker_id}:1", hardware=hardware, protocol_version=PROTOCOL_VERSION, addresses=list(addresses))


def stage(name: str) -> StageAssignment:
    return StageAssignment(worker_id=name, url=f"http://{name}:1", start_layer=0, end_layer=1)


class Pool:
    """A coordinator whose stages answer probes with what a test says."""

    def __init__(self, *names: str, addresses: dict[str, tuple[str, ...]]) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.answers: dict[str, dict] = {}
        self.plane = Orchestrator()
        self.plane.mesh = None
        self.plane.workers = {name: WorkerRecord(registration=registration(name, addresses.get(name, ())), last_seen=time.monotonic()) for name in names}
        self.plane.route = [stage(name) for name in names]
        self.plane.epoch = 4

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.host, request.url.path, body))
        if request.url.path.endswith("/probe"):
            return httpx.Response(200, json=self.answers.get(request.url.host, {"100": {"median_ms": 200.0}}))
        if request.url.path.endswith("/open"):
            return httpx.Response(200, json={"engine": True, "address": f"{request.url.host}:7", "port": 7, "fingerprint": request.url.host})
        return httpx.Response(200, json={})

    def aim(self) -> None:
        async def run() -> None:
            self.plane.client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
            self.plane.hops_checked_at = float("-inf")
            await self.plane._aim_hops()

        asyncio.run(run())

    def told(self, host: str) -> list[dict]:
        return [body for name, path, body in self.calls if name == host and path == "/internal/hop"]


def reading(ms: float) -> dict:
    return {"proved": True, "100": {"median_ms": ms}}


def test_a_stage_hears_the_fastest_proven_address_of_its_next_stage_alone() -> None:
    pool = Pool("a", "b", addresses={"b": ("http://10.0.0.5:2", "http://10.0.0.6:2")})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {"http://10.0.0.5:2": reading(21.0), "http://10.0.0.6:2": reading(40.0)}}
    pool.aim()
    assert pool.told("a") == [{"worker_id": "b", "url": "http://10.0.0.5:2"}]
    assert pool.told("b") == []  # the last stage passes nothing on over HTTP
    assert pool.plane.links["a", "b"] == 21.0
    assert pool.plane.link_readings["a", "b"]["direct"] is True


def test_each_direction_is_probed_from_the_end_that_sends() -> None:
    pool = Pool("a", "b", addresses={"a": ("http://10.0.0.4:2",), "b": ("http://10.0.0.5:2",)})
    pool.aim()
    probes = {host: body for host, path, body in pool.calls if path.endswith("/probe")}
    assert probes["a"]["addresses"] == ["http://10.0.0.5:2"] and probes["a"]["worker_id"] == "b"
    assert probes["b"]["addresses"] == ["http://10.0.0.4:2"] and probes["b"]["url"] == "http://a:1"


def test_only_the_pairs_on_the_route_are_probed() -> None:
    pool = Pool("a", "b", "c", addresses={"c": ("http://10.0.0.5:2",), "a": ("http://10.0.0.6:2",)})
    pool.aim()
    assert sorted(host for host, path, _ in pool.calls if path.endswith("/probe")) == ["b", "c"]


def test_a_pool_with_no_other_addresses_sends_nothing_new() -> None:
    pool = Pool("a", "b", addresses={})
    pool.aim()
    assert pool.calls == []


def test_a_fast_address_that_proved_nothing_does_not_count() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: {"proved": False, "100": {"median_ms": 1.0}}}}
    pool.aim()
    assert pool.told("a") == [] and pool.plane.hops == {}
    assert pool.plane.links["a", "b"] == 205.0


def test_the_stage_is_sent_back_when_a_proven_address_stops_proving() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0)}}
    pool.aim()
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: {"proved": False}}}
    pool.aim()
    assert pool.told("a")[-1] == {"worker_id": "b", "url": None}
    assert pool.plane.link_readings["a", "b"]["direct"] is False


def test_noise_does_not_move_a_live_link() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED, "http://10.0.0.6:2")})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0), "http://10.0.0.6:2": reading(30.0)}}
    pool.aim()
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0), "http://10.0.0.6:2": reading(20.0)}}
    pool.aim()
    assert pool.told("a")[-1]["url"] == REUSED
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0), "http://10.0.0.6:2": reading(10.0)}}
    pool.aim()
    assert pool.told("a")[-1]["url"] == "http://10.0.0.6:2"


def test_an_old_address_that_died_does_not_stop_a_proven_one() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"addresses": {REUSED: reading(21.0)}}
    pool.aim()
    assert pool.told("a") == [{"worker_id": "b", "url": REUSED}]


def test_a_later_measuring_keeps_the_shorter_links_reading() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0)}}
    pool.aim()
    pool.plane.links_checked.clear()
    pool.answers["a"] = {"100": {"median_ms": 205.0}}
    asyncio.run(pool.plane.measure_links())
    assert pool.plane.links["a", "b"] == 21.0
    assert ("b", "a") in pool.plane.links_checked


def status_text(plane: Orchestrator) -> str:
    return json.dumps(plane.status(), default=str)


def test_no_status_shows_an_address_only_a_stage_should_hear() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0)}}
    pool.aim()
    assert pool.plane.hops
    assert "10.0.0.5" not in status_text(pool.plane)
    assert "10.0.0.5" not in json.dumps(list(pool.plane.events))


def ring(pool: Pool) -> None:
    async def run() -> None:
        pool.plane.client = httpx.AsyncClient(transport=httpx.MockTransport(pool.handler))
        await pool.plane._start_ring()

    asyncio.run(run())


def dialled(pool: Pool, host: str) -> list[dict]:
    return [body for name, path, body in pool.calls if name == host and path.endswith("/connect")]


def test_a_ring_link_dials_the_proven_host_on_the_door_port_and_keeps_its_pin() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0)}}
    ring(pool)
    assert dialled(pool, "a")[0]["next"] == {"address": "10.0.0.5:7", "fingerprint": "b"}
    assert dialled(pool, "b")[0]["entry"] == {"address": "a:7", "fingerprint": "a"}  # the way back proved nothing
    assert pool.plane.ring_epoch == 4


def test_the_closing_link_uses_its_own_proven_address() -> None:
    pool = Pool("a", "b", addresses={"a": (REUSED,)})
    pool.answers["b"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0)}}
    ring(pool)
    assert dialled(pool, "b")[0]["entry"]["address"] == "10.0.0.5:7"
    assert pool.told("b") == []  # the ring closes here, and HTTP passes do not


def test_a_ring_that_cannot_dial_a_proven_address_opens_on_the_advertised_ones() -> None:
    pool = Pool("a", "b", addresses={"b": (REUSED,)})
    pool.answers["a"] = {"100": {"median_ms": 205.0}, "addresses": {REUSED: reading(21.0)}}
    answer = pool.handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/connect") and b"10.0.0.5" in request.content:
            pool.calls.append((request.url.host, request.url.path, json.loads(request.content)))
            return httpx.Response(502, json={"detail": "could not reach the next stage"})
        return answer(request)

    async def run() -> None:
        pool.plane.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await pool.plane._start_ring()

    asyncio.run(run())
    assert dialled(pool, "a")[-1]["next"]["address"] == "b:7"
    assert pool.plane.ring_epoch == 4


def test_a_better_address_waits_for_a_quiet_moment_to_move_the_ring() -> None:
    pool = Pool("a", "b", addresses={})
    plane = pool.plane
    plane.ring_epoch, plane.ring_tried, plane.ring_moved, plane.replies_in_flight = 4, 4, True, 1
    plane._redial_moved_ring()
    assert plane.ring_epoch == 4 and plane.ring_moved
    plane.replies_in_flight = 0
    plane._redial_moved_ring()
    assert plane.ring_epoch is None and plane.ring_tried is None and not plane.ring_moved


def test_a_broken_ring_checks_its_addresses_before_it_redials() -> None:
    pool = Pool("a", "b", addresses={})
    plane = pool.plane
    plane.ring_epoch, plane.hops_checked_at = 4, time.monotonic()
    plane.workers["a"].health = {"telemetry": {"ring": {"epoch": 4, "next_up": False}}}
    plane._check_ring()
    assert plane.ring_epoch is None and plane.hops_checked_at == float("-inf")


def stage_at(handler, hop_handler) -> worker_module.WorkerRuntime:
    item = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    item.worker_id = "a"
    item.pool_token = TOKEN
    item.assignment = stage("a")
    item.route = [stage("a"), stage("b")]
    item.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    item.hop_handler = hop_handler
    return item


def aim(item: worker_module.WorkerRuntime, monkeypatch) -> None:
    monkeypatch.setattr(worker_module, "direct_client", lambda _: httpx.AsyncClient(transport=httpx.MockTransport(item.hop_handler)))


async def send(item: worker_module.WorkerRuntime) -> httpx.Response:
    hop = item._hop_to(item.route[1])
    assert hop is not None
    return await item._over_hop(hop, b"x", {}, httpx.Timeout(30.0))


def run_on_hop(item: worker_module.WorkerRuntime, monkeypatch, work) -> None:
    async def run() -> None:
        monkeypatch.setattr(worker_module, "HOP_WATCH_SECONDS", 30.0)
        aim(item, monkeypatch)
        await item.aim_hop({"worker_id": "b", "url": REUSED})
        await work()
        if item.hop is not None:
            item._forget_hop(item.hop)
        await asyncio.sleep(0)

    asyncio.run(run())


def test_a_pass_goes_by_the_shorter_way_with_the_advertised_node_named(monkeypatch) -> None:
    seen = []

    def on_hop(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"token_id": 7})

    item = stage_at(lambda _: httpx.Response(500), on_hop)

    async def work() -> None:
        assert (await send(item)).json() == {"token_id": 7}

    run_on_hop(item, monkeypatch, work)
    assert str(seen[0].url) == REUSED + "/internal/forward"
    assert seen[0].extensions[AUDIENCE] == "b:1" and seen[0].extensions["sni_hostname"] == "b"


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused at 10.0.0.5"), httpx.ReadTimeout("no answer from 10.0.0.5")])
def test_a_failed_shorter_way_asks_for_a_replay_and_goes_back_to_the_old_address(monkeypatch, failure) -> None:
    def on_hop(_: httpx.Request) -> httpx.Response:
        raise failure

    item = stage_at(lambda _: httpx.Response(200), on_hop)

    async def work() -> None:
        with pytest.raises(HTTPException) as caught:
            await send(item)
        assert caught.value.status_code == 503 and caught.value.detail["retry"]
        assert "10.0.0.5" not in str(caught.value.detail)
        assert item.hop is None and item._hop_to(item.route[1]) is None
        assert item.hop_fallbacks == 1

    run_on_hop(item, monkeypatch, work)


def test_a_refusal_from_whoever_has_the_address_now_is_not_the_next_stages_fault(monkeypatch) -> None:
    item = stage_at(lambda _: httpx.Response(200), lambda _: httpx.Response(401, json={"detail": "who are you"}))

    async def work() -> None:
        with pytest.raises(HTTPException) as caught:
            await send(item)
        assert caught.value.detail["retry"]
        assert item.hop is None

    run_on_hop(item, monkeypatch, work)


def test_a_report_from_further_down_the_route_passes_through_as_it_would_have(monkeypatch) -> None:
    report = {"failed_worker": "c", "message": "gone"}
    item = stage_at(lambda _: httpx.Response(200), lambda _: httpx.Response(502, json={"detail": report}))

    async def work() -> None:
        assert (await send(item)).status_code == 502
        assert item.hop is not None  # the link did its job

    run_on_hop(item, monkeypatch, work)


def test_a_route_that_moved_on_is_not_a_failed_link(monkeypatch) -> None:
    item = stage_at(lambda _: httpx.Response(200), lambda _: httpx.Response(409))

    async def work() -> None:
        assert (await send(item)).status_code == 409
        assert item.hop is not None

    run_on_hop(item, monkeypatch, work)


def test_a_pass_on_a_silent_link_gives_up_when_the_watch_does(monkeypatch) -> None:
    async def on_hop(request: httpx.Request) -> httpx.Response:
        if request.url.path == module.ECHO_PATH:
            raise httpx.ReadTimeout("silence")
        await asyncio.sleep(60)
        return httpx.Response(200)

    item = stage_at(lambda _: httpx.Response(200), on_hop)
    monkeypatch.setattr(worker_module, "HOP_SILENCE_SECONDS", 0.05)
    monkeypatch.setattr(worker_module, "HOP_WATCH_SECONDS", 0.01)

    async def run() -> None:
        aim(item, monkeypatch)
        await item.aim_hop({"worker_id": "b", "url": REUSED})
        began = time.monotonic()
        with pytest.raises(HTTPException) as caught:
            await send(item)
        assert caught.value.detail["retry"]
        assert time.monotonic() - began < 5
        assert item.hop is None and item.hop_fallbacks == 1

    asyncio.run(run())


def test_a_stage_takes_an_address_only_for_its_next_stage(monkeypatch) -> None:
    item = stage_at(lambda _: httpx.Response(200), lambda _: httpx.Response(200))

    async def run() -> None:
        aim(item, monkeypatch)
        with pytest.raises(HTTPException) as caught:
            await item.aim_hop({"worker_id": "c", "url": REUSED})
        assert caught.value.status_code == 409 and item.hop is None

    asyncio.run(run())


def test_an_address_for_an_earlier_route_is_dropped(monkeypatch) -> None:
    item = stage_at(lambda _: httpx.Response(200), lambda _: httpx.Response(200))

    async def work() -> None:
        assert item._hop_to(stage("c")) is None
        assert item.hop is None and item.hop_fallbacks == 0

    run_on_hop(item, monkeypatch, work)


def test_being_told_the_old_address_ends_the_shorter_way(monkeypatch) -> None:
    item = stage_at(lambda _: httpx.Response(200), lambda _: httpx.Response(200))

    async def work() -> None:
        await item.aim_hop({"worker_id": "b", "url": None})
        assert item.hop is None and item.hop_fallbacks == 0

    run_on_hop(item, monkeypatch, work)


def probing(monkeypatch, old_way, proved_address: bool = True) -> dict:
    """What a stage reports for one address, with the old way doing what ``old_way`` does."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(old_way))

    async def try_address(url, via, worker_id, sizes, rounds):
        return {"proved": proved_address}

    item = SimpleNamespace(client=client, try_address=try_address, pool_token=TOKEN, requires_identity=False, revoked_nodes=set())
    item.time_link = lambda *args, **options: worker_module.WorkerRuntime.time_link(item, *args, **options)
    monkeypatch.setattr(worker_module, "runtime", item)
    body = {"url": "http://b:1", "worker_id": "b", "addresses": [REUSED], "sizes": [100], "rounds": 1}
    return asyncio.run(worker_module.diag_probe(body, f"Bearer {TOKEN}"))


def test_a_probe_reports_addresses_beside_the_old_way(monkeypatch) -> None:
    answer = probing(monkeypatch, lambda _: httpx.Response(200, json={}))
    assert answer["addresses"] == {REUSED: {"proved": True}} and "100" in answer


def test_a_dead_old_way_leaves_only_its_reading_out(monkeypatch) -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gone")

    assert probing(monkeypatch, refuse) == {"addresses": {REUSED: {"proved": True}}}


def test_a_plain_probe_still_fails_as_it_did(monkeypatch) -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gone")

    item = SimpleNamespace(client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)), pool_token=TOKEN, requires_identity=False, revoked_nodes=set())
    item.time_link = lambda *args, **options: worker_module.WorkerRuntime.time_link(item, *args, **options)
    monkeypatch.setattr(worker_module, "runtime", item)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(worker_module.diag_probe({"url": "http://b:1", "sizes": [100], "rounds": 1}, f"Bearer {TOKEN}"))


def test_an_address_that_will_not_prove_itself_is_never_timed(monkeypatch, no_authority) -> None:
    item = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
    item.proof, item.revoked_nodes, item.pool_token = NodeProof(TOKEN, None), set(), TOKEN
    item.client = SimpleNamespace(auth=None)
    seen = []

    def stranger(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    monkeypatch.setattr(worker_module, "direct_client", lambda _: httpx.AsyncClient(transport=httpx.MockTransport(stranger)))
    answer = asyncio.run(item.try_address(REUSED, "http://b:1", "b", [100], 1))
    assert answer == {"proved": False}
    assert seen == [module.PROVE_PATH]
