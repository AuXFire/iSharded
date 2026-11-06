"""Finding a mesh: answers are untrusted, and malformed ones must do nothing."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from isharded import discovery

GOOD = {"url": "http://10.0.0.5:8209/", "authority": "a" * 64, "name": "first"}


def test_a_good_answer_is_kept() -> None:
    assert discovery.valid_answer(json.dumps(GOOD).encode()) == {"url": "http://10.0.0.5:8209", "authority": "a" * 64, "name": "first"}


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b"{}",
        json.dumps({**GOOD, "url": "file:///etc/passwd"}).encode(),
        json.dumps({**GOOD, "url": "http://no-port"}).encode(),
        json.dumps({**GOOD, "authority": "short"}).encode(),
    ],
)
def test_a_bad_answer_is_dropped(raw: bytes) -> None:
    assert discovery.valid_answer(raw) is None


def test_a_beacon_answers_only_a_real_probe_and_a_search_finds_it() -> None:
    async def scenario() -> list[dict[str, str]]:
        transport = await discovery.start_beacon(lambda: GOOD, port=18208)
        assert transport is not None
        try:
            return await asyncio.to_thread(discovery.ask_network, 0.6, 18208, ["127.0.0.1"])
        finally:
            transport.close()

    found = asyncio.run(scenario())
    assert found and found[0]["name"] == "first"


def test_a_beacon_ignores_anything_else() -> None:
    sent = []
    beacon = discovery.Beacon(lambda: GOOD)
    beacon.connection_made(SimpleNamespace(sendto=lambda data, address: sent.append(data)))
    beacon.datagram_received(b"hello", ("10.0.0.9", 5))
    assert not sent
    beacon.datagram_received(discovery.PROBE, ("10.0.0.9", 5))
    assert len(sent) == 1


def test_only_online_tailnet_peers_are_asked(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "tailscale_tool", lambda: "tailscale")
    status = {"Peer": {"a": {"Online": True, "TailscaleIPs": ["100.64.0.1"]}, "b": {"Online": False, "TailscaleIPs": ["100.64.0.2"]}, "c": {"Online": True, "TailscaleIPs": []}}}
    answer = SimpleNamespace(stdout=json.dumps(status))
    assert discovery.tailnet_addresses(run=lambda *_, **__: answer) == ["100.64.0.1"]


def test_no_tailscale_means_no_peers(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "tailscale_tool", lambda: None)
    assert discovery.tailnet_addresses() == []


def test_a_mesh_answering_twice_is_listed_once(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "ask_network", lambda wait=0: [{**GOOD, "url": "http://a:1"}, {**GOOD, "url": "http://b:1"}])
    monkeypatch.setattr(discovery, "ask_tailnet", lambda: [{**GOOD, "authority": "b" * 64}])
    assert len(discovery.find_meshes()) == 2
