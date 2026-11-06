"""A node gets its weights from whichever source is fastest, and nothing is assumed."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import pytest
import torch
from safetensors.torch import save_file

from isharded import transfers, weights
from isharded.protocol import WeightUnit
from isharded.transfers import MODEL_HOST, SlowSource, SourcePicker
from isharded.weights import FetchProgress, LocalReader, UnitCache, fetch_unit, has_model_host, read_header

MIB = 1024**2
PEER = "https://peer:8200"
OTHER = "https://other:8200"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def run(picker: SourcePicker, clock: Clock, transfer, rate: float, seconds: float) -> None:
    """Delivers bytes at a steady rate, a second at a time."""
    for _ in range(int(seconds)):
        clock.now += 1.0
        transfer(int(rate))


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def picker(clock: Clock) -> SourcePicker:
    return SourcePicker(clock)


def test_an_untried_host_goes_first_and_then_the_nodes_get_measured(picker: SourcePicker) -> None:
    first = picker.transfer(picker.pick([MODEL_HOST, PEER]), [], MIB, lambda count: None)
    assert first.source == MODEL_HOST
    with first:
        assert picker.pick([MODEL_HOST, PEER]) == PEER


def test_a_faster_host_is_chosen_over_a_peer(picker: SourcePicker) -> None:
    picker.note(MODEL_HOST, 50e6)
    picker.note(PEER, 2e6)
    assert picker.pick([MODEL_HOST, PEER]) == MODEL_HOST


def test_a_much_faster_peer_is_chosen_over_the_host(picker: SourcePicker) -> None:
    picker.note(MODEL_HOST, 12e6)
    picker.note(PEER, 110e6)
    assert picker.pick([MODEL_HOST, PEER]) == PEER


def test_a_near_tie_goes_to_the_host_to_spare_a_nodes_uplink(picker: SourcePicker) -> None:
    picker.note(MODEL_HOST, 50e6)
    picker.note(PEER, 55e6)
    assert picker.pick([MODEL_HOST, PEER]) == MODEL_HOST


def test_parallel_downloads_use_both_sources_when_that_is_faster(picker: SourcePicker) -> None:
    picker.note(MODEL_HOST, 60e6)
    picker.note(PEER, 50e6)
    with picker.transfer(MODEL_HOST, [PEER], MIB, lambda count: None):
        assert picker.pick([MODEL_HOST, PEER]) == PEER


def test_a_slow_peer_is_given_up_for_a_faster_host(picker: SourcePicker, clock: Clock) -> None:
    picker.note(MODEL_HOST, 50e6)
    with pytest.raises(SlowSource), picker.transfer(PEER, [MODEL_HOST], 2000 * MIB, lambda count: None) as transfer:
        run(picker, clock, transfer, 1.7 * MIB, 30)
    assert clock.now <= transfers.SETTLE_SECONDS + 1
    assert picker.pick([MODEL_HOST, PEER]) == MODEL_HOST


def test_a_transfer_that_starts_fast_then_crawls_is_given_up(picker: SourcePicker, clock: Clock) -> None:
    picker.note(MODEL_HOST, 50e6)
    with pytest.raises(SlowSource), picker.transfer(PEER, [MODEL_HOST], 4000 * MIB, lambda count: None) as transfer:
        run(picker, clock, transfer, 60e6, 8)
        run(picker, clock, transfer, 1e6, 30)


def test_a_transfer_almost_done_is_left_to_finish(picker: SourcePicker, clock: Clock) -> None:
    picker.note(MODEL_HOST, 50e6)
    with picker.transfer(PEER, [MODEL_HOST], 100 * MIB, lambda count: None) as transfer:
        transfer(95 * MIB)
        run(picker, clock, transfer, 0.5 * MIB, 8)


def test_a_slow_source_with_no_alternative_is_never_given_up(picker: SourcePicker, clock: Clock) -> None:
    """A closed network may have one slow holder, and it is still the only way."""
    with picker.transfer(PEER, [], 2000 * MIB, lambda count: None) as transfer:
        run(picker, clock, transfer, 1 * MIB, 60)


def test_a_crawl_tries_a_source_nobody_has_measured(picker: SourcePicker, clock: Clock) -> None:
    with pytest.raises(SlowSource), picker.transfer(PEER, [OTHER], 2000 * MIB, lambda count: None) as transfer:
        run(picker, clock, transfer, 1 * MIB, 30)


def test_a_failed_source_stays_a_last_resort(picker: SourcePicker) -> None:
    picker.fail(MODEL_HOST)
    assert picker.pick([MODEL_HOST, PEER]) == PEER
    assert picker.pick([MODEL_HOST]) == MODEL_HOST


def test_a_second_visit_is_never_given_up(picker: SourcePicker, clock: Clock) -> None:
    picker.note(MODEL_HOST, 50e6)
    with picker.transfer(PEER, [MODEL_HOST], 2000 * MIB, lambda count: None, patient=True) as transfer:
        run(picker, clock, transfer, 1 * MIB, 30)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("org/model", True), ("org/model:file.gguf", True), ("file.gguf", False), (r"D:\elsewhere\file.gguf", False)],
)
def test_a_model_named_only_by_its_file_has_no_host(source: str, expected: bool) -> None:
    assert has_model_host(source) is expected


class Throttled(BaseHTTPRequestHandler):
    """Serves a file at a set speed, as a real host or a real peer would."""

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        payload, rate = self.server.payload, self.server.rate
        self.server.served += 1
        if "/internal/shards/file" in self.path:
            start, end = 0, len(payload)
            self.send_response(200)
        else:
            first, last = self.headers["Range"].removeprefix("bytes=").split("-")
            start, end = int(first), int(last) + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end - 1}/{len(payload)}")
        self.send_header("Content-Length", str(end - start))
        self.end_headers()
        try:
            step = max(64 * 1024, int(rate / 50))  # sleeps shorter than this are not kept on Windows
            for position in range(start, end, step):
                self.wfile.write(payload[position : min(position + step, end)])
                time.sleep(step / rate)
        except OSError:
            pass


def serve(payload: bytes, rate: float) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Throttled)
    server.payload, server.rate, server.served = payload, rate, 0
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def address(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def model(tmp_path: Path) -> tuple[WeightUnit, bytes, bytes]:
    """One 16 MiB unit, the bytes a host serves, and the unit file a peer serves."""
    folder = tmp_path / "model"
    folder.mkdir()
    save_file({"a": torch.zeros(4 * MIB, dtype=torch.float32)}, folder / "model.safetensors")
    unit = WeightUnit(name="unit", kind="dense", layer=0, tensors=list(read_header(LocalReader(str(folder)), "model.safetensors").values()))
    built = tmp_path / "built"
    fetch_unit(LocalReader(str(folder)), unit, built, lambda count: None)
    return unit, (folder / "model.safetensors").read_bytes(), built.read_bytes()


@pytest.fixture
def quick(monkeypatch) -> None:
    monkeypatch.setattr(transfers, "SETTLE_SECONDS", 0.3)
    monkeypatch.setattr(transfers, "CHECK_SECONDS", 0.1)


def test_over_real_connections_a_crawling_peer_loses_to_the_host(model, tmp_path: Path, monkeypatch, quick) -> None:
    unit, hosted, built = model
    host, peer = serve(hosted, 200 * MIB), serve(built, 1 * MIB)
    monkeypatch.setenv("HF_ENDPOINT", address(host))
    monkeypatch.setattr(weights, "client_verify", lambda: False)
    monkeypatch.setattr(weights, "client_auth", lambda: None)
    cache = UnitCache(tmp_path / "cache")
    picker = SourcePicker()
    picker.note(MODEL_HOST, 200 * MIB)  # as it would be once another unit had arrived
    monkeypatch.setattr(weights, "SourcePicker", lambda: picker)
    monkeypatch.setattr(SourcePicker, "pick", lambda self, sources: sources[-1] if len(sources) == 2 and not self.busy else sources[0])
    progress = FetchProgress("job")
    began = time.monotonic()
    cache.fetch("org/model", "m", "r", [unit], progress, peers={"unit": [address(peer)]})
    assert cache.has("m", "r", unit)
    assert progress.bytes_from_peers == 0
    assert progress.bytes_done == unit.nbytes
    assert time.monotonic() - began < 8  # the peer alone would take 16 s
    assert peer.served == 1 and host.served >= 1


def test_over_real_connections_a_fast_peer_beats_a_crawling_host(model, tmp_path: Path, monkeypatch, quick) -> None:
    unit, hosted, built = model
    host, peer = serve(hosted, 1 * MIB), serve(built, 200 * MIB)
    monkeypatch.setenv("HF_ENDPOINT", address(host))
    monkeypatch.setattr(weights, "client_verify", lambda: False)
    monkeypatch.setattr(weights, "client_auth", lambda: None)
    cache = UnitCache(tmp_path / "cache")
    progress = FetchProgress("job")
    began = time.monotonic()
    cache.fetch("org/model", "m", "r", [unit], progress, peers={"unit": [address(peer)]})
    assert cache.has("m", "r", unit)
    assert progress.bytes_from_peers == unit.nbytes
    assert progress.bytes_done == unit.nbytes
    assert time.monotonic() - began < 8


def test_on_a_closed_network_the_host_is_skipped_quickly_and_peers_deliver(model, tmp_path: Path, monkeypatch, quick) -> None:
    unit, _, built = model
    peer = serve(built, 200 * MIB)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(weights, "client_verify", lambda: False)
    monkeypatch.setattr(weights, "client_auth", lambda: None)
    cache = UnitCache(tmp_path / "cache")
    progress = FetchProgress("job")
    began = time.monotonic()
    cache.fetch("org/model", "m", "r", [unit], progress, peers={"unit": [address(peer)]})
    assert progress.bytes_from_peers == unit.nbytes
    assert time.monotonic() - began < 10  # retrying the host would take over a minute


def test_a_cancelled_load_stops_at_once(model, tmp_path: Path, monkeypatch, quick) -> None:
    unit, _, built = model
    peer = serve(built, 2 * MIB)
    monkeypatch.setattr(weights, "client_verify", lambda: False)
    monkeypatch.setattr(weights, "client_auth", lambda: None)
    progress = FetchProgress("job")
    threading.Timer(0.3, lambda: setattr(progress, "cancelled", True)).start()
    with pytest.raises(weights.WeightSourceError, match="cancelled"):
        UnitCache(tmp_path / "cache").fetch("only-in-the-pool.gguf", "m", "r", [unit], progress, peers={"unit": [address(peer)]})
