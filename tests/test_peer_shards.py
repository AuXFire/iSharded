"""Workers reuse what the pool already has before touching the model host."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from shardmesh import worker as worker_module
from shardmesh.planner import Candidate, ModelProfile, plan_placement
from shardmesh.protocol import MemoryPool, TensorRef, WeightUnit
from shardmesh.weights import (
    FetchProgress,
    LocalReader,
    UnitCache,
    fetch_from_peer,
    fetch_unit,
    load_unit,
    read_header,
)

GIB = 1024**3
MIB = 1024**2


def source(directory: Path) -> WeightUnit:
    directory.mkdir(parents=True, exist_ok=True)
    save_file({"a": torch.arange(3000, dtype=torch.float32), "b": torch.arange(50, dtype=torch.bfloat16)}, directory / "model.safetensors")
    header = read_header(LocalReader(str(directory)), "model.safetensors")
    return WeightUnit(name="unit", kind="dense", layer=0, tensors=list(header.values()))


def serving(file: Path, status: int = 200, cut: int = 0, tamper: bool = False) -> httpx.Client:
    payload = file.read_bytes()
    if tamper:
        payload = b"\0" * 32 + payload[32:]
    if cut:
        payload = payload[:-cut]
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, content=payload)))


@pytest.fixture
def held(tmp_path: Path) -> tuple[WeightUnit, Path]:
    unit = source(tmp_path / "model")
    file = tmp_path / "held.safetensors"
    fetch_unit(LocalReader(str(tmp_path / "model")), unit, file, lambda count: None)
    return unit, file


def test_a_peer_copy_is_identical_and_loads(held, tmp_path: Path) -> None:
    unit, file = held
    destination = tmp_path / "copy" / "unit.safetensors"
    moved: list[int] = []
    assert fetch_from_peer("http://peer", None, "m", "r", unit, destination, moved.append, serving(file))
    assert destination.read_bytes() == file.read_bytes()
    assert sum(moved) == file.stat().st_size
    assert set(load_unit(destination, "cpu", None, lambda name: False).tensors) == {"a", "b"}


@pytest.mark.parametrize("options", [{"status": 404}, {"status": 500}, {"cut": 10}, {"tamper": True}], ids=["missing", "error", "short", "wrong-header"])
def test_a_bad_peer_leaves_nothing_behind(held, tmp_path: Path, options: dict) -> None:
    unit, file = held
    destination = tmp_path / "copy" / "unit.safetensors"
    moved: list[int] = []
    assert not fetch_from_peer("http://peer", None, "m", "r", unit, destination, moved.append, serving(file, **options))
    assert not destination.exists()
    assert sum(moved) == 0
    assert not list(destination.parent.glob("*.part"))


def test_an_unreachable_peer_falls_back_to_the_model_host(tmp_path: Path) -> None:
    unit = source(tmp_path / "model")
    cache = UnitCache(tmp_path / "cache")
    progress = FetchProgress("job")
    cache.fetch(str(tmp_path / "model"), "m", "r", [unit], progress, peers={"unit": ["http://127.0.0.1:1"]})
    assert cache.has("m", "r", unit)
    assert progress.bytes_from_peers == 0
    assert progress.bytes_done == unit.nbytes


def test_progress_matches_the_plan_when_a_peer_delivers(held, tmp_path: Path, monkeypatch) -> None:
    unit, file = held
    payload = file.read_bytes()

    def copy(peer, token, model_id, revision, item, destination, on_bytes, client=None) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        on_bytes(len(payload))
        return True

    monkeypatch.setattr("shardmesh.weights.fetch_from_peer", copy)
    cache = UnitCache(tmp_path / "cache")
    progress = FetchProgress("job")
    cache.fetch("unused-source", "m", "r", [unit], progress, peers={"unit": ["http://peer"]})
    assert progress.bytes_done == unit.nbytes
    assert progress.bytes_from_peers == unit.nbytes
    assert progress.units_done == 1


@pytest.fixture
def worker_app(held, tmp_path: Path, monkeypatch) -> TestClient:
    unit, file = held
    cache = UnitCache(tmp_path / "served")
    destination = cache.path("org/model", "abc", "unit")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(file.read_bytes())
    secret = tmp_path / "served" / "secret.safetensors"
    secret.write_bytes(b"outside the model folder")
    monkeypatch.setattr(worker_module.runtime, "unit_cache", cache)
    monkeypatch.setattr(worker_module.runtime, "peer_serve", True)
    monkeypatch.setattr(worker_module.runtime, "pool_token", None)
    return TestClient(worker_module.app)


def query(unit: str = "unit", revision: str = "abc") -> dict[str, str]:
    return {"model_id": "org/model", "revision": revision, "unit": unit}


def test_workers_list_and_serve_what_they_hold(worker_app: TestClient, held) -> None:
    _, file = held
    index = worker_app.get("/internal/shards/index", params={"model_id": "org/model", "revision": "abc"}).json()
    assert index == {"units": {"unit": file.stat().st_size}, "serves_peers": True}
    response = worker_app.get("/internal/shards/file", params=query())
    assert response.status_code == 200
    assert response.content == file.read_bytes()


@pytest.mark.parametrize(
    "unit, revision",
    [("../../secret", "abc"), ("..", "abc"), ("unit", "../.."), ("a/b", "abc"), ("a\\b", "abc"), ("", "abc")],
)
def test_served_paths_cannot_leave_the_cache(worker_app: TestClient, unit: str, revision: str) -> None:
    assert worker_app.get("/internal/shards/file", params=query(unit, revision)).status_code == 404


def test_a_node_can_refuse_to_serve_peers(worker_app: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(worker_module.runtime, "peer_serve", False)
    assert worker_app.get("/internal/shards/file", params=query()).status_code == 403


def test_shards_need_the_pool_token(worker_app: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(worker_module.runtime, "pool_token", "secret")
    assert worker_app.get("/internal/shards/file", params=query()).status_code == 401
    allowed = worker_app.get("/internal/shards/file", params=query(), headers={"Authorization": "Bearer secret"})
    assert allowed.status_code == 200


def unit(name: str, kind: str, size: int, layer: int | None = None) -> WeightUnit:
    return WeightUnit(
        name=name,
        kind=kind,
        layer=layer,
        tensors=[TensorRef(name=name, file="f", dtype="BF16", shape=[size // 2], start=0, end=size)],
    )


def profile(amortize: int = 20_000) -> ModelProfile:
    units = [unit("embed", "embed", 50 * MIB), unit("head", "head", 50 * MIB)]
    units += [unit(f"layer-{index:05d}.dense", "dense", 500 * MIB, index) for index in range(8)]
    return ModelProfile(
        units=units,
        num_layers=8,
        expert_active_fraction=1.0,
        kv_bytes_per_token=[0] * 8,
        state_bytes=[0] * 8,
        amortize_tokens=amortize,
    )


def worker(name: str, speed: float, cached: set[str], rate: float = 8e6, peers: frozenset[str] = frozenset()) -> Candidate:
    pool = MemoryPool(
        pool_id="cuda:0",
        kind="vram",
        device="cuda:0",
        capacity_group=name,
        budget_bytes=8 * GIB,
        free_bytes=8 * GIB,
        bandwidth_gbps=speed,
    )
    return Candidate(name, f"http://{name}", [pool], frozenset(cached), peers, rate)


def layers_of(route) -> dict[str, tuple[int, int]]:
    return {stage.worker_id: (stage.start_layer, stage.end_layer) for stage in route}


def names(start: int, end: int, *ends: str) -> set[str]:
    return {f"layer-{index:05d}.dense" for index in range(start, end)} | set(ends)


def test_near_ties_stay_where_the_shards_already_are() -> None:
    a = worker("a", 300, names(0, 2, "embed"))
    b = worker("b", 301, names(2, 8, "head"))
    assert layers_of(plan_placement([a, b], [], profile())) == {"a": (0, 2), "b": (2, 8)}


def test_the_same_workers_rebalance_freely_when_downloads_are_ignored() -> None:
    a = worker("a", 300, names(0, 2, "embed"))
    b = worker("b", 301, names(2, 8, "head"))
    assert layers_of(plan_placement([a, b], [], profile(amortize=0))) != {"a": (0, 2), "b": (2, 8)}


def test_a_large_speedup_still_earns_its_download() -> None:
    slow = worker("slow", 2, names(0, 8, "embed", "head"))
    fast = worker("fast", 500, set(), rate=100e6)
    route = plan_placement([slow, fast], [], profile())
    spans = layers_of(route)
    assert spans["fast"][1] - spans["fast"][0] > spans["slow"][1] - spans["slow"][0]


def test_a_shard_a_peer_holds_is_cheaper_than_downloading_it() -> None:
    with_peers = worker("a", 300, set(), rate=1e6, peers=frozenset(names(0, 8, "embed", "head")))
    alone = worker("a", 300, set(), rate=1e6)
    other = worker("b", 300, names(0, 8, "embed", "head"))
    near = plan_placement([with_peers, other], [], profile())
    far = plan_placement([alone, other], [], profile())
    assert layers_of(near)["a"][1] - layers_of(near)["a"][0] >= layers_of(far)["a"][1] - layers_of(far)["a"][0]
