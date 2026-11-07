from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import pytest

from isharded import node as node_module
from isharded.lanes import (
    LANE_ASK_ATTEMPTS,
    LANE_GPU_OVERHEAD_GB,
    LANE_PORT_STEP,
    LaneDoor,
    clean_lanes,
    lane_budgets,
    lane_url,
    read_lanes,
    spare_budgets,
    write_lanes,
)
from isharded.mesh import ControlMesh
from isharded.node import LaneKeeper, add_node_arguments, lane_arguments, node_specs
from isharded.protocol import MemoryPool

GIB = 1024**3


def node_args(*extra: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_node_arguments(parser)
    args = parser.parse_args(["--node-id", "alpha", "--advertise-host", "10.0.0.1", "--gpu-device", "none", "--cpu-threads", "2", "--weights-ram-gb", "4", *extra])
    return args


def test_a_lane_is_the_same_node_a_fixed_step_up() -> None:
    assert lane_url("http://10.0.0.2:8300", 2) == f"http://10.0.0.2:{8300 + 2 * LANE_PORT_STEP}"
    assert lane_url("https://[fd7a::1]:8000", 1) == f"https://[fd7a::1]:{8000 + LANE_PORT_STEP}"


def test_only_well_formed_lanes_survive() -> None:
    lanes = clean_lanes(
        {
            "small": {"slot": 1, "budgets": {"alpha": {"gpu_gb": 1, "ram_gb": -2}}},
            "Bad Name": {"slot": 2},
            "../escape": {"slot": 2},
            "toohigh": {"slot": 99},
            "noslot": {},
            "notadict": 3,
        }
    )
    assert list(lanes) == ["small"]
    assert lanes["small"]["budgets"]["alpha"] == {"gpu_gb": 1.0, "ram_gb": 0.0}
    assert clean_lanes("nonsense") == {}


def test_the_file_survives_a_round_trip_and_a_bad_one_means_no_lanes(tmp_path: Path) -> None:
    path = tmp_path / "folder" / "lanes.json"
    write_lanes(path, {"small": {"slot": 1, "budgets": {}, "by": "n-1"}}, 4)
    assert read_lanes(path) == ({"small": {"slot": 1, "budgets": {}, "by": "n-1"}}, 4)
    path.write_text("{broken", encoding="utf-8")
    assert read_lanes(path) == ({}, 0)
    assert read_lanes(tmp_path / "missing.json") == ({}, 0)


def record(worker_id: str, pools: list[MemoryPool], used: dict[str, int], live: bool = True) -> SimpleNamespace:
    hardware = SimpleNamespace(memory_pools=pools)
    return SimpleNamespace(
        live=live,
        health={"memory_pools": [{"pool_id": key, "used_bytes": value} for key, value in used.items()]},
        registration=SimpleNamespace(worker_id=worker_id, hardware=hardware),
    )


def pool(pool_id: str, kind: str, group: str, budget_gb: float) -> MemoryPool:
    return MemoryPool(pool_id=pool_id, kind=kind, device="cpu", capacity_group=group, budget_bytes=int(budget_gb * GIB))


def test_spare_memory_stays_under_what_the_owner_set_and_counts_each_resource_once() -> None:
    workers = {
        "alpha-gpu": record("alpha-gpu", [pool("v", "vram", "alpha-vram", 8), pool("r", "ram", "alpha-ram", 6)], {"v": 5 * GIB, "r": 1 * GIB}),
        "alpha-cpu": record("alpha-cpu", [pool("r2", "ram", "alpha-ram", 6)], {"r2": 2 * GIB}),
        "beta-gpu": record("beta-gpu", [pool("v", "vram", "beta-vram", 24)], {"v": 30 * GIB}),
        "gone-gpu": record("gone-gpu", [pool("v", "vram", "gone-vram", 24)], {}, live=False),
    }
    spare = spare_budgets(workers)
    assert spare["alpha"] == {"gpu_gb": 3.0, "ram_gb": 3.0}
    assert spare["beta"]["gpu_gb"] == 0.0
    assert "gone" not in spare


def test_lanes_share_what_is_spare_and_never_exceed_it() -> None:
    spare = {"alpha": {"gpu_gb": 4.0, "ram_gb": 10.0}}
    budgets = lane_budgets(spare, {})
    assert budgets["alpha"] == {"gpu_gb": 4.0 - LANE_GPU_OVERHEAD_GB, "ram_gb": 10.0}
    existing = {"one": {"slot": 1, "budgets": {"alpha": {"gpu_gb": 3.0, "ram_gb": 6.0}}}}
    assert lane_budgets(spare, existing)["alpha"] == {"gpu_gb": 0.5, "ram_gb": 4.0}
    assert lane_budgets(spare, {}, gpu_gb=1.0, ram_gb=2.0)["alpha"] == {"gpu_gb": 1.0, "ram_gb": 2.0}
    assert lane_budgets({"alpha": {"gpu_gb": 0.2, "ram_gb": 0.0}}, {})["alpha"] == {"gpu_gb": 0.0, "ram_gb": 0.0}


def door_to_a_lane(monkeypatch, *failures: Exception) -> tuple[LaneDoor, list[str]]:
    """A door to a lane whose coordinator fails these ways before it answers."""
    left, asked = list(failures), []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.host)
        if left:
            raise left.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    async def no_wait(_: float) -> None:
        pass

    monkeypatch.setattr("isharded.lanes.asyncio.sleep", no_wait)
    door = LaneDoor(
        SimpleNamespace(
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            mesh=SimpleNamespace(node_id="n", lanes={}, url="http://n:1"),
            access=SimpleNamespace(authority=None),
            pool_token=None,
        )
    )
    door.seen["a"] = {"leader_url": "http://leader", "leader": "n"}
    return door, asked


@pytest.mark.parametrize("failure", [httpx.ConnectError("down"), httpx.ReadError("reset"), httpx.RemoteProtocolError("cut off")], ids=["connect", "read", "protocol"])
def test_asking_a_lane_waits_out_a_lost_coordinator(monkeypatch, failure: Exception) -> None:
    door, asked = door_to_a_lane(monkeypatch, failure)
    assert asyncio.run(door.ask("a", {}, None)) == "hello"
    assert len(asked) == 2


def test_asking_a_lane_gives_up_the_same_way_as_with_no_coordinator(monkeypatch) -> None:
    door, asked = door_to_a_lane(monkeypatch, *[httpx.ConnectError("down") for _ in range(LANE_ASK_ATTEMPTS)])
    with pytest.raises(HTTPException) as refused:
        asyncio.run(door.ask("a", {}, None))
    assert refused.value.status_code == 503
    assert len(asked) == LANE_ASK_ATTEMPTS


def test_a_lane_silent_for_the_whole_wait_is_not_asked_again(monkeypatch) -> None:
    door, asked = door_to_a_lane(monkeypatch, httpx.ReadTimeout("silent"))
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(door.ask("a", {}, None))
    assert len(asked) == 1


def test_followers_learn_lanes_from_the_leader_only() -> None:
    now = [100.0]
    leader = ControlMesh("alpha", "http://10.0.0.1:8000", (), clock=lambda: now[0])
    follower = ControlMesh("beta", "http://10.0.0.2:8000", ("http://10.0.0.1:8000",), clock=lambda: now[0])
    leader.leader_id = "alpha"
    leader.publish_lanes({"small": {"slot": 1, "budgets": {}}})
    stranger = {**leader.message(), "node_id": "gamma", "url": "http://10.0.0.3:8000", "leader_id": "alpha", "lanes": {"evil": {"slot": 1}}, "lanes_version": 9}
    follower.observe(leader.message())
    assert follower.lanes == {"small": {"slot": 1, "budgets": {}}}
    follower.observe(stranger)
    assert "evil" not in follower.lanes
    assert follower.message()["lanes"] is None


def test_a_lane_runs_on_higher_ports_with_its_own_records_and_only_what_is_spare() -> None:
    args = node_args("--join", "http://10.0.0.2:8309", "--base-port", "8200", "--control-port", "8000")
    lane = {"slot": 2, "budgets": {"alpha": {"gpu_gb": 0.0, "ram_gb": 1.5}}}
    changed = lane_arguments(args, "small", lane)
    specs = dict(node_specs(changed))
    control = specs["Control plane"]
    assert control[control.index("--port") + 1] == str(8000 + 2 * LANE_PORT_STEP)
    assert control[control.index("--join") + 1] == f"http://10.0.0.2:{8309 + 2 * LANE_PORT_STEP}"
    assert control[control.index("--lane") + 1] == "small"
    assert "alpha-small" in control[control.index("--state-file") + 1]
    assert "small" in control[control.index("--keys-file") + 1]
    worker = specs["CPU inference"]
    assert worker[worker.index("--port") + 1] == str(8200 + 2 * LANE_PORT_STEP)
    assert worker[worker.index("--weights-ram-gb") + 1] == "1.5"
    assert "--standby" not in worker
    assert args.base_port == 8200  # the node's own settings are left alone


def test_a_node_with_nothing_spare_still_answers_for_the_lane() -> None:
    changed = lane_arguments(node_args(), "small", {"slot": 1, "budgets": {}})
    assert [label for label, _ in node_specs(changed)] == ["Control plane"]


class FakeService:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started = 0
        self.stopped = 0
        self.supervised = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def supervise(self) -> None:
        self.supervised += 1


def test_the_keeper_starts_and_stops_lanes_as_the_pool_asks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(node_module, "data_folder", lambda: tmp_path)
    path = tmp_path / "lanes.json"
    made: list[FakeService] = []

    def make(label: str, _: list[str]) -> FakeService:
        made.append(FakeService(label))
        return made[-1]

    keeper = LaneKeeper(node_args(), path, make)
    keeper.supervise()
    assert not made
    write_lanes(path, {"small": {"slot": 1, "budgets": {"alpha": {"gpu_gb": 0, "ram_gb": 2}}}}, 1)
    keeper.supervise()
    keeper.supervise()
    assert [service.started for service in made] == [1, 1]  # started once, not each pass
    assert all(service.supervised == 2 for service in made)
    state = node_module.lane_state_file("alpha", "small")
    state.write_text("{}", encoding="utf-8")
    write_lanes(path, {}, 2)
    keeper.supervise()
    assert all(service.stopped == 1 for service in made)
    assert not state.exists()  # a lane made later must not bring back the old model
