"""The scale harness (tests/scale): its pieces, and a five-node run of the real control plane."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

SCALE = Path(__file__).parent / "scale"
sys.path.insert(0, str(SCALE))

import harness  # noqa: E402
import host  # noqa: E402
import report  # noqa: E402


def test_calls_are_sorted_by_what_they_are():
    assert host.kind_of("/internal/gossip") == "gossip"
    assert host.kind_of("/internal/register") == host.kind_of("/internal/heartbeat") == "checkin"
    assert host.kind_of("/health") == "health"
    assert host.kind_of("/admin/status") == "status"
    assert host.kind_of("/v1/chat/completions") == "other"


def test_a_silent_link_costs_what_a_real_one_does():
    request = httpx.Request("POST", "http://peer/", extensions={"timeout": {"connect": 2.5, "read": 2.5}})
    # An open connection: the answer never comes, and the read times out once.
    assert host.silent_wait(request, 3, kept=True) == (2.5, "ReadTimeout")
    # No connection: four connect attempts with httpcore's backoff of 0, 0.5 and 1 s between them.
    assert host.silent_wait(request, 3, kept=False) == (4 * 2.5 + 1.5, "ConnectTimeout")
    assert host.silent_wait(request, 0, kept=False) == (2.5, "ConnectTimeout")


def test_a_partition_cuts_only_across_sides():
    faults = host.Faults()
    assert not faults.cut("a", "b")
    faults.sides = {"a": 0, "b": 0, "c": 1}
    assert not faults.cut("a", "b")
    assert faults.cut("a", "c") and faults.cut("c", "b")
    assert not faults.cut("a", None)  # an address no node owns is never cut


def test_growth_law_is_the_log_log_slope():
    assert report.slope([(10, 5.0), (50, 25.0)]) == pytest.approx(1.0)
    assert report.slope([(10, 1.0), (100, 100.0)]) == pytest.approx(2.0)
    assert report.slope([(10, 3.0), (50, 3.0), (100, 3.0)]) == pytest.approx(0.0)
    assert report.slope([(10, 0.0), (50, 3.0)]) is None  # one usable point says nothing
    assert report.law(1.08).startswith("O(N)")
    assert report.law(1.95).startswith("O(N^2)")
    assert report.law(1.5) == "N^1.50"


def test_growth_table_reads_results():
    def result(nodes, gossip):
        return {"nodes": nodes, "phases": {"steady": {"per_node": {"out:gossip:calls": {"mean": gossip}}}}}

    table = report.growth_table([result(50, 16.3), result(10, 3.0)])
    row = next(line for line in table.splitlines() if line.startswith("|   gossip"))
    assert "| 3.00 | 16.30 | O(N)" in row
    assert any(line.startswith("|   model") for line in table.splitlines())


def test_whatever_a_phase_kills_or_freezes_runs_alone(tmp_path):
    args = harness.parse_args(["--nodes", "12", "--phases", "all", "--per-process", "4", "--run-dir", str(tmp_path)])
    pool = harness.Pool(args, 12)
    try:
        groups = pool.layout()
        alone = {group[0] for group in groups if len(group) == 1}
        assert {pool.leader, pool.second, pool.leaver} <= alone
        assert sorted(name for group in groups for name in group) == pool.names[:-1]
        assert pool.newcomer not in {name for group in groups for name in group}
        assert max(len(group) for group in groups) == 4
        assert pool.leader == "n000" and pool.newcomer == "n012"
    finally:
        asyncio.run(pool.close())


def test_a_repeated_phase_is_recorded_again():
    args = harness.parse_args(["--nodes", "5"])
    assert args.phases.count("steady") == 2
    with pytest.raises(SystemExit):
        harness.parse_args(["--nodes", "5", "--phases", "steady,teleport"])


def test_five_nodes_settle_join_leave_and_elect(tmp_path):
    """The real control plane on five emulated nodes, in about a minute and a half.

    It checks the harness measures what the code says should happen: gossip to every
    peer every 3 s, a check-in with every control plane every 5 s, a health probe of every
    worker every 10 s, and one election when the coordinator dies.
    """
    out = tmp_path / "scale-5.json"
    args = harness.parse_args([
        "--nodes", "5",
        "--phases", "steady,join,leave,kill-leader",
        "--steady-seconds", "9",
        "--settle-seconds", "1",
        "--settle-timeout", "120",
        "--run-dir", str(tmp_path),
        "--out", str(out),
    ])  # fmt: skip
    result = asyncio.run(harness.run(args, 5))
    out.write_text(json.dumps(result), encoding="utf-8")
    assert "error" not in result, result.get("error")
    assert result["startup"]["leader"] == "n000"

    steady = result["phases"]["steady"]
    rates = {key: value["mean"] for key, value in steady["per_node"].items()}
    assert steady["nodes"] == 5
    # Three gossip rounds in 9 s, to four peers, give 1.33 calls a second; rounds also take time.
    assert 0.8 <= rates["out:gossip:calls"] <= 1.8
    assert 0.7 <= rates["out:checkin:calls"] <= 1.3  # five control planes every 5 s
    assert 0.25 <= rates["out:health:calls"] <= 0.75  # five workers every 10 s
    assert rates["total:out:failures"] == 0
    assert rates["total:out:bytes"] > 1000 and rates["total:in:bytes"] > 1000
    assert steady["fanout"]["gossip"]["max"] == 4
    assert steady["fanout"]["checkin"]["max"] == 5
    assert steady["fanout"]["health"]["max"] == 5
    assert steady["cpu_cores_per_node"] > 0

    join = result["phases"]["join"]
    assert 0 <= join["mesh_s"] < 30 and join["elections"] == 0
    assert join["worker_live_everywhere_s"] < 30 and join["controls_everywhere_s"] < 45

    leave = result["phases"]["leave"]
    # A peer silent for 12 s is gone, and a worker silent for 15 s is not live.
    assert 10 <= leave["mesh_s"] < 60
    assert 13 <= leave["worker_dropped_everywhere_s"] < 60
    assert leave["elections"] == 0

    kill = result["phases"]["kill-leader"]
    assert kill["old"] == "n000" and kill["new"] == "n001"
    assert kill["elections"] == 1
    assert 10 <= kill["agreed_s"] < 90
