"""Tabulate scale-harness results against N, with the growth law each metric follows.

    python tests/scale/report.py scale-results/scale-*.json

The law is the slope of log(metric) against log(N), fitted over every size given:
about 0 is flat, 1 grows with N, 2 with N squared. Two sizes give a slope from two
points; more sizes make the fit trustworthy.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# The code's own periods, for the model row: gossip to every peer every 3 s
# (mesh.GOSSIP_SECONDS), a check-in from the worker to every control plane every
# 5 s (worker.CHECK_IN_SECONDS), and a health probe of every worker every 10 s
# (orchestrator.MAINTENANCE_SECONDS).
GOSSIP_SECONDS = 3.0
CHECK_IN_SECONDS = 5.0
PROBE_SECONDS = 10.0

METRICS: list[tuple[str, tuple[str, ...]]] = [
    ("calls sent /s per node", ("phases", "steady", "per_node", "total:out:calls", "mean")),
    ("  gossip", ("phases", "steady", "per_node", "out:gossip:calls", "mean")),
    ("  check-in", ("phases", "steady", "per_node", "out:checkin:calls", "mean")),
    ("  health probe", ("phases", "steady", "per_node", "out:health:calls", "mean")),
    ("  model: (N-1)/3 + N/5 + N/10", ("model",)),
    ("calls received /s per node", ("phases", "steady", "per_node", "total:in:calls", "mean")),
    ("bytes sent /s per node", ("phases", "steady", "per_node", "total:out:bytes", "mean")),
    ("  gossip", ("phases", "steady", "per_node", "out:gossip:bytes", "mean")),
    ("  check-in", ("phases", "steady", "per_node", "out:checkin:bytes", "mean")),
    ("  health probe", ("phases", "steady", "per_node", "out:health:bytes", "mean")),
    ("bytes received /s per node", ("phases", "steady", "per_node", "total:in:bytes", "mean")),
    ("bytes per gossip call (both ways)", ("ratio", "out:gossip")),
    ("bytes per check-in call (both ways)", ("ratio", "out:checkin")),
    ("bytes per health probe (both ways)", ("ratio", "out:health")),
    ("failed calls /s per node", ("phases", "steady", "per_node", "total:out:failures", "mean")),
    ("live peers falsely dropped in the window", ("phases", "steady", "live_peers_dropped")),
    ("live workers falsely dropped in the window", ("phases", "steady", "live_workers_dropped")),
    ("gossip fan-out (peers called)", ("phases", "steady", "fanout", "gossip", "mean")),
    ("check-in fan-out", ("phases", "steady", "fanout", "checkin", "mean")),
    ("health fan-out", ("phases", "steady", "fanout", "health", "mean")),
    ("after a leave: calls sent /s per node", ("phases", "steady-2", "per_node", "total:out:calls", "mean")),
    ("after a leave: gossip calls /s per node", ("phases", "steady-2", "per_node", "out:gossip:calls", "mean")),
    ("after a leave: failed calls /s per node", ("phases", "steady-2", "per_node", "total:out:failures", "mean")),
    ("after a leave: live peers falsely dropped", ("phases", "steady-2", "live_peers_dropped")),
    ("after a silent leave too: gossip calls /s per node", ("phases", "steady-3", "per_node", "out:gossip:calls", "mean")),
    ("after a silent leave too: failed calls /s per node", ("phases", "steady-3", "per_node", "total:out:failures", "mean")),
    ("after a silent leave too: live peers falsely dropped", ("phases", "steady-3", "live_peers_dropped")),
    ("machine CPU % (every node shares it)", ("phases", "steady", "machine_cpu_percent")),
    ("CPU cores per node", ("phases", "steady", "cpu_cores_per_node")),
    ("RSS MB per node", ("phases", "steady", "rss_mb_per_node")),
    ("loop lag p99 ms (worst process)", ("phases", "steady", "loop_lag_p99_ms")),
    ("status on leader p50 ms", ("phases", "status", "leader", "p50_ms")),
    ("status on leader p99 ms", ("phases", "status", "leader", "p99_ms")),
    ("status on follower p50 ms", ("phases", "status", "follower", "p50_ms")),
    ("status reply bytes", ("phases", "status", "leader", "bytes")),
    ("link pass: pairs to time", ("phases", "links", "pairs_expected")),
    ("link pass: pairs timed", ("phases", "links", "pairs_measured")),
    ("link pass: s", ("phases", "links", "seconds")),
    ("link pass: echo calls", ("phases", "links", "echo_calls")),
    ("link pass: echo calls per worker", ("phases", "links", "echo_calls_per_worker", "mean")),
    ("link pass: CPU core-seconds", ("phases", "links", "cpu_core_seconds")),
    ("settle after start s", ("startup", "settle_s")),
    ("start: elections before agreeing", ("startup", "elections")),
    ("join: all see it s", ("phases", "join", "mesh_s")),
    ("join: its worker live everywhere s", ("phases", "join", "worker_live_everywhere_s")),
    ("join: every worker has its control s", ("phases", "join", "controls_everywhere_s")),
    ("leave: all drop it s", ("phases", "leave", "mesh_s")),
    ("leave: its worker dropped everywhere s", ("phases", "leave", "worker_dropped_everywhere_s")),
    ("silent leave: all drop it s", ("phases", "silent-leave", "mesh_s")),
    ("silent leave: its worker dropped everywhere s", ("phases", "silent-leave", "worker_dropped_everywhere_s")),
    ("leader kill: agreed on new s", ("phases", "kill-leader", "agreed_s")),
    ("leader kill: elections", ("phases", "kill-leader", "elections")),
    ("leader kill: step-downs", ("phases", "kill-leader", "stepdowns")),
    ("leader freeze: new leader s", ("phases", "freeze-leader", "new_leader_s")),
    ("leader freeze: one leader after resume s", ("phases", "freeze-leader", "one_leader_after_resume_s")),
    ("leader freeze: elections", ("phases", "freeze-leader", "elections")),
    ("leader freeze: step-downs", ("phases", "freeze-leader", "stepdowns")),
    ("partition: far side leader s", ("phases", "partition", "far_side_leader_s")),
    ("partition: one leader after heal s", ("phases", "partition", "one_leader_after_heal_s")),
    ("partition: elections", ("phases", "partition", "elections")),
]


def pick(result: dict[str, Any], path: tuple[str, ...]) -> float | None:
    if path == ("model",):
        n = result["nodes"]
        return (n - 1) / GOSSIP_SECONDS + n / CHECK_IN_SECONDS + n / PROBE_SECONDS
    if path[0] == "ratio":
        # Bytes per call, from the steady window's rates: what one call costs, however
        # many calls a loaded machine managed to make.
        rates = result.get("phases", {}).get("steady", {}).get("per_node", {})
        calls, size = (rates.get(f"{path[1]}:{what}", {}).get("mean") for what in ("calls", "bytes"))
        return size / calls if calls and size is not None else None
    value: Any = result
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def slope(points: list[tuple[int, float]]) -> float | None:
    """Least-squares slope of log(value) against log(N), over positive values."""
    usable = [(math.log(n), math.log(value)) for n, value in points if value and value > 0]
    if len(usable) < 2 or len({x for x, _ in usable}) < 2:
        return None
    mean_x = sum(x for x, _ in usable) / len(usable)
    mean_y = sum(y for _, y in usable) / len(usable)
    spread = sum((x - mean_x) ** 2 for x, _ in usable)
    return sum((x - mean_x) * (y - mean_y) for x, y in usable) / spread


def law(exponent: float | None) -> str:
    if exponent is None:
        return "-"
    nearest = round(exponent)
    if abs(exponent - nearest) <= 0.3:
        return {0: "O(1)", 1: "O(N)", 2: "O(N^2)", 3: "O(N^3)"}.get(nearest, f"N^{exponent:.2f}") + f" ({exponent:.2f})"
    return f"N^{exponent:.2f}"


def number(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 10000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def growth_table(results: list[dict[str, Any]]) -> str:
    results = sorted(results, key=lambda item: item["nodes"])
    header = ["metric", *[f"N={item['nodes']}" for item in results], "growth"]
    rows = []
    for label, path in METRICS:
        values = [pick(item, path) for item in results]
        if all(value is None for value in values):
            continue
        exponent = slope([(item["nodes"], value) for item, value in zip(results, values) if value is not None])
        rows.append([label, *map(number, values), law(exponent)])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main(paths: list[str]) -> None:
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    print(growth_table(results))


if __name__ == "__main__":
    main(sys.argv[1:])
