"""Scale harness: run N emulated nodes' control planes and measure what the control plane costs.

Each emulated node is the real control plane and a real worker with no model (see
host.py). This driver starts them as processes, waits for the mesh to settle, then
runs these phases and records what it saw:

  steady        (run twice by default: before and after a leave, as every node keeps
                calling a dead member) calls and bytes per node per second by kind (gossip, check-in, health
                probe), the fan-out of each, CPU and memory per node, loop lag
  status        latency of GET /admin/status on the leader and on a follower
  links         the coordinator times every link between workers, as before planning a
                route: N(N-1) probes and their echoes, how long, and whether it ran out of time
  join          a new node joins: time until every node sees it, and its worker is live
                at every control plane and has every control plane
  leave         a node is killed: time until every node has dropped it and its worker
  silent-leave  a node is unplugged (every call to it goes silent): the same times as for
                leave; a steady window after it shows what a silent dead member costs
  kill-leader   the coordinator is killed: time until the rest agree on a new one, and
                how many elections and step-downs that took
  freeze-leader the coordinator is suspended (silent, the worst kind of failure), then
                resumed: election time and count, then time back to one leader
  partition     the nodes split in two halves that cannot hear each other (calls go
                silent), then heal: elections on each side, then time back to one leader

One run:   python tests/scale/harness.py --nodes 50
A sweep:   python tests/scale/harness.py --sweep 10,50,100,200 --per-process 1 --out-dir scale-results
The sweep prints a table of every metric against N with its fitted growth law; report.py
does the same for result files written earlier.

--delay-ms, --jitter-ms and --loss emulate the network inside each node's HTTP client:
each call waits the delay both ways, and a lost call goes silent until the caller's own
timeout, as on a real network. On Linux, WSL network namespaces with netem do the same
at the packet level, but they need root on every box; this runs anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from report import growth_table  # noqa: E402

# The second steady window runs with a dead member, which every node keeps calling.
DEFAULT_PHASES = ("steady", "status", "links", "join", "leave", "steady", "kill-leader")
ALL_PHASES = (*DEFAULT_PHASES, "silent-leave", "steady", "freeze-leader", "partition")
POLL_SECONDS = 0.5
# Phases that still mean something when the mesh never agreed: the rest wait for it to.
UNSETTLED_PHASES = {"steady", "status", "links"}
# A steady window in which some host's event loop stopped this long is measured again.
STALL_MS = 2000.0


def free_ports(count: int) -> list[int]:
    """Ports nothing listens on. All are held until all are chosen, so none repeats."""
    held = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        return [sock.getsockname()[1] for sock in held]
    finally:
        for sock in held:
            sock.close()


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(int(len(ordered) * share), len(ordered) - 1)], 2)


def spread(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None}
    return {"mean": round(statistics.fmean(values), 3), "max": round(max(values), 3)}


class Pool:
    """The host processes of one run, and the driver's record of what each node saw."""

    def __init__(self, args: argparse.Namespace, nodes: int) -> None:
        self.args = args
        self.count = nodes
        self.phases = args.phases
        self.token = secrets.token_hex(16)
        self.run_dir = Path(args.run_dir or tempfile.mkdtemp(prefix="isharded-scale-")).resolve() / f"n{nodes}-{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        width = max(3, len(str(nodes)))
        self.names = [f"n{index:0{width}d}" for index in range(nodes + 1)]  # the last one joins later
        ports = free_ports(2 * len(self.names))
        self.nodes = {
            name: {"name": name, "control_port": ports[2 * index], "worker_port": ports[2 * index + 1], "join": []}
            for index, name in enumerate(self.names)
        }
        seeds = [f"http://127.0.0.1:{self.nodes[name]['control_port']}" for name in self.names[: max(args.seeds, 1)]]
        for name in self.names:
            self.nodes[name]["join"] = [seed for seed in seeds if not seed.endswith(f":{self.nodes[name]['control_port']}")]
        self.owners = {f"127.0.0.1:{node[key]}": name for name, node in self.nodes.items() for key in ("control_port", "worker_port")}
        self.leader = self.names[0]
        self.second = self.names[1]
        self.newcomer = self.names[-1]
        self.leaver = self.names[-2]
        self.processes: dict[str, dict[str, Any]] = {}  # host id -> process, observe port, nodes
        self.host_of: dict[str, str] = {}
        self.events: dict[str, list[list[Any]]] = {name: [] for name in self.names}
        self.reports: dict[str, dict[str, Any]] = {}
        self.client = httpx.AsyncClient(timeout=30.0, trust_env=False)
        self.status_client = httpx.AsyncClient(timeout=120.0, trust_env=False, headers={"Authorization": f"Bearer {self.token}"})
        self.down: set[str] = set()
        # Nodes cut off from everyone, as if unplugged. They run on, unheard.
        self.cut_off: set[str] = set()
        self.unplugged = self.names[-3]

    # Layout and processes

    def layout(self) -> list[list[str]]:
        """Which nodes share a process. Anything a phase kills or freezes runs alone."""
        alone = {self.leader}
        if "freeze-leader" in self.phases:
            alone.add(self.second)
        if "leave" in self.phases:
            alone.add(self.leaver)
        initial = self.names[:-1]
        rest = [name for name in initial if name not in alone]
        per = self.args.per_process or max(1, math.ceil(self.count / (os.cpu_count() or 1)))
        groups = [[name] for name in initial if name in alone]
        groups += [rest[index : index + per] for index in range(0, len(rest), per)]
        return groups

    def start_host(self, names: list[str]) -> None:
        host = f"h{len(self.processes)}"
        port = free_ports(1)[0]
        spec = {
            "run_dir": str(self.run_dir),
            "token": self.token,
            "parent": os.getpid(),
            "observe_port": port,
            "nodes": [self.nodes[name] for name in names],
            "owners": self.owners,
            "network": {"delay_ms": self.args.delay_ms, "jitter_ms": self.args.jitter_ms, "loss": self.args.loss},
            "low_priority": not self.args.normal_priority,
        }
        path = self.run_dir / f"{host}.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        log = open(self.run_dir / f"{host}.log", "w", encoding="utf-8")
        # One BLAS thread: otherwise each process reserves about 480 MB of memory it never
        # uses for numpy's thread buffers, and a hundred processes run out of commit.
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        environment.pop("ISHARDED_POOL_TOKEN_FILE", None)
        process = subprocess.Popen([sys.executable, str(HERE / "host.py"), str(path)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=environment)
        self.processes[host] = {"process": process, "port": port, "nodes": names, "log": log, "ps": psutil.Process(process.pid)}
        for name in names:
            self.host_of[name] = host

    def live_hosts(self) -> dict[str, dict[str, Any]]:
        return {host: entry for host, entry in self.processes.items() if entry["process"].poll() is None}

    def kill(self, name: str) -> float:
        entry = self.processes[self.host_of[name]]
        assert entry["nodes"] == [name], f"{name} shares a process and cannot be killed alone"
        killed_at = time.time()
        entry["process"].kill()
        entry["process"].wait(30)
        self.down.add(name)
        return killed_at

    def suspend(self, name: str) -> float:
        entry = self.processes[self.host_of[name]]
        assert entry["nodes"] == [name], f"{name} shares a process and cannot be frozen alone"
        at = time.time()
        entry["ps"].suspend()
        return at

    def resume(self, name: str) -> float:
        at = time.time()
        self.processes[self.host_of[name]]["ps"].resume()
        return at

    async def close(self) -> None:
        for entry in self.processes.values():
            if entry["process"].poll() is None:
                try:
                    entry["ps"].resume()
                except psutil.Error:
                    pass
                entry["process"].kill()
        for entry in self.processes.values():
            try:
                entry["process"].wait(30)
            except subprocess.TimeoutExpired:
                pass
            entry["log"].close()
        await self.client.aclose()
        await self.status_client.aclose()

    # Watching

    async def poll(self, skip: set[str] = frozenset()) -> None:
        """Bring the driver's copy of every node's events and counters up to date."""

        async def one(host: str, entry: dict[str, Any]) -> None:
            since = {name: len(self.events[name]) for name in entry["nodes"]}
            try:
                response = await self.client.get(f"http://127.0.0.1:{entry['port']}/snapshot", params={"since": json.dumps(since)})
                body = response.json()
            except (httpx.HTTPError, ValueError):
                return
            entry["lag"] = body["lag"]
            for report in body["nodes"]:
                self.events[report["name"]].extend(report.pop("events"))
                self.reports[report["name"]] = report

        hosts = {host: entry for host, entry in self.live_hosts().items() if not set(entry["nodes"]) & skip}
        await asyncio.gather(*(one(host, entry) for host, entry in hosts.items()))

    async def until(self, check: Any, timeout: float, what: str, skip: set[str] = frozenset()) -> float:
        began = time.time()
        while True:
            await self.poll(skip)
            if check():
                return time.time() - began
            if time.time() - began > timeout:
                raise TimeoutError(f"{what} did not happen in {timeout:.0f} s: {self.describe()}")
            await asyncio.sleep(POLL_SECONDS)

    def describe(self) -> str:
        rows = [f"{name}: leader={report['leader']} term={report['term']} peers={report['peers']} workers={report['workers_live']} controls={report['controls']}" for name, report in sorted(self.reports.items()) if name not in self.down]
        return "; ".join(rows[:12]) + (" ..." if len(rows) > 12 else "")

    def members(self) -> list[str]:
        return [name for name in self.reports if name not in self.down and name not in self.cut_off]

    async def set_sides(self, sides: dict[str, int]) -> None:
        """Tell every process who can hear whom. A node cut off stays cut off."""
        sides = {**sides, **{name: -1 for name in self.cut_off}}
        await asyncio.gather(*(self.client.post(f"http://127.0.0.1:{entry['port']}/network", json={"sides": sides}) for entry in self.live_hosts().values()))

    def settled(self, expected: list[str]) -> bool:
        """Every expected node is up, sees the rest, agrees on one leader, and every worker is everywhere."""
        if any(name not in self.reports or self.reports[name]["ready_at"] is None for name in expected):
            return False
        size = len(expected)
        leaders = {self.reports[name]["leader"] for name in expected}
        return (
            len(leaders) == 1
            and next(iter(leaders)) in expected
            and all(self.reports[name]["peers"] == size - 1 for name in expected)
            and all(self.reports[name]["workers_live"] == size for name in expected)
            and all(self.reports[name]["controls"] == size for name in expected)
        )

    def last_event(self, name: str, kinds: set[str], items: set[str] | None = None, after: float = 0.0) -> float | None:
        times = [event[0] for event in self.events[name] if event[1] in kinds and event[0] >= after and (items is None or event[2] in items)]
        return max(times) if times else None

    def control_url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.nodes[name]['control_port']}"

    def worker_of(self, name: str) -> str:
        return f"{name}-cpu"

    # Phases

    async def start(self) -> dict[str, Any]:
        launched = time.time()
        for names in self.layout():
            self.start_host(names)
        expected = self.names[:-1]
        await self.until(lambda: all(name in self.reports and self.reports[name]["ready_at"] for name in expected), self.args.settle_timeout, "every node starting")
        ready = max(self.reports[name]["ready_at"] for name in expected)
        try:
            await self.until(lambda: self.settled(expected), self.args.settle_timeout, "the mesh settling")
            settled = True
        except TimeoutError:
            # A mesh that never agrees is a result too: what it costs is still measured.
            settled = False
        settled_at = max(self.last_event(name, {"peers+", "workers+", "controls+", "leader"}) or 0 for name in expected)
        return {
            "processes": len(self.processes),
            "start_s": round(ready - launched, 2),
            "settled": settled,
            "settle_s": round(settled_at - ready, 2) if settled else None,
            "view": self.view(expected),
            "leader": self.reports[self.leader]["leader"],
            # A node that hears no leader within the join grace elects itself, so a slow start
            # shows here as extra elections and step-downs before the mesh agrees.
            "elections": sum(len(self.reports[name]["takeovers"]) for name in expected),
            "stepdowns": sum(len(self.reports[name]["stepdowns"]) for name in expected),
        }

    def view(self, names: list[str]) -> dict[str, Any]:
        """How far each node's view is from the whole: peers and live workers it sees, and leaders."""
        reports = [self.reports[name] for name in names if name in self.reports]
        return {
            "peers_seen": spread([float(report["peers"]) for report in reports]),
            "peers_seen_min": min((report["peers"] for report in reports), default=None),
            "workers_live_min": min((report["workers_live"] for report in reports), default=None),
            "leaders": len({report["leader"] for report in reports}),
        }

    def cpu_seconds(self) -> dict[str, float]:
        used = {}
        for host, entry in self.live_hosts().items():
            try:
                times = entry["ps"].cpu_times()
                used[host] = times.user + times.system
            except psutil.Error:
                pass
        return used

    def memory_bytes(self) -> int:
        total = 0
        for entry in self.live_hosts().values():
            try:
                total += entry["ps"].memory_info().rss
            except psutil.Error:
                pass
        return total

    async def mark(self) -> None:
        await asyncio.gather(*(self.client.post(f"http://127.0.0.1:{entry['port']}/mark") for entry in self.live_hosts().values()))

    def counts(self) -> dict[str, dict[str, list[int]]]:
        """A copy of every node's counters as last polled."""
        return {name: {key: list(value) for key, value in report["meter"]["counts"].items()} for name, report in self.reports.items()}

    def rates(self, before: dict[str, dict[str, list[int]]], seconds: float, members: list[str]) -> dict[str, list[float]]:
        """Each node's calls, bytes and failures per second by kind since ``before``, and totals."""
        rates: dict[str, list[float]] = {}
        for name in members:
            after = self.reports[name]["meter"]["counts"]
            totals = {"out:calls": 0.0, "in:calls": 0.0, "out:bytes": 0.0, "in:bytes": 0.0, "out:failures": 0.0}
            for key, (calls, size, failures) in after.items():
                start = before[name].get(key, [0, 0, 0])
                direction, kind = key.split(":", 1)
                call_rate, byte_rate = (calls - start[0]) / seconds, (size - start[1]) / seconds
                rates.setdefault(f"{direction}:{kind}:calls", []).append(call_rate)
                rates.setdefault(f"{direction}:{kind}:bytes", []).append(byte_rate)
                totals[f"{direction}:calls"] += call_rate
                totals[f"{direction}:bytes"] += byte_rate
                if direction == "out":
                    totals["out:failures"] += (failures - start[2]) / seconds
            for key, value in totals.items():
                rates.setdefault(f"total:{key}", []).append(value)
        return rates

    async def steady(self) -> dict[str, Any]:
        """Measure a steady window, and again if a process stalled through it.

        A host whose event loop stops for seconds (paging, or a machine that suspends
        background processes) sends a third of its calls and says nothing about the code.
        Such a window is measured again, up to ``--stall-retries`` times, and the count of
        windows thrown away is kept with the result.
        """
        thrown: list[float] = []
        while True:
            window = await self.steady_window()
            worst = window["loop_lag_max_ms"] or 0.0
            if worst < STALL_MS or len(thrown) >= self.args.stall_retries:
                break
            print(f"[N={self.count}] a process stalled {worst:.0f} ms in that window; measuring again", flush=True)
            thrown.append(worst)
        window["stalled"] = (window["loop_lag_max_ms"] or 0.0) >= STALL_MS
        window["windows_thrown_away"] = len(thrown)
        return window

    async def steady_window(self) -> dict[str, Any]:
        await asyncio.sleep(self.args.settle_seconds)
        await self.mark()
        await self.poll()
        before = self.counts()
        psutil.cpu_percent(None)
        cpu_before, began = self.cpu_seconds(), time.time()
        await asyncio.sleep(self.args.steady_seconds)
        await self.poll()
        cpu_after, seconds = self.cpu_seconds(), time.time() - began
        machine = psutil.cpu_percent(None)
        members = [name for name in self.members() if name in before]
        rates = self.rates(before, seconds, members)
        fanout: dict[str, list[float]] = {}
        for name in members:
            for kind, count in self.reports[name]["meter"]["fanout"].items():
                fanout.setdefault(kind, []).append(count)
        cpu = {host: (cpu_after[host] - cpu_before[host]) / seconds for host in cpu_after if host in cpu_before}
        nodes = len(members)
        peer_flaps, worker_flaps = self.false_drops(members, began, began + seconds)
        lags = [entry.get("lag", {}) for entry in self.live_hosts().values()]
        return {
            "seconds": round(seconds, 1),
            "nodes": nodes,
            "per_node": {key: spread(values) for key, values in sorted(rates.items())},
            "fanout": {kind: spread(values) for kind, values in sorted(fanout.items())},
            "live_peers_dropped": peer_flaps,
            "live_workers_dropped": worker_flaps,
            "cpu_cores_per_node": round(sum(cpu.values()) / nodes, 4),
            "cpu_busiest_process_cores": round(max(cpu.values()), 3) if cpu else None,
            "rss_mb_per_node": round(self.memory_bytes() / nodes / 2**20, 1),
            # Every node shares this machine: near 100%, the nodes slow each other down.
            "machine_cpu_percent": machine,
            "view": self.view(members),
            "loop_lag_p99_ms": max((lag.get("p99_ms", 0.0) for lag in lags), default=None),
            "loop_lag_max_ms": max((lag.get("max_ms", 0.0) for lag in lags), default=None),
        }

    def false_drops(self, members: list[str], began: float, ended: float) -> tuple[int, int]:
        """How many times, between two moments, a node dropped a live member or its worker.

        Every such drop is a false alarm. It happens when a gossip round or a health pass
        takes longer than the silence after which a member counts as gone.
        """
        live, live_workers = set(members), {self.worker_of(name) for name in members}

        def count(kind: str, alive: set[str]) -> int:
            return sum(1 for name in members for event in self.events[name] if event[1] == kind and event[2] in alive and began <= event[0] <= ended)

        return count("peers-", live), count("workers-", live_workers)

    async def status(self) -> dict[str, Any]:
        """The dashboard's call. It probes every worker before it answers, so its time grows with N."""
        follower = self.names[len(self.names) // 2]
        results: dict[str, dict[str, Any]] = {}
        for name in (self.reports[self.leader]["leader"] or self.leader, follower):
            times, sizes = [], []
            ends = time.time() + self.args.status_seconds / 2
            while time.time() < ends:
                began = time.perf_counter()
                try:
                    response = await self.status_client.get(self.control_url(name) + "/admin/status")
                    response.raise_for_status()
                    times.append((time.perf_counter() - began) * 1000)
                    sizes.append(len(response.content))
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(max(1.0 - (time.perf_counter() - began), 0.0))
            results["leader" if name != follower else "follower"] = {
                "samples": len(times),
                "p50_ms": percentile(times, 0.5),
                "p99_ms": percentile(times, 0.99),
                "max_ms": round(max(times), 1) if times else None,
                "bytes": int(statistics.fmean(sizes)) if sizes else None,
            }
        return results

    async def links(self) -> dict[str, Any]:
        """The coordinator times every link between live workers, as it does before planning a route.

        It asks each worker to time its link to every other one (``preview_links``, which
        leaves the running model's costs alone), so a full pass is N(N-1) probes, each a few
        echoes, inside a time limit; a pass that runs out of time plans with what it measured.
        """
        coordinator = self.reports[self.members()[0]]["leader"] or self.leader
        await self.mark()
        await self.poll()
        before, cpu_before, began = self.counts(), self.cpu_seconds(), time.time()
        entry = self.processes[self.host_of[coordinator]]
        response = await self.client.post(f"http://127.0.0.1:{entry['port']}/links", json={"node": coordinator}, timeout=self.args.settle_timeout)
        response.raise_for_status()
        body = response.json()
        seconds, workers = time.time() - began, int(body["workers"])
        await self.poll()
        cpu_after = self.cpu_seconds()
        members = [name for name in self.members() if name in before]
        sent = {name: {key: self.reports[name]["meter"]["counts"].get(key, [0, 0, 0])[0] - before[name].get(key, [0, 0, 0])[0] for key in ("out:link_probe", "out:link_echo")} for name in members}
        echoes = [sent[name]["out:link_echo"] for name in members]
        cpu = sum(cpu_after[host] - cpu_before[host] for host in cpu_after if host in cpu_before)
        return {
            "workers": workers,
            "pairs_expected": workers * (workers - 1),
            "pairs_measured": body["pairs"],
            "pairs_unreachable": body["unreachable"],
            "cut_short": body["pairs"] < workers * (workers - 1),
            "seconds": round(body["seconds"], 2),
            "probe_calls": sent[coordinator]["out:link_probe"],
            "echo_calls": sum(echoes),
            "echo_calls_per_worker": spread([float(value) for value in echoes]),
            "fanout": {kind: spread([float(value) for name in members if (value := self.reports[name]["meter"]["fanout"].get(kind)) is not None]) for kind in ("link_probe", "link_echo")},
            "cpu_core_seconds": round(cpu, 2),
            "cpu_cores_while_measuring": round(cpu / seconds, 3) if seconds else None,
        }

    async def join(self) -> dict[str, Any]:
        new = self.newcomer
        launched = time.time()
        self.start_host([new])
        expected = [name for name in self.names if name not in self.down]
        await self.until(lambda: self.settled(expected), self.args.settle_timeout, f"{new} joining")
        ready = self.reports[new]["ready_at"]
        others = [name for name in expected if name != new]
        seen = [self.last_event(name, {"peers+"}, {new}, ready) for name in others]
        sees = self.last_event(new, {"peers+"}, set(others), ready)
        worker = [self.last_event(name, {"workers+"}, {self.worker_of(new)}, ready) for name in expected]
        controls = [self.last_event(name, {"controls+"}, {self.control_url(new)}, ready) for name in others]
        controls.append(self.last_event(new, {"controls+"}, None, ready))
        def since_ready(times: list[float | None]) -> float | None:
            known = [at for at in times if at is not None]
            return round(max(known) - ready, 2) if known else None

        return {
            "start_s": round(ready - launched, 2),
            "mesh_s": since_ready([*seen, sees]),
            "worker_live_everywhere_s": since_ready(worker),
            "controls_everywhere_s": since_ready(controls),
            "elections": sum(1 for name in expected for at in self.reports[name]["takeovers"] if at >= launched),
        }

    async def leave(self) -> dict[str, Any]:
        """A node's process is killed. Its port then refuses, which the system says at once."""
        gone = self.leaver
        killed = self.kill(gone)
        return await self.dropped(gone, killed)

    async def silent_leave(self) -> dict[str, Any]:
        """A node is unplugged: every call to it goes silent, as to a machine switched off."""
        gone = self.unplugged
        unplugged = time.time()
        self.cut_off.add(gone)
        await self.set_sides({})
        return await self.dropped(gone, unplugged)

    async def dropped(self, gone: str, killed: float) -> dict[str, Any]:
        """How long the rest took to drop a node that left, and its worker."""
        survivors = [name for name in self.members() if name != gone]
        worker = self.worker_of(gone)

        def dropped() -> bool:
            return all(self.reports[name]["peers"] == len(survivors) - 1 and self.reports[name]["workers_live"] == len(survivors) for name in survivors)

        await self.until(dropped, self.args.settle_timeout, f"everyone dropping {gone}", {gone})
        # A survivor that had already dropped it, falsely, before it left has no drop to time.
        mesh = [at for name in survivors if (at := self.last_event(name, {"peers-"}, {gone}, killed)) is not None]
        workers = [at for name in survivors if (at := self.last_event(name, {"workers-"}, {worker}, killed)) is not None]
        return {
            "mesh_s": round(max(mesh) - killed, 2) if mesh else None,
            "worker_dropped_everywhere_s": round(max(workers) - killed, 2) if workers else None,
            "had_dropped_it_already": len(survivors) - len(mesh),
            "elections": sum(1 for name in survivors for at in self.reports[name]["takeovers"] if at >= killed),
        }

    async def kill_leader(self) -> dict[str, Any]:
        old = self.reports[self.leader]["leader"]
        old_term = self.reports[old]["term"]
        killed = self.kill(old)
        survivors = [name for name in self.members() if name != old]

        def agreed() -> bool:
            leaders = {(self.reports[name]["leader"], self.reports[name]["term"]) for name in survivors}
            return len(leaders) == 1 and next(iter(leaders))[0] not in (None, old) and next(iter(leaders))[1] > old_term

        await self.until(agreed, self.args.settle_timeout, f"a new leader after {old}", {old})
        changed = [self.last_event(name, {"leader"}, None, killed) for name in survivors]
        return {
            "old": old,
            "new": self.reports[survivors[0]]["leader"],
            "agreed_s": round(max(time for time in changed if time) - killed, 2),
            "elections": sum(1 for name in survivors for at in self.reports[name]["takeovers"] if at >= killed),
            "stepdowns": sum(1 for name in survivors for at in self.reports[name]["stepdowns"] if at >= killed),
        }

    async def freeze_leader(self) -> dict[str, Any]:
        """Suspend the leader: it says nothing and answers nothing, then comes back unaware."""
        frozen_name = self.reports[self.members()[0]]["leader"]
        old_term = self.reports[frozen_name]["term"]
        frozen = self.suspend(frozen_name)
        others = [name for name in self.members() if name != frozen_name]

        def replaced() -> bool:
            leaders = {(self.reports[name]["leader"], self.reports[name]["term"]) for name in others}
            return len(leaders) == 1 and next(iter(leaders))[0] not in (None, frozen_name) and next(iter(leaders))[1] > old_term

        try:
            await self.until(replaced, self.args.settle_timeout, f"a leader in place of frozen {frozen_name}", {frozen_name})
            elected = max(self.last_event(name, {"leader"}, None, frozen) or 0 for name in others)
            hold = max(self.args.freeze_seconds - (time.time() - frozen), 0.0)
            await asyncio.sleep(hold)
        finally:
            resumed = self.resume(frozen_name)
        everyone = self.members()
        await self.until(lambda: self.settled(everyone), self.args.settle_timeout, f"one leader after {frozen_name} resumed")
        back = max(self.last_event(name, {"leader", "peers+", "workers+", "controls+"}, None, resumed) or resumed for name in everyone)
        return {
            "frozen": frozen_name,
            "frozen_s": round(resumed - frozen, 1),
            "new_leader_s": round(elected - frozen, 2),
            "one_leader_after_resume_s": round(back - resumed, 2),
            "leader_after": self.reports[everyone[0]]["leader"],
            "elections": sum(1 for name in everyone for at in self.reports[name]["takeovers"] if at >= frozen),
            "stepdowns": sum(1 for name in everyone for at in self.reports[name]["stepdowns"] if at >= frozen),
        }

    async def partition(self) -> dict[str, Any]:
        """Split the nodes into two halves whose calls to each other go silent, then heal."""
        everyone = sorted(self.members())
        half = everyone[len(everyone) // 2 :]
        sides = {name: (1 if name in half else 0) for name in everyone}
        cut = time.time()
        await self.set_sides(sides)

        def split() -> bool:
            leaders = {self.reports[name]["leader"] for name in half}
            return len(leaders) == 1 and next(iter(leaders)) in half and all(self.reports[name]["peers"] == len(half) - 1 for name in half)

        await self.until(split, self.args.settle_timeout, "the far side electing its own leader")
        far_leader = max(self.last_event(name, {"leader"}, None, cut) or cut for name in half)
        await asyncio.sleep(max(self.args.partition_seconds - (time.time() - cut), 0.0))
        healed = time.time()
        await self.set_sides({})
        await self.until(lambda: self.settled(everyone), self.args.settle_timeout, "one leader after healing")
        back = max(self.last_event(name, {"leader", "peers+", "workers+", "controls+"}, None, healed) or healed for name in everyone)
        return {
            "far_side_leader_s": round(far_leader - cut, 2),
            "one_leader_after_heal_s": round(back - healed, 2),
            "elections": sum(1 for name in everyone for at in self.reports[name]["takeovers"] if at >= cut),
            "stepdowns": sum(1 for name in everyone for at in self.reports[name]["stepdowns"] if at >= cut),
        }


PHASES = {
    "steady": Pool.steady,
    "status": Pool.status,
    "links": Pool.links,
    "join": Pool.join,
    "leave": Pool.leave,
    "silent-leave": Pool.silent_leave,
    "kill-leader": Pool.kill_leader,
    "freeze-leader": Pool.freeze_leader,
    "partition": Pool.partition,
}


async def run(args: argparse.Namespace, nodes: int) -> dict[str, Any]:
    pool = Pool(args, nodes)
    result: dict[str, Any] = {
        "nodes": nodes,
        "date": time.strftime("%Y-%m-%d"),
        "platform": sys.platform,
        "cpus": os.cpu_count(),
        "per_process": args.per_process or max(1, math.ceil(nodes / (os.cpu_count() or 1))),
        "network": {"delay_ms": args.delay_ms, "jitter_ms": args.jitter_ms, "loss": args.loss},
        "phases": {},
    }
    try:
        result["startup"] = await pool.start()
        settled = result["startup"]["settled"]
        print(f"[N={nodes}] startup: {json.dumps(result['startup'])}", flush=True)
        for phase in args.phases:
            # A phase run again is recorded as steady-2, steady-3 and so on.
            runs = sum(name == phase or (name.startswith(phase + "-") and name[len(phase) + 1 :].isdigit()) for name in result["phases"])
            key = f"{phase}-{runs + 1}" if runs else phase
            if not settled and phase not in UNSETTLED_PHASES:
                result["phases"][key] = {"skipped": "the mesh never settled"}
                continue
            print(f"[N={nodes}] {key} ...", flush=True)
            result["phases"][key] = await PHASES[phase](pool)
            print(f"[N={nodes}] {key}: {json.dumps(result['phases'][key])[:400]}", flush=True)
    except Exception as exc:  # a failed phase still leaves the results so far, and no processes
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[N={nodes}] {result['error']}", flush=True)
    finally:
        await pool.close()
    result["run_dir"] = str(pool.run_dir)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the control plane of N emulated nodes.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--nodes", type=int, help="how many nodes to start")
    size.add_argument("--sweep", help="several sizes, comma separated, such as 10,50,100,200")
    parser.add_argument("--phases", default=",".join(DEFAULT_PHASES), help=f"comma separated, from {','.join(PHASES)}; all is {','.join(ALL_PHASES)}")
    parser.add_argument("--per-process", type=int, default=0, help="nodes sharing one process; default spreads N over the cores, 1 is one process a node")
    parser.add_argument("--seeds", type=int, default=1, help="how many of the first nodes everyone joins through")
    parser.add_argument("--steady-seconds", type=float, default=60.0, help="length of the steady window")
    parser.add_argument("--settle-seconds", type=float, default=10.0, help="wait after settling before the steady window")
    parser.add_argument("--status-seconds", type=float, default=20.0, help="time spent asking for status, half on the leader")
    parser.add_argument("--freeze-seconds", type=float, default=30.0, help="how long the leader stays suspended")
    parser.add_argument("--partition-seconds", type=float, default=30.0, help="how long the halves stay apart")
    parser.add_argument("--stall-retries", type=int, default=1, help="steady windows measured again when a process stalls through one")
    parser.add_argument("--settle-timeout", type=float, default=300.0, help="longest wait for any phase to settle")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="emulated one-way delay on every call")
    parser.add_argument("--jitter-ms", type=float, default=0.0, help="emulated jitter on that delay")
    parser.add_argument("--loss", type=float, default=0.0, help="share of calls that go silent, 0 to 1")
    parser.add_argument("--run-dir", default=None, help="where nodes keep their files and logs")
    parser.add_argument("--out", default=None, help="write one run's results here as JSON")
    parser.add_argument("--out-dir", default=None, help="write each size's results here, as scale-N.json")
    parser.add_argument("--normal-priority", action="store_true", help="do not lower the hosts' priority")
    args = parser.parse_args(argv)
    args.phases = list(ALL_PHASES) if args.phases == "all" else [item.strip() for item in args.phases.split(",") if item.strip()]
    unknown = set(args.phases) - set(PHASES)
    if unknown:
        parser.error(f"unknown phases: {', '.join(sorted(unknown))}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sizes = [args.nodes] if args.nodes else [int(item) for item in args.sweep.split(",")]
    if min(sizes) < 3:
        raise SystemExit("a run needs at least 3 nodes: one to lead, one to leave, one to stay")
    results = []
    for nodes in sizes:
        result = asyncio.run(run(args, nodes))
        results.append(result)
        text = json.dumps(result, indent=2)
        if args.out and len(sizes) == 1:
            Path(args.out).write_text(text, encoding="utf-8")
        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / f"scale-{nodes}.json").write_text(text, encoding="utf-8")
        if not args.out and not args.out_dir:
            print(text)
    if len(results) > 1:
        print(growth_table(results))
    return 1 if any("error" in result or not result.get("startup", {}).get("settled") for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
