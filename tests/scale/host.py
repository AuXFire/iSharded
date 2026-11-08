"""One process of the scale harness: it runs one or more emulated nodes.

An emulated node is the real control plane (``isharded.orchestrator`` with its
``ControlMesh``) and one real worker (``isharded.worker``: its check-in loop and
its ``/health``), with no model and no compute. Only the worker's hardware probe
is replaced, by a fixed CPU-and-RAM description, so nothing touches a GPU or the
user's caches. Every line of gossip, check-in, health probe and election is the
code a real node runs.

Several nodes can share a process. Each extra node gets its own copy of the
orchestrator and worker modules, loaded under a private name, so each has its own
``runtime`` and ``app`` exactly as a process of its own would. Shared libraries
are loaded once, so a node costs about 5 MB this way instead of about 85 MB.

Nothing here changes the control plane. What the harness adds sits around it:

- a counting transport under each node's ``httpx`` client, which counts calls and
  bytes by kind and by destination, and can delay, drop or cut calls to emulate a
  slow, lossy or partitioned network. A cut or lost call goes silent for as long as
  a real one would (see ``silent_wait``) rather than failing fast;
- a counting ASGI wrapper around each app, for the calls a node receives;
- a counter on each node's ``_take_over`` and ``_step_down``, to count elections;
- a sampler that writes down, ten times a second, every change in what each node
  sees: peers alive, the leader and term, live workers, and the worker's control
  links. The driver works out convergence times from these.

The driver (``harness.py``) talks to this process over a small observer server on
its own port, which is not counted.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_SECONDS = 0.1


def kind_of(path: str) -> str:
    """Which control-plane traffic a call is."""
    if path == "/internal/gossip":
        return "gossip"
    if path in ("/internal/register", "/internal/heartbeat"):
        return "checkin"
    if path == "/health":
        return "health"
    if path == "/admin/status":
        return "status"
    if path == "/internal/diag/probe":
        return "link_probe"
    return "other"


def header_bytes(pairs: Any) -> int:
    return sum(len(name) + len(value) + 4 for name, value in pairs)


class Meter:
    """Calls and bytes one node sent and received, by kind, since it started."""

    def __init__(self) -> None:
        self.counts: dict[str, list[int]] = {}  # "out:gossip" -> [calls, bytes, failures]
        self.fanout: dict[str, set[str]] = {}  # "gossip" -> destinations since the last mark

    def add(self, direction: str, kind: str, size: int, failed: bool = False) -> None:
        row = self.counts.setdefault(f"{direction}:{kind}", [0, 0, 0])
        row[0] += 1
        row[1] += size
        row[2] += failed

    def reach(self, kind: str, destination: str) -> None:
        self.fanout.setdefault(kind, set()).add(destination)

    def snapshot(self) -> dict[str, Any]:
        return {"counts": {key: list(value) for key, value in self.counts.items()}, "fanout": {key: len(value) for key, value in self.fanout.items()}}


class Faults:
    """The emulated network, for every node in this process. The driver sets it."""

    def __init__(self) -> None:
        self.delay_ms = 0.0
        self.jitter_ms = 0.0
        self.loss = 0.0
        # Node name to side. Two nodes talk only if they are on the same side.
        self.sides: dict[str, int] = {}

    def cut(self, source: str, target: str | None) -> bool:
        if target is None or not self.sides:
            return False
        return self.sides.get(source, 0) != self.sides.get(target, 0)

    def one_way(self) -> float:
        if not self.delay_ms and not self.jitter_ms:
            return 0.0
        return max(self.delay_ms + random.uniform(-self.jitter_ms, self.jitter_ms), 0.0) / 1000


FAULTS = Faults()


def counting_stream(inner: Any, done: Any) -> Any:
    """A response body that reports its size once the caller has read and closed it."""
    import httpx

    class CountingStream(httpx.AsyncByteStream):
        size = 0

        async def __aiter__(self):
            async for chunk in inner:
                self.size += len(chunk)
                yield chunk

        async def aclose(self) -> None:
            try:
                await inner.aclose()
            finally:
                done(self.size)

    return CountingStream()


def silent_wait(request: Any, retries: int, kept: bool) -> tuple[float, str]:
    """How long a call into a silent network takes to fail, and how it fails.

    Over a connection already open, the request goes out and no answer comes: the read
    times out, once. With no connection, each connect attempt times out, and the
    transport tries again ``retries`` times with httpcore's backoff of 0, 0.5, 1, 2 s.
    """
    timeout = request.extensions.get("timeout") or {}
    if kept:
        return float(timeout.get("read") or 5.0), "ReadTimeout"
    connect = float(timeout.get("connect") or 5.0)
    backoff = sum(0.5 * 2 ** (attempt - 1) for attempt in range(1, retries))
    return connect * (1 + retries) + backoff, "ConnectTimeout"


def counting_transport(inner: Any, node: str, meter: Meter, owners: dict[str, str]) -> Any:
    """Wrap a node's transport: count each call, and apply the emulated network."""
    import httpx

    retries = int(getattr(getattr(inner, "_pool", None), "_retries", 0) or 0)
    kept: set[str] = set()  # destinations this client holds an open connection to

    class CountingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            kind = kind_of(request.url.path)
            destination = f"{request.url.host}:{request.url.port}"
            meter.reach(kind, destination)
            try:
                body = len(request.content)
            except httpx.RequestNotRead:
                body = int(request.headers.get("content-length") or 0)
            sent = len(request.method) + len(request.url.raw_path) + 11 + header_bytes(request.headers.raw) + body
            if FAULTS.cut(node, owners.get(destination)) or (FAULTS.loss and random.random() < FAULTS.loss):
                seconds, error = silent_wait(request, retries, destination in kept)
                kept.discard(destination)  # a timed-out connection is closed
                await asyncio.sleep(seconds)
                meter.add("out", kind, sent, failed=True)
                raise getattr(httpx, error)("emulated silent network", request=request)
            delay = FAULTS.one_way()
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await inner.handle_async_request(request)
            except BaseException:
                kept.discard(destination)
                meter.add("out", kind, sent, failed=True)
                raise
            kept.add(destination)
            delay = FAULTS.one_way()
            if delay:
                await asyncio.sleep(delay)
            head = 15 + header_bytes(response.headers.raw)
            failed = response.status_code >= 400
            response.stream = counting_stream(response.stream, lambda size: meter.add("out", kind, sent + head + size, failed))
            return response

        async def aclose(self) -> None:
            await inner.aclose()

    return CountingTransport()


class CountingApp:
    """Count what a node's server receives and answers."""

    def __init__(self, app: Any, meter: Meter) -> None:
        self.app = app
        self.meter = meter

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        size = len(scope.get("method", "")) + len(scope.get("raw_path") or b"") + 11 + header_bytes(scope.get("headers", []))
        status = [0]

        async def counted_receive() -> Any:
            nonlocal size
            message = await receive()
            if message.get("type") == "http.request":
                size += len(message.get("body", b""))
            return message

        async def counted_send(message: dict[str, Any]) -> None:
            nonlocal size
            if message["type"] == "http.response.start":
                status[0] = message["status"]
                size += 15 + header_bytes(message.get("headers", []))
            elif message["type"] == "http.response.body":
                size += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, counted_receive, counted_send)
        finally:
            self.meter.add("in", kind_of(scope["path"]), size, failed=status[0] >= 400 or status[0] == 0)


def load_module(module: str, tag: str | None) -> types.ModuleType:
    """The real module for a process's first node, and a private copy for each other one."""
    import isharded

    if tag is None:
        return importlib.import_module(f"isharded.{module}")
    name = f"isharded._scale_{module}_{tag}"
    spec = importlib.util.spec_from_file_location(name, Path(isharded.__file__).parent / f"{module}.py")
    assert spec is not None and spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    sys.modules[name] = copy
    spec.loader.exec_module(copy)
    return copy


def node_environment(node: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """What ``isharded node`` would set for this node's control plane and worker."""
    home = run_dir / node["name"]
    home.mkdir(parents=True, exist_ok=True)
    control = f"http://127.0.0.1:{node['control_port']}"
    return {
        "ISHARDED_UP_HOME": str(home),
        "ISHARDED_KEYS_FILE": str(home / "access.json"),
        "ISHARDED_NODE_ID": node["name"],
        "ISHARDED_NODE_URL": control,
        "ISHARDED_JOIN": ",".join(node["join"]),
        "ISHARDED_LANE": "",
        "ISHARDED_STATE_FILE": str(home / "orchestrator-state.json"),
        "ISHARDED_AUDIT_FILE": str(home / "audit.jsonl"),
        "ISHARDED_WORKER_ID": f"{node['name']}-cpu",
        "ISHARDED_HOST_ID": node["name"],
        "ISHARDED_ORCHESTRATOR_URL": ",".join(dict.fromkeys([control, *node["join"]])),
        "ISHARDED_ADVERTISE_URL": f"http://127.0.0.1:{node['worker_port']}",
        "ISHARDED_LOCATION": "",
        "ISHARDED_ROLES": "inference",
        "ISHARDED_CPU_THREADS": "1",
        "ISHARDED_CACHE_DIR": str(home / "cache"),
        "ISHARDED_STATE_DISK_GB": "0",
        "ISHARDED_STATE_RAM_GB": "0",
        "ISHARDED_STATE_COPY_MB": "0",
        "ISHARDED_REACH_URLS": "",
        "ISHARDED_GGUF_DIRS": "",
        "ISHARDED_PEER_FETCH": "0",
    }


STUB_RAM_BYTES = 16 * 1024**3
STUB_BANDWIDTH_GBPS = 20.0


def stub_setup(worker_module: types.ModuleType, runtime: Any) -> None:
    """HARNESS STUB: the worker's hardware probe, replaced by a fixed CPU and 16 GiB of RAM.

    The real ``setup`` looks for accelerators, sizes pools from free memory, and times
    memory bandwidth with PyTorch, which costs 850 MB a process. A hundred emulated
    nodes on one machine must do none of that, and must all look alike.
    """
    from isharded.protocol import MemoryPool
    from isharded.statestore import StateStore

    if runtime.hardware is not None:
        return
    runtime.held = StateStore(None, 0, 0)
    pool = MemoryPool(
        pool_id="ram",
        kind="ram",
        device="cpu",
        label="System memory",
        capacity_group=f"{runtime.host_id}:ram",
        total_bytes=STUB_RAM_BYTES,
        budget_bytes=STUB_RAM_BYTES,
        free_bytes=STUB_RAM_BYTES,
        bandwidth_gbps=STUB_BANDWIDTH_GBPS,
        dtype=runtime.dtype_name,
    )
    runtime.pools = {pool.pool_id: pool}
    runtime.hardware = worker_module.hardware_info([], [pool], runtime.host_id, runtime.roles, 1, 0, 0, False)


class Node:
    """One emulated node: a control plane and a worker, and what the harness keeps on them."""

    def __init__(self, spec: dict[str, Any], run_dir: Path, owners: dict[str, str], first: bool) -> None:
        self.name = spec["name"]
        self.spec = spec
        saved = dict(os.environ)
        os.environ.update(node_environment(spec, run_dir))
        try:
            tag = None if first else self.name
            # A real module imported before this node's settings existed must be set up again.
            stale = {name for name in ("orchestrator", "worker") if first and f"isharded.{name}" in sys.modules}
            self.orchestrator_module = load_module("orchestrator", tag)
            self.worker_module = load_module("worker", tag)
            if "orchestrator" in stale:
                self.orchestrator_module.runtime = self.orchestrator_module.Orchestrator()
            if "worker" in stale:
                self.worker_module.runtime = self.worker_module.WorkerRuntime()
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.orchestrator = self.orchestrator_module.runtime
        self.worker = self.worker_module.runtime
        self.worker.setup = types.MethodType(lambda runtime: stub_setup(self.worker_module, runtime), self.worker)
        self.meter = Meter()
        for client in (self.orchestrator.client, self.worker.client):
            client._transport = counting_transport(client._transport, self.name, self.meter, owners)
        self.takeovers: list[float] = []
        self.stepdowns: list[float] = []
        take_over, step_down = self.orchestrator._take_over, self.orchestrator._step_down

        def counted_take_over() -> None:
            self.takeovers.append(time.time())
            take_over()

        def counted_step_down() -> None:
            self.stepdowns.append(time.time())
            step_down()

        self.orchestrator._take_over = counted_take_over
        self.orchestrator._step_down = counted_step_down
        self.events: list[list[Any]] = []
        self.seen: dict[str, Any] = {"peers": set(), "leader": None, "workers": set(), "controls": set()}
        self.ready_at: float | None = None

    def apps(self) -> list[tuple[Any, int]]:
        return [
            (CountingApp(self.orchestrator_module.app, self.meter), self.spec["control_port"]),
            (CountingApp(self.worker_module.app, self.meter), self.spec["worker_port"]),
        ]

    def sample(self) -> None:
        """Write down what changed in this node's view since the last look."""
        now = round(time.time(), 3)
        mesh = self.orchestrator.mesh
        peers = set(mesh.alive()) - {self.name}
        leader = (mesh.leader_id, mesh.term)
        workers = {name for name, record in self.orchestrator.workers.items() if record.live}
        controls = {url for url, link in self.worker.control_urls.items() if link.registered}
        for key, current in (("peers", peers), ("workers", workers), ("controls", controls)):
            before = self.seen[key]
            for item in sorted(current - before):
                self.events.append([now, key + "+", item])
            for item in sorted(before - current):
                self.events.append([now, key + "-", item])
            self.seen[key] = current
        if leader != self.seen["leader"]:
            self.events.append([now, "leader", leader[0], leader[1]])
            self.seen["leader"] = leader

    def report(self, since: int) -> dict[str, Any]:
        mesh = self.orchestrator.mesh
        return {
            "name": self.name,
            "ready_at": self.ready_at,
            "leader": mesh.leader_id,
            "term": mesh.term,
            "peers": len(self.seen["peers"]),
            "workers_live": len(self.seen["workers"]),
            "workers_known": len(self.orchestrator.workers),
            "controls": len(self.seen["controls"]),
            "takeovers": self.takeovers,
            "stepdowns": self.stepdowns,
            "meter": self.meter.snapshot(),
            "events": self.events[since:],
            "event_count": len(self.events),
        }


class Lag:
    """How late this process's event loop runs, which says when it is saturated."""

    def __init__(self) -> None:
        self.samples: deque[float] = deque(maxlen=20000)

    async def run(self) -> None:
        interval = 0.05
        while True:
            started = time.perf_counter()
            await asyncio.sleep(interval)
            self.samples.append(max(time.perf_counter() - started - interval, 0.0) * 1000)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"p50_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        ordered = sorted(self.samples)
        return {
            "p50_ms": round(ordered[len(ordered) // 2], 2),
            "p99_ms": round(ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)], 2),
            "max_ms": round(ordered[-1], 2),
        }


def observer_app(nodes: list[Node], lag: Lag) -> Any:
    """The driver's view into this process. Its calls are not counted."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def snapshot(request: Request) -> JSONResponse:
        since = json.loads(request.query_params.get("since") or "{}")
        return JSONResponse({
            "pid": os.getpid(),
            "time": time.time(),
            "lag": lag.summary(),
            "nodes": [node.report(int(since.get(node.name, 0))) for node in nodes],
        })

    async def mark(_: Request) -> JSONResponse:
        """Start a new window: forget destinations seen and loop lag so far."""
        for node in nodes:
            node.meter.fanout.clear()
        lag.samples.clear()
        return JSONResponse({"ok": True})

    async def network(request: Request) -> JSONResponse:
        body = await request.json()
        FAULTS.delay_ms = float(body.get("delay_ms", FAULTS.delay_ms))
        FAULTS.jitter_ms = float(body.get("jitter_ms", FAULTS.jitter_ms))
        FAULTS.loss = float(body.get("loss", FAULTS.loss))
        if "sides" in body:
            FAULTS.sides = {str(name): int(side) for name, side in (body["sides"] or {}).items()}
        return JSONResponse({"ok": True})

    return Starlette(routes=[
        Route("/snapshot", snapshot),
        Route("/mark", mark, methods=["POST"]),
        Route("/network", network, methods=["POST"]),
    ])


def lower_priority() -> None:
    import psutil

    try:
        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
    except (OSError, psutil.Error):
        pass


def raise_file_limit() -> None:
    """Every node keeps a connection to every other, so a big run needs many sockets."""
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft < wanted:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (ValueError, OSError):
            pass


def watch_parent(parent: int) -> None:
    """Leave when the driver is gone, so a crashed run leaves nothing behind."""
    import psutil

    def watch() -> None:
        while psutil.pid_exists(parent):
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


async def serve(nodes: list[Node], observe_port: int, lag: Lag) -> None:
    import uvicorn

    from isharded import serving

    servers: list[Any] = []

    def server(app: Any, port: int, quiet: bool = False) -> Any:
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            http=serving.NoDelayHTTPProtocol,
            access_log=False,
            timeout_keep_alive=serving.KEEP_ALIVE_SECONDS,
            limit_concurrency=serving.MAX_CONNECTIONS,
            log_level="error" if quiet else "warning",
            lifespan="on",
        )
        instance = uvicorn.Server(config)
        instance.install_signal_handlers = lambda: None  # several servers share this loop
        servers.append(instance)
        return instance

    asyncio.create_task(lag.run())
    observer = server(observer_app(nodes, lag), observe_port, quiet=True)
    running = [asyncio.create_task(observer.serve())]
    for node in nodes:
        pair = [server(app, port) for app, port in node.apps()]
        running += [asyncio.create_task(instance.serve()) for instance in pair]

        async def mark_ready(node: Node = node, pair: list[Any] = pair) -> None:
            while not all(instance.started for instance in pair):
                await asyncio.sleep(0.02)
            node.ready_at = time.time()

        asyncio.create_task(mark_ready())

    async def sampler() -> None:
        while True:
            for node in nodes:
                if node.ready_at is not None:
                    node.sample()
            await asyncio.sleep(SAMPLE_SECONDS)

    asyncio.create_task(sampler())
    await asyncio.gather(*running)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", help="JSON file the driver wrote for this process")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if spec.get("low_priority", True):
        lower_priority()
    raise_file_limit()
    if spec.get("parent"):
        watch_parent(int(spec["parent"]))
    os.environ["ISHARDED_POOL_TOKEN"] = spec["token"]
    os.environ.pop("ISHARDED_DISCOVERABLE", None)  # no beacons on the harness's network
    os.environ.pop("ISHARDED_CA_FILE", None)
    os.environ.pop("ISHARDED_OWN_ADDRESS", None)
    from isharded import serving

    serving.keep_listening()
    run_dir = Path(spec["run_dir"])
    owners = {key: value for key, value in spec["owners"].items()}
    nodes = [Node(node, run_dir, owners, first=index == 0) for index, node in enumerate(spec["nodes"])]
    network = spec.get("network") or {}
    FAULTS.delay_ms = float(network.get("delay_ms", 0))
    FAULTS.jitter_ms = float(network.get("jitter_ms", 0))
    FAULTS.loss = float(network.get("loss", 0))
    asyncio.run(serve(nodes, int(spec["observe_port"]), Lag()))


if __name__ == "__main__":
    main()
