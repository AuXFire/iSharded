"""Several models at once, each in its own lane.

A lane is a full copy of a node's control plane and workers, on ports a fixed
step above the node's own, with a mesh of its own. Everything that keeps one
model running keeps a lane running, and a failure in one lane stays there.

The main control plane is the front door. It holds the keys, decides which lanes
exist, and passes a request to the lane serving the model it names. Lanes reach
each other the way nodes already do, so they add no new kind of trust.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .access import node_of

LANE_PORT_STEP = 1000
MAX_LANES = 7
LANE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
LANE_REFRESH_SECONDS = 3.0
LANE_WORKERS_WAIT_SECONDS = 120.0
# A new process needs room for its own runtime before any weights arrive.
LANE_GPU_OVERHEAD_GB = 0.5
GIB = 1024**3


def lane_port(port: int, slot: int) -> int:
    return port + LANE_PORT_STEP * slot


def lane_url(url: str, slot: int) -> str:
    """The same node's address in a lane."""
    parts = urlsplit(url)
    if parts.port is None:
        raise ValueError(f"{url} has no port")
    host = parts.netloc.rpartition(":")[0]
    return urlunsplit((parts.scheme, f"{host}:{lane_port(parts.port, slot)}", parts.path, "", ""))


def clean_lanes(raw: Any) -> dict[str, dict[str, Any]]:
    """Keep only well-formed lanes. A supervisor starts processes from this."""
    lanes: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return lanes
    for name, lane in raw.items():
        try:
            slot = int(lane["slot"])
            budgets = {
                str(node): {"gpu_gb": max(float(budget.get("gpu_gb", 0)), 0.0), "ram_gb": max(float(budget.get("ram_gb", 0)), 0.0)}
                for node, budget in (lane.get("budgets") or {}).items()
            }
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if LANE_NAME.fullmatch(str(name)) and 1 <= slot <= MAX_LANES:
            lanes[str(name)] = {"slot": slot, "budgets": budgets}
    return lanes


def read_lanes(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return clean_lanes(document.get("lanes")), int(document.get("version", 0))
    except (OSError, ValueError, AttributeError, TypeError):
        return {}, 0


def write_lanes(path: Path, lanes: dict[str, dict[str, Any]], version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps({"version": version, "lanes": lanes}), encoding="utf-8")
    partial.replace(path)


def spare_budgets(workers: dict[str, Any]) -> dict[str, dict[str, float]]:
    """What each node has left under the limits its owner set."""
    groups: dict[str, dict[str, Any]] = {}
    for record in workers.values():
        if not record.live:
            continue
        reported = {item["pool_id"]: item for item in (record.health or {}).get("memory_pools", [])}
        for pool in record.registration.hardware.memory_pools:
            group = groups.setdefault(
                pool.capacity_group,
                {"node": node_of(record.registration.worker_id), "kind": pool.kind, "budget": 0, "used": 0},
            )
            group["budget"] = max(group["budget"], pool.budget_bytes)
            group["used"] += int(reported.get(pool.pool_id, {}).get("used_bytes", 0))
    spare: dict[str, dict[str, float]] = {}
    for group in groups.values():
        entry = spare.setdefault(group["node"], {"gpu_gb": 0.0, "ram_gb": 0.0})
        left = max(group["budget"] - group["used"], 0) / GIB
        if group["kind"] in {"vram", "unified"}:
            entry["gpu_gb"] += left
        elif group["kind"] == "ram":
            entry["ram_gb"] += left
    return spare


def lane_budgets(
    spare: dict[str, dict[str, float]],
    lanes: dict[str, dict[str, Any]],
    gpu_gb: float | None = None,
    ram_gb: float | None = None,
) -> dict[str, dict[str, float]]:
    """Share what is left between lanes, never more than a node's owner allows."""
    budgets: dict[str, dict[str, float]] = {}
    for node, left in spare.items():
        taken_gpu = sum(lane["budgets"].get(node, {}).get("gpu_gb", 0.0) for lane in lanes.values())
        taken_ram = sum(lane["budgets"].get(node, {}).get("ram_gb", 0.0) for lane in lanes.values())
        gpu = max(left["gpu_gb"] - taken_gpu - LANE_GPU_OVERHEAD_GB, 0.0)
        ram = max(left["ram_gb"] - taken_ram, 0.0)
        budgets[node] = {
            "gpu_gb": round(gpu if gpu_gb is None else min(gpu, gpu_gb), 2),
            "ram_gb": round(ram if ram_gb is None else min(ram, ram_gb), 2),
        }
    return budgets


class LaneDoor:
    """The front door's view of the lanes, on the main control plane only."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        configured = os.getenv("SHARDMESH_LANES_FILE")
        self.path = Path(configured) if configured and not os.getenv("SHARDMESH_LANE") else None
        self.seen: dict[str, dict[str, Any]] = {}
        self.problems: dict[str, str] = {}
        self.written = -1
        self.pushed_access = ""

    @property
    def active(self) -> bool:
        return self.path is not None and self.runtime.mesh is not None

    def restore(self) -> None:
        """Lanes come back after a restart, like the model on the main route."""
        if self.active:
            lanes, version = read_lanes(self.path)
            mesh = self.runtime.mesh
            if version > mesh.lanes_version:
                mesh.lanes, mesh.lanes_version = lanes, version
            self.written = version

    def sync(self) -> None:
        """Tell this node's supervisor what the pool wants."""
        mesh = self.runtime.mesh
        if self.active and mesh.lanes_version != self.written:
            write_lanes(self.path, clean_lanes(mesh.lanes), mesh.lanes_version)
            self.written = mesh.lanes_version

    def _sibling(self, name: str) -> str | None:
        lane = clean_lanes(self.runtime.mesh.lanes).get(name)
        return lane_url(self.runtime.mesh.url, lane["slot"]) if lane else None

    def _headers(self, caller: Any = None) -> dict[str, str]:
        """Node to node, carrying who is asking the way forwarding does."""
        from .tls import BEHALF_HEADER, FORWARD_MARK

        runtime = self.runtime
        headers = {FORWARD_MARK: runtime.mesh.node_id}
        if runtime.access.authority is not None:
            if caller is not None:
                headers[BEHALF_HEADER] = f"{caller.role}:{caller.name}"
        elif runtime.pool_token:
            headers["Authorization"] = f"Bearer {runtime.pool_token}"
        return headers

    async def _look(self, name: str) -> None:
        url = self._sibling(name)
        if url is None:
            return
        try:
            response = await self.runtime.client.get(url + "/internal/lane", headers=self._node_headers(), timeout=5.0)
            response.raise_for_status()
            self.seen[name] = response.json()
        except (httpx.HTTPError, ValueError):
            self.seen.pop(name, None)

    def _node_headers(self) -> dict[str, str]:
        token = self.runtime.pool_token
        return {"Authorization": f"Bearer {token}"} if token and self.runtime.access.authority is None else {}

    async def _share_access(self, name: str) -> None:
        """Signed changes reach a lane's nodes through their own front doors."""
        url = self._sibling(name)
        if url is None or self.runtime.access.authority is None:
            return
        document = self.runtime.access.document()
        try:
            await self.runtime.client.post(url + "/internal/lane/access", json=document, headers=self._node_headers(), timeout=5.0)
        except httpx.HTTPError:
            pass

    async def run(self) -> None:
        while True:
            await asyncio.sleep(LANE_REFRESH_SECONDS)
            try:
                self.sync()
                names = list(clean_lanes(self.runtime.mesh.lanes))
                for gone in set(self.seen) - set(names):
                    self.seen.pop(gone, None)
                await asyncio.gather(*(self._look(name) for name in names))
                document = json.dumps(self.runtime.access.document(), sort_keys=True, default=str)
                if document != self.pushed_access:
                    await asyncio.gather(*(self._share_access(name) for name in names))
                    self.pushed_access = document
            except Exception as exc:  # noqa: BLE001 - lanes must not stop the main route's upkeep
                self.runtime.add_event("failure", "Lane upkeep failed", str(exc)[:200])

    def serving(self, model: str | None) -> str | None:
        """The lane to send a request to, or None for the main route."""
        runtime = self.runtime
        if not self.active or (model and model == runtime.model_id):
            return None
        if model:
            return next((name for name, info in self.seen.items() if info.get("model") == model), None)
        if runtime.route:
            return None
        return next((name for name, info in self.seen.items() if info.get("ready")), None)

    def models(self) -> list[str]:
        return sorted({str(info["model"]) for info in self.seen.values() if info.get("ready") and info.get("model")})

    def _target(self, name: str) -> str:
        info = self.seen.get(name, {})
        target = self._sibling(name) if info.get("ready") else info.get("leader_url")
        if not target:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=f"lane {name} has no coordinator yet; try again in a few seconds")
        return str(target).rstrip("/")

    async def forward(self, name: str, path: str, payload: dict[str, Any], caller: Any) -> Any:
        """Relay a reply, streamed or not, exactly as the lane sends it."""
        from fastapi import HTTPException
        from starlette.background import BackgroundTask
        from starlette.responses import StreamingResponse

        client = self.runtime.client
        request = client.build_request("POST", self._target(name) + path, json=payload, headers=self._headers(caller))
        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"lane {name} did not answer: {exc}") from exc
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.aclose),
        )

    def create(self, name: str, gpu_gb: float | None, ram_gb: float | None) -> None:
        from fastapi import HTTPException

        if not self.active:
            raise HTTPException(status_code=409, detail="this node does not run lanes; start it with shardmesh node")
        if not LANE_NAME.fullmatch(name):
            raise HTTPException(status_code=422, detail="a lane name is lowercase letters, digits, and hyphens")
        mesh = self.runtime.mesh
        lanes = clean_lanes(mesh.lanes)
        if name in lanes:
            return
        slot = next((slot for slot in range(1, MAX_LANES + 1) if slot not in {lane["slot"] for lane in lanes.values()}), None)
        if slot is None:
            raise HTTPException(status_code=409, detail=f"this pool already runs {MAX_LANES} lanes")
        budgets = lane_budgets(spare_budgets(self.runtime.workers), lanes, gpu_gb, ram_gb)
        if not any(budget["gpu_gb"] or budget["ram_gb"] for budget in budgets.values()):
            raise HTTPException(status_code=409, detail="no node has memory left under the limits its owner set")
        lanes[name] = {"slot": slot, "budgets": budgets}
        mesh.publish_lanes(lanes)
        self.sync()
        self.runtime.add_event("pool", "Lane created", f"{name}: {budgets}")

    async def _start_load(self, name: str, payload: dict[str, Any], caller: Any) -> None:
        """Wait for the lane's workers to come up, then load in it."""
        loop = asyncio.get_running_loop()
        lane = clean_lanes(self.runtime.mesh.lanes)[name]
        expected = sum(1 for budget in lane["budgets"].values() if budget["gpu_gb"] or budget["ram_gb"])
        deadline = loop.time() + LANE_WORKERS_WAIT_SECONDS
        while True:
            await self._look(name)
            info = self.seen.get(name, {})
            if info.get("leader_url") and int(info.get("workers", 0)) >= expected:
                break
            if loop.time() > deadline:
                if info.get("leader_url") and int(info.get("workers", 0)) > 0:
                    break
                raise RuntimeError(f"lane {name} started no workers in {LANE_WORKERS_WAIT_SECONDS:.0f} s")
            await asyncio.sleep(1.0)
        url = str(self.seen[name]["leader_url"]).rstrip("/") + "/admin/load"
        response = await self.runtime.client.post(url, json=payload, headers=self._headers(caller), timeout=60.0)
        if response.status_code >= 400:
            raise RuntimeError(response.text[:300])

    async def load(self, body: Any, caller: Any) -> dict[str, Any]:
        name = body.lane
        self.create(name, body.lane_gpu_gb, body.lane_ram_gb)
        self.problems.pop(name, None)
        payload = body.model_dump(exclude_none=True, exclude={"lane", "lane_gpu_gb", "lane_ram_gb"})
        payload["wait"] = False

        async def start() -> None:
            try:
                await self._start_load(name, payload, caller)
            except (RuntimeError, httpx.HTTPError, KeyError) as exc:
                self.problems[name] = str(exc) or type(exc).__name__
                self.runtime.add_event("failure", "Lane load failed", f"{name}: {self.problems[name][:200]}")

        self.runtime.spawn(start())
        return {"status": "started", "lane": name}

    def remove(self, name: str) -> None:
        from fastapi import HTTPException

        mesh = self.runtime.mesh
        lanes = clean_lanes(mesh.lanes) if self.active else {}
        if name not in lanes:
            raise HTTPException(status_code=404, detail=f"no lane named {name}")
        del lanes[name]
        mesh.publish_lanes(lanes)
        self.sync()
        self.seen.pop(name, None)
        self.problems.pop(name, None)
        self.runtime.add_event("pool", "Lane removed", name)

    async def tell_all(self, path: str, caller: Any) -> None:
        """Pass a change on to every lane's coordinator, such as a cordon."""
        for name, info in list(self.seen.items()):
            target = info.get("leader_url")
            if not target:
                continue
            try:
                await self.runtime.client.post(str(target).rstrip("/") + path, headers=self._headers(caller), timeout=30.0)
            except httpx.HTTPError as exc:
                self.runtime.add_event("failure", "Lane did not take a change", f"{name}: {exc}")

    def status(self) -> list[dict[str, Any]]:
        if not self.active:
            return []
        rows = []
        for name, lane in sorted(clean_lanes(self.runtime.mesh.lanes).items()):
            info = self.seen.get(name, {})
            rows.append(
                {
                    "name": name,
                    "model": info.get("model"),
                    "ready": bool(info.get("ready")),
                    "workers": int(info.get("workers", 0)),
                    "route": info.get("route", []),
                    "load_job": info.get("load_job"),
                    "error": self.problems.get(name),
                    "budgets": lane["budgets"],
                }
            )
        return rows
