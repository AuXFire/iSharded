from __future__ import annotations

import argparse
import asyncio
import gc
import hmac
import importlib.metadata
import json
import os
import platform
import socket
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
import torch
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from .memory import (
    AUTO_RAM_SHARE,
    Accelerator,
    _synchronize,
    apply_accelerator_limits,
    build_pools,
    discover_accelerators,
    live_free_bytes,
)
from .models import Adapter, GgufAdapter, create_adapter
from .models.base import layer_unit
from .protocol import (
    pass_seconds,
    META_HEADER,
    PROTOCOL_VERSION,
    CacheRestoreRequest,
    ConfigureRequest,
    HardwareInfo,
    DraftModel as DraftSource,
    GenerateRequest,
    InferRequest,
    MemoryPool,
    PrepareRequest,
    ResourceContribution,
    StageAssignment,
    WorkerRegistration,
    auth_headers,
    decode_activation,
    decode_meta,
    decode_tensors,
    encode_activation,
    encode_meta,
    encode_tensors,
)
from .gguf import parse_source
from .gguf_stage import INITIAL_GUESSES, DraftModel, GgufStage, PromptPlan, next_guess_count, plan_prompt
from . import __version__
from .llamacpp import LlamaError, StageLost, ensure_build, pick_build, rocm_library_paths
from .pool import check_exposure, load_token
from .serving import MAX_STAGE_BYTES, BodyLimit, serve
from .slots import SlotGate
from .tls import PEER, client_verify
from .weights import FetchProgress, LoadedUnit, UnitCache, WeightSourceError, gguf_reader, load_unit, map_unit

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "shardmesh" / "units"
SUPPORTED_DTYPES = ["bfloat16", "float32", "float16"]
# Snapshots waiting to send. Past this the store is too slow to keep up, and
# recovery falls back to replaying the prompt.
MAX_PENDING_CHECKPOINTS = 32


def choose_dtype_name(requested: str) -> str:
    """BF16 unless a node asks otherwise, for example FP32 for full precision."""
    name = "bfloat16" if requested == "auto" else requested
    if name not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported tensor dtype {requested!r}")
    return name


def choose_dtype(requested: str) -> torch.dtype | None:
    """Torch dtype for a pool, or None when weights must stay as stored."""
    if requested == "checkpoint":
        return None
    return getattr(torch, choose_dtype_name(requested))


def parse_ram_budget(value: str) -> int | None:
    """Read a RAM budget: unset, ``auto``, or a size in GiB."""
    if not value:
        return None
    if value == "auto":
        return int(psutil.virtual_memory().available * AUTO_RAM_SHARE)
    return int(float(value) * 1024**3)


def hardware_info(
    accelerators: list[Accelerator],
    pools: list[MemoryPool],
    host_id: str,
    roles: set[str],
    cpu_threads: int,
    ram_budget_bytes: int,
    accelerator_memory_budget_bytes: int,
    standby: bool,
) -> HardwareInfo:
    primary = accelerators[0] if accelerators else None
    accelerator_memory = sum(item.total_bytes for item in accelerators)
    return HardwareInfo(
        host_id=host_id,
        device=primary.device if primary else "cpu",
        backend=primary.backend if primary else "cpu",
        runtime_version=primary.runtime_version if primary else platform.python_version(),
        supported_dtypes=SUPPORTED_DTYPES,
        cpu_cores=os.cpu_count() or 1,
        ram_total_bytes=psutil.virtual_memory().total,
        accelerator_name=" + ".join(item.name for item in accelerators) or None,
        accelerator_memory_bytes=accelerator_memory,
        contribution=ResourceContribution(
            roles=sorted(roles),
            cpu_threads=cpu_threads,
            ram_budget_bytes=ram_budget_bytes,
            accelerator_memory_budget_bytes=min(
                accelerator_memory_budget_bytes or accelerator_memory,
                accelerator_memory,
            ),
            standby=standby,
        ),
        memory_pools=pools,
    )


# Past this, a new reply is refused rather than left waiting indefinitely.
MAX_WAITING_REPLIES = 32
# A coordinator that asked for a drain and then died must not block replies forever.
DRAIN_LIMIT_SECONDS = 180.0
# A control plane this worker only heard about, and that stays silent, is forgotten.
CONTROL_LINK_FAILURES = 60


@dataclass(slots=True)
class ControlLink:
    """This worker's standing with one control plane."""

    registered: bool = False
    failures: int = 0


@dataclass(slots=True)
class CacheEntry:
    """One session's position and per-layer state on this worker."""

    epoch: int
    slot: int = 0
    states: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    length: int = 0


@dataclass(slots=True)
class StoredSnapshot:
    payload: bytes
    metadata: dict[str, Any]


class MemoryStore:
    """Bounded LRU store for committed, layer-local KV snapshots."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = max(0, budget_bytes)
        self.used_bytes = 0
        self.evictions = 0
        self.items: OrderedDict[str, StoredSnapshot] = OrderedDict()

    def put(self, key: str, payload: bytes, metadata: dict[str, Any]) -> None:
        if not self.budget_bytes:
            raise ValueError("this worker does not contribute a RAM cache budget")
        if len(payload) > self.budget_bytes:
            raise ValueError("snapshot exceeds the contributed RAM cache budget")
        previous = self.items.pop(key, None)
        if previous:
            self.used_bytes -= len(previous.payload)
        while self.items and self.used_bytes + len(payload) > self.budget_bytes:
            _, evicted = self.items.popitem(last=False)
            self.used_bytes -= len(evicted.payload)
            self.evictions += 1
        self.items[key] = StoredSnapshot(payload=payload, metadata=metadata)
        self.used_bytes += len(payload)

    def get_layers(
        self,
        session_id: str,
        epoch: int,
        start_layer: int,
        end_layer: int,
    ) -> bytes:
        chunks: dict[str, dict[int, torch.Tensor]] = {}
        touched: list[str] = []
        for key, snapshot in self.items.items():
            meta = snapshot.metadata
            if meta.get("session_id") != session_id or meta.get("epoch") != epoch:
                continue
            tensors = decode_tensors(snapshot.payload)
            start_position = int(meta.get("start_position", 0))
            for name, tensor in tensors.items():
                try:
                    layer = int(name.split("_", 2)[1])
                except (IndexError, ValueError):
                    continue
                if start_layer <= layer < end_layer:
                    chunks.setdefault(name, {})[start_position] = tensor
                    touched.append(key)
        expected = {
            f"layer_{layer}_{kind}"
            for layer in range(start_layer, end_layer)
            for kind in ("key", "value")
        }
        if set(chunks) != expected:
            missing = sorted(expected - set(chunks))
            raise KeyError(f"KV snapshot is incomplete; missing {missing}")
        selected: dict[str, torch.Tensor] = {}
        for name, parts in chunks.items():
            cursor = 0
            ordered: list[torch.Tensor] = []
            for start, tensor in sorted(parts.items()):
                if start != cursor:
                    raise KeyError(f"KV snapshot for {name} has a gap at position {cursor}")
                ordered.append(tensor)
                cursor += tensor.shape[-2]
            selected[name] = torch.cat(ordered, dim=-2)
        for key in dict.fromkeys(touched):
            self.items.move_to_end(key)
        return encode_tensors(selected)

    def clear_session(self, session_id: str) -> None:
        for key in list(self.items):
            if self.items[key].metadata.get("session_id") == session_id:
                snapshot = self.items.pop(key)
                self.used_bytes -= len(snapshot.payload)

    def stats(self) -> dict[str, int]:
        return {
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "snapshots": len(self.items),
            "evictions": self.evictions,
        }


@dataclass(slots=True)
class MergedStep:
    """One conversation's next token, waiting to share a pass through this stage."""

    hidden: torch.Tensor
    request: InferRequest
    outcome: asyncio.Future[Any]


class WorkerRuntime:
    def __init__(self) -> None:
        self.worker_id = os.getenv("SHARDMESH_WORKER_ID", socket.gethostname())
        self.host_id = os.getenv("SHARDMESH_HOST_ID", socket.gethostname())
        # Any control plane will do to get in, and each one answers with the rest.
        configured = os.getenv("SHARDMESH_ORCHESTRATOR_URL", "http://127.0.0.1:8000")
        self.seed_urls = {url.strip().rstrip("/") for url in configured.split(",") if url.strip()}
        self.control_urls: dict[str, ControlLink] = {url: ControlLink() for url in self.seed_urls}
        self.term = 0
        self.advertised_url = os.getenv("SHARDMESH_ADVERTISE_URL", "http://127.0.0.1:8100").rstrip("/")
        self.pool_token = load_token()
        self.requires_identity = bool(os.getenv("SHARDMESH_CA_FILE"))
        self.revoked_nodes: set[str] = set()
        self.roles = {
            role.strip()
            for role in os.getenv("SHARDMESH_ROLES", "inference").split(",")
            if role.strip()
        }
        self.cpu_threads = max(0, int(os.getenv("SHARDMESH_CPU_THREADS", "0")))
        self.ram_budget_bytes = max(
            0,
            int(float(os.getenv("SHARDMESH_RAM_BUDGET_GB", "0")) * 1024**3),
        )
        self.accelerator_memory_budget_bytes = max(
            0,
            int(float(os.getenv("SHARDMESH_GPU_MEMORY_GB", "0")) * 1024**3),
        )
        self.weights_ram_bytes = parse_ram_budget(os.getenv("SHARDMESH_WEIGHTS_RAM_GB", ""))
        self.weights_disk_bytes = int(float(os.getenv("SHARDMESH_WEIGHTS_DISK_GB", "0")) * 1024**3)
        self.standby = os.getenv("SHARDMESH_STANDBY", "0").lower() in {"1", "true", "yes"}
        self.requested_device = os.getenv("SHARDMESH_DEVICE", "auto")
        self.dtype_name = choose_dtype_name(os.getenv("SHARDMESH_DTYPE", "auto"))
        self.cache_dir = Path(os.getenv("SHARDMESH_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.unit_cache = UnitCache(
            self.cache_dir,
            budget_bytes=int(float(os.getenv("SHARDMESH_CACHE_BUDGET_GB", "0")) * 1024**3),
            concurrency=int(os.getenv("SHARDMESH_FETCH_CONCURRENCY", "4")),
            local_dirs=tuple(Path(item) for item in os.getenv("SHARDMESH_GGUF_DIRS", "").split(os.pathsep) if item),
        )
        self.peer_fetch = os.getenv("SHARDMESH_PEER_FETCH", "1") != "0"
        self.peer_serve = os.getenv("SHARDMESH_PEER_SERVE", "1") != "0"
        self.hardware: HardwareInfo | None = None
        self.pools: dict[str, MemoryPool] = {}
        self.llama_dir = os.getenv("SHARDMESH_LLAMA_DIR") or None
        self.gguf: GgufStage | None = None
        self.gguf_logits: torch.Tensor | None = None
        self.merge_passes = False
        self.merge_queue: list[MergedStep] = []
        self.merge_task: asyncio.Task[None] | None = None
        self.merged_passes = 0
        self.merged_rows = 0
        self.gguf_choices: list[int] = []
        self.draft: DraftModel | None = None
        self.speculate_tokens = 0
        self.draft_passes = 0
        self.draft_accepted = 0
        self.stage_restarts = 0
        self.draft_request: tuple[ConfigureRequest, str, int] | None = None
        # One conversation at a time per stage process. Requests from any node
        # queue here, and a pool change waits here for the reply in progress.
        self.gate = SlotGate(1)
        self.waiting = 0
        self.drain_timer: asyncio.TimerHandle | None = None
        # Per sequence, the conversation prefix the stages can restore: key and tokens.
        self.checkpoints: dict[int, tuple[str, list[int]]] = {}
        self.slot_used_at: dict[int, float] = {}
        self.gguf_signature: tuple[Any, ...] | None = None
        self.stages_kept = 0
        self.model_id: str | None = None
        self.adapter: Adapter | None = None
        self.epoch = 0
        self.assignment: StageAssignment | None = None
        self.route: list[StageAssignment] = []
        self.cache_url: str | None = None
        # Resident units by model, revision, unit, and pool, so that a
        # rebalance keeps whatever did not move.
        self.loaded: dict[str, LoadedUnit] = {}
        self.unit_weights: dict[str, dict[str, torch.Tensor]] = {}
        self.pool_used: dict[str, int] = {}
        self.prepare_jobs: OrderedDict[str, FetchProgress] = OrderedDict()
        self.background: set[asyncio.Task[None]] = set()
        self.materialize_total = 0
        self.materialize_done = 0
        self.caches: dict[str, CacheEntry] = {}
        self.lock = asyncio.Lock()
        # A disk-tier prompt can take minutes, and a slow hop is not a failure.
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=10.0), verify=client_verify())
        self.registration_task: asyncio.Task[None] | None = None
        self.inference_calls = 0
        self.tokens_processed = 0
        self.activation_bytes_sent = 0
        self.compute_ms_ema: float | None = None
        self.downstream_ms_ema: float | None = None
        self.last_error: str | None = None
        self.kv_snapshots_sent = 0
        self.kv_snapshot_bytes_sent = 0
        self.kv_restores = 0
        self.memory_store = MemoryStore(self.ram_budget_bytes)
        # Moving averages of where a token's time goes on this worker, in ms.
        self.phases: dict[str, float] = {}
        self.checkpoint_tail: asyncio.Task[None] | None = None
        self.checkpoints_pending = 0
        self.checkpoints_dropped = 0

    def setup(self) -> None:
        """Probe hardware. Separate from ``__init__`` so importing stays cheap."""
        if self.hardware is not None:
            return
        if self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        computes = "inference" in self.roles
        accelerators = discover_accelerators(self.requested_device) if computes else []
        pools = (
            build_pools(
                self.host_id,
                accelerators,
                accelerator_budget_bytes=self.accelerator_memory_budget_bytes,
                compute_dtype=self.dtype_name,
                ram_budget_bytes=self.weights_ram_bytes,
                disk_budget_bytes=self.weights_disk_bytes,
                cache_dir=self.cache_dir,
            )
            if computes
            else []
        )
        apply_accelerator_limits(pools, explicit_budget=bool(self.accelerator_memory_budget_bytes))
        self.pools = {pool.pool_id: pool for pool in pools}
        self.hardware = hardware_info(
            accelerators,
            pools,
            self.host_id,
            self.roles,
            self.cpu_threads,
            self.ram_budget_bytes,
            self.accelerator_memory_budget_bytes,
            self.standby,
        )

    @staticmethod
    def _ema(previous: float | None, value: float, weight: float = 0.2) -> float:
        return value if previous is None else previous * (1 - weight) + value * weight

    def _record(self, phase: str, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000
        self.phases[phase] = round(self._ema(self.phases.get(phase), elapsed), 2)

    async def close(self) -> None:
        if self.registration_task:
            self.registration_task.cancel()
        await self.client.aclose()

    def registration(self) -> WorkerRegistration:
        assert self.hardware is not None
        return WorkerRegistration(
            worker_id=self.worker_id,
            advertised_url=self.advertised_url,
            hardware=self.hardware,
            protocol_version=PROTOCOL_VERSION,
            code_version=__version__,
        )

    async def _check_in(self, url: str, link: ControlLink) -> None:
        """Register with one control plane, or tell it this worker is still here."""
        path = "/internal/heartbeat" if link.registered else "/internal/register"
        body: dict[str, Any] = (
            {"worker_id": self.worker_id} if link.registered else self.registration().model_dump(mode="json")
        )
        try:
            response = await self.client.post(
                url + path, json=body, headers=auth_headers(self.pool_token), timeout=4.0
            )
            if response.status_code == 404 and link.registered:
                link.registered = False
            elif response.status_code == 426:
                link.registered = False
                message = str(response.json().get("detail"))
                if self.last_error != message:
                    print(f"Refused by {url}: {message}", flush=True)
                self.last_error = message
            else:
                response.raise_for_status()
                link.registered = True
                link.failures = 0
                answer = response.json()
                self.revoked_nodes = set(answer.get("revoked_nodes", []))
                for other in answer.get("controls", []):
                    self.control_urls.setdefault(str(other).rstrip("/"), ControlLink())
        except (httpx.HTTPError, OSError, ValueError):
            link.registered = False
            link.failures += 1
            if link.failures >= CONTROL_LINK_FAILURES and url not in self.seed_urls:
                self.control_urls.pop(url, None)

    async def registration_loop(self) -> None:
        """Stay known to every control plane, so a new coordinator needs no introduction."""
        while True:
            await asyncio.gather(*(self._check_in(url, link) for url, link in list(self.control_urls.items())))
            await asyncio.sleep(5)

    def pool_status(self) -> list[dict[str, Any]]:
        """Live view of each pool, for the planner and the dashboard."""
        status: list[dict[str, Any]] = []
        for pool in self.pools.values():
            status.append(
                {
                    **pool.model_dump(),
                    "free_bytes": live_free_bytes(pool, self.cache_dir),
                    "used_bytes": self.pool_used.get(pool.pool_id, 0),
                }
            )
        return status

    def start_prepare(self, request: PrepareRequest) -> FetchProgress:
        """Begin downloading in the background. Serving is not interrupted."""
        progress = FetchProgress(request.job_id)
        self.prepare_jobs[request.job_id] = progress
        while len(self.prepare_jobs) > 8:
            self.prepare_jobs.popitem(last=False)
        task = asyncio.create_task(self._run_prepare(request, progress))
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        return progress

    async def _run_prepare(self, request: PrepareRequest, progress: FetchProgress) -> None:
        try:
            await asyncio.to_thread(
                self.unit_cache.fetch,
                request.source,
                request.model_id,
                request.revision,
                request.units,
                progress,
                request.source_revision,
                request.peers if self.peer_fetch else None,
                self.pool_token,
                request.file_sizes,
                request.file_sources,
            )
            progress.phase = "ready"
        except (WeightSourceError, httpx.HTTPError, OSError, ValueError) as exc:
            progress.phase = "failed"
            progress.error = f"{type(exc).__name__}: {exc}"
            self.last_error = progress.error
        finally:
            progress.finished_at = time.monotonic()

    def _close_gguf(self) -> None:
        if self.draft is not None:
            self.draft.close()
            self.draft = None
        if self.gguf is not None:
            self.gguf.close()
            self.gguf = None
        self.gguf_logits = None

    def _draft_file(self, draft: DraftSource) -> Path:
        """A draft model's file: a copy already on this machine, or a download."""
        local = self.unit_cache.local_copy(draft.file, draft.file_size)
        if local is not None:
            return local
        target = self.unit_cache.directory(draft.source, draft.revision) / draft.file
        if target.is_file() and target.stat().st_size == draft.file_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        reader = gguf_reader(draft.source, draft.revision, draft.file_sources, self.pool_token)
        try:
            with partial.open("wb") as handle:
                for chunk in reader.iter_range(draft.file, 0, draft.file_size):
                    handle.write(chunk)
        finally:
            reader.close()
        partial.replace(target)
        return target

    def _load_draft(self, request: ConfigureRequest, folder: str, threads: int) -> None:
        """Speculation only speeds things up, so a draft that fails must not stop the load."""
        self.speculate_tokens = request.speculate_tokens
        if request.draft is None or request.assignment.start_layer != 0:
            return
        self.draft_request = (request, folder, threads)
        try:
            path = self._draft_file(request.draft)
            self.draft = DraftModel(folder, rocm_library_paths(), path, request.context_tokens, threads, request.slots)
        except (LlamaError, WeightSourceError, httpx.HTTPError, OSError, ValueError) as exc:
            self.last_error = f"draft model unavailable, running without it: {exc}"

    async def replace_stage(self, lost: StageLost) -> HTTPException:
        """Start a new stage process in place of a dead one.

        Only this process failed, so the node keeps its place: the answer asks
        for a replay and says nothing against the worker. A stage that will not
        restart is a real failure, and reads as one.
        """
        self.last_error = str(lost)
        self.caches.clear()
        try:
            assert self.gguf is not None
            await asyncio.to_thread(self.gguf.respawn)
        except (LlamaError, OSError, AssertionError) as exc:
            self.last_error = f"stage would not restart: {exc}"
            return HTTPException(status_code=500, detail=self.last_error)
        self.stage_restarts += 1
        return HTTPException(status_code=503, detail={"retry": True, "message": str(lost)})

    async def replace_draft(self, reason: Exception) -> None:
        """A draft is only an accelerator: lose it, try for another, keep serving."""
        self.last_error = f"draft model lost: {reason}"
        draft, self.draft = self.draft, None
        if draft is not None:
            await asyncio.to_thread(draft.close)
        if self.draft_request is not None:
            await asyncio.to_thread(self._load_draft, *self.draft_request)

    @staticmethod
    def _stage_signature(request: ConfigureRequest) -> tuple[Any, ...]:
        stage = request.assignment
        return (
            request.model_id,
            request.revision,
            stage.start_layer,
            stage.end_layer,
            tuple(sorted((item.unit, item.pool_id, item.bytes) for item in stage.placements)),
            request.context_tokens,
            request.speculate_tokens,
            request.slots,
            request.draft.file if request.draft and stage.start_layer == 0 else None,
        )

    async def hold_gate(self, seconds: float) -> bool:
        """Let the reply in progress finish, and start no new one.

        Released by the next configure, or by the timer if the coordinator that
        asked for this dies first, because nothing else would ever release it.
        """
        if self.drain_timer is not None:
            return True
        if not await self.gate.hold(seconds):
            return False
        loop = asyncio.get_running_loop()
        self.drain_timer = loop.call_later(DRAIN_LIMIT_SECONDS, lambda: loop.create_task(self.release_gate()))
        return True

    async def release_gate(self) -> None:
        if self.drain_timer is not None:
            self.drain_timer.cancel()
            self.drain_timer = None
            await self.gate.resume()

    def choose_slot(self, free: set[int], prompt: list[int]) -> int:
        """Prefer the sequence holding this conversation, then the one idle longest."""
        best, reach = None, -1
        for slot in free:
            held = self.checkpoints.get(slot)
            if held and len(held[1]) > reach and len(held[1]) < len(prompt) and prompt[: len(held[1])] == held[1]:
                best, reach = slot, len(held[1])
        if best is not None:
            return best
        return min(free, key=lambda slot: self.slot_used_at.get(slot, 0.0))

    def _materialize_gguf(self, request: ConfigureRequest, adapter: GgufAdapter) -> None:
        """Assemble this stage's file from cached units and load it into llama.cpp."""
        for key in list(self.loaded):
            self.loaded.pop(key).release()
        self.unit_weights = {}
        placements = [item for item in request.assignment.placements if item.bytes]
        self.materialize_total = len(placements)
        self.materialize_done = 0
        folder = self.llama_dir or str(ensure_build(pick_build(self.requested_device)))
        threads = self.cpu_threads or max((os.cpu_count() or 4) // 2, 1)
        self.gguf = GgufStage(folder, rocm_library_paths(), self.unit_cache, request, adapter, self.pools, threads)
        self._load_draft(request, folder, threads)
        used: dict[str, int] = {}
        for item in placements:
            used[item.pool_id] = used.get(item.pool_id, 0) + item.bytes
        self.pool_used = used
        self.materialize_done = len(placements)

    def _materialize(self, request: ConfigureRequest, adapter: Adapter) -> None:
        """Move fetched units into their pools, keeping what is already there."""
        signature = self._stage_signature(request)
        if isinstance(adapter, GgufAdapter) and self.gguf is not None and signature == self.gguf_signature:
            # Nothing about this stage changed, so its weights stay where they are.
            self.stages_kept += 1
            return
        self._close_gguf()
        self.checkpoints = {}
        if isinstance(adapter, GgufAdapter):
            self._materialize_gguf(request, adapter)
            self.gguf_signature = signature
            return
        prefix = f"{request.model_id}@{request.revision}"
        placements = request.assignment.placements
        wanted = {
            f"{prefix}/{item.unit}/{item.pool_id}/{item.compact}": item for item in placements if item.bytes
        }
        self.unit_weights = {}
        for key in [key for key in self.loaded if key not in wanted]:
            self.loaded.pop(key).release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.materialize_total = len(wanted)
        self.materialize_done = 0
        used: dict[str, int] = {}
        for key, item in wanted.items():
            if key not in self.loaded:
                pool = self.pools.get(item.pool_id)
                if pool is None:
                    raise ValueError(f"this worker has no memory pool {item.pool_id!r}")
                path = self.unit_cache.path(request.model_id, request.revision, item.unit)
                if not path.is_file():
                    raise ValueError(f"unit {item.unit} was not fetched before configure")
                if pool.kind == "disk":
                    self.loaded[key] = map_unit(path)
                else:
                    self.loaded[key] = load_unit(
                        path,
                        pool.device,
                        choose_dtype(pool.dtype),
                        adapter.keep_float32,
                        compact=item.compact,
                    )
            self.unit_weights[item.unit] = {
                adapter.local_name(name): tensor for name, tensor in self.loaded[key].tensors.items()
            }
            used[item.pool_id] = used.get(item.pool_id, 0) + item.bytes
            self.materialize_done += 1
        # A zero-byte placement means this unit shares the head's tensors.
        for item in placements:
            if not item.bytes:
                self.unit_weights[item.unit] = self.unit_weights["head"]
        self.pool_used = used

    def _release_weights(self) -> None:
        """Give back everything after a failed load.

        A worker that stopped halfway belongs to no route, and memory it keeps
        would count against it on the next attempt.
        """
        self._close_gguf()
        self.gguf_signature = None
        self.checkpoints = {}
        for key in list(self.loaded):
            self.loaded.pop(key).release()
        self.unit_weights = {}
        self.pool_used = {}
        self.adapter = None
        self.model_id = None
        self.assignment = None
        self.route = []
        self.caches.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def configure(self, request: ConfigureRequest) -> None:
        if "inference" not in self.roles:
            raise ValueError("this worker does not contribute an inference executor")
        if request.assignment.worker_id != self.worker_id:
            raise ValueError("assignment targets a different worker")
        if request.term < self.term:
            raise ValueError(f"order from term {request.term}, but term {self.term} already took over")
        self.term = request.term
        adapter = create_adapter(request.architecture)
        if not (0 <= request.assignment.start_layer < request.assignment.end_layer <= adapter.num_layers):
            raise ValueError("invalid layer interval")
        try:
            await asyncio.to_thread(self._materialize, request, adapter)
        except Exception:
            self._release_weights()
            raise
        self.adapter = adapter
        self.model_id = request.model_id
        self.epoch = request.epoch
        self.assignment = request.assignment
        self.route = request.route
        self.cache_url = request.cache_url.rstrip("/") if request.cache_url else None
        self.caches.clear()
        self.merge_passes = request.merge_passes
        if self.gate.size != request.slots:
            self.gate.resize(request.slots)
        await self.release_gate()

    def _next_stage(self) -> StageAssignment | None:
        assert self.assignment is not None
        index = next(i for i, stage in enumerate(self.route) if stage.worker_id == self.worker_id)
        return self.route[index + 1] if index + 1 < len(self.route) else None

    def _stages_after(self) -> int:
        index = next(i for i, stage in enumerate(self.route) if stage.worker_id == self.worker_id)
        return len(self.route) - index - 1

    def _session(self, request: InferRequest) -> CacheEntry:
        entry = self.caches.get(request.session_id)
        restoring = request.restore_length is not None
        stale = entry is None or entry.epoch != request.epoch
        if self.gguf is not None and stale and not request.reset and not restoring:
            # llama.cpp holds one sequence and this session's is gone. Starting
            # from nothing would answer from the wrong state, so ask for a replay.
            raise HTTPException(status_code=503, detail={"retry": True, "message": "this session's state is gone"})
        if request.reset or restoring or stale:
            entry = CacheEntry(epoch=request.epoch, slot=request.slot)
            if self.gguf is not None:
                if restoring:
                    try:
                        self.gguf.restore(request.checkpoint_key, request.slot)
                    except StageLost:
                        raise
                    except LlamaError as exc:
                        raise HTTPException(status_code=503, detail={"retry": True, "message": str(exc)}) from exc
                    entry.length = request.restore_length
                else:
                    self.gguf.clear(request.slot)
                # A new session takes over its sequence from whoever had it.
                self.caches = {key: item for key, item in self.caches.items() if item.slot != request.slot}
            self.caches[request.session_id] = entry
        return entry

    def _prepare_entry(self, request: InferRequest) -> torch.Tensor:
        if self.gguf is not None:
            self._session(request)
            return torch.from_numpy(self.gguf.embed(request.input_ids).copy()).unsqueeze(0)
        if self.adapter is None or "embed" not in self.unit_weights:
            raise ValueError("this worker is not the entry stage")
        entry = self._session(request)
        input_ids = torch.tensor([request.input_ids], dtype=torch.long)
        positions = torch.arange(entry.length, entry.length + input_ids.shape[1], dtype=torch.long)
        with torch.inference_mode():
            return self.adapter.embed(self.unit_weights["embed"], input_ids, positions.unsqueeze(0))

    def _run_local(self, hidden: torch.Tensor, request: InferRequest) -> torch.Tensor:
        assert self.adapter is not None and self.assignment is not None
        started = time.perf_counter()
        # The entry stage already reset this session before embedding.
        entry = self.caches.get(request.session_id) if self.assignment.start_layer == 0 else None
        entry = entry or self._session(request)
        tokens = hidden.shape[1]
        if self.gguf is not None:
            if request.rollback_to is not None and entry.length > request.rollback_to:
                self.gguf.truncate(request.rollback_to, request.slot)
                entry.length = request.rollback_to
            rows = hidden[0].float().numpy()
            if self.gguf.last and request.verify:
                self.gguf_choices = self.gguf.verify(
                    rows,
                    entry.length,
                    guesses=request.input_ids[1:],
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    slot=request.slot,
                )
            else:
                output = self.gguf.forward(rows, entry.length, request.slot)
                if self.gguf.last:
                    self.gguf_logits = torch.from_numpy(output.copy())
                else:
                    hidden = torch.from_numpy(output.copy()).unsqueeze(0)
        else:
            positions = torch.arange(entry.length, entry.length + tokens, dtype=torch.long)
            with torch.inference_mode():
                for layer in range(self.assignment.start_layer, self.assignment.end_layer):
                    hidden = self.adapter.run_layer(
                        layer,
                        self.unit_weights[layer_unit(layer, "dense")],
                        self.unit_weights.get(layer_unit(layer, "experts")),
                        hidden,
                        entry.states.setdefault(layer, {}),
                        positions,
                    )
                # GPU work is asynchronous. Without this, the time would surface
                # later, in whatever first touches the result.
                _synchronize(hidden.device)
        entry.length += tokens
        if request.save_checkpoint and self.gguf is not None:
            self.gguf.save(request.save_checkpoint, request.slot)
        self.inference_calls += 1
        self.tokens_processed += tokens
        self.compute_ms_ema = self._ema(self.compute_ms_ema, (time.perf_counter() - started) * 1000)
        return hidden

    def _sample(self, hidden: torch.Tensor, request: InferRequest, logits: torch.Tensor | None = None) -> int:
        gguf_logits = logits if logits is not None else self.gguf_logits if self.gguf is not None else None
        if gguf_logits is None and (self.adapter is None or "head" not in self.unit_weights):
            raise ValueError("route ended on a worker without the output head")
        with torch.inference_mode():
            if gguf_logits is not None:
                logits = gguf_logits
            else:
                logits = self.adapter.logits(self.unit_weights["head"], hidden[:, -1:, :])[0, -1].float()
            if request.temperature == 0:
                return int(torch.argmax(logits).item())
            if request.top_k:
                cutoff = torch.topk(logits, min(request.top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = torch.softmax(logits / request.temperature, dim=-1)
            if request.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > request.top_p
                sorted_probs[remove] = 0
                probabilities = torch.zeros_like(probabilities).scatter(0, sorted_indices, sorted_probs)
                probabilities /= probabilities.sum()
            if request.seed is not None:
                generator = torch.Generator(device=probabilities.device)
                generator.manual_seed(request.seed + self.caches[request.session_id].length)
                return int(torch.multinomial(probabilities, 1, generator=generator).item())
            return int(torch.multinomial(probabilities, 1).item())

    def _cache_payload(self, session_id: str, token_count: int) -> tuple[bytes, int]:
        if self.assignment is None:
            raise ValueError("worker has no stage assignment")
        entry = self.caches[session_id]
        tensors: dict[str, torch.Tensor] = {}
        start_position = 0
        for layer, state in entry.states.items():
            if "key" not in state:
                continue
            total = state["key"].shape[-2]
            count = min(token_count, total)
            start_position = total - count
            tensors[f"layer_{layer}_key"] = state["key"][..., -count:, :]
            tensors[f"layer_{layer}_value"] = state["value"][..., -count:, :]
        if not tensors:
            raise ValueError("session has no KV state to checkpoint")
        return encode_tensors(tensors), start_position

    def schedule_checkpoint(self, request: InferRequest) -> None:
        """Queue a KV snapshot without making the token wait for it.

        The payload is taken now, because the cache moves on with the next
        token, and sends stay in order so a store never sees a gap it can avoid.
        Recovery checks the length, so a store that is behind costs a replay.
        """
        if not self.cache_url or not self.assignment or request.session_id not in self.caches:
            return
        if self.checkpoints_pending >= MAX_PENDING_CHECKPOINTS:
            self.checkpoints_dropped += 1
            return
        try:
            payload, start_position = self._cache_payload(request.session_id, len(request.input_ids))
        except (KeyError, TypeError, ValueError) as exc:
            self.last_error = f"KV checkpoint failed: {exc}"
            return
        previous = self.checkpoint_tail
        self.checkpoints_pending += 1

        async def send() -> None:
            try:
                if previous is not None:
                    await previous
                await self.checkpoint_cache(request, payload, start_position)
            finally:
                self.checkpoints_pending -= 1

        self.checkpoint_tail = asyncio.create_task(send())

    async def checkpoint_cache(self, request: InferRequest, payload: bytes, start_position: int) -> None:
        started = time.perf_counter()
        try:
            metadata = {
                "session_id": request.session_id,
                "epoch": request.epoch,
                "worker_id": self.worker_id,
                "model_id": self.model_id,
                "start_layer": self.assignment.start_layer,
                "end_layer": self.assignment.end_layer,
                "start_position": start_position,
                "token_count": len(request.input_ids),
            }
            headers = auth_headers(self.pool_token)
            headers[META_HEADER] = encode_meta(metadata)
            headers["content-type"] = "application/x-safetensors"
            response = await self.client.put(
                self.cache_url + "/internal/cache/kv",
                content=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            self.kv_snapshots_sent += 1
            self.kv_snapshot_bytes_sent += len(payload)
            self._record("checkpoint_send", started)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            # Cache replication must never take the inference path down. The
            # orchestrator will fall back to committed-prefix replay.
            self.last_error = f"KV checkpoint failed: {exc}"

    async def restore_cache(self, request: CacheRestoreRequest) -> None:
        if self.assignment is None:
            raise ValueError("worker has no stage assignment")
        response = await self.client.get(
            request.cache_url.rstrip("/") + f"/internal/cache/kv/{request.session_id}/{request.source_epoch}",
            params={
                "start_layer": self.assignment.start_layer,
                "end_layer": self.assignment.end_layer,
            },
            headers=auth_headers(self.pool_token),
            timeout=60.0,
        )
        response.raise_for_status()
        tensors = decode_tensors(response.content)
        entry = CacheEntry(epoch=request.target_epoch)
        for layer in range(self.assignment.start_layer, self.assignment.end_layer):
            # Any full-width tensor shows this layer's device and dtype.
            weights = self.unit_weights[layer_unit(layer, "dense")].values()
            reference = next(tensor for tensor in weights if tensor.element_size() > 1)
            key = tensors[f"layer_{layer}_key"].to(reference.device, reference.dtype)
            value = tensors[f"layer_{layer}_value"].to(reference.device, reference.dtype)
            entry.states[layer] = {"key": key, "value": value}
            entry.length = key.shape[-2]
        if entry.length != request.expected_length:
            raise ValueError(f"store has {entry.length} tokens, session has {request.expected_length}")
        self.caches[request.session_id] = entry
        self.kv_restores += 1

    async def _step_alone(self, hidden: torch.Tensor, request: InferRequest) -> torch.Tensor | dict[str, Any]:
        """This stage's share of one request. Hidden states, or the answer at the end."""
        next_stage = self._next_stage()
        # Compute runs off the event loop: a disk-tier token can take seconds,
        # and a blocked loop would stop heartbeats and look like a dead node.
        started = time.perf_counter()
        hidden = await asyncio.to_thread(self._run_local, hidden, request)
        self._record("compute", started)
        if next_stage is None and request.verify:
            return {"token_ids": self.gguf_choices}
        if next_stage is None and request.prefill_only:
            return {"token_id": -1}
        if next_stage is None:
            started = time.perf_counter()
            result = {"token_id": await asyncio.to_thread(self._sample, hidden, request)}
            self._record("sample", started)
            self.schedule_checkpoint(request)
            return result
        return hidden

    def _can_merge(self, hidden: torch.Tensor, request: InferRequest) -> bool:
        """Ordinary next-token steps can share a pass. Anything with its own state work cannot.

        Off unless asked for. Answers are right, and one model on one GPU gained
        2.8x at four conversations, but a two-node pool measured slower with it.
        """
        return (
            self.merge_passes
            and self.gguf is not None
            and hidden.shape[1] == 1
            and not (request.verify or request.reset or request.prefill_only or request.save_checkpoint)
            and request.restore_length is None
        )

    async def _step_together(self, hidden: torch.Tensor, request: InferRequest) -> torch.Tensor | dict[str, Any]:
        """Join whichever other conversations are waiting for this stage.

        There is no window to wait out. Steps gather while the stage is busy, so a
        lone conversation goes straight through and a busy stage takes what it has.
        """
        step = MergedStep(hidden, request, asyncio.get_running_loop().create_future())
        self.merge_queue.append(step)
        if self.merge_task is None or self.merge_task.done():
            self.merge_task = asyncio.ensure_future(self._run_merge_queue())
        return await step.outcome

    async def _run_merge_queue(self) -> None:
        while self.merge_queue:
            async with self.lock:
                steps = [step for step in self.merge_queue if not step.outcome.done()]
                self.merge_queue = []
                stale = [step for step in steps if step.request.epoch != self.epoch or self.gguf is None]
                steps = [step for step in steps if step not in stale]
                for step in stale:
                    step.outcome.set_exception(HTTPException(status_code=409, detail=f"pipeline epoch mismatch; worker has {self.epoch}"))
                try:
                    if len(steps) == 1:
                        results: list[Any] = [await self._step_alone(steps[0].hidden, steps[0].request)]
                    elif steps:
                        started = time.perf_counter()
                        results = await asyncio.to_thread(self._run_merged, steps)
                        self._record("compute", started)
                        for step, result in zip(steps, results):
                            if isinstance(result, dict):
                                self.schedule_checkpoint(step.request)
                    else:
                        results = []
                except StageLost as exc:
                    # One of them restarts the stage. The rest just need a replay.
                    results = [exc] + [HTTPException(status_code=503, detail={"retry": True, "message": str(exc)})] * (len(steps) - 1)
                except Exception as exc:  # noqa: BLE001 - each waiting reply must hear about it
                    results = [exc] * len(steps)
                for step, result in zip(steps, results):
                    if step.outcome.done():
                        continue
                    if isinstance(result, BaseException):
                        step.outcome.set_exception(result)
                    else:
                        step.outcome.set_result(result)

    def _run_merged(self, steps: list["MergedStep"]) -> list[Any]:
        assert self.gguf is not None
        started = time.perf_counter()
        results: dict[int, Any] = {}
        ready: list[tuple[int, MergedStep, CacheEntry]] = []
        for index, step in enumerate(steps):
            request = step.request
            try:
                entry = self.caches.get(request.session_id) if self.assignment.start_layer == 0 else None
                entry = entry or self._session(request)
            except HTTPException as exc:
                results[index] = exc
                continue
            if request.rollback_to is not None and entry.length > request.rollback_to:
                self.gguf.truncate(request.rollback_to, request.slot)
                entry.length = request.rollback_to
            ready.append((index, step, entry))
        ready.sort(key=lambda item: item[1].request.slot)
        if ready:
            rows = np.concatenate([step.hidden[0].float().numpy() for _, step, _ in ready])
            output = self.gguf.forward_many(rows, [entry.length for _, _, entry in ready], [step.request.slot for _, step, _ in ready])
            for row, (index, step, entry) in zip(output, ready):
                entry.length += 1
                if self.gguf.last:
                    results[index] = {"token_id": self._sample(step.hidden, step.request, torch.from_numpy(row.copy()))}
                else:
                    results[index] = torch.from_numpy(row.copy()).reshape(1, 1, -1)
            self.merged_passes += 1
            self.merged_rows += len(ready)
            self.inference_calls += len(ready)
            self.tokens_processed += len(ready)
        self.compute_ms_ema = self._ema(self.compute_ms_ema, (time.perf_counter() - started) * 1000)
        return [results[index] for index in range(len(steps))]

    async def execute(self, hidden: torch.Tensor, request: InferRequest) -> dict[str, Any]:
        if self.assignment is None or self.model_id is None:
            raise HTTPException(status_code=409, detail="worker is not configured")
        if request.epoch != self.epoch:
            raise HTTPException(status_code=409, detail=f"pipeline epoch mismatch; worker has {self.epoch}")
        next_stage = self._next_stage()
        if self._can_merge(hidden, request):
            outcome = await self._step_together(hidden, request)
        else:
            # The lock covers this node's compute and nothing else. While one reply
            # is away at the next stage, this stage can work on another.
            async with self.lock:
                outcome = await self._step_alone(hidden, request)
        if isinstance(outcome, dict):
            return outcome
        hidden = outcome

        started = time.perf_counter()
        headers = auth_headers(self.pool_token)
        headers[META_HEADER] = encode_meta(request.model_dump())
        headers["content-type"] = "application/x-safetensors"
        # A batch of FP32 rows outgrows a cold TCP window and costs extra round
        # trips. Single tokens stay exact; llama.cpp stages turn BF16 back to FP32.
        if self.gguf is not None and hidden.shape[1] > 1 and self.dtype_name != "float32":
            hidden = hidden.to(torch.bfloat16)
        payload = encode_activation(hidden)
        self._record("encode", started)
        self.activation_bytes_sent += len(payload)
        started = time.perf_counter()
        try:
            response = await self.client.post(
                next_stage.url.rstrip("/") + "/internal/forward",
                content=payload,
                headers=headers,
                timeout=httpx.Timeout(pass_seconds(hidden.shape[1], self._stages_after()), connect=10.0),
            )
            if response.status_code == 503:
                detail = response.json().get("detail")
                if isinstance(detail, dict) and detail.get("retry"):
                    raise HTTPException(status_code=503, detail=detail)
            response.raise_for_status()
            self.downstream_ms_ema = self._ema(
                self.downstream_ms_ema,
                (time.perf_counter() - started) * 1000,
            )
            self._record("downstream", started)
            self.last_error = None
            result = response.json()
            self.schedule_checkpoint(request)
            return result
        except httpx.HTTPStatusError as exc:
            # The next stage is up and refused this request, so blame the request.
            self.last_error = str(exc)
            raise HTTPException(status_code=500, detail={"stage_error": str(exc)}) from exc
        except httpx.HTTPError as exc:
            self.last_error = str(exc)
            raise HTTPException(
                status_code=502,
                detail={"failed_worker": next_stage.worker_id, "message": str(exc)},
            ) from exc


runtime = WorkerRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.setup()
    runtime.registration_task = asyncio.create_task(runtime.registration_loop())
    try:
        yield
    finally:
        await runtime.close()


app = FastAPI(title="ShardMesh worker", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    BodyLimit,
    large={"/internal/forward": MAX_STAGE_BYTES, "/internal/diag/echo": MAX_STAGE_BYTES},
)


def authorized(authorization: str | None) -> bool:
    peer = PEER.get()
    if peer is not None:
        return peer not in runtime.revoked_nodes
    if runtime.requires_identity:
        return False  # the token alone would let a revoked node straight back in
    if runtime.pool_token:
        return hmac.compare_digest(authorization or "", f"Bearer {runtime.pool_token}")
    return not runtime.requires_identity


def verify_token(authorization: str | None) -> None:
    if not authorized(authorization):
        raise HTTPException(status_code=401, detail="a pool token or node certificate is required")


def verify_source(source: str) -> None:
    """A model on this machine must be somewhere this machine offers models from.

    Peers name the source, and anything under it can be served back to them.
    """
    if not (runtime.pool_token or runtime.requires_identity):
        return
    located = parse_source(source)
    folder = located[0] if located else source
    if not folder or not Path(folder).exists():
        return  # a hub repository, or a file held by another node
    offered = [*runtime.unit_cache.local_dirs, runtime.unit_cache.root]
    offered += [Path(item) for item in os.getenv("SHARDMESH_MODEL_DIRS", "").split(os.pathsep) if item]
    resolved = Path(folder).resolve()
    if not any(resolved == base.resolve() or base.resolve() in resolved.parents for base in offered):
        raise HTTPException(status_code=403, detail=f"{folder} is not a folder this node offers; see --model-dir")


@app.get("/health")
async def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorized(authorization):
        return {"status": "ok"}  # hardware and layout are nobody else's business
    assignment = runtime.assignment.model_dump() if runtime.assignment else None
    return {
        "status": "ok",
        "worker_id": runtime.worker_id,
        "code_version": __version__,
        "term": runtime.term,
        "model": runtime.model_id,
        "epoch": runtime.epoch,
        "assignment": assignment,
        "hardware": runtime.hardware.model_dump() if runtime.hardware else None,
        "sessions": len(runtime.caches),
        "roles": sorted(runtime.roles),
        "memory_store": runtime.memory_store.stats(),
        "memory_pools": runtime.pool_status(),
        "materialize": {"done": runtime.materialize_done, "total": runtime.materialize_total},
        "telemetry": {
            "inference_calls": runtime.inference_calls,
            "merged_passes": runtime.merged_passes,
            "merged_rows": runtime.merged_rows,
            "tokens_processed": runtime.tokens_processed,
            "activation_bytes_sent": runtime.activation_bytes_sent,
            "compute_ms_ema": runtime.compute_ms_ema,
            "downstream_ms_ema": runtime.downstream_ms_ema,
            "last_error": runtime.last_error,
            "kv_snapshots_sent": runtime.kv_snapshots_sent,
            "kv_snapshot_bytes_sent": runtime.kv_snapshot_bytes_sent,
            "kv_restores": runtime.kv_restores,
            "checkpoints_pending": runtime.checkpoints_pending,
            "stage_restarts": runtime.stage_restarts,
            "stages_kept": runtime.stages_kept,
            "draft_passes": runtime.draft_passes,
            "draft_tokens_per_pass": round(runtime.draft_accepted / runtime.draft_passes, 2) if runtime.draft_passes else None,
            "checkpoints_dropped": runtime.checkpoints_dropped,
            "phases_ms": runtime.phases,
        },
    }


@app.post("/internal/prepare")
async def prepare(body: PrepareRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    verify_source(body.source)
    return runtime.start_prepare(body).snapshot()


@app.get("/internal/prepare/{job_id}")
async def prepare_status(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    progress = runtime.prepare_jobs.get(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="unknown prepare job")
    return progress.snapshot()


@app.post("/internal/diag/echo")
async def diag_echo(request: Request, authorization: str | None = Header(default=None)) -> dict[str, int]:
    verify_token(authorization)
    return {"bytes": len(await request.body())}


@app.post("/internal/diag/probe")
async def diag_probe(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Time requests from this worker to another, to measure a link from this end."""
    verify_token(authorization)
    target = str(body["url"]).rstrip("/") + "/internal/diag/echo"
    results: dict[str, Any] = {}
    for size in body.get("sizes", [100, 6144, 236544]):
        payload = bytes(int(size))
        times: list[float] = []
        for _ in range(int(body.get("rounds", 10))):
            started = time.perf_counter()
            response = await runtime.client.post(target, content=payload, headers=auth_headers(runtime.pool_token))
            response.raise_for_status()
            times.append((time.perf_counter() - started) * 1000)
        times.sort()
        results[str(size)] = {"median_ms": round(times[len(times) // 2], 1), "min_ms": round(times[0], 1), "max_ms": round(times[-1], 1)}
    return results


@app.get("/internal/diag/runtime")
async def diag_runtime(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Report why this process might be slow, independent of any network."""
    verify_token(authorization)
    started = time.perf_counter()
    for _ in range(200):
        await asyncio.sleep(0)
    loop_step_us = (time.perf_counter() - started) / 200 * 1e6

    started = time.perf_counter()
    total = 0
    for value in range(1_000_000):
        total += value
    python_loop_ms = (time.perf_counter() - started) * 1000

    process = psutil.Process()
    before = {thread.id: thread.user_time + thread.system_time for thread in process.threads()}
    others = list(psutil.process_iter(["name"]))
    for other in others:
        try:
            other.cpu_percent(None)
        except psutil.Error:
            pass
    psutil.cpu_percent(None)
    await asyncio.sleep(1.0)
    busy = sorted(
        (
            (round((thread.user_time + thread.system_time - before.get(thread.id, 0)) * 100, 1), thread.id)
            for thread in process.threads()
        ),
        reverse=True,
    )
    consumers: dict[str, float] = {}
    for other in others:
        try:
            name = other.info["name"] or "?"
            consumers[name] = consumers.get(name, 0.0) + other.cpu_percent(None)
        except psutil.Error:
            pass
    frames = []
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        frames.append(f"{names.get(ident, ident)}: {Path(frame.f_code.co_filename).name}:{frame.f_lineno} {frame.f_code.co_name}")
    versions = {}
    for package in ("uvicorn", "httpx", "httpcore", "anyio", "starlette", "fastapi", "torch", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "source": str(Path(__file__).resolve()),
        "event_loop": type(asyncio.get_running_loop()).__name__,
        "switch_interval_ms": sys.getswitchinterval() * 1000,
        "loop_step_us": round(loop_step_us, 1),
        "python_1m_loop_ms": round(python_loop_ms, 1),
        "system_cpu_percent": psutil.cpu_percent(None),
        "cpu_count": psutil.cpu_count(),
        "process_threads": process.num_threads(),
        "busiest_threads_percent_of_core": busy[:6],
        "python_thread_frames": frames,
        "top_cpu_processes": sorted(((round(value, 1), name) for name, value in consumers.items()), reverse=True)[:8],
        "torch_threads": torch.get_num_threads(),
        "versions": versions,
        "env": {key: value for key, value in os.environ.items() if key.startswith(("OMP_", "MKL_", "HIP_", "HSA_", "PYTHON", "ROC"))},
    }


@app.get("/internal/shards/index")
async def shard_index(
    model_id: str,
    revision: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    held = await asyncio.to_thread(runtime.unit_cache.held, model_id, revision)
    return {"units": held, "serves_peers": runtime.peer_serve}


@app.get("/internal/shards/file")
async def shard_file(
    model_id: str,
    revision: str,
    unit: str,
    authorization: str | None = Header(default=None),
) -> FileResponse:
    verify_token(authorization)
    if not runtime.peer_serve:
        raise HTTPException(status_code=403, detail="this worker does not serve shards to peers")
    path = runtime.unit_cache.peer_file(model_id, revision, unit)
    if path is None:
        raise HTTPException(status_code=404, detail="unit not held here")
    return FileResponse(path, media_type="application/octet-stream")


@app.delete("/internal/prepare/{job_id}")
async def cancel_prepare(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_token(authorization)
    progress = runtime.prepare_jobs.get(job_id)
    if progress is not None:
        progress.cancelled = True
    return {"status": "cancelling"}


@app.get("/internal/gguf/index")
async def gguf_index(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Whole GGUF files on this machine, which other nodes may read."""
    verify_token(authorization)
    if not runtime.peer_serve:
        return {"files": []}
    files = await asyncio.to_thread(runtime.unit_cache.local_files)
    return {"files": [{"name": name, "size": path.stat().st_size} for name, path in files.items()]}


@app.get("/internal/gguf/range")
async def gguf_range(name: str, start: int, end: int, authorization: str | None = Header(default=None)) -> StreamingResponse:
    verify_token(authorization)
    # Only a name from the index resolves, so a request cannot reach other files.
    path = runtime.unit_cache.local_files().get(name) if runtime.peer_serve else None
    if path is None or not 0 <= start < end <= path.stat().st_size:
        raise HTTPException(status_code=404, detail="no such file or range")

    def read() -> Iterator[bytes]:
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start
            while remaining:
                chunk = handle.read(min(remaining, 4 * 1024 * 1024))
                if not chunk:
                    return
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(read(), media_type="application/octet-stream")


@app.post("/internal/drain")
async def drain(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, bool]:
    """Ahead of a pool change: wait for the reply in progress, and hold new ones."""
    verify_token(authorization)
    return {"drained": await runtime.hold_gate(float(body.get("seconds", 30)))}


@app.post("/internal/configure")
async def configure(body: ConfigureRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    if body.draft is not None:
        verify_source(body.draft.source)
    async with runtime.lock:
        try:
            await runtime.configure(body)
        except (ValueError, OSError, RuntimeError) as exc:
            # RuntimeError covers an accelerator running out of memory.
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            raise HTTPException(status_code=400, detail=runtime.last_error) from exc
    return {
        "status": "configured",
        "epoch": runtime.epoch,
        "assignment": body.assignment.model_dump(),
        "cache_url": runtime.cache_url,
    }


@app.delete("/internal/sessions/{session_id}")
async def clear_session(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_token(authorization)
    runtime.caches.pop(session_id, None)
    runtime.memory_store.clear_session(session_id)
    return {"status": "cleared"}


@app.put("/internal/cache/kv")
async def store_kv(
    request: Request,
    authorization: str | None = Header(default=None),
    x_shardmesh_meta: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    if "kv-cache" not in runtime.roles:
        raise HTTPException(status_code=409, detail="worker does not contribute a KV-cache service")
    if not x_shardmesh_meta:
        raise HTTPException(status_code=400, detail=f"missing {META_HEADER} header")
    try:
        metadata = decode_meta(x_shardmesh_meta)
        payload = await request.body()
        # Decode once at admission so malformed tensors never enter the store.
        decode_tensors(payload)
        key = (
            f"{metadata['session_id']}:{metadata['epoch']}:"
            f"{metadata['worker_id']}:{metadata.get('start_position', 0)}"
        )
        runtime.memory_store.put(key, payload, metadata)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "stored", **runtime.memory_store.stats()}


@app.get("/internal/cache/kv/{session_id}/{epoch}")
async def fetch_kv(
    session_id: str,
    epoch: int,
    start_layer: int,
    end_layer: int,
    authorization: str | None = Header(default=None),
) -> Response:
    verify_token(authorization)
    try:
        payload = runtime.memory_store.get_layers(session_id, epoch, start_layer, end_layer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=payload, media_type="application/x-safetensors")


@app.post("/internal/restore")
async def restore_kv(
    body: CacheRestoreRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    async with runtime.lock:
        try:
            await runtime.restore_cache(body)
        except (KeyError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "restored", "session_id": body.session_id, "epoch": body.target_epoch}


@app.post("/v1/infer")
async def infer(body: InferRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    if runtime.assignment is None or runtime.assignment.start_layer != 0:
        raise HTTPException(status_code=409, detail="this worker is not the entry stage")
    if not body.input_ids:
        raise HTTPException(status_code=400, detail="input_ids cannot be empty")
    try:
        arrived = time.perf_counter()
        async with runtime.lock:
            hidden = await asyncio.to_thread(runtime._prepare_entry, body)
        runtime._record("embed", arrived)
        result = await runtime.execute(hidden, body)
        runtime._record("handler", arrived)
        return result
    except HTTPException:
        raise
    except StageLost as exc:
        raise await runtime.replace_stage(exc) from exc
    except Exception as exc:
        runtime.last_error = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=500, detail=runtime.last_error) from exc


@app.post("/v1/generate")
async def generate(body: GenerateRequest, authorization: str | None = Header(default=None)) -> StreamingResponse:
    """Run the token loop here and stream ids out, one JSON object per line.

    Each token still crosses the stages, but no longer makes a round trip to the
    orchestrator. A failure ends the stream, and the orchestrator recovers by
    replaying through ``/v1/infer``.
    """
    verify_token(authorization)
    if runtime.assignment is None or runtime.assignment.start_layer != 0:
        raise HTTPException(status_code=409, detail="this worker is not the entry stage")
    if not body.input_ids:
        raise HTTPException(status_code=400, detail="input_ids cannot be empty")
    if runtime.waiting >= MAX_WAITING_REPLIES:
        raise HTTPException(status_code=429, detail="too many replies are waiting; try again shortly")
    stops = set(body.stop_token_ids)

    async def step(request: InferRequest) -> dict[str, Any]:
        arrived = time.perf_counter()
        async with runtime.lock:
            hidden = await asyncio.to_thread(runtime._prepare_entry, request)
        runtime._record("embed", arrived)
        result = await runtime.execute(hidden, request)
        runtime._record("handler", arrived)
        return result

    async def first_token(request: InferRequest, plan: PromptPlan) -> int:
        """Run what the stages have not seen, stopping to checkpoint if asked."""
        prompt, start = body.input_ids, plan.start
        where = {"reset": not start, "restore_length": start or None, "checkpoint_key": plan.key}
        if plan.mark is not None:
            await step(
                request.model_copy(
                    update={
                        **where,
                        "input_ids": prompt[start : plan.mark],
                        "save_checkpoint": plan.new_key,
                        "prefill_only": True,
                    }
                )
            )
            runtime.checkpoints[request.slot] = (plan.new_key, prompt[: plan.mark])
            start = plan.mark
            where = {"reset": False, "restore_length": None, "checkpoint_key": None}
        return (await step(request.model_copy(update={**where, "input_ids": prompt[start:]})))["token_id"]

    async def plain(request: InferRequest, plan: PromptPlan) -> AsyncIterator[int]:
        token = await first_token(request, plan)
        yield token
        for _ in range(body.max_tokens - 1):
            if token in stops:
                return
            request = request.model_copy(
                update={"input_ids": [token], "reset": False, "restore_length": None, "checkpoint_key": None}
            )
            token = (await step(request))["token_id"]
            yield token

    async def speculative(request: InferRequest, plan: PromptPlan) -> AsyncIterator[int]:
        """Guess ahead on the entry worker, and check every guess in one pass.

        The first token comes from the normal path, which also resets each stage.
        After that, ``position`` is where the newest token goes: it has not run
        yet, and every stage forgets anything at or past it before the next pass.
        """
        draft = runtime.draft
        assert draft is not None
        # The draft reads the prompt while the real model does.
        prepared = asyncio.create_task(asyncio.to_thread(draft.prepare, body.input_ids, plan, request.slot))
        token = await first_token(request, plan)
        yield token
        try:
            await prepared
        except LlamaError as exc:
            await runtime.replace_draft(exc)
            raise RuntimeError("the draft model was lost; replaying without it") from exc
        history = [*body.input_ids, token]
        position = len(body.input_ids)
        produced = 1
        ahead = min(INITIAL_GUESSES, runtime.speculate_tokens)
        while produced < body.max_tokens and token not in stops:
            room = body.max_tokens - produced
            started = time.perf_counter()
            try:
                guesses = await asyncio.to_thread(draft.guess, history, min(ahead, room), request.slot)
            except LlamaError as exc:
                await runtime.replace_draft(exc)
                raise RuntimeError("the draft model was lost; replaying without it") from exc
            runtime._record("draft", started)
            result = await step(
                request.model_copy(
                    update={
                        "input_ids": [token, *guesses],
                        "reset": False,
                        "restore_length": None,
                        "checkpoint_key": None,
                        "verify": True,
                        "rollback_to": position,
                    }
                )
            )
            accepted = result["token_ids"]
            runtime.draft_passes += 1
            runtime.draft_accepted += len(accepted)
            ahead = next_guess_count(ahead, len(accepted) - 1, runtime.speculate_tokens)
            for token in accepted[:room]:
                yield token
                produced += 1
                if token in stops:
                    return
            history.extend(accepted)
            position += len(accepted)

    async def tokens() -> AsyncIterator[bytes]:
        request = InferRequest(**body.model_dump(exclude={"max_tokens", "stop_token_ids", "checkpoint_at"}))
        runtime.waiting += 1
        try:
            slot = await runtime.gate.acquire(lambda free: runtime.choose_slot(free, body.input_ids))
        finally:
            runtime.waiting -= 1
        request = request.model_copy(update={"slot": slot})
        try:
            if body.epoch != runtime.epoch:
                yield json.dumps({"error": f"pipeline epoch mismatch; worker has {runtime.epoch}"}).encode() + b"\n"
                return
            held = runtime.checkpoints.get(slot) if runtime.gguf is not None else None
            plan = plan_prompt(held, body.input_ids, body.checkpoint_at)
            use_draft = runtime.draft is not None and runtime.speculate_tokens
            source = speculative(request, plan) if use_draft else plain(request, plan)
            try:
                async for token in source:
                    yield json.dumps({"token_id": token}).encode() + b"\n"
                return
            except HTTPException as exc:
                failure: Any = exc.detail
            except StageLost as exc:
                failure = (await runtime.replace_stage(exc)).detail
            except Exception as exc:
                runtime.last_error = f"{type(exc).__name__}: {exc}"
                failure = runtime.last_error
            # A stage may have lost the checkpoint, and a replay needs no help from it.
            runtime.checkpoints.pop(slot, None)
            yield json.dumps({"error": failure}).encode() + b"\n"
        finally:
            runtime.slot_used_at[slot] = time.monotonic()
            await runtime.gate.release(slot)

    return StreamingResponse(tokens(), media_type="application/x-ndjson")


@app.post("/internal/forward")
async def forward(
    request: Request,
    authorization: str | None = Header(default=None),
    x_shardmesh_meta: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    if not x_shardmesh_meta:
        raise HTTPException(status_code=400, detail=f"missing {META_HEADER} header")
    arrived = time.perf_counter()
    try:
        meta = InferRequest.model_validate(decode_meta(x_shardmesh_meta))
        hidden = decode_activation(await request.body())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runtime._record("receive", arrived)
    try:
        result = await runtime.execute(hidden, meta)
        runtime._record("handler", arrived)
        return result
    except HTTPException:
        raise
    except StageLost as exc:
        raise await runtime.replace_stage(exc) from exc
    except Exception as exc:
        runtime.last_error = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=500, detail=runtime.last_error) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ShardMesh compute worker")
    parser.add_argument("--host", default=os.getenv("SHARDMESH_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SHARDMESH_PORT", "8100")))
    parser.add_argument("--worker-id", default=os.getenv("SHARDMESH_WORKER_ID", socket.gethostname()))
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("SHARDMESH_ORCHESTRATOR_URL", "http://127.0.0.1:8000"),
        help="one or more control planes, comma separated; the rest are found from them",
    )
    parser.add_argument("--advertise-url", default=os.getenv("SHARDMESH_ADVERTISE_URL"))
    parser.add_argument("--device", default=os.getenv("SHARDMESH_DEVICE", "auto"))
    parser.add_argument("--roles", default=os.getenv("SHARDMESH_ROLES", "inference"))
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("SHARDMESH_CPU_THREADS", "0")))
    parser.add_argument("--ram-budget-gb", type=float, default=float(os.getenv("SHARDMESH_RAM_BUDGET_GB", "0")))
    parser.add_argument("--gpu-memory-gb", type=float, default=float(os.getenv("SHARDMESH_GPU_MEMORY_GB", "0")))
    parser.add_argument("--weights-ram-gb", default=None, help="system RAM for model weights, or 'auto'")
    parser.add_argument("--weights-disk-gb", type=float, default=0, help="disk for memory-mapped model weights")
    parser.add_argument("--cache-dir", default=os.getenv("SHARDMESH_CACHE_DIR"), help="where fetched weights live")
    parser.add_argument("--cache-budget-gb", type=float, default=0, help="evict old models past this size")
    parser.add_argument("--no-peer-fetch", action="store_true", help="always download from the model host")
    parser.add_argument("--no-peer-serve", action="store_true", help="do not send shards to other workers")
    parser.add_argument("--llama-dir", default=os.getenv("SHARDMESH_LLAMA_DIR"), help="a llama.cpp build for GGUF models")
    parser.add_argument("--open", action="store_true", help="serve a network with no credentials, trusting everyone on it")
    parser.add_argument("--gguf-dir", action="append", default=None, help="folder holding GGUF files; repeatable")
    parser.add_argument("--trusted-network", action="store_true", help="send pool credentials unencrypted, because nobody else can read this network")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float32", "float16"],
        default="auto",
        help="auto is bfloat16; float32 runs this node at full precision",
    )
    parser.add_argument("--standby", action="store_true", default=False)
    args = parser.parse_args()
    os.environ["SHARDMESH_WORKER_ID"] = args.worker_id
    os.environ["SHARDMESH_ORCHESTRATOR_URL"] = args.orchestrator_url
    os.environ["SHARDMESH_ADVERTISE_URL"] = args.advertise_url or f"http://127.0.0.1:{args.port}"
    os.environ["SHARDMESH_DEVICE"] = args.device
    os.environ["SHARDMESH_ROLES"] = args.roles
    os.environ["SHARDMESH_CPU_THREADS"] = str(args.cpu_threads)
    os.environ["SHARDMESH_RAM_BUDGET_GB"] = str(args.ram_budget_gb)
    os.environ["SHARDMESH_GPU_MEMORY_GB"] = str(args.gpu_memory_gb)
    os.environ["SHARDMESH_WEIGHTS_RAM_GB"] = "" if args.weights_ram_gb is None else str(args.weights_ram_gb)
    os.environ["SHARDMESH_WEIGHTS_DISK_GB"] = str(args.weights_disk_gb)
    os.environ["SHARDMESH_CACHE_DIR"] = args.cache_dir or ""
    os.environ["SHARDMESH_CACHE_BUDGET_GB"] = str(args.cache_budget_gb)
    os.environ["SHARDMESH_PEER_FETCH"] = "0" if args.no_peer_fetch else "1"
    os.environ["SHARDMESH_PEER_SERVE"] = "0" if args.no_peer_serve else "1"
    os.environ["SHARDMESH_LLAMA_DIR"] = args.llama_dir or ""
    check_exposure(args.host, load_token(), os.getenv("SHARDMESH_CA_FILE"), args.open, args.advertise_url, args.trusted_network)
    os.environ["SHARDMESH_GGUF_DIRS"] = os.pathsep.join(args.gguf_dir or [])
    os.environ["SHARDMESH_DTYPE"] = args.dtype
    os.environ["SHARDMESH_STANDBY"] = "1" if args.standby else "0"
    global runtime
    runtime = WorkerRuntime()
    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
