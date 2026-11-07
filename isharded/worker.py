from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import hmac
import importlib.metadata
import json
import os
import platform
import re
import secrets
import socket
import sys
import threading
import time
from urllib.parse import urlsplit
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Collection, Iterator
import contextlib
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from .inventory import inventory, ram
from .memory import (
    AUTO_RAM_SHARE,
    Accelerator,
    _synchronize,
    apply_accelerator_limits,
    build_pools,
    discover_accelerators,
    live_free_bytes,
)
from .engine import REFUSE_EPOCH, REFUSE_STAGE_ERROR, EngineClient, EngineRefusal
from .geo import country_code, node_country
from .models import Adapter, GgufAdapter, create_adapter
from .models.base import layer_unit
from .optional import NO_TORCH, has_torch, torch, torch_loaded
from .protocol import (
    host_model_patterns,
    hosts_model,
    pass_seconds,
    META_HEADER,
    PROTOCOL_VERSION,
    CacheRestoreRequest,
    ConfigureRequest,
    HardwareInfo,
    DraftModel as DraftSource,
    EmbedPass,
    GenerateRequest,
    ImagesMeasure,
    InferRequest,
    MemoryPool,
    PrepareRequest,
    ResourceContribution,
    StageAssignment,
    WorkerRegistration,
    auth_headers,
    decode_meta,
    decode_pass,
    decode_rows,
    decode_tensors,
    encode_activation,
    encode_meta,
    encode_rows,
    encode_tensors,
    needs_pytorch,
)
from .gguf import parse_source
from .gguf_stage import (
    GUESSED,
    INITIAL_GUESSES,
    PIPELINE_STRETCH,
    PIPELINED,
    PLAIN,
    DraftChoice,
    DraftModel,
    GgufStage,
    ImageEncoder,
    PromptPlan,
    draft_apart,
    next_guess_count,
    plan_prompt,
)
from .pipeline import Pipeline, RowClock
from . import __version__
from .llamacpp import PINNED_BUILD, LlamaError, StageLost, ensure_build, pick_build, rocm_library_paths, speculative_choice, step_draws, step_random
from .pool import check_exposure, load_token
from .serving import MAX_STAGE_BYTES, MAX_STATE_BYTES, BodyLimit, serve, where_everything_is
from .slots import SlotGate
from .statecopy import CopySender, fetch_file
from .statestore import HISTORY, PARTIAL, PrefixIndex, StateStore, clear_other_stores, prompt_parts, stage_namespace
from .access import load_authority
from .addresses import ECHO_PATH, NodeProof, claim, direct_client, host_port, prove, reach_options, reach_urls
from .tickets import TicketIdentity, client_auth
from .tls import client_verify, node_proven, node_transport, PEER, stay_on_this_machine_async
from .vision import PromptPart, PromptRows, lay_out
from .weights import FetchProgress, LoadedUnit, UnitCache, WeightSourceError, gguf_reader, has_model_host, load_unit, map_unit

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "isharded" / "units"
SUPPORTED_DTYPES = ["bfloat16", "float32", "float16"]
# Snapshots waiting to send. Past this the store is too slow to keep up, and
# recovery falls back to replaying the prompt.
MAX_PENDING_CHECKPOINTS = 32
# Times a reply plans again after a stage turns out not to hold a state.
PLAN_ATTEMPTS = 3
# Written tokens a resuming reply keeps on the ring at once. A pass is one row, so this
# fills any route without crowding out other replies for long.
RESUME_WINDOW = 16
# Where a node keeps copies of other stages' states, beside its own store.
HELD_FOLDER = "held"
# After a route change, how long a reply waits to hear what the stages can restore.
INDEX_SECONDS = 3.0
PEER_SECONDS = 2.0
# How long a reply waits for a stage to fetch a copy, before reading the prompt instead.
FETCH_SECONDS = 30.0
NAME = re.compile(r"[0-9a-f]{1,64}")


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


def env_gib(name: str, default: str = "0") -> int:
    """Bytes, from a size in GiB in the environment."""
    return int(float(os.getenv(name, default)) * 1024**3)


def parse_ram_budget(value: str) -> int | None:
    """Read a RAM budget: unset, ``auto``, or a size in GiB."""
    if not value:
        return None
    if value == "auto":
        return int(ram().available * AUTO_RAM_SHARE)
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
        ram_total_bytes=ram().total,
        accelerator_name=" + ".join(item.name for item in accelerators) or None,
        accelerator_memory_bytes=accelerator_memory,
        pytorch=has_torch(),
        contribution=ResourceContribution(
            roles=sorted(roles),
            cpu_threads=cpu_threads,
            ram_budget_bytes=ram_budget_bytes,
            accelerator_memory_budget_bytes=min(
                accelerator_memory_budget_bytes or accelerator_memory,
                accelerator_memory,
            ),
            standby=standby,
            host_models=host_model_patterns(),
        ),
        memory_pools=pools,
    )


# Past this, a new reply is refused rather than left waiting indefinitely.
MAX_WAITING_REPLIES = 32
# A coordinator that asked for a drain and then died must not block replies forever.
DRAIN_LIMIT_SECONDS = 180.0
# A control plane this worker only heard about, and that stays silent, is forgotten.
CONTROL_LINK_FAILURES = 60
CHECK_IN_SECONDS = 5.0
# How long a link may go without an acknowledgement, and how long a stage keeps dialling.
RING_ACK_MS = 5000
# A stage that loses a link says so inside the limit above, and an answer the closing link
# held comes back by walking soon after. This leaves room for that, however fast the ring.
RING_ANSWER_FLOOR_SECONDS = 2 * RING_ACK_MS / 1000
# How often a wait on the ring looks up to see whether it, or the stage under it, is still there.
SETTLE_LOOK_SECONDS = 0.25
RING_DIAL_MS = 10_000
# A silent link answers nothing, so a shorter address is checked while it is in use. It
# gets as long without an answer as a ring link gets without an acknowledgement.
HOP_WATCH_SECONDS = 1.0
HOP_SILENCE_SECONDS = RING_ACK_MS / 1000
# Longest one address may take to prove itself and be timed.
ADDRESS_PROBE_SECONDS = 6.0
# What a shorter address can do wrong. A certificate failure arrives as a plain OSError,
# and whoever has the address may answer with anything.
LINK_FAILURES = (httpx.HTTPError, OSError, ValueError)
# How a kept connection fails when the far side closed it, or a router forgot it. The stage
# behind it may be fine, unlike one that will not connect or does not answer.
BROKEN_CONNECTION = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)
# Off is for checking results against a prompt that went out whole.
PROMPT_PARTS = os.getenv("ISHARDED_PROMPT_PARTS", "1") != "0"
UNAVAILABLE_STATUSES = {502, 503, 504}
# What every pass of a reply after its first has in common.
GOING_ON = {"reset": False, "restore_length": None, "checkpoint_key": None}


def retry_error(message: str) -> HTTPException:
    """Asks for a replay, and says nothing against any worker."""
    return HTTPException(status_code=503, detail={"retry": True, "message": message})


def epoch_error(epoch: int) -> HTTPException:
    return HTTPException(status_code=409, detail=f"pipeline epoch mismatch; worker has {epoch}")


@dataclass(slots=True, eq=False)
class NextHop:
    """A proven shorter way to the next stage. Its advertised address stays as the way back."""

    worker_id: str
    via: str
    url: str
    client: httpx.AsyncClient
    lost: asyncio.Event = field(default_factory=asyncio.Event)
    watch: asyncio.Task[None] | None = None
    retired: bool = False
    passes: int = 0  # on their way now

    @property
    def options(self) -> dict[str, Any]:
        return reach_options(self.via)


def free_memory() -> None:
    gc.collect()
    if torch_loaded() and torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    # This reply's passes left the Python path, so the later stages' own counts are stale.
    ring: bool = False


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

    def get_layers(self, session_id: str, epoch: int, start_layer: int, end_layer: int) -> bytes:
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
    hop: NextHop | None = None
    hop_fallbacks = 0

    def __init__(self) -> None:
        self.worker_id = os.getenv("ISHARDED_WORKER_ID", socket.gethostname())
        self.host_id = os.getenv("ISHARDED_HOST_ID", socket.gethostname())
        # Any control plane will do to get in, and each one answers with the rest.
        configured = os.getenv("ISHARDED_ORCHESTRATOR_URL", "http://127.0.0.1:8000")
        self.seed_urls = {url.strip().rstrip("/") for url in configured.split(",") if url.strip()}
        self.control_urls: dict[str, ControlLink] = {url: ControlLink() for url in self.seed_urls}
        self.term = 0
        self.advertised_url = os.getenv("ISHARDED_ADVERTISE_URL", "http://127.0.0.1:8100").rstrip("/")
        self.draft_choice = DraftChoice()
        # The stage that would not restart, for as long as it is still this worker's stage.
        self.stage_down: Any = None
        self.row_clock = RowClock()
        self.ring_live = False
        self.ring_epoch: int | None = None
        self.ring_door: dict[str, Any] | None = None
        self.ring_client: EngineClient | None = None
        self.ring_guard: asyncio.Task[None] | None = None
        self.country = node_country(self.advertised_url, os.getenv("ISHARDED_LOCATION"))
        self.pool_token = load_token()
        self.requires_identity = bool(os.getenv("ISHARDED_CA_FILE"))
        self.authority = load_authority()
        self.revoked_nodes: set[str] = set()
        self.roles = {role.strip() for role in os.getenv("ISHARDED_ROLES", "inference").split(",") if role.strip()}
        self.cpu_threads = max(0, int(os.getenv("ISHARDED_CPU_THREADS", "0")))
        self.ram_budget_bytes = max(0, env_gib("ISHARDED_RAM_BUDGET_GB"))
        self.accelerator_memory_budget_bytes = max(0, env_gib("ISHARDED_GPU_MEMORY_GB"))
        self.weights_ram_bytes = parse_ram_budget(os.getenv("ISHARDED_WEIGHTS_RAM_GB", ""))
        self.weights_disk_bytes = env_gib("ISHARDED_WEIGHTS_DISK_GB")
        self.standby = os.getenv("ISHARDED_STANDBY", "0").lower() in {"1", "true", "yes"}
        self.host_models = host_model_patterns()
        self.requested_device = os.getenv("ISHARDED_DEVICE", "auto")
        self.dtype_name = choose_dtype_name(os.getenv("ISHARDED_DTYPE", "auto"))
        self.cache_dir = Path(os.getenv("ISHARDED_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.unit_cache = UnitCache(
            self.cache_dir,
            budget_bytes=env_gib("ISHARDED_CACHE_BUDGET_GB"),
            concurrency=int(os.getenv("ISHARDED_FETCH_CONCURRENCY", "4")),
            local_dirs=tuple(Path(item) for item in os.getenv("ISHARDED_GGUF_DIRS", "").split(os.pathsep) if item),
        )
        self.peer_fetch = os.getenv("ISHARDED_PEER_FETCH", "1") != "0"
        self.peer_serve = os.getenv("ISHARDED_PEER_SERVE", "1") != "0"
        self.hardware: HardwareInfo | None = None
        self.pools: dict[str, MemoryPool] = {}
        self.llama_dir = os.getenv("ISHARDED_LLAMA_DIR") or None
        self.gguf: GgufStage | None = None
        self.gguf_logits: torch.Tensor | None = None
        self.merge_passes = False
        self.replacing = asyncio.Lock()
        self.merge_queue: list[MergedStep] = []
        self.merge_task: asyncio.Task[None] | None = None
        self.merged_passes = 0
        self.merged_rows = 0
        self.gguf_choices: list[int] = []
        self.draft: DraftModel | None = None
        self.encoder: ImageEncoder | None = None
        self.speculate_tokens = 0
        self.prompt_lookup = False
        self.rewinds = False
        self.draft_passes = 0
        self.draft_accepted = 0
        self.pipes: dict[int, Pipeline] = {}
        # The newest part of a prompt each slot has run, and word of the next one.
        self.parts_run: dict[int, tuple[str, int]] = {}
        self.part_ran = asyncio.Condition()
        self.pipeline_tokens = 0
        self.pipeline_misses = 0
        self.pipeline_lookups = 0
        self.pipeline_depth = 0
        self.stage_restarts = 0
        self.draft_request: tuple[ConfigureRequest, str, int] | None = None
        # One conversation at a time per stage process. Requests from any node
        # queue here, and a pool change waits here for the reply in progress.
        self.gate = SlotGate(1)
        self.waiting = 0
        self.drain_timer: asyncio.TimerHandle | None = None
        # What the stages can restore, on an entry worker. Any sequence can take any state.
        self.prefix = PrefixIndex()
        self.prompt_cache = True
        self.state_ram_bytes = env_gib("ISHARDED_STATE_RAM_GB", "1")
        self.state_disk_bytes = env_gib("ISHARDED_STATE_DISK_GB", "8")
        self.prompt_tokens_computed = 0
        self.tokens_reused: dict[str, int] = {}
        self.state_misses = 0
        # Copies this node keeps for stages elsewhere, opened in setup().
        self.held = StateStore(None, 0, 0)
        self.namespace: str | None = None
        self.state_peers: list[str] = []
        self.fetched: set[str] = set()
        self.copies_fetched = 0
        self.index_epoch = -1
        self.index_lock = asyncio.Lock()
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
        self.client = httpx.AsyncClient(transport=node_transport(), event_hooks={"request": [stay_on_this_machine_async]}, timeout=httpx.Timeout(1800.0, connect=10.0), verify=client_verify(), auth=client_auth())
        rate = int(float(os.getenv("ISHARDED_STATE_COPY_MB", "10")) * 1024**2)
        self.copies = CopySender(self.client, lambda: auth_headers(self.pool_token), rate)
        self.reach_urls = [url for url in os.getenv("ISHARDED_REACH_URLS", "").split(",") if url]
        self.proof = NodeProof(self.pool_token, getattr(self.client.auth, "maker", None))
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
        self.held = StateStore(self.state_root / HELD_FOLDER, 0, self.state_disk_bytes)
        if self.cpu_threads and has_torch():
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
        if self.hop is not None:
            self._forget_hop(self.hop)
        await self.client.aclose()

    def registration(self) -> WorkerRegistration:
        assert self.hardware is not None
        return WorkerRegistration(
            worker_id=self.worker_id,
            advertised_url=self.advertised_url,
            hardware=self.hardware,
            protocol_version=PROTOCOL_VERSION,
            code_version=__version__,
            country=self.country,
            addresses=self.reach_urls,
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
                self._take_revocations(answer)
                for other in answer.get("controls", []):
                    self.control_urls.setdefault(str(other).rstrip("/"), ControlLink())
        except (httpx.HTTPError, OSError, ValueError):
            link.registered = False
            link.failures += 1
            if link.failures >= CONTROL_LINK_FAILURES and url not in self.seed_urls:
                self.control_urls.pop(url, None)

    def _take_revocations(self, answer: dict[str, Any]) -> None:
        """Revocations only ever add up. With an authority, only signed ones count.

        Any control plane answers this, so one must not be able to clear the
        list, or to lock its peers out by naming them.
        """
        if self.authority is None:
            self.revoked_nodes |= set(answer.get("revoked_nodes", []))
            return
        for signed in answer.get("signed_revocations", []):
            entry = self.authority.verified(signed)
            if entry is not None and entry.get("type") == "revoke":
                self.revoked_nodes.add(str(entry.get("name")))

    async def registration_loop(self) -> None:
        """Stay known to every control plane, so a new coordinator needs no introduction.

        Each link keeps its own time. Waiting for all of them let one flaky link hold
        up every check-in, until this worker looked dead even to the node beside it.
        """
        running: dict[str, asyncio.Task[None]] = {}
        while True:
            for url, link in list(self.control_urls.items()):
                if url not in running or running[url].done():
                    running[url] = asyncio.create_task(self._check_in(url, link))
            await asyncio.sleep(CHECK_IN_SECONDS)

    def pool_status(self) -> list[dict[str, Any]]:
        """Live view of each pool, for the planner and the dashboard."""
        return [
            {
                **pool.model_dump(),
                "free_bytes": live_free_bytes(pool, self.cache_dir),
                "used_bytes": self.pool_used.get(pool.pool_id, 0),
            }
            for pool in self.pools.values()
        ]

    def start_prepare(self, request: PrepareRequest) -> FetchProgress:
        """Begin downloading in the background. Serving is not interrupted."""
        running = self.prepare_jobs.get(request.job_id)
        if running is not None:
            return running  # asked twice, as a retry after a lost reply would
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
                request.file_sources if self.peer_fetch or not has_model_host(request.source) else None,
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
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None
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
        self.draft_request = None  # a stage restart must not bring back the last model's draft
        if request.draft is None or request.assignment.start_layer != 0:
            return
        self.draft_request = (request, folder, threads)
        try:
            path = self._draft_file(request.draft)
            beside = None if self.gguf is None or draft_apart() else self.gguf.host
            self.draft = DraftModel(folder, rocm_library_paths(), path, request.context_tokens, threads, request.slots, beside)
        except (LlamaError, WeightSourceError, httpx.HTTPError, OSError, ValueError) as exc:
            self.last_error = f"draft model unavailable, running without it: {exc}"

    def _load_projector(self, request: ConfigureRequest, adapter: GgufAdapter, folder: str, threads: int) -> None:
        """Unlike a draft, a projector that fails fails the load: this model was asked to see."""
        if request.projector is None or request.assignment.start_layer != 0:
            return
        assert self.gguf is not None
        try:
            path = self._draft_file(request.projector)
        except (WeightSourceError, httpx.HTTPError) as exc:
            raise RuntimeError(f"could not fetch the projector {request.projector.file}: {exc}") from exc
        self.encoder = ImageEncoder(
            folder, rocm_library_paths(), self.gguf.path, path, adapter.header.hidden_size, threads, self.gguf.uses_gpu
        )

    async def prompt_rows(self, tokens: list[int], images: list[str]) -> PromptRows:
        """Encode a prompt's images and work out every row of it."""
        if self.encoder is None:
            raise HTTPException(status_code=409, detail="this worker has no projector")
        encoder = self.encoder
        encoded = [await asyncio.to_thread(encoder.encode, base64.b64decode(item)) for item in images]
        return lay_out(tokens, encoded)

    async def replace_stage(self, lost: StageLost) -> HTTPException:
        """Start a new stage process in place of a dead one.

        Only this process failed, so the node keeps its place: the answer asks
        for a replay and says nothing against the worker. A stage that will not
        restart is a real failure, and reads as one.
        """
        retry = retry_error(str(lost))
        # Several replies can find the same stage dead. Starting a second copy would
        # kill the fresh one, and two loading at once can run a small GPU out of memory.
        async with self.replacing:
            if self.gguf is not None and lost.host is not None and lost.host is not self.gguf.host:
                return retry
            self.last_error = str(lost)
            self.caches.clear()
            self.ring_stop()  # its links died with the process, and this says so to whoever reopens them
            draft = self.draft
            try:
                assert self.gguf is not None
                await asyncio.to_thread(self.gguf.respawn)
            except (LlamaError, OSError, AssertionError) as exc:
                self.last_error = f"stage would not restart: {exc}"
                self.stage_down = self.gguf
                # Not a 500, which reads as a request the stage refused: that ends the reply
                # and blames nobody. This node has lost its stage, so it goes and the reply goes on.
                return HTTPException(status_code=502, detail={"failed_worker": self.worker_id, "message": self.last_error})
            self.stage_down = None
            self.stage_restarts += 1
        if self.draft_request is not None and not draft_apart():
            await self.replace_draft(lost, draft)  # it lived in that process
        return retry

    async def replace_draft(self, reason: Exception, failed: DraftModel | None = None) -> RuntimeError:
        """A draft is only an accelerator: lose it, try for another, and replay without it.

        Every reply on a dead draft ends up here, and only the first should load a new one.
        A draft that shares a stage's process holds memory there until it is closed.
        """
        lost = RuntimeError("the draft model was lost; replaying without it")
        async with self.replacing:
            if failed is not None and failed is not self.draft:
                return lost
            self.last_error = f"draft model lost: {reason}"
            draft, self.draft = self.draft, None
            if draft is not None:
                await asyncio.to_thread(draft.close)
            if self.draft_request is not None:
                await asyncio.to_thread(self._load_draft, *self.draft_request)
        return lost

    async def draft_failed(self, reason: LlamaError, draft: DraftModel) -> Exception:
        """What a reply raises for a draft that failed.

        A draft in the stage's process often hears first when that process dies. That is a
        lost stage, which asks for a replay, and not a lost draft, which ends the stream.
        """
        shared = self.gguf is not None and draft.host.process is self.gguf.host.process
        if isinstance(reason, StageLost) and shared and draft is self.draft:
            return StageLost(str(reason), self.gguf.host)
        return await self.replace_draft(reason, draft)

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
            tuple(sorted(request.unit_hashes.items())),
            request.embedding,
            request.projector.file if request.projector and stage.start_layer == 0 else None,
            request.prompt_cache,
            request.kv_type,
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

    def choose_slot(self, free: set[int]) -> int:
        """The sequence idle longest. A saved state can go on any of them."""
        return min(free, key=lambda slot: self.slot_used_at.get(slot, 0.0))

    @property
    def state_root(self) -> Path:
        """This worker's own, because workers on a node share a cache and clear what is not theirs."""
        return self.cache_dir / "states" / re.sub(r"[^A-Za-z0-9._-]", "_", self.worker_id)

    @staticmethod
    def state_namespace(request: ConfigureRequest) -> str:
        """Everything that decides what a stage's saved state means."""
        stage = request.assignment
        return stage_namespace(
            PINNED_BUILD, request.model_id, request.revision, stage.start_layer, stage.end_layer, request.speculate_tokens, request.kv_type
        )

    def cache_report(self) -> dict[str, Any]:
        """What the cache saved and holds. ``tokens_reused`` says where each came from."""
        held = self.gguf.states if self.gguf is not None else {}
        return {
            "prompt_tokens_computed": self.prompt_tokens_computed,
            "tokens_reused": dict(self.tokens_reused),
            "state_misses": self.state_misses,
            "states": held.get("states", 0),
            "ram_bytes": held.get("ram_bytes", 0),
            "disk_bytes": held.get("disk_bytes", 0),
            "saved_bytes": held.get("saved_bytes", 0),
            "evicted": held.get("evicted", 0),
            "held_bytes": self.held.stats()["disk_bytes"],
            "copies_fetched": self.copies_fetched,
            **self.copies.report(),
        }

    async def restorable(self) -> dict[str, dict[str, Any]]:
        """What this stage can restore: what it holds, then what others keep for these layers."""
        if self.gguf is None or self.namespace is None:
            return {}
        try:
            answer, _ = await asyncio.to_thread(self.gguf.host.call, "state_list")
        except LlamaError:
            return {}
        found = {item["key"]: {**item, "elsewhere": False} for item in answer["states"]}

        async def ask(url: str) -> list[dict[str, Any]]:
            try:
                response = await self.client.get(
                    f"{url}/internal/state/held/{self.namespace}", headers=auth_headers(self.pool_token), timeout=PEER_SECONDS
                )
                response.raise_for_status()
                return response.json()["states"]
            except (httpx.HTTPError, ValueError, KeyError):
                return []  # gone or silent, and this reply is not going to wait for it

        for listing in await asyncio.gather(*(ask(url) for url in self.state_peers)):
            for item in listing:
                found.setdefault(item["key"], {**item, "elsewhere": True})
        return found

    async def sync_index(self) -> None:
        """Learn what every stage can restore, once per route.

        An entry worker new to this route knows nothing, and even one that stayed
        cannot know what came back or moved. Only what all stages can reach is any use.
        """
        if self.index_epoch == self.epoch or self.gguf is None or not self.prompt_cache:
            return
        async with self.index_lock:
            epoch = self.epoch
            if self.index_epoch == epoch:
                return

            async def listing(stage: StageAssignment) -> dict[str, tuple[int, str, bool]]:
                try:
                    if stage.worker_id == self.worker_id:
                        items = list((await asyncio.wait_for(self.restorable(), INDEX_SECONDS)).values())
                    else:
                        response = await self.client.get(
                            stage.url.rstrip("/") + "/internal/state/index", headers=auth_headers(self.pool_token), timeout=INDEX_SECONDS
                        )
                        response.raise_for_status()
                        items = response.json()["states"]
                    return {str(item["key"]): (int(item["length"]), str(item["kind"]), bool(item.get("elsewhere"))) for item in items}
                except (httpx.HTTPError, ValueError, KeyError, TypeError, asyncio.TimeoutError):
                    return {}

            listings = await asyncio.gather(*(listing(stage) for stage in self.route))
            for key in set.intersection(*(set(item) for item in listings)) if listings else ():
                length, kind, _ = listings[0][key]
                self.prefix.add(key, length, kind, any(found[key][2] for found in listings))
            self.index_epoch = epoch

    async def ensure_state(self, key: str) -> bool:
        """Have this state here, fetching a copy if this stage took over from a lost one."""
        if self.gguf is None or self.namespace is None:
            return False
        try:
            answer, _ = await asyncio.to_thread(self.gguf.host.call, "state_has", key=key)
            if answer["held"]:
                return True
            target = self.state_root / self.namespace / (key + PARTIAL)
            for url in self.state_peers:
                source = f"{url}/internal/state/held/{self.namespace}/{key}"
                if await fetch_file(self.client, source, target, auth_headers(self.pool_token)):
                    answer, _ = await asyncio.to_thread(self.gguf.host.call, "state_adopt", key=key)
                    if answer["adopted"]:
                        self.fetched.add(key)
                        self.copies_fetched += 1
                        return True
        except (LlamaError, OSError):
            pass  # a stage that cannot answer has nothing to restore, and a reread needs no help
        return False

    async def prepare_state(self, key: str) -> bool:
        """Get a state onto every stage before a reply counts on it.

        This runs before the pass, so a slow fetch never holds a stage's lock. A
        stage that cannot get it in time means reading the prompt instead.
        """

        async def ready(stage: StageAssignment) -> bool:
            if stage.worker_id == self.worker_id:
                return await self.ensure_state(key)
            try:
                response = await self.client.post(
                    stage.url.rstrip("/") + "/internal/state/prepare",
                    json={"key": key},
                    headers=auth_headers(self.pool_token),
                    timeout=httpx.Timeout(FETCH_SECONDS, connect=PEER_SECONDS),
                )
                response.raise_for_status()
                return bool(response.json()["ready"])
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                return False

        try:
            return all(await asyncio.wait_for(asyncio.gather(*(ready(stage) for stage in self.route)), FETCH_SECONDS))
        except asyncio.TimeoutError:
            return False

    def _materialize_gguf(self, request: ConfigureRequest, adapter: GgufAdapter) -> None:
        """Assemble this stage's file from cached units and load it into llama.cpp."""
        self._drop_units()
        placements = [item for item in request.assignment.placements if item.bytes]
        self.materialize_total = len(placements)
        self.materialize_done = 0
        on_vulkan = any(pool.device.startswith("vulkan") for pool in self.pools.values())
        folder = self.llama_dir or str(ensure_build(pick_build("vulkan" if on_vulkan else self.requested_device)))
        threads = self.cpu_threads or max((os.cpu_count() or 4) // 2, 1)
        # A state means nothing to another model or other layers, so those go now.
        namespace = self.state_namespace(request)
        clear_other_stores(self.state_root, {namespace, HELD_FOLDER})
        caching = request.prompt_cache and not request.embedding
        self.gguf = GgufStage(
            folder, rocm_library_paths(), self.unit_cache, request, adapter, self.pools, threads,
            self.state_root / namespace if caching else None,
            self.state_ram_bytes if caching else 0,
            self.state_disk_bytes if caching else 0,
            self.dtype_name != "float32",
        )
        self._load_draft(request, folder, threads)
        self._load_projector(request, adapter, folder, threads)
        used: dict[str, int] = {}
        for item in placements:
            used[item.pool_id] = used.get(item.pool_id, 0) + item.bytes
        self.pool_used = used
        self.materialize_done = len(placements)

    def _drop_units(self, keep: Collection[str] = ()) -> None:
        for key in [key for key in self.loaded if key not in keep]:
            self.loaded.pop(key).release()
        self.unit_weights = {}

    def _load_unit(self, request: ConfigureRequest, adapter: Adapter, item: Any) -> LoadedUnit:
        pool = self.pools.get(item.pool_id)
        if pool is None:
            raise ValueError(f"this worker has no memory pool {item.pool_id!r}")
        path = self.unit_cache.path(request.model_id, request.revision, item.unit)
        if not path.is_file():
            raise ValueError(f"unit {item.unit} was not fetched before configure")
        if pool.kind == "disk":
            return map_unit(path)
        return load_unit(path, pool.device, choose_dtype(pool.dtype), adapter.keep_float32, compact=item.compact)

    def _materialize(self, request: ConfigureRequest, adapter: Adapter) -> None:
        """Move fetched units into their pools, keeping what is already there."""
        signature = self._stage_signature(request)
        if isinstance(adapter, GgufAdapter) and self.gguf is not None and signature == self.gguf_signature:
            # Nothing about this stage changed, so its weights stay where they are.
            self.stages_kept += 1
            return
        self._close_gguf()
        if isinstance(adapter, GgufAdapter):
            self._materialize_gguf(request, adapter)
            self.gguf_signature = signature
            return
        prefix = f"{request.model_id}@{request.revision}"
        placements = request.assignment.placements
        wanted = {
            f"{prefix}/{item.unit}/{item.pool_id}/{item.compact}": item for item in placements if item.bytes
        }
        self._drop_units(keep=wanted)
        free_memory()

        self.materialize_total = len(wanted)
        self.materialize_done = 0
        used: dict[str, int] = {}
        for key, item in wanted.items():
            if key not in self.loaded:
                self.loaded[key] = self._load_unit(request, adapter, item)
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
        self.prefix.clear()
        self._drop_units()
        self.pool_used = {}
        self.adapter = None
        self.model_id = None
        self.assignment = None
        self.route = []
        self.caches.clear()
        free_memory()

    async def configure(self, request: ConfigureRequest) -> None:
        if "inference" not in self.roles:
            raise ValueError("this worker does not contribute an inference executor")
        if not hosts_model(self.host_models, request.model_id):
            raise ValueError(f"this node's owner does not host {request.model_id}")
        if request.assignment.worker_id != self.worker_id:
            raise ValueError("assignment targets a different worker")
        if request.term < self.term:
            raise ValueError(f"order from term {request.term}, but term {self.term} already took over")
        self.term = request.term
        self.ring_stop()
        self.draft_choice = DraftChoice()  # a new route can change which way is faster
        self.row_clock = RowClock()
        adapter = create_adapter(request.architecture)
        if not isinstance(adapter, GgufAdapter) and not has_torch():
            raise ValueError(NO_TORCH)
        if not (0 <= request.assignment.start_layer < request.assignment.end_layer <= adapter.num_layers):
            raise ValueError("invalid layer interval")
        try:
            await asyncio.to_thread(self._materialize, request, adapter)
        except Exception:
            self._release_weights()
            raise
        self.adapter = adapter
        if request.model_id != self.model_id:
            # The planner trusts these counts to say how this model's guesses go.
            self.pipeline_tokens = self.pipeline_misses = 0
        self.model_id = request.model_id
        self.epoch = request.epoch
        self.assignment = request.assignment
        self.route = request.route
        self.rewinds = request.rewinds  # here, because a kept stage loads nothing
        self.prompt_lookup = request.prompt_lookup
        self.cache_url = request.cache_url.rstrip("/") if request.cache_url else None
        self.caches.clear()
        self.prefix.clear()  # the stages behind this one may have changed
        self.prompt_cache = request.prompt_cache
        self.namespace = self.state_namespace(request) if self.gguf is not None and request.prompt_cache and not request.embedding else None
        self.state_peers = [url.rstrip("/") for url in request.state_peers]
        folder = None if self.namespace is None or not self.state_disk_bytes else self.state_root / self.namespace
        self.copies.aim(request.copy_url, folder, self.namespace or "")
        self.merge_passes = request.merge_passes
        if self.gate.size != request.slots:
            self.gate.resize(request.slots)
        await self.release_gate()

    def ring_stop(self) -> None:
        """Leave the ring. Replies go on over HTTP, or replay if one was using it."""
        self.ring_live = False
        self.ring_epoch = None
        self.ring_door = None
        if self.ring_client is not None:
            self.ring_client.close()
            self.ring_client = None

    async def ring_open(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.gguf is None or body["epoch"] != self.epoch:
            raise HTTPException(status_code=409, detail="this worker is not on that route")
        self.ring_stop()
        answer = await asyncio.to_thread(
            self.gguf.ring_call,
            "ring_open",
            host=os.getenv("ISHARDED_HOST", "0.0.0.0"),
            port=int(os.getenv("ISHARDED_DATA_PORT", "0")),
            epoch=body["epoch"],
            index=body["index"],
            count=body["count"],
            key=body["key"],
            ack_timeout_ms=body.get("ack_timeout_ms", RING_ACK_MS),
            silence_timeout_ms=body.get("silence_timeout_ms", RING_ACK_MS),
            merge_passes=self.merge_passes,
        )
        if not answer.get("engine"):
            return {"engine": False}
        self.ring_door = answer
        self.ring_epoch = body["epoch"]
        host = urlsplit(self.advertised_url).hostname or "127.0.0.1"
        return {"engine": True, "address": host_port(host, answer["port"]), "port": answer["port"], "fingerprint": answer["fingerprint"]}

    async def ring_connect(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.gguf is None or self.ring_door is None or body["epoch"] != self.ring_epoch:
            raise HTTPException(status_code=409, detail="this worker has not opened that ring")
        links = {"next": body.get("next"), "entry": body.get("entry"), "timeout_ms": RING_DIAL_MS}
        await asyncio.to_thread(self.gguf.ring_call, "ring_connect", links=links)
        give_up = time.monotonic() + RING_DIAL_MS / 500
        state = "dialling"
        while state == "dialling" and time.monotonic() < give_up:
            await asyncio.sleep(0.02)
            stats = await asyncio.to_thread(self.gguf.ring_stats)
            state = (stats or {}).get("connect", state)
        if state != "up":
            raise HTTPException(status_code=502, detail=f"could not reach the next stage: {state}")
        if self.assignment is not None and self.assignment.start_layer == 0:
            self.ring_client = await EngineClient.open(self.ring_door["local_port"], bytes(self.ring_door["local_key"]))
            self.ring_guard = asyncio.create_task(self._close_with_stage(self.ring_client, self.gguf.host.stopped))
        return {"connected": True}

    async def _close_with_stage(self, client: EngineClient, stopped: threading.Event) -> None:
        """A process that takes no kill keeps the local door open, so nothing would tell a reply."""
        while client.alive and not stopped.is_set():
            await asyncio.sleep(SETTLE_LOOK_SECONDS)
        client.close()

    def ring_go(self, body: dict[str, Any]) -> None:
        if body["epoch"] != self.ring_epoch or self.ring_client is None:
            raise HTTPException(status_code=409, detail="this worker has not joined that ring")
        self.ring_live = True

    def ring_ready(self, request: InferRequest) -> bool:
        """Whether this pass can take the native path.

        Only plain next-token and verify passes do. Prompts, restores and images change
        what a stage holds, and those stay on the path that knows how.
        """
        return (
            self.ring_live
            and self.ring_client is not None
            and self.ring_client.alive
            and request.epoch == self.ring_epoch
            and not (request.reset or request.prompt_rows or request.saves or request.cuts or request.images)
            and request.restore_length is None
            and request.next_position is None
        )

    def ring_pipelines(self, request: InferRequest) -> bool:
        """Whether this reply can keep several passes on the route.

        A wrong guess is undone by rewinding past passes that have finished. The ring means
        the engine is there, and it can do that for recurrent state too. One stage has nothing
        to overlap with.
        """
        return len(self.route) > 1 and self.ring_ready(request)

    def _ring_entry(self, request: InferRequest) -> CacheEntry:
        """This session's entry, marked as having left the Python path."""
        entry = self.caches.get(request.session_id)
        if entry is None or entry.epoch != request.epoch:
            raise retry_error("this session's state is gone")
        entry.ring = True
        return entry

    @staticmethod
    def _ring_pass(request: InferRequest, position: int, rollback_to: int | None, tokens: list[int], verify: bool = False) -> dict[str, Any]:
        return {
            "epoch": request.epoch,
            "slot": request.slot,
            "position": position,
            "rollback_to": rollback_to,
            "tokens": tokens,
            "guesses": tokens[1:] if verify else [],
            "verify": verify,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "seed": request.seed,
        }

    def _ring_silent(self) -> HTTPException:
        self.ring_live = False
        return retry_error("a pass did not come back")

    def ring_send(self, request: InferRequest, token: int, position: int, rollback_to: int | None) -> asyncio.Future[list[int]]:
        """One pass along the ring, not waited for. Passes run in the order they are sent."""
        assert self.ring_client is not None
        entry = self._ring_entry(request)
        entry.length = position + 1
        self.inference_calls += 1
        self.tokens_processed += 1
        try:
            # Attention rewinds by itself, so only recurrent state pays for a snapshot.
            return self.ring_client.send(**self._ring_pass(request, position, rollback_to, [token]), keep=not self.rewinds)
        except EngineRefusal as refused:
            raise self.ring_refused(refused) from refused

    async def ring_answer(self, answer: asyncio.Future[list[int]]) -> int:
        round_trip = self.ring_client.round_trip if self.ring_client is not None else None
        if round_trip is None:
            seconds = pass_seconds(1, len(self.route))
        else:
            seconds = max(RING_ANSWER_FLOOR_SECONDS, RING_ACK_MS / 1000 + 2 * round_trip)
        try:
            return (await asyncio.wait_for(answer, seconds))[0]
        except EngineRefusal as refused:
            raise self.ring_refused(refused) from refused
        except asyncio.TimeoutError as exc:
            raise self._ring_silent() from exc

    async def ring_settle(self, slot: int) -> None:
        """Waits for a slot's passes, so none reaches a stage after the next reply has begun.

        A ring known to be lost is not waited on. Replays leave it, so nothing of this slot's
        can reach them, and this runs before the slot is free for the replay to take.
        """
        client = self.ring_client
        if client is None:
            return
        left = pass_seconds(1, len(self.route))
        while self.ring_live and left > 0:
            if await client.settled(slot, min(left, SETTLE_LOOK_SECONDS)):
                return
            left -= SETTLE_LOOK_SECONDS
        self.ring_live = False
        await client.settled(slot, 0)  # it ends the reply's generation, which a lost ring still needs

    def ring_refused(self, refused: EngineRefusal) -> HTTPException:
        """The error the HTTP path would have raised, so recovery is the same."""
        if refused.refusal == REFUSE_EPOCH:
            return epoch_error(self.epoch)
        if refused.refusal == REFUSE_STAGE_ERROR:
            return HTTPException(status_code=500, detail={"stage_error": refused.message})
        # A dead link does not say a worker is dead: its stage may just need restarting. So
        # nobody is blamed here. The replay runs over HTTP, which finds out what really
        # happened, and a worker that is still there keeps its place and its weights.
        self.ring_live = False
        return retry_error(refused.message)

    async def ring_step(self, request: InferRequest) -> dict[str, Any]:
        """One pass along the ring. Nothing on this machine waits on a stage but this."""
        assert self.ring_client is not None
        entry = self._ring_entry(request)
        position = entry.length if request.rollback_to is None else min(entry.length, request.rollback_to)
        try:
            tokens = await self.ring_client.run(
                pass_seconds(len(request.input_ids), len(self.route)),
                **self._ring_pass(request, position, request.rollback_to, request.input_ids, request.verify),
            )
        except EngineRefusal as refused:
            raise self.ring_refused(refused) from refused
        except asyncio.TimeoutError as exc:
            raise self._ring_silent() from exc
        entry.length = position + len(request.input_ids)
        self.inference_calls += 1
        self.tokens_processed += len(request.input_ids)
        return {"token_ids": tokens} if request.verify else {"token_id": tokens[0]}

    def _next_stage(self) -> StageAssignment | None:
        assert self.assignment is not None
        after = self._stages_after()
        return self.route[-after] if after else None

    def _stages_after(self) -> int:
        index = next(i for i, stage in enumerate(self.route) if stage.worker_id == self.worker_id)
        return len(self.route) - index - 1

    async def aim_hop(self, body: dict[str, Any]) -> None:
        """Take the address the coordinator proved for the next stage, or go back to the old one."""
        next_stage = self._next_stage() if self.assignment is not None else None
        if next_stage is None or next_stage.worker_id != body["worker_id"]:
            raise HTTPException(status_code=409, detail="that worker is not the next stage")
        via, url = next_stage.url.rstrip("/"), body.get("url")
        if self.hop is not None and self.hop.url == url and self.hop.via == via:
            return
        hop = NextHop(next_stage.worker_id, via, str(url).rstrip("/"), direct_client(self.client.auth)) if url else None
        # Any node may call this route, so its word alone must not move a stage's traffic.
        proved = hop is not None and await self._proves(hop.client, hop.url, via, hop.worker_id)
        if self.hop is not None:
            self._retire_hop(self.hop)
        if proved:
            self.hop = hop
            hop.watch = asyncio.create_task(self._watch_hop(hop))
        elif hop is not None:
            await hop.client.aclose()

    async def _proves(self, client: httpx.AsyncClient, url: str, via: str, worker_id: str) -> bool:
        try:
            return await prove(client, self.proof, url, via, worker_id, self.revoked_nodes)
        except LINK_FAILURES:
            return False

    def _hop_to(self, next_stage: StageAssignment) -> NextHop | None:
        """The shorter way to this stage. One learned for an earlier route is not it."""
        hop = self.hop
        if hop is not None and (hop.worker_id != next_stage.worker_id or hop.via != next_stage.url.rstrip("/")):
            self._retire_hop(hop)
            return None
        return hop

    def _forget_hop(self, hop: NextHop) -> None:
        hop.lost.set()
        if self.hop is hop:
            self.hop = None

    def _retire_hop(self, hop: NextHop) -> None:
        """Nothing failed, so passes on their way finish there and no reply has to replay."""
        hop.retired = True
        if self.hop is hop:
            self.hop = None
        if not hop.passes:
            hop.lost.set()

    def _hop_failed(self, hop: NextHop) -> None:
        self.hop_fallbacks += self.hop is hop
        self._forget_hop(hop)

    async def _watch_hop(self, hop: NextHop) -> None:
        """A link that goes silent raises nothing, and a pass on it would wait out its deadline."""
        heard = time.monotonic()
        try:
            while not hop.lost.is_set():
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(hop.lost.wait(), HOP_WATCH_SECONDS)
                    return
                try:
                    answer = await hop.client.post(hop.url + ECHO_PATH, headers=auth_headers(self.pool_token), extensions=hop.options, timeout=HOP_WATCH_SECONDS * 2)
                    answer.raise_for_status()
                    heard = time.monotonic()
                except LINK_FAILURES:
                    if time.monotonic() - heard >= HOP_SILENCE_SECONDS:
                        self._hop_failed(hop)
        finally:
            self._forget_hop(hop)  # its client closes here, and no pass may be sent to that
            await hop.client.aclose()

    async def _over_hop(self, hop: NextHop, payload: bytes, headers: dict[str, str], timeout: httpx.Timeout) -> httpx.Response:
        """One pass by the shorter way. If that fails, the reply is replayed by the old one.

        A pass that went unanswered may have run, and only a replay is safe to do twice.
        The error says nothing of the address, because it can travel as far as a status.
        """
        sending = asyncio.ensure_future(hop.client.post(hop.url + "/internal/forward", content=payload, headers=headers, timeout=timeout, extensions=hop.options))
        lost = asyncio.ensure_future(hop.lost.wait())
        hop.passes += 1
        try:
            await asyncio.wait({sending, lost}, return_when=asyncio.FIRST_COMPLETED)
            response = sending.result() if sending.done() else None
        except LINK_FAILURES:
            response = None
        finally:
            sending.cancel()
            lost.cancel()
            hop.passes -= 1
            if hop.retired and not hop.passes:
                hop.lost.set()
        # A refusal here may come from whoever has this address now, not from the next stage.
        if response is None or (response.status_code >= 300 and response.status_code != 409 and failure_report(response) is None):
            self._hop_failed(hop)
            raise retry_error("the shorter way to the next stage failed")
        return response

    async def time_link(self, client: httpx.AsyncClient, url: str, sizes: list[int], rounds: int, **options: Any) -> dict[str, Any]:
        """Time echoes to one address, from this end."""
        results: dict[str, Any] = {}
        for size in sizes:
            payload = bytes(int(size))
            times: list[float] = []
            for _ in range(rounds):
                started = time.perf_counter()
                response = await client.post(url + ECHO_PATH, content=payload, headers=auth_headers(self.pool_token), **options)
                response.raise_for_status()
                times.append((time.perf_counter() - started) * 1000)
            times.sort()
            results[str(size)] = {"median_ms": round(times[len(times) // 2], 1), "min_ms": round(times[0], 1), "max_ms": round(times[-1], 1)}
        return results

    async def try_address(self, url: str, via: str, worker_id: str, sizes: list[int], rounds: int) -> dict[str, Any]:
        """Time one address of another worker. It hears no credential until it has proved whose it is."""

        async def attempt() -> dict[str, Any]:
            async with direct_client(self.client.auth) as client:
                if not await self._proves(client, url, via, worker_id):
                    return {"proved": False}
                return {"proved": True, **await self.time_link(client, url, sizes, rounds, extensions=reach_options(via))}

        try:
            return await asyncio.wait_for(attempt(), ADDRESS_PROBE_SECONDS)
        except (asyncio.TimeoutError, *LINK_FAILURES):
            return {"proved": False}

    def _session(self, request: InferRequest) -> CacheEntry:
        entry = self.caches.get(request.session_id)
        restoring = request.restore_length is not None
        stale = entry is None or entry.epoch != request.epoch
        if self.gguf is not None and stale and not request.reset and not restoring:
            # llama.cpp holds one sequence and this session's is gone. Starting
            # from nothing would answer from the wrong state, so ask for a replay.
            raise retry_error("this session's state is gone")
        if request.reset or restoring or stale:
            entry = CacheEntry(epoch=request.epoch, slot=request.slot)
            if self.gguf is not None:
                if restoring:
                    try:
                        tier = self.gguf.restore(request.checkpoint_key, request.slot)
                    except StageLost:
                        raise
                    except LlamaError as exc:
                        detail = {"retry": True, "missing": request.checkpoint_key, "message": str(exc)}
                        raise HTTPException(status_code=503, detail=detail) from exc
                    entry.length = request.restore_length
                    if request.checkpoint_key in self.fetched:
                        self.fetched.discard(request.checkpoint_key)
                        tier = "copy"
                    source = f"{request.restore_kind or HISTORY}_{tier}"
                    self.tokens_reused[source] = self.tokens_reused.get(source, 0) + request.reused_rows
                else:
                    self.gguf.clear(request.slot)
                # A new session takes over its sequence from whoever had it.
                self.caches = {key: item for key, item in self.caches.items() if item.slot != request.slot}
            self.caches[request.session_id] = entry
        return entry

    def _prepare_entry(self, request: InferRequest, part: PromptPart | None = None) -> torch.Tensor:
        if self.gguf is not None:
            self._session(request)
            if part is None:
                return self.gguf.embed(request.input_ids).copy()[None]
            tokens = np.array(request.input_ids, dtype=np.int64)
            rows = np.empty((tokens.size, self.gguf.width), dtype=np.float32)
            text = tokens >= 0
            if text.any():
                rows[text] = self.gguf.embed(tokens[text].tolist())
            rows[part.images] = part.vectors
            return rows[None]
        if self.adapter is None or "embed" not in self.unit_weights:
            raise ValueError("this worker is not the entry stage")
        entry = self._session(request)
        input_ids = torch.tensor([request.input_ids], dtype=torch.long)
        positions = torch.arange(entry.length, entry.length + input_ids.shape[1], dtype=torch.long)
        with torch.inference_mode():
            return self.adapter.embed(self.unit_weights["embed"], input_ids, positions.unsqueeze(0))

    def _pass_entry(self, request: InferRequest) -> CacheEntry:
        """This pass's session, having forgotten anything past where the pass starts."""
        # The entry stage already reset this session before embedding.
        entry = self.caches.get(request.session_id) if self.assignment.start_layer == 0 else None
        entry = entry or self._session(request)
        if self.gguf is not None and request.rollback_to is not None and entry.length > request.rollback_to:
            self.gguf.truncate(request.rollback_to, request.slot)
            entry.length = request.rollback_to
        return entry

    def _run_local(self, hidden: torch.Tensor, request: InferRequest, places: np.ndarray | None = None) -> torch.Tensor:
        assert self.adapter is not None and self.assignment is not None
        started = time.perf_counter()
        entry = self._pass_entry(request)
        tokens = hidden.shape[1]
        if self.gguf is not None:
            rows = np.asarray(hidden[0], dtype=np.float32)
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
                output = self.gguf.forward(rows, entry.length, request.slot, places, request.cuts, request.saves)
                if self.gguf.last:
                    self.gguf_logits = output.copy()
                else:
                    hidden = output.copy()[None]
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
        # An image's rows take fewer positions than rows, so the pass says where it ended.
        entry.length = entry.length + tokens if request.next_position is None else request.next_position
        self.inference_calls += 1
        self.tokens_processed += tokens
        if request.prompt_rows:
            self.prompt_tokens_computed += tokens
        self.compute_ms_ema = self._ema(self.compute_ms_ema, (time.perf_counter() - started) * 1000)
        return hidden

    def _sample(self, hidden: torch.Tensor, request: InferRequest, logits: torch.Tensor | None = None) -> int:
        if logits is None and self.gguf is not None:
            logits = self.gguf_logits
        if logits is None and (self.adapter is None or "head" not in self.unit_weights):
            raise ValueError("route ended on a worker without the output head")
        if self.gguf is not None:
            # A reply leaves the ring for this path when a stage fails, so a seed has
            # to draw here what it draws there.
            rows = np.asarray(logits)
            if request.temperature == 0:
                return int(np.argmax(rows))
            position = 0 if request.seed is None else self.caches[request.session_id].length - 1
            draws = step_draws(request.seed, position)
            return speculative_choice(lambda _: rows, [], request.temperature, request.top_p, request.top_k, draws)[0]
        with torch.inference_mode():
            if logits is None:
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
                generator.manual_seed(int(step_random(request.seed, self.caches[request.session_id].length).integers(2**63)))
                return int(torch.multinomial(probabilities, 1, generator=generator).item())
            return int(torch.multinomial(probabilities, 1).item())

    def _tensor_headers(self, metadata: dict[str, Any]) -> dict[str, str]:
        return {**auth_headers(self.pool_token), META_HEADER: encode_meta(metadata), "content-type": "application/x-safetensors"}

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
            response = await self.client.put(
                self.cache_url + "/internal/cache/kv",
                content=payload,
                headers=self._tensor_headers(metadata),
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

    async def _step_alone(
        self, hidden: torch.Tensor, request: InferRequest, places: np.ndarray | None = None
    ) -> torch.Tensor | dict[str, Any]:
        """This stage's share of one request. Hidden states, or the answer at the end."""
        next_stage = self._next_stage()
        # Compute runs off the event loop: a disk-tier token can take seconds,
        # and a blocked loop would stop heartbeats and look like a dead node.
        started = time.perf_counter()
        hidden = await asyncio.to_thread(self._run_local, hidden, request, places)
        self._record("compute", started)
        if next_stage is not None:
            return hidden
        if request.verify:
            return {"token_ids": self.gguf_choices}
        started = time.perf_counter()
        result = {"token_id": await asyncio.to_thread(self._sample, hidden, request)}
        self._record("sample", started)
        self.schedule_checkpoint(request)
        return result

    def _can_merge(self, hidden: torch.Tensor, request: InferRequest) -> bool:
        """Ordinary next-token steps can share a pass. Anything with its own state work cannot."""
        return (
            self.merge_passes
            and self.gguf is not None
            and hidden.shape[1] == 1
            and not (request.verify or request.reset or request.prompt_rows)
            and request.restore_length is None
            and request.next_position is None
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

    def _take_merge_queue(self) -> list[MergedStep]:
        """The steps that can still run. A step from another route hears so now."""
        waiting, self.merge_queue = self.merge_queue, []
        steps: list[MergedStep] = []
        for step in waiting:
            if step.outcome.done():
                continue
            if step.request.epoch != self.epoch or self.gguf is None:
                step.outcome.set_exception(epoch_error(self.epoch))
            else:
                steps.append(step)
        return steps

    async def _merged_results(self, steps: list[MergedStep]) -> list[Any]:
        """Each step's answer, or the exception for it, so every waiting reply hears."""
        try:
            if len(steps) == 1:
                return [await self._step_alone(steps[0].hidden, steps[0].request)]
            if not steps:
                return []
            started = time.perf_counter()
            results = await asyncio.to_thread(self._run_merged, steps)
            self._record("compute", started)
            for step, result in zip(steps, results):
                if isinstance(result, dict):
                    self.schedule_checkpoint(step.request)
            return results
        except StageLost as exc:
            # One of them restarts the stage. The rest just need a replay.
            return [exc] + [retry_error(str(exc))] * (len(steps) - 1)
        except Exception as exc:
            return [exc] * len(steps)

    async def _run_merge_queue(self) -> None:
        while self.merge_queue:
            async with self.lock:
                steps = self._take_merge_queue()
                for step, result in zip(steps, await self._merged_results(steps)):
                    if step.outcome.done():
                        continue
                    if isinstance(result, BaseException):
                        step.outcome.set_exception(result)
                    else:
                        step.outcome.set_result(result)

    def _run_merged(self, steps: list[MergedStep]) -> list[Any]:
        assert self.gguf is not None
        started = time.perf_counter()
        results: dict[int, Any] = {}
        ready: list[tuple[int, MergedStep, CacheEntry]] = []
        for index, step in enumerate(steps):
            try:
                ready.append((index, step, self._pass_entry(step.request)))
            except HTTPException as exc:
                results[index] = exc
        ready.sort(key=lambda item: item[1].request.slot)
        if ready:
            rows = np.concatenate([np.asarray(step.hidden[0], dtype=np.float32) for _, step, _ in ready])
            try:
                output = list(self.gguf.forward_many(rows, [entry.length for _, _, entry in ready], [step.request.slot for _, step, _ in ready]))
            except StageLost:
                raise
            except LlamaError:
                # llama.cpp refuses a whole batch over one bad row. Alone, only that row is refused.
                output = [self._forward_alone(step, entry) for _, step, entry in ready]
            for row, (index, step, entry) in zip(output, ready):
                if isinstance(row, LlamaError):
                    results[index] = row
                    continue
                entry.length += 1
                if self.gguf.last:
                    try:
                        results[index] = {"token_id": self._sample(step.hidden, step.request, row.copy())}
                    except (RuntimeError, ValueError) as exc:
                        results[index] = exc  # one reply's sampling is no reason to fail the others
                else:
                    results[index] = row.copy().reshape(1, 1, -1)
            self.merged_passes += 1
            self.merged_rows += len(ready)
            self.inference_calls += len(ready)
            self.tokens_processed += len(ready)
        self.compute_ms_ema = self._ema(self.compute_ms_ema, (time.perf_counter() - started) * 1000)
        return [results[index] for index in range(len(steps))]

    def _forward_alone(self, step: MergedStep, entry: CacheEntry) -> np.ndarray | LlamaError:
        """One step of a merge the stage refused, or why the stage refuses it too."""
        assert self.gguf is not None
        try:
            return self.gguf.forward(np.asarray(step.hidden[0], dtype=np.float32), entry.length, step.request.slot).reshape(-1)
        except StageLost:
            raise
        except LlamaError as exc:
            return exc

    async def await_part(self, request: InferRequest, index: int, seconds: float) -> None:
        """Waits for a part of a prompt to have run on this stage."""
        wanted = (request.part_of, index)
        async with self.part_ran:
            try:
                await asyncio.wait_for(self.part_ran.wait_for(lambda: self.parts_run.get(request.slot) == wanted), seconds)
            except asyncio.TimeoutError as exc:
                # The part before never came: this prompt failed, or was replaced by a replay.
                raise retry_error("a part of the prompt was lost") from exc

    async def note_part(self, request: InferRequest) -> None:
        async with self.part_ran:
            self.parts_run[request.slot] = (request.part_of, request.part_index)
            self.part_ran.notify_all()

    async def run_in_parts(self, whole: InferRequest, pieces: list[tuple[int, int]], step: Any) -> dict[str, Any]:
        """Runs a prompt a part at a time, without waiting for each to cross the route.

        A part leaves as soon as this stage has run the one before, so the next stage reads
        while this one does, and a long prompt takes about one stage's share of the time.
        """
        group = secrets.token_hex(8)
        running: list[asyncio.Task[dict[str, Any]]] = []
        try:
            for index, (first, end) in enumerate(pieces):
                update: dict[str, Any] = {
                    "input_ids": whole.input_ids[first:end],
                    "cuts": [cut - first for cut in whole.cuts if first < cut < end],
                    "saves": [save.model_copy(update={"at": save.at - first}) for save in whole.saves if first < save.at <= end],
                    "part_of": group,
                    "part_index": index,
                    "part_count": len(pieces),
                }
                if index:
                    update.update({"reset": False, "restore_length": None, "checkpoint_key": None, "reused_rows": 0, "restore_kind": None})
                part = whole.model_copy(update=update)
                running.append(asyncio.ensure_future(step(part)))
                if index + 1 < len(pieces):
                    here = asyncio.ensure_future(self.await_part(part, index, pass_seconds(len(whole.input_ids), 1)))
                    try:
                        await asyncio.wait({here, running[-1]}, return_when=asyncio.FIRST_COMPLETED)
                        (here if here.done() else running[-1]).result()
                    finally:
                        here.cancel()
            return (await asyncio.gather(*running))[-1]
        except BaseException:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

    async def execute(self, hidden: torch.Tensor, request: InferRequest, places: np.ndarray | None = None) -> dict[str, Any]:
        if self.assignment is None or self.model_id is None:
            raise HTTPException(status_code=409, detail="worker is not configured")
        if request.epoch != self.epoch:
            raise epoch_error(self.epoch)
        next_stage = self._next_stage()
        # The rows of a whole prompt, since one part can wait behind all the others.
        rows = hidden.shape[1] * request.part_count
        if request.part_index:
            await self.await_part(request, request.part_index - 1, pass_seconds(rows, len(self.route)))
        if self._can_merge(hidden, request):
            outcome = await self._step_together(hidden, request)
        else:
            # The lock covers this node's compute and nothing else. While one reply
            # is away at the next stage, this stage can work on another.
            async with self.lock:
                outcome = await self._step_alone(hidden, request, places)
        if request.part_of is not None:
            await self.note_part(request)
        if request.saves:
            self.copies.add(request.saves)
        if isinstance(outcome, dict):
            return outcome
        return await self._pass_on(outcome, request, places, next_stage, rows)

    async def _pass_on(
        self, hidden: torch.Tensor, request: InferRequest, places: np.ndarray | None, next_stage: StageAssignment, rows: int
    ) -> dict[str, Any]:
        """Sends this stage's rows to the next one, and brings back the route's answer."""
        started = time.perf_counter()
        headers = self._tensor_headers(request.model_dump(exclude={"images"}))
        # A batch of FP32 rows outgrows a cold TCP window and costs extra round
        # trips. Single tokens stay exact; llama.cpp stages turn BF16 back to FP32.
        # Prompt rows always go as BF16, or a restored state that left one row to run
        # would not match a run that read the whole prompt.
        if self.gguf is not None:
            payload = encode_rows(hidden, places, (hidden.shape[1] > 1 or request.prompt_rows) and self.dtype_name != "float32")
        else:
            payload = encode_activation(hidden, None if places is None else torch.from_numpy(places))
        self._record("encode", started)
        self.activation_bytes_sent += len(payload)
        started = time.perf_counter()
        timeout = httpx.Timeout(pass_seconds(rows, self._stages_after()), connect=10.0)
        hop = self._hop_to(next_stage)
        try:
            if hop is not None:
                response = await self._over_hop(hop, payload, headers, timeout)
            else:
                response = await self.client.post(next_stage.url.rstrip("/") + "/internal/forward", content=payload, headers=headers, timeout=timeout)
            report = failure_report(response)
            if report is not None:
                # Said further down the route. Only the stage that saw it knows who failed.
                raise HTTPException(status_code=response.status_code, detail=report)
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
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            if status == 409:
                # The route moved on. Nothing failed, and the reply can follow it.
                raise epoch_error(self.epoch) from exc
            self.last_error = str(exc)
            if status is not None and status not in UNAVAILABLE_STATUSES:
                # The next stage is up and refused this request, so blame the request.
                raise HTTPException(status_code=500, detail={"stage_error": str(exc)}) from exc
            detail = {"failed_worker": next_stage.worker_id, "message": str(exc)}
            if isinstance(exc, BROKEN_CONNECTION):
                detail["lost"] = next_stage.worker_id  # which may be all that went, so someone should ask first
            raise HTTPException(status_code=502, detail=detail) from exc


def failure_report(response: httpx.Response) -> dict[str, Any] | None:
    """What a stage further down says failed, so that it reaches the entry unchanged."""
    if response.status_code not in UNAVAILABLE_STATUSES:
        return None
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail if isinstance(detail, dict) and (detail.get("retry") or detail.get("failed_worker")) else None


runtime = WorkerRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.setup()
    runtime.registration_task = asyncio.create_task(runtime.registration_loop())
    try:
        yield
    finally:
        await runtime.close()


app = FastAPI(title="iSharded worker", version="0.2.0", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TicketIdentity)
app.add_middleware(
    BodyLimit,
    paths={"/internal/forward": MAX_STAGE_BYTES, "/internal/diag/echo": MAX_STAGE_BYTES, "/internal/state/copy": MAX_STATE_BYTES},
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
    node_proven()


def require_entry(input_ids: list[int]) -> None:
    if runtime.assignment is None or runtime.assignment.start_layer != 0:
        raise HTTPException(status_code=409, detail="this worker is not the entry stage")
    if not input_ids:
        raise HTTPException(status_code=400, detail="input_ids cannot be empty")


async def stage_failure(exc: Exception) -> HTTPException:
    """What a caller hears of a pass that failed here, once a lost stage has been replaced."""
    if isinstance(exc, StageLost):
        return await runtime.replace_stage(exc)
    runtime.last_error = f"{type(exc).__name__}: {exc}"
    return HTTPException(status_code=500, detail=runtime.last_error)


async def entry_pass(request: InferRequest, part: PromptPart | None, arrived: float) -> dict[str, Any]:
    """One pass from the top of the route."""
    async with runtime.lock:
        hidden = await asyncio.to_thread(runtime._prepare_entry, request, part)
    runtime._record("embed", arrived)
    result = await runtime.execute(hidden, request, None if part is None else part.places)
    runtime._record("handler", arrived)
    return result


async def admitted(choose: Callable[[set[int]], int] = min) -> int:
    """A sequence for this request, once one is free."""
    runtime.waiting += 1
    try:
        return await runtime.gate.acquire(choose)
    finally:
        runtime.waiting -= 1


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
    offered += [Path(item) for item in os.getenv("ISHARDED_MODEL_DIRS", "").split(os.pathsep) if item]
    resolved = Path(folder).resolve()
    if not any(resolved == base.resolve() or base.resolve() in resolved.parents for base in offered):
        raise HTTPException(status_code=403, detail=f"{folder} is not a folder this node offers; see --model-dir")


@app.get("/health")
async def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorized(authorization):
        return {"status": "ok"}  # hardware and layout are nobody else's business
    if runtime.gguf is not None:
        if runtime.stage_down is runtime.gguf:
            # A report that this node lost its stage is checked here, and must not hear all is well.
            raise HTTPException(status_code=503, detail=runtime.last_error)
        await asyncio.to_thread(runtime.gguf.refresh_states)
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
            "pipeline": {"tokens": runtime.pipeline_tokens, "misses": runtime.pipeline_misses, "depth": runtime.pipeline_depth, "lookups": runtime.pipeline_lookups},
            "draft_ways": {way: round(runtime.draft_choice.rate(way), 2) for way in runtime.draft_choice.seconds},
            "checkpoints_dropped": runtime.checkpoints_dropped,
            "cache": runtime.cache_report(),
            "phases_ms": runtime.phases,
            "ring": await asyncio.to_thread(runtime.gguf.ring_stats) if runtime.gguf is not None else None,
            "ring_live": runtime.ring_live,
            "hop_direct": runtime.hop is not None,
            "hop_fallbacks": runtime.hop_fallbacks,
        },
    }


@app.post("/internal/prepare")
async def prepare(body: PrepareRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    verify_source(body.source)
    if needs_pytorch(body.model_id) and not has_torch():
        raise HTTPException(status_code=400, detail=NO_TORCH)  # before it downloads what it cannot run
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
    """Time requests from this worker to another, to measure a link from this end.

    With ``addresses`` it also tries each of them. Those can go on when the old address
    has failed, so here that one's failure is only a missing reading.
    """
    verify_token(authorization)
    via = str(body["url"]).rstrip("/")
    sizes, rounds = body.get("sizes", [100, 6144, 236544]), int(body.get("rounds", 10))
    addresses = [str(url).rstrip("/") for url in body.get("addresses", [])]
    if not addresses:
        return await runtime.time_link(runtime.client, via, sizes, rounds)

    async def old_way() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(runtime.time_link(runtime.client, via, sizes, rounds), ADDRESS_PROBE_SECONDS)
        except (asyncio.TimeoutError, *LINK_FAILURES):
            return {}

    tried = [runtime.try_address(url, via, str(body["worker_id"]), sizes, rounds) for url in addresses]
    old, *found = await asyncio.gather(old_way(), *tried)
    return {**old, "addresses": dict(zip(addresses, found))}


@app.post("/internal/diag/prove")
async def diag_prove(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str | None]:
    """Sign for one of this worker's addresses.

    A credential must not go to an address until it has answered this, so a caller shows
    it is in the pool by signing the same challenge, which is good for nothing else.
    """
    nonce, url, worker_id = str(body["nonce"]), str(body["url"]).rstrip("/"), str(body["worker_id"])
    signer = runtime.proof.signer((authorization or "").removeprefix("Bearer "), claim("ask", nonce, url, worker_id))
    if signer is None or signer in runtime.revoked_nodes:
        raise HTTPException(status_code=401, detail="a pool token or node certificate is required")
    if worker_id != runtime.worker_id or url not in {runtime.advertised_url, *runtime.reach_urls}:
        raise HTTPException(status_code=404, detail="not an address of this worker")
    return {"proof": runtime.proof.sign(claim("answer", nonce, url, worker_id))}


@app.post("/internal/hop")
async def aim_hop(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_token(authorization)
    await runtime.aim_hop(body)
    return {"status": "aimed"}


@app.get("/internal/inventory")
async def node_inventory(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    return await asyncio.to_thread(inventory)  # asking a GPU tool can take a moment


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
        "torch_threads": torch.get_num_threads() if has_torch() else None,
        "versions": versions,
        "env": {key: value for key, value in os.environ.items() if key.startswith(("OMP_", "MKL_", "HIP_", "HSA_", "PYTHON", "ROC"))},
    }


@app.get("/internal/shards/index")
async def shard_index(model_id: str, revision: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    held = await asyncio.to_thread(runtime.unit_cache.held, model_id, revision)
    return {"units": held, "serves_peers": runtime.peer_serve}


@app.get("/internal/shards/file")
async def shard_file(model_id: str, revision: str, unit: str, authorization: str | None = Header(default=None)) -> FileResponse:
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


@app.get("/internal/diag/tasks")
async def diag_tasks(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Where everything in this process is waiting, for a reply that has gone quiet."""
    verify_token(authorization)
    return where_everything_is()


@app.post("/internal/release")
async def release(authorization: str | None = Header(default=None)) -> dict[str, str]:
    """The pool change this worker was drained for is over."""
    verify_token(authorization)
    await runtime.release_gate()
    return {"status": "released"}


@app.post("/internal/configure")
async def configure(body: ConfigureRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    if body.draft is not None:
        verify_source(body.draft.source)
    if body.projector is not None:
        verify_source(body.projector.source)
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


@app.put("/internal/state/copy")
async def receive_copy(
    request: Request,
    authorization: str | None = Header(default=None),
    x_isharded_meta: str | None = Header(default=None),
) -> dict[str, str]:
    """Keep a copy of another stage's state, straight to disk as it arrives."""
    verify_token(authorization)
    if runtime.held.folder is None:
        raise HTTPException(status_code=409, detail="this node keeps no saved states on disk")
    try:
        meta = decode_meta(x_isharded_meta or "")
        namespace, key, replaces = meta["namespace"], meta["key"], meta.get("replaces")
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"bad {META_HEADER} header") from exc
    if not NAME.fullmatch(str(namespace)) or not NAME.fullmatch(str(key)) or (replaces and not NAME.fullmatch(str(replaces))):
        raise HTTPException(status_code=400, detail="a state is named by a digest")
    size = int(request.headers.get("content-length", 0))
    if size > runtime.held.disk_bytes:
        raise HTTPException(status_code=413, detail="this state is larger than this node keeps")
    if not runtime.held.has_room(size):
        raise HTTPException(status_code=507, detail="this node's disk is too full to keep a copy")
    name = f"{namespace}.{key}"
    partial = runtime.held.folder / (name + PARTIAL)
    try:
        with partial.open("wb") as handle:
            async for chunk in request.stream():
                await asyncio.to_thread(handle.write, chunk)
    except (OSError, ClientDisconnect) as exc:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise HTTPException(status_code=507, detail="this copy could not be kept") from exc
    if not await asyncio.to_thread(runtime.held.adopt, name, partial):
        raise HTTPException(status_code=400, detail="this copy does not match its checksum")
    if replaces:
        runtime.held.drop(f"{namespace}.{replaces}")
    return {"status": "kept"}


@app.get("/internal/state/held/{namespace}")
async def held_states(namespace: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    start = namespace + "."
    states = [{**item, "key": item["key"][len(start) :]} for item in runtime.held.listing() if item["key"].startswith(start)]
    return {"states": states}


@app.get("/internal/state/held/{namespace}/{key}")
async def held_state(namespace: str, key: str, authorization: str | None = Header(default=None)) -> Response:
    verify_token(authorization)
    path = runtime.held.file_of(f"{namespace}.{key}") if NAME.fullmatch(namespace) and NAME.fullmatch(key) else None
    if path is None:
        raise HTTPException(status_code=404, detail="no copy of that state here")
    return FileResponse(path, media_type="application/octet-stream")


@app.get("/internal/state/index")
async def state_index(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    return {"states": list((await runtime.restorable()).values())}


@app.post("/internal/state/prepare")
async def prepare_state(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, bool]:
    verify_token(authorization)
    key = str(body.get("key"))
    return {"ready": bool(NAME.fullmatch(key)) and await runtime.ensure_state(key)}


@app.post("/internal/state/keep")
async def keep_states(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Forget copies for anything no stage is running any more."""
    verify_token(authorization)
    wanted = set(body.get("namespaces", []))
    for item in runtime.held.listing():
        if item["key"].split(".", 1)[0] not in wanted:
            runtime.held.drop(item["key"])
    return {"status": "kept"}


@app.put("/internal/cache/kv")
async def store_kv(
    request: Request,
    authorization: str | None = Header(default=None),
    x_isharded_meta: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    if "kv-cache" not in runtime.roles:
        raise HTTPException(status_code=409, detail="worker does not contribute a KV-cache service")
    if not x_isharded_meta:
        raise HTTPException(status_code=400, detail=f"missing {META_HEADER} header")
    try:
        metadata = decode_meta(x_isharded_meta)
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
async def fetch_kv(session_id: str, epoch: int, start_layer: int, end_layer: int, authorization: str | None = Header(default=None)) -> Response:
    verify_token(authorization)
    try:
        payload = runtime.memory_store.get_layers(session_id, epoch, start_layer, end_layer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=payload, media_type="application/x-safetensors")


@app.post("/internal/restore")
async def restore_kv(body: CacheRestoreRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
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
    require_entry(body.input_ids)
    try:
        arrived = time.perf_counter()
        part = None
        if body.images:
            # A replay of a prompt with images in it. Later passes have no need of them.
            rows = await runtime.prompt_rows(body.input_ids, body.images)
            part = rows.part(0, len(body.input_ids))
            body = body.model_copy(update={"images": [], "next_position": part.after})
        return await entry_pass(body, part, arrived)
    except HTTPException:
        raise
    except Exception as exc:
        raise await stage_failure(exc) from exc


def token_line(token: int) -> bytes:
    """The same bytes ``json.dumps`` would give, without building them for every token."""
    return b'{"token_id": %d}\n' % token


def error_line(detail: Any) -> bytes:
    return json.dumps({"error": detail}).encode() + b"\n"


async def reply_step(request: InferRequest, part: PromptPart | None = None) -> dict[str, Any]:
    """One pass of a reply, along the ring where it can take it."""
    arrived = time.perf_counter()
    if part is None and runtime.ring_ready(request):
        result = await runtime.ring_step(request)
        runtime._record("handler", arrived)
        return result
    held = runtime.caches.get(request.session_id)
    if held is not None and held.ring and not request.reset and request.restore_length is None:
        # The other stages' own counts stopped when this reply took the ring.
        raise retry_error("the ring went down during this reply")
    return await entry_pass(request, part, arrived)


async def read_written(request: InferRequest, written: list[int], position: int) -> int:
    """Runs what a reply had already written, and returns the token that comes next.

    Each token gets its own one-row pass at its own position, because that is what plain
    decoding ran, and a batched pass rounds differently. Whatever the model answers along
    the way, the written tokens stand, so on the ring no pass has to wait for the one before.

    A reply that stops here leaves its passes out, uncancelled: ``finish_reply`` waits for
    the newest of them, so none reaches a stage after the slot has gone to someone else.
    """
    if not runtime.ring_ready(request):
        for token in written:
            answer = (await reply_step(request.model_copy(update={"input_ids": [token]})))["token_id"]
        return answer
    flights: deque[asyncio.Future[list[int]]] = deque()
    for place, token in enumerate(written, position):
        if len(flights) == RESUME_WINDOW:
            await runtime.ring_answer(flights.popleft())
        flights.append(runtime.ring_send(request, token, place, None))
    while len(flights) > 1:
        await runtime.ring_answer(flights.popleft())
    return await runtime.ring_answer(flights.popleft())


async def state_in_reach(key: str) -> bool:
    """Gets a state held elsewhere onto every stage, or forgets about it."""
    try:
        ready = await runtime.prepare_state(key)
    except Exception as exc:  # a reread needs none of this, so it must never cost a reply
        runtime.last_error = f"could not fetch a saved state: {exc}"
        ready = False
    if ready:
        runtime.prefix.elsewhere.discard(key)
    else:
        runtime.prefix.drop(key)
        runtime.state_misses += 1
    return ready


async def finish_reply(slot: int) -> None:
    try:
        pipe = runtime.pipes.pop(slot, None)
        try:
            if pipe is not None:
                await pipe.close()
        finally:
            await runtime.ring_settle(slot)  # it ends the reply's generation, so it always runs
    finally:
        runtime.slot_used_at[slot] = time.monotonic()
        await runtime.gate.release(slot)


@app.post("/v1/generate")
async def generate(body: GenerateRequest, authorization: str | None = Header(default=None)) -> StreamingResponse:
    """Run the token loop here and stream ids out, one JSON object per line.

    Each token still crosses the stages, but no longer makes a round trip to the
    orchestrator. A failure ends the stream, and the orchestrator recovers by opening
    a new one with ``written`` set, or by replaying through ``/v1/infer``.
    """
    verify_token(authorization)
    require_entry(body.input_ids)
    if runtime.waiting >= MAX_WAITING_REPLIES:
        raise HTTPException(status_code=429, detail="too many replies are waiting; try again shortly")
    stops = set(body.stop_token_ids)

    async def first_token(request: InferRequest, plan: PromptPlan, rows: PromptRows | None = None) -> int:
        """Run what the stages have not seen, in one trip along the route.

        Each stage stops at the same cuts and saves at the same places, so saving
        costs no extra pass and no extra hop.
        """
        prompt, start = body.input_ids, plan.start
        part = rows.part(start, len(prompt)) if rows else None
        whole = request.model_copy(
            update={
                "input_ids": prompt[start:],
                "reset": not start,
                "restore_length": (rows.position_at(start) if rows else start) or None,
                "checkpoint_key": plan.key,
                "reused_rows": start,
                "restore_kind": plan.kind,
                "prompt_rows": True,
                "cuts": plan.cuts,
                "saves": plan.saves,
                "next_position": part.after if part else None,
            }
        )
        pieces = prompt_parts(plan.cuts, len(prompt) - start) if PROMPT_PARTS and rows is None and len(runtime.route) > 1 else []
        result = await (runtime.run_in_parts(whole, pieces, reply_step) if len(pieces) > 1 else reply_step(whole, part))
        for save in plan.saves:
            if save.replaces:
                runtime.prefix.drop(save.replaces)
            runtime.prefix.add(save.key, save.length, save.kind)
        return result["token_id"]

    async def next_new_token(request: InferRequest, plan: PromptPlan, rows: PromptRows | None = None) -> int:
        token = await first_token(request, plan, rows)
        if body.written:
            token = await read_written(request.model_copy(update=GOING_ON), body.written, len(body.input_ids))
        return token

    async def plain(request: InferRequest, plan: PromptPlan, rows: PromptRows | None = None) -> AsyncIterator[int]:
        token = await next_new_token(request, plan, rows)
        yield token
        following = request.model_copy(update=GOING_ON)
        for _ in range(body.max_tokens - 1):
            if token in stops:
                return
            token = (await reply_step(following.model_copy(update={"input_ids": [token]})))["token_id"]
            yield token

    async def speculative(request: InferRequest, plan: PromptPlan) -> AsyncIterator[int]:
        """Guess ahead on the entry worker, and check every guess in one pass.

        The first token comes from the normal path, which also resets each stage.
        After that, ``position`` is where the newest token goes: it has not run
        yet, and every stage forgets anything at or past it before the next pass.
        """
        draft = runtime.draft
        # The draft reads the prompt while the real model does.
        prepared = asyncio.create_task(asyncio.to_thread(draft.prepare, body.input_ids, plan, request.slot)) if draft else None
        try:
            token = await next_new_token(request, plan)
        except Exception:
            # A reply may plan again now, and two of these on one sequence would tangle it.
            if prepared:
                await asyncio.gather(prepared, return_exceptions=True)
            raise
        yield token
        try:
            if prepared:
                await prepared
        except LlamaError as exc:
            raise await runtime.draft_failed(exc, draft) from exc
        history = [*body.input_ids, *body.written, token]
        produced = 1
        ahead = min(INITIAL_GUESSES, runtime.speculate_tokens)
        following = request.model_copy(update=GOING_ON)

        async def draft_guesses(fresh: list[int], count: int) -> list[int]:
            started = time.perf_counter()
            try:
                guesses = await asyncio.to_thread(draft.guess, fresh, count, request.slot)
            except LlamaError as exc:
                raise await runtime.draft_failed(exc, draft) from exc
            runtime._record("draft", started)
            return guesses

        async def next_guess(fresh: list[int]) -> int:
            return (await draft_guesses(fresh, 1))[0]

        async def rewind_draft(keep: int) -> None:
            draft.rewind(keep, request.slot)

        pipe: Pipeline | None = None
        while produced < body.max_tokens and token not in stops:
            room = body.max_tokens - produced
            began = time.perf_counter()
            # Checking guesses in one pass runs several rows at once, which rounds a little
            # differently. A pipeline and plain decoding run one row, so they agree exactly.
            # So a guess from the text only ever goes out on a pipeline.
            ways = [PIPELINED, PLAIN] if runtime.ring_pipelines(following) else [GUESSED, PLAIN] if draft else [PLAIN]
            way = runtime.draft_choice.pick(ways)
            if way == PIPELINED:
                if pipe is None:
                    pipe = Pipeline(
                        history,
                        room,
                        stops,
                        len(runtime.route),
                        runtime.speculate_tokens,
                        send=lambda item, place, rollback: runtime.ring_send(following, item, place, rollback),
                        wait=runtime.ring_answer,
                        guess=next_guess if draft else None,
                        rewind=rewind_draft if draft else None,
                        sharing=lambda: runtime.gate.size - len(runtime.gate.free),
                        told=draft.seen(request.slot) if draft else 0,
                        lookup=runtime.prompt_lookup,
                        clock=runtime.row_clock,
                    )
                    runtime.pipes[request.slot] = pipe
                made, misses, lookups = pipe.tokens_made, pipe.misses, pipe.lookups
                async for token in pipe.tokens(PIPELINE_STRETCH):
                    produced += 1
                    yield token
                runtime.pipeline_tokens += pipe.tokens_made - made
                runtime.pipeline_misses += pipe.misses - misses
                runtime.pipeline_lookups += pipe.lookups - lookups
                runtime.pipeline_depth = pipe.depth()
                runtime.draft_choice.note(way, pipe.tokens_made - made, time.perf_counter() - began)
                continue
            landing = pipe is not None
            if pipe is not None:
                await pipe.land()
                pipe = None
                del runtime.pipes[request.slot]
            drafting = way == GUESSED
            guesses = await draft_guesses(history[draft.seen(request.slot) :], min(ahead, room)) if drafting else []
            result = await reply_step(
                following.model_copy(update={"input_ids": [token, *guesses], "verify": drafting, "rollback_to": len(history) - 1})
            )
            accepted = result["token_ids"] if drafting else [result["token_id"]]
            if drafting:
                runtime.draft_passes += 1
                runtime.draft_accepted += len(accepted)
                ahead = next_guess_count(ahead, len(accepted) - 1, runtime.speculate_tokens)
            # The first pass off a pipeline waits behind what was out, which says nothing about
            # this way, and a trial is too short to carry that. The next pass is the trial.
            if not landing:
                runtime.draft_choice.note(way, len(accepted), time.perf_counter() - began)
            for token in accepted[:room]:
                yield token
                produced += 1
                if token in stops:
                    return
            history.extend(accepted)
            if drafting:
                draft.rewind(len(history) - 1, request.slot)  # the newest token is the model's own

    async def tokens() -> AsyncIterator[bytes]:
        request = InferRequest(**body.model_dump(exclude={"max_tokens", "stop_token_ids", "checkpoint_at", "cut_token", "written", "images"}))
        slot = await admitted(runtime.choose_slot)
        request = request.model_copy(update={"slot": slot})
        try:
            if body.epoch != runtime.epoch:
                yield error_line(epoch_error(runtime.epoch).detail)
                return
            # Guesses come from text, so they cannot follow a prompt with images in it.
            use_guesses = runtime.speculate_tokens and not body.images and (runtime.draft is not None or (runtime.prompt_lookup and len(runtime.route) > 1))
            caching = runtime.gguf is not None and runtime.prompt_cache
            if caching:
                try:
                    await runtime.sync_index()
                except Exception as exc:  # this only saves work, so it must never cost a reply
                    runtime.last_error = f"could not learn what the stages hold: {exc}"
            rows: PromptRows | None = None
            answered = False
            failure: Any = None
            for attempt in range(PLAN_ATTEMPTS):
                # The last attempt reads the whole prompt, which depends on no state at all.
                plan = plan_prompt(runtime.prefix, body.input_ids, body.cut_token, body.checkpoint_at, caching and attempt < PLAN_ATTEMPTS - 1)
                if plan.key in runtime.prefix.elsewhere and not await state_in_reach(plan.key):
                    continue
                try:
                    if body.images and rows is None:
                        rows = await runtime.prompt_rows(body.input_ids, body.images)
                    source = speculative(request, plan) if use_guesses else plain(request, plan, rows)
                    async for token in source:
                        answered = True
                        yield token_line(token)
                    return
                except HTTPException as exc:
                    failure = exc.detail
                except Exception as exc:
                    failure = (await stage_failure(exc)).detail
                # A stage that no longer holds a state is no failure: plan without it.
                missing = failure.get("missing") if isinstance(failure, dict) else None
                if missing is None or answered:
                    break
                runtime.prefix.drop(missing)
                runtime.state_misses += 1
            yield error_line(failure)
        finally:
            await finish_reply(slot)

    return StreamingResponse(tokens(), media_type="application/x-ndjson")


@app.post("/internal/images/measure")
async def measure_images(body: ImagesMeasure, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """What images will cost, so a control plane can check them against the context."""
    verify_token(authorization)
    encoder = runtime.encoder
    if encoder is None or body.epoch != runtime.epoch:
        raise epoch_error(runtime.epoch)
    try:
        return {"images": [await asyncio.to_thread(encoder.measure, base64.b64decode(item)) for item in body.images]}
    except LlamaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/embed")
async def embed(body: EmbedPass, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """One pass of an embedding model. The control plane tokenizes and packs the inputs."""
    verify_token(authorization)
    if not body.inputs or not all(body.inputs):
        raise HTTPException(status_code=400, detail="inputs cannot be empty")
    if runtime.waiting >= MAX_WAITING_REPLIES:
        raise HTTPException(status_code=429, detail="too many requests are waiting; try again shortly")
    slot = await admitted()
    try:
        if body.epoch != runtime.epoch:
            raise epoch_error(runtime.epoch)
        if runtime.gguf is None or not runtime.gguf.embedding:
            raise HTTPException(status_code=409, detail="this worker is not running an embedding model")
        started = time.perf_counter()
        vectors = await asyncio.to_thread(runtime.gguf.pool, body.inputs)
        runtime._record("compute", started)
        runtime.inference_calls += 1
        runtime.tokens_processed += sum(map(len, body.inputs))
        return {"vectors": base64.b64encode(vectors.tobytes()).decode("ascii")}
    except LlamaError as exc:
        raise await stage_failure(exc) from exc
    finally:
        await runtime.gate.release(slot)


@app.post("/internal/ring/open")
async def ring_open(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    return await runtime.ring_open(body)


@app.post("/internal/ring/connect")
async def ring_connect(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    return await runtime.ring_connect(body)


@app.post("/internal/ring/close")
async def ring_close(authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_token(authorization)
    runtime.ring_stop()
    if runtime.gguf is not None:
        await asyncio.to_thread(runtime.gguf.ring_call, "ring_close")
    return {"status": "closed"}


@app.post("/internal/ring/go")
async def ring_go(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_token(authorization)
    runtime.ring_go(body)
    return {"status": "live"}


@app.post("/internal/forward")
async def forward(
    request: Request,
    authorization: str | None = Header(default=None),
    x_isharded_meta: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    if not x_isharded_meta:
        raise HTTPException(status_code=400, detail=f"missing {META_HEADER} header")
    arrived = time.perf_counter()
    try:
        meta = InferRequest.model_validate(decode_meta(x_isharded_meta))
        # A stage with no PyTorch has only one way to read, even before it has a model.
        decode = decode_rows if runtime.gguf is not None or not torch_loaded() else decode_pass
        hidden, places = decode(await request.body())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runtime._record("receive", arrived)
    try:
        result = await runtime.execute(hidden, meta, None if places is None else np.asarray(places))
        runtime._record("handler", arrived)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise await stage_failure(exc) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an iSharded compute worker")
    parser.add_argument("--host", default=os.getenv("ISHARDED_HOST", "0.0.0.0"), help="address to listen on")
    parser.add_argument("--port", type=int, default=int(os.getenv("ISHARDED_PORT", "8100")), help="port to listen on")
    parser.add_argument("--worker-id", default=os.getenv("ISHARDED_WORKER_ID", socket.gethostname()), help="unique name; use node-gpu, node-cpu, or node-memory so certificates match")
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("ISHARDED_ORCHESTRATOR_URL", "http://127.0.0.1:8000"),
        help="one or more control planes, comma separated; the rest are found from them",
    )
    parser.add_argument("--advertise-url", default=os.getenv("ISHARDED_ADVERTISE_URL"), help="how other nodes reach this worker")
    parser.add_argument("--reach-hosts", default=os.getenv("ISHARDED_REACH_HOSTS", "auto"), help="other hosts this worker answers on, comma separated; auto finds this machine's, none registers only the advertised one")
    parser.add_argument("--location", type=country_code, default=os.getenv("ISHARDED_LOCATION") or None, help="two letter country code to report, or none to report no country; default is looked up from the advertised address")
    parser.add_argument("--device", default=os.getenv("ISHARDED_DEVICE", "auto"), help="auto, cpu, cuda:0, mps, or xpu:0")
    parser.add_argument("--roles", default=os.getenv("ISHARDED_ROLES", "inference"), help="inference, or kv-cache,checkpoint for a memory worker")
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("ISHARDED_CPU_THREADS", "0")), help="CPU threads; 0 uses half the cores for a stage")
    parser.add_argument("--ram-budget-gb", type=float, default=float(os.getenv("ISHARDED_RAM_BUDGET_GB", "0")), help="RAM for KV checkpoints, not weights")
    parser.add_argument("--state-ram-gb", type=float, default=float(os.getenv("ISHARDED_STATE_RAM_GB", "1")), help="RAM for saved prompt states on each stage")
    parser.add_argument("--state-disk-gb", type=float, default=float(os.getenv("ISHARDED_STATE_DISK_GB", "8")), help="disk for saved prompt states; 0 keeps them in RAM only")
    parser.add_argument("--state-copy-mb", type=float, default=float(os.getenv("ISHARDED_STATE_COPY_MB", "10")), help="MB per second for copies of saved states to another node; 0 sends none")
    parser.add_argument("--gpu-memory-gb", type=float, default=float(os.getenv("ISHARDED_GPU_MEMORY_GB", "0")), help="VRAM budget per accelerator; 0 means auto")
    parser.add_argument("--weights-ram-gb", default=None, help="system RAM for model weights, or 'auto'")
    parser.add_argument("--weights-disk-gb", type=float, default=0, help="disk for memory-mapped model weights")
    parser.add_argument("--cache-dir", default=os.getenv("ISHARDED_CACHE_DIR"), help="where fetched weights live")
    parser.add_argument("--cache-budget-gb", type=float, default=0, help="evict old models past this size")
    parser.add_argument("--no-peer-fetch", action="store_true", help="always download from the model host")
    parser.add_argument("--no-peer-serve", action="store_true", help="do not send shards to other workers")
    parser.add_argument("--llama-dir", default=os.getenv("ISHARDED_LLAMA_DIR"), help="a llama.cpp build for GGUF models")
    parser.add_argument("--open", action="store_true", help="serve a network with no credentials, trusting everyone on it")
    parser.add_argument("--gguf-dir", action="append", default=None, help="folder holding GGUF files; repeatable")
    parser.add_argument("--trusted-network", action="store_true", help="send pool credentials unencrypted, because nobody else can read this network")
    parser.add_argument(
        "--dtype",
        choices=["auto", *SUPPORTED_DTYPES],
        default="auto",
        help="auto is bfloat16; float32 runs this node at full precision",
    )
    parser.add_argument("--host-models", default=os.getenv("ISHARDED_HOST_MODELS", ""), help="comma-separated patterns; host only models that match")
    parser.add_argument("--standby", action="store_true", default=False, help="only use this worker when a model does not fit without it")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["ISHARDED_WORKER_ID"] = args.worker_id
    os.environ["ISHARDED_ORCHESTRATOR_URL"] = args.orchestrator_url
    os.environ["ISHARDED_ADVERTISE_URL"] = args.advertise_url or f"http://127.0.0.1:{args.port}"
    os.environ["ISHARDED_LOCATION"] = args.location or ""
    os.environ["ISHARDED_DEVICE"] = args.device
    os.environ["ISHARDED_ROLES"] = args.roles
    os.environ["ISHARDED_CPU_THREADS"] = str(args.cpu_threads)
    os.environ["ISHARDED_RAM_BUDGET_GB"] = str(args.ram_budget_gb)
    os.environ["ISHARDED_STATE_RAM_GB"] = str(args.state_ram_gb)
    os.environ["ISHARDED_STATE_DISK_GB"] = str(args.state_disk_gb)
    os.environ["ISHARDED_STATE_COPY_MB"] = str(args.state_copy_mb)
    os.environ["ISHARDED_GPU_MEMORY_GB"] = str(args.gpu_memory_gb)
    os.environ["ISHARDED_WEIGHTS_RAM_GB"] = "" if args.weights_ram_gb is None else str(args.weights_ram_gb)
    os.environ["ISHARDED_WEIGHTS_DISK_GB"] = str(args.weights_disk_gb)
    os.environ["ISHARDED_CACHE_DIR"] = args.cache_dir or ""
    os.environ["ISHARDED_CACHE_BUDGET_GB"] = str(args.cache_budget_gb)
    os.environ["ISHARDED_PEER_FETCH"] = "0" if args.no_peer_fetch else "1"
    os.environ["ISHARDED_PEER_SERVE"] = "0" if args.no_peer_serve else "1"
    os.environ["ISHARDED_LLAMA_DIR"] = args.llama_dir or ""
    check_exposure(args.host, load_token(), os.getenv("ISHARDED_CA_FILE"), args.open, args.advertise_url, args.trusted_network)
    # With no identity to check, nothing could prove an address, so none would ever count.
    provable = bool(load_token() or os.getenv("ISHARDED_CA_FILE"))
    urls, left_out = reach_urls(args.reach_hosts if provable else "none", os.environ["ISHARDED_ADVERTISE_URL"], args.host, args.port, args.open or args.trusted_network)
    if left_out:
        print(f"{left_out} of this machine's addresses stay unregistered: they would carry credentials unencrypted. Use TLS, or --trusted-network if nobody else can read them.", flush=True)
    os.environ["ISHARDED_REACH_URLS"] = ",".join(urls)
    os.environ["ISHARDED_GGUF_DIRS"] = os.pathsep.join(args.gguf_dir or [])
    os.environ["ISHARDED_DTYPE"] = args.dtype
    os.environ["ISHARDED_STANDBY"] = "1" if args.standby else "0"
    os.environ["ISHARDED_HOST_MODELS"] = args.host_models
    global runtime
    runtime = WorkerRuntime()
    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
