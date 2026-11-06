from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import __version__
from .access import INVITE_FIELDS, load_authority, NeedsAuthority, node_of, ROLE_ADMIN, ROLE_MEMBER, ROLE_NODE, ROLE_USER, AccessControl, AuditLog, Caller
from .gguf import parse_source
from . import formats
from .lanes import LaneDoor
from .mesh import GOSSIP_SECONDS, ControlMesh
from .models import Adapter, GgufAdapter, SUPPORTED_TYPES, UnsupportedModel, create_adapter
from .planner import AMORTIZE_TOKENS, HOP_MS, Candidate, ModelProfile, PlacementError, plan_placement
from .protocol import (
    hosts_model,
    pass_seconds,
    CacheRestoreRequest,
    ChatCompletionRequest,
    CompletionRequest,
    ConfigureRequest,
    PROTOCOL_VERSION,
    DraftModel,
    GenerateRequest,
    Heartbeat,
    InferRequest,
    LoadModelRequest,
    MemoryPool,
    PrepareRequest,
    SessionState,
    StageAssignment,
    WeightUnit,
    WorkerRegistration,
    auth_headers,
)
from .pool import check_exposure, home as pool_home, load_token, read_membership_request
from .serving import BodyLimit, serve, where_everything_is
from .mesh import clean_url
from .tickets import PREFIX as TICKET_PREFIX, NodeTickets, TicketChecker, TicketIdentity, client_auth, make_checker, reply_audience
from .tls import BEHALF, BEHALF_HEADER, FORWARD_MARK, INTERCEPTED, intercepted, PEER, client_verify
from .tokenizer import ChatTokenizer, IncrementalText, load_tokenizer
from .weights import apply_model_hashes, Manifest, WeightSourceError, build_manifest

# Replans allowed when workers fail mid-load before the job gives up.
MAX_DEPLOY_ATTEMPTS = 4
PROGRESS_POLL_SECONDS = 0.5
# Missed progress polls in a row before a fetching worker counts as gone.
MAX_MISSED_POLLS = 6

# Least time a layer takes on a GPU, measured at 0.5 to 0.75 ms on an RX 7900 XT.
LAYER_FLOOR_MS = 0.5
# Small models that share a vocabulary with a family, for speculative decoding.
AUTO_DRAFTS = {
    "qwen35": "lmstudio-community/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q8_0.gguf",
    "qwen35moe": "lmstudio-community/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q8_0.gguf",
}
# llama.cpp's compute buffers on a GPU, on top of weights and context.
GGUF_STAGE_WORKSPACE_BYTES = 768 * 1024**2
DRAFT_WORKSPACE_BYTES = 512 * 1024**2
# How often the orchestrator looks at the pool when nothing is asking it to.
MAINTENANCE_SECONDS = 10.0
LOGGED_EVENTS = {"failure", "recovery", "mesh", "rebalance", "security"}
# A live worker that failed to activate gets this many more tries, this far apart.
ACTIVATE_RETRIES = 2
ACTIVATE_RETRY_SECONDS = 5.0
# Replanning around a worker that is alive but failed may move this much at most.
REPLAN_MOVE_LIMIT_BYTES = 2 * 1024**3
# Healthy checks before a quarantined worker may return, doubling per offense.
PROBATION_PASSES = 3
MAX_PROBATION_PASSES = 48
# Least time between repairs the orchestrator starts by itself.
REPAIR_COOLDOWN_SECONDS = 120.0
REPAIR_RETRY_SECONDS = 15.0
# Wi-Fi drops out. A stage has to stay gone this long before weights move for it.
REPAIR_GRACE_SECONDS = 30.0
# How long a reply waits for a repair another node is carrying out, and how long
# a pool change waits for the reply in progress before going ahead anyway.
ROUTE_WAIT_SECONDS = 120.0
COORDINATOR_WAIT_SECONDS = 3.0
MAX_ROUTE_MOVES = 8
# New streams a reply may open while recovering, before going one token at a time.
STREAM_ATTEMPTS = 6
DRAIN_WAIT_SECONDS = 45.0
# Long enough for a planned move to finish, for a reply that arrives during one.
MOVE_WAIT_SECONDS = 90.0
# How often a reply with nothing arriving checks that its entry worker is still there.
STREAM_CHECK_SECONDS = 5.0
# How long workers get to come back before a saved route is rebuilt from scratch.
ADOPTION_GRACE_SECONDS = 60.0
CATALOG = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "family": "Qwen2.5", "kind": "dense", "size": "1 GB"},
    {"id": "Qwen/Qwen3-0.6B", "family": "Qwen3", "kind": "dense", "size": "1.5 GB"},
    {"id": "Qwen/Qwen3.5-0.8B", "family": "Qwen3.5", "kind": "hybrid", "size": "1.7 GB"},
    {"id": "Qwen/Qwen3-1.7B", "family": "Qwen3", "kind": "dense", "size": "4 GB"},
    {"id": "Qwen/Qwen3.5-4B", "family": "Qwen3.5", "kind": "hybrid", "size": "9 GB"},
    {"id": "Qwen/Qwen3-8B", "family": "Qwen3", "kind": "dense", "size": "16 GB"},
    {"id": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "family": "Qwen1.5", "kind": "MoE", "size": "29 GB"},
    {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "family": "Qwen3", "kind": "MoE", "size": "61 GB"},
    {"id": "Qwen/Qwen3.6-35B-A3B", "family": "Qwen3.6", "kind": "hybrid MoE", "size": "72 GB"},
    {"id": "Qwen/Qwen3.8-27B", "family": "Qwen3.8", "kind": "hybrid", "size": "54 GB"},
    {"id": "Qwen/Qwen3.8-27B-FP8", "family": "Qwen3.8", "kind": "hybrid FP8", "size": "29 GB"},
    {"id": "Qwen/Qwen3.6-35B-A3B-FP8", "family": "Qwen3.6", "kind": "hybrid MoE FP8", "size": "36 GB"},
    {"id": "ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q4_K_M.gguf", "family": "Qwen3.8", "kind": "hybrid GGUF Q4", "size": "19 GB"},
    {"id": "ggml-org/Qwen3.8-27B-GGUF:Qwen3.8-27B-Q8_0.gguf", "family": "Qwen3.8", "kind": "hybrid GGUF Q8", "size": "29 GB"},
    {"id": "lmstudio-community/Qwen3.6-27B-GGUF:Qwen3.6-27B-Q4_K_M.gguf", "family": "Qwen3.6", "kind": "hybrid GGUF Q4", "size": "17 GB"},
    {"id": "lmstudio-community/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-Q4_K_M.gguf", "family": "Qwen3.6", "kind": "hybrid MoE GGUF Q4", "size": "21 GB"},
    {"id": "distilbert/distilgpt2", "family": "GPT-2", "kind": "dense", "size": "0.3 GB"},
]


class WorkerSilent(Exception):
    """A worker stopped answering without any error, as a lost link does."""


@dataclass(slots=True)
class WorkerRecord:
    registration: WorkerRegistration
    last_seen: float
    quarantined: bool = False
    quarantine_reason: str | None = None
    # Healthy checks in a row while quarantined, and how often it has been sent
    # there: a node that keeps failing has to prove itself for longer.
    probation_passes: int = 0
    quarantine_count: int = 0
    control_latency_ms: float | None = None
    health: dict[str, Any] | None = None
    # Measured download speed from the model host, once a load has shown it.
    fetch_bytes_per_second: float = 0.0

    @property
    def live(self) -> bool:
        return time.monotonic() - self.last_seen < 15.0


@dataclass(slots=True)
class ModelTarget:
    """A model resolved far enough to plan and fetch it."""

    model_id: str
    manifest: Manifest
    adapter: Adapter
    units: dict[str, WeightUnit]
    profile: ModelProfile
    spread: bool
    tokenizer: ChatTokenizer | None = None
    source_revision: str = "main"
    worker_model_ids: dict[str, str] = field(default_factory=dict)
    entry_worker: str | None = None
    exit_worker: str | None = None
    reuse_cached: bool = True
    peer_fetch: bool = True
    # Which units each worker holds on disk, refreshed before every plan.
    held: dict[str, set[str]] = field(default_factory=dict)
    # Workers willing to send those units to others.
    serving: set[str] = field(default_factory=set)
    draft: DraftModel | None = None
    speculate_tokens: int = 0
    request: LoadModelRequest | None = None
    # Workers that hold the whole GGUF, for a pool that cannot reach a model host.
    file_sources: list[str] = field(default_factory=list)
    slots: int = 1
    merge_passes: bool = True

    @property
    def file_sizes(self) -> dict[str, int]:
        config = self.manifest.config
        return {config["gguf_file"]: config["gguf_file_size"]} if "gguf_file" in config else {}

    def source_for(self, worker_id: str) -> str:
        return self.worker_model_ids.get(worker_id, self.model_id)


@dataclass(slots=True)
class LoadJob:
    """One attempt to bring a model onto the pool, visible while it runs."""

    job_id: str
    model_id: str
    phase: str = "resolving"  # resolving, fetching, loading, ready, failed, cancelled
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    route: list[StageAssignment] = field(default_factory=list)
    workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_bytes: int = 0
    cancelled: bool = False

    def snapshot(self) -> dict[str, Any]:
        wanted = sum(item.get("bytes_total", 0) for item in self.workers.values())
        done = sum(item.get("bytes_done", 0) for item in self.workers.values())
        return {
            "job_id": self.job_id,
            "model": self.model_id,
            "phase": self.phase,
            "error": self.error,
            "started_at": int(self.started_at),
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 1),
            "model_bytes": self.total_bytes,
            "bytes_total": wanted,
            "bytes_done": done,
            "bytes_reused": sum(item.get("bytes_reused", 0) for item in self.workers.values()),
            "bytes_from_peers": sum(item.get("bytes_from_peers", 0) for item in self.workers.values()),
            "bytes_per_second": sum(item.get("bytes_per_second", 0) for item in self.workers.values()),
            "workers": self.workers,
            "route": [stage.model_dump() for stage in self.route],
        }


def build_target(
    request: LoadModelRequest,
    reuse_cached: bool = True,
    peer_fetch: bool = True,
    sources: dict[str, list[str]] | None = None,
    token: str | None = None,
) -> ModelTarget:
    """Read a checkpoint's headers and work out its units. Downloads no weights.

    The two flags are defaults: a request that sets either one wins.
    """
    reuse_cached = reuse_cached if request.reuse_cached is None else request.reuse_cached
    peer_fetch = peer_fetch if request.peer_fetch is None else request.peer_fetch
    control_source = request.control_model_id or request.model_id
    sources = sources or {}
    holders = holders_of(sources, control_source)
    manifest = build_manifest(control_source, request.revision, holders, token)
    adapter = create_adapter(manifest.config)
    draft = resolve_draft(request, manifest, adapter, sources, token)
    speculate_tokens = request.draft_tokens if draft else 0
    slots = request.parallel if isinstance(adapter, GgufAdapter) else 1
    units = adapter.partition(manifest.tensors)
    if request.model_hashes:
        apply_model_hashes(manifest, units, request.model_hashes)
    by_kind = {unit.kind: unit for unit in units if unit.kind in {"embed", "head"}}
    embedding = by_kind["embed"].tensors
    head_tensors = {tensor.name for tensor in by_kind["head"].tensors}
    profile = ModelProfile(
        units=units,
        num_layers=adapter.num_layers,
        expert_active_fraction=adapter.expert_active_fraction,
        kv_bytes_per_token=[adapter.kv_bytes_per_token(i, 2) * slots for i in range(adapter.num_layers)],
        # Rewinding needs a snapshot of recurrent state for each guess.
        state_bytes=[adapter.recurrent_state_bytes(i) * (1 + speculate_tokens) * slots for i in range(adapter.num_layers)],
        context_tokens=request.context_tokens,
        # Only a lone embedding table can be shared: GPT-2's unit also holds
        # position embeddings, which the head does not have.
        tied_embeddings=len(embedding) == 1 and embedding[0].name in head_tensors,
        amortize_tokens=AMORTIZE_TOKENS if reuse_cached else 0,
        stage_workspace_bytes=GGUF_STAGE_WORKSPACE_BYTES if isinstance(adapter, GgufAdapter) else 0,
        entry_extra_bytes=draft.file_size + DRAFT_WORKSPACE_BYTES if draft else 0,
        layer_step=max(getattr(adapter, "interval", 1), 1),
    )
    hosted_copy = manifest.revision.startswith("local-") and control_source != request.model_id
    return ModelTarget(
        model_id=request.model_id,
        manifest=manifest,
        adapter=adapter,
        units={unit.name: unit for unit in units},
        profile=profile,
        spread=request.placement == "spread",
        source_revision=request.revision if hosted_copy else manifest.revision,
        worker_model_ids=dict(request.worker_model_ids),
        entry_worker=request.entry_worker,
        exit_worker=request.exit_worker,
        reuse_cached=reuse_cached,
        peer_fetch=peer_fetch,
        draft=draft,
        speculate_tokens=speculate_tokens,
        request=request,
        file_sources=holders,
        slots=slots,
        merge_passes=request.merge_passes,
    )


def holders_of(sources: dict[str, list[str]], model: str) -> list[str]:
    located = parse_source(model)
    return sources.get(located[1], []) if located else []


def resolve_draft(
    request: LoadModelRequest,
    manifest: Manifest,
    adapter: Adapter,
    sources: dict[str, list[str]] | None = None,
    token: str | None = None,
) -> DraftModel | None:
    """Pick a draft model, or None. "auto" never fails a load; a named draft can."""
    named = request.draft_model not in {"auto", "none"}
    if request.draft_model == "none" or not isinstance(adapter, GgufAdapter):
        if named:
            raise UnsupportedModel("speculative decoding needs a GGUF model")
        return None
    choice = request.draft_model if named else AUTO_DRAFTS.get(adapter.architecture)
    if not choice or parse_source(choice) == parse_source(request.model_id):
        return None
    if (parse_source(choice) or ("", ""))[1] == (parse_source(request.model_id) or ("", ""))[1]:
        return None  # the same file under another name would only guess at itself
    try:
        holders = holders_of(sources or {}, choice)
        draft = build_manifest(choice, "main", holders, token)
        if draft.config.get("gguf_vocabulary") != manifest.config.get("gguf_vocabulary"):
            raise UnsupportedModel(f"draft model {choice} has a different vocabulary from {request.model_id}")
    except (UnsupportedModel, WeightSourceError, OSError, httpx.HTTPError, KeyError, ValueError):
        if named:
            raise
        return None
    located = parse_source(choice)
    assert located is not None
    return DraftModel(
        source=choice,
        file=located[1],
        file_size=draft.config["gguf_file_size"],
        revision=draft.revision,
        file_sources=holders,
    )


def describe_plan(target: ModelTarget, route: list[StageAssignment]) -> dict[str, Any]:
    """Summarize a plan for people: what goes where, and what it will cost."""
    # Bandwidth alone flatters small active sets, such as a MoE's: every layer still
    # costs its kernel launches, and every stage boundary costs a network hop.
    milliseconds = sum(
        max(stage.estimated_ms_per_token, (stage.end_layer - stage.start_layer) * LAYER_FLOOR_MS) for stage in route
    ) + HOP_MS * (len(route) - 1)
    download = sum(
        target.units[item.unit].nbytes
        for stage in route
        for item in stage.placements
        if item.bytes and item.unit not in target.held.get(stage.worker_id, set())
    )
    return {
        "model": target.model_id,
        "architecture": target.adapter.config.get("model_type"),
        "layers": target.adapter.num_layers,
        "experts": getattr(target.adapter, "num_experts", 0),
        "model_bytes": target.profile.total_bytes,
        "draft_model": target.draft.source if target.draft else None,
        "download_bytes": download,
        "route": [stage.model_dump() for stage in route],
        "entry_worker": route[0].worker_id,
        "exit_worker": route[-1].worker_id,
        "estimated_tokens_per_second": round(1000 / milliseconds, 1) if milliseconds else None,
    }


class Orchestrator:
    def __init__(self) -> None:
        self.pool_token = load_token()
        self.reuse_cached = os.getenv("SHARDMESH_REUSE_CACHED", "1") != "0"
        self.peer_fetch = os.getenv("SHARDMESH_PEER_FETCH", "1") != "0"
        self.workers: dict[str, WorkerRecord] = {}
        self.model_id: str | None = None
        self.planning_model: str | None = None
        self.active: ModelTarget | None = None
        self.tokenizer: ChatTokenizer | None = None
        self.route: list[StageAssignment] = []
        self.active_cache_url: str | None = None
        self.epoch = 0
        self.job: LoadJob | None = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0), verify=client_verify(), auth=client_auth())
        authority = load_authority()
        self.access = AccessControl.load(
            self.pool_token,
            os.getenv("SHARDMESH_KEYS_FILE"),
            identities=authority is not None,
            authority=authority,
        )
        self.audit = AuditLog(os.getenv("SHARDMESH_AUDIT_FILE"), os.getenv("SHARDMESH_NODE_ID") or socket.gethostname())
        self.control_lock = asyncio.Lock()
        self.generation_lock = asyncio.Lock()
        # Survives a restart: what was running, so the pool can be picked back up.
        self.state_path = Path(os.environ["SHARDMESH_STATE_FILE"]) if os.getenv("SHARDMESH_STATE_FILE") else None
        self.saved_state: dict[str, Any] | None = None
        self.saved_state_seen_at = 0.0
        # Route members that dropped out, to bring back when they recover.
        self.lost_from_route: set[str] = set()
        self.absent_since: dict[str, float] = {}
        self.need_reconfigure: set[str] = set()
        # The route this node has installed, which followers keep up to date.
        self.installed_epoch: int | None = None
        self.following = False
        self.follow_retry_at = 0.0
        self.last_repair_at = float("-inf")  # monotonic time starts near zero after a reboot
        self.stage_restarts = 0
        self.stream_fallbacks = 0
        self.stage_errors = 0
        self.replies_in_flight = 0
        self.warned_interception: set[str] = set()
        self.drained: set[str] = set()
        self.reply_checker: TicketChecker | None = None
        self.maintenance_task: asyncio.Task[None] | None = None
        self.gossip_task: asyncio.Task[None] | None = None
        self.mesh: ControlMesh | None = None
        seen_as = os.getenv("SHARDMESH_NODE_URL")
        if seen_as:
            seeds = tuple(url.strip() for url in os.getenv("SHARDMESH_JOIN", "").split(",") if url.strip())
            self.mesh = ControlMesh(os.getenv("SHARDMESH_NODE_ID") or socket.gethostname(), seen_as, seeds)
        self.background: set[asyncio.Task[Any]] = set()
        self.lanes = LaneDoor(self)
        self.round_trip_ms: dict[str, float] = {}
        self.lane_name = os.getenv("SHARDMESH_LANE") or None
        self.started_at = time.time()
        self.requests_total = 0
        self.tokens_generated = 0
        self.recoveries_total = 0
        self.token_latency_ms_ema: float | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=40)
        self.add_event("pool", "Orchestrator online", "Control plane is accepting workers")

    @staticmethod
    def _ema(previous: float | None, value: float, weight: float = 0.2) -> float:
        return value if previous is None else previous * (1 - weight) + value * weight

    def add_event(self, kind: str, title: str, detail: str) -> None:
        if kind in LOGGED_EVENTS:
            # Events live in memory, so the reason for a failure would die with
            # this process. It is usually the one thing someone needs afterwards.
            try:
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {kind}: {title}: {detail}", flush=True)
            except (OSError, ValueError):
                pass  # a full disk or an odd console must not fail what is being logged
        self.events.appendleft(
            {"id": uuid.uuid4().hex[:10], "at": int(time.time()), "kind": kind, "title": title, "detail": detail}
        )

    def spawn(self, work: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(work)
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        return task

    async def close(self) -> None:
        for task in (self.maintenance_task, self.gossip_task):
            if task is not None:
                task.cancel()
        await self.client.aclose()

    def load_saved_state(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            LoadModelRequest.model_validate(state["request"])
            self.epoch = int(state["epoch"])
            if self.mesh is not None:
                # Only whoever ends up coordinating acts on this.
                self.mesh.replica = state
                self.mesh.learn_term(int(state.get("term", 0)))
                return
            self.saved_state = state
            self.saved_state_seen_at = time.monotonic()
            self.add_event("pool", "Recovering", f"Waiting for the workers of {state['request']['model_id']}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.add_event("failure", "Saved state unreadable", str(exc)[:200])

    def save_state(self, target: ModelTarget) -> None:
        if target.request is None:
            return
        state = {
            "request": target.request.model_dump(),
            "revision": target.manifest.revision,
            "epoch": self.epoch,
            "route": [stage.model_dump() for stage in self.route],
            "term": self.mesh.term if self.mesh else 0,
        }
        if self.mesh is not None:
            self.mesh.publish(state)
        self._write_state(state)

    def _write_state(self, state: dict[str, Any]) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.state_path.with_suffix(".part")
        partial.write_text(json.dumps(state), encoding="utf-8")
        partial.replace(self.state_path)

    @property
    def coordinating(self) -> bool:
        return self.mesh is None or self.mesh.is_leader

    def gossip_message(self) -> dict[str, Any]:
        assert self.mesh is not None
        return {**self.mesh.message(), "access": self.access.document()}

    def receive_gossip(self, message: dict[str, Any]) -> None:
        assert self.mesh is not None
        if str(message.get("node_id")) in self.access.revoked:
            return
        held = self.mesh.replica_version
        lost_leadership = self.mesh.observe(message)
        self.access.merge(message.get("access"))
        self.apply_revocations()
        if self.mesh.replica_version != held and self.mesh.replica:
            self._write_state(self.mesh.replica)
        if lost_leadership:
            self._step_down()

    def _reply_is_genuine(self, reply: dict[str, Any]) -> bool:
        """A reply must prove it comes from the node it names, as a request must."""
        if self.access.authority is None:
            return True
        assert self.mesh is not None
        if self.reply_checker is None:
            self.reply_checker = make_checker({reply_audience(self.mesh.url)})
        proof = str(reply.get("proof", ""))
        return proof.startswith(TICKET_PREFIX) and self.reply_checker.name(proof) == reply.get("node_id")

    def apply_revocations(self) -> None:
        """Act on revoked nodes, wherever the news came from."""
        if self.mesh:
            self.mesh.forget(self.access.revoked)
        for worker_id, record in self.workers.items():
            if node_of(worker_id) in self.access.revoked and not record.quarantined:
                self._quarantine(worker_id, "its node was revoked")

    def _step_down(self) -> None:
        """Another node outranks this one, so it stops coordinating at once.

        It keeps its route: replies in progress need it, and it serves as a
        follower from here, until the new coordinator's state replaces it.
        """
        self.saved_state = None
        if self.job is not None:
            self.job.cancelled = True
        leader = self.mesh.leader_id if self.mesh else None
        self.add_event("mesh", "Stepped down", f"{leader} coordinates now")

    def _take_over(self) -> None:
        assert self.mesh is not None
        state = self.mesh.replica
        self.add_event("mesh", "Coordinating", f"{self.mesh.node_id} leads term {self.mesh.term}")
        if state:
            self.saved_state = state
            self.saved_state_seen_at = time.monotonic()
            self.epoch = max(self.epoch, int(state["epoch"]))

    async def _gossip_round(self) -> None:
        assert self.mesh is not None

        async def talk(url: str) -> None:
            started = time.monotonic()
            try:
                response = await self.client.post(
                    url + "/internal/gossip",
                    json=self.gossip_message(),
                    headers=auth_headers(self.pool_token),
                    timeout=2.5,
                )
                response.raise_for_status()
                reply = response.json()
                if self._reply_is_genuine(reply):
                    self.round_trip_ms[str(reply.get("node_id"))] = round((time.monotonic() - started) * 1000, 1)
                    self.receive_gossip(reply)
            except (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError, TypeError) as exc:
                # Silence is how a peer's absence is noticed, so this says nothing
                # unless the cause is one a person has to fix.
                if intercepted(exc) and url not in self.warned_interception:
                    self.warned_interception.add(url)
                    self.add_event("failure", f"Cannot trust {url}", INTERCEPTED)

        # One bad address must never stop the election below.
        await asyncio.gather(*(talk(url) for url in self.mesh.targets()), return_exceptions=True)
        for record in self.workers.values():
            self.mesh.learn_term(int((record.health or {}).get("term", 0) or 0))
        if self.mesh.elect():
            self._take_over()
        if self._should_follow():
            self.following = True
            self.spawn(self._follow())

    def _should_follow(self) -> bool:
        state = self.mesh.replica if self.mesh else None
        return (
            state is not None
            and not self.coordinating
            and not self.following
            and int(state["epoch"]) != self.installed_epoch
            and time.monotonic() >= self.follow_retry_at
        )

    async def _follow(self) -> None:
        """Keep a local copy of the route, so this node can serve replies itself."""
        try:
            await self._install(self.mesh.replica)
        except Exception as exc:  # a follower that cannot install yet forwards instead
            self.follow_retry_at = time.monotonic() + 15.0
            self.add_event("failure", "Could not follow the route", f"{type(exc).__name__}: {exc}"[:200])
        finally:
            self.following = False

    async def gguf_sources(self) -> dict[str, list[str]]:
        """Which workers hold which whole GGUF files right now."""

        async def ask(record: WorkerRecord) -> tuple[str, list[str]]:
            url = str(record.registration.advertised_url).rstrip("/")
            try:
                response = await self.client.get(
                    url + "/internal/gguf/index", headers=auth_headers(self.pool_token), timeout=30.0
                )
                response.raise_for_status()
                return url, [item["name"] for item in response.json()["files"]]
            except (httpx.HTTPError, ValueError, KeyError):
                return url, []

        answers = await asyncio.gather(*(ask(record) for record in self.workers.values() if record.live))
        holders: dict[str, list[str]] = {}
        for url, names in answers:
            for name in names:
                holders.setdefault(name, []).append(url)
        return holders

    async def _resolve(self, request: LoadModelRequest) -> ModelTarget:
        """Read a model's description, from inside the pool if anyone there has it."""
        sources = await self.gguf_sources()
        target = await asyncio.to_thread(
            build_target, request, self.reuse_cached, self.peer_fetch, sources, self.pool_token
        )
        source = request.control_model_id or request.model_id
        target.tokenizer = await asyncio.to_thread(
            load_tokenizer, source, request.revision, holders_of(sources, source), self.pool_token
        )
        return target

    async def _install(self, state: dict[str, Any]) -> None:
        """Make a replicated route this node's own, without touching any worker."""
        request = LoadModelRequest.model_validate(state["request"])
        held = self.active
        if held is None or held.request != request or held.manifest.revision != state.get("revision"):
            # Only a new model needs reading. A new route for this one is just data.
            target = await self._resolve(request)
            self.active = target
            self.model_id = target.model_id
            self.tokenizer = target.tokenizer
        self.route = [StageAssignment.model_validate(item) for item in state["route"]]
        self.active_cache_url = None
        self.epoch = int(state["epoch"])
        self.installed_epoch = self.epoch

    async def _gossip(self) -> None:
        while True:
            try:
                await self._gossip_round()
            except Exception as exc:  # membership must outlive any one bad message
                self.add_event("failure", "Gossip error", f"{type(exc).__name__}: {exc}"[:200])
            await asyncio.sleep(GOSSIP_SECONDS)

    def start_maintenance(self) -> None:
        self.load_saved_state()
        self.maintenance_task = asyncio.create_task(self._maintain())
        if self.mesh is not None:
            self.gossip_task = asyncio.create_task(self._gossip())
        if self.lanes.active:
            self.lanes.restore()
            self.spawn(self.lanes.run())

    async def _maintain(self) -> None:
        while True:
            await asyncio.sleep(MAINTENANCE_SECONDS)
            try:
                await self.maintenance_pass()
            except HTTPException as exc:
                self.add_event("failure", "Automatic repair failed", str(exc.detail)[:200])
            except Exception as exc:  # this loop is what keeps the pool healthy, so it must outlive any one failure
                self.add_event("failure", "Maintenance error", f"{type(exc).__name__}: {exc}"[:200])

    async def maintenance_pass(self) -> None:
        await self.refresh_worker_metrics()
        if not self.coordinating:
            return
        self._review_quarantine()
        await self._adopt_saved_route()
        await self._repair_route()

    def _review_quarantine(self) -> None:
        """Let a quarantined worker back once it has stayed healthy for a while."""
        for worker_id, record in self.workers.items():
            if not record.quarantined:
                continue
            if any(worker_id == node or worker_id.startswith(node + "-") for node in self.access.revoked):
                continue  # only an admin brings a revoked node back
            healthy = record.live and (record.health or {}).get("status") == "ok"
            record.probation_passes = record.probation_passes + 1 if healthy else 0
            needed = min(PROBATION_PASSES * 2 ** max(record.quarantine_count - 1, 0), MAX_PROBATION_PASSES)
            if record.probation_passes >= needed:
                record.quarantined = False
                record.quarantine_reason = None
                record.probation_passes = 0
                self.add_event("recovery", f"{worker_id} returned", f"Healthy for {needed} checks in a row")

    async def _adopt_saved_route(self) -> None:
        """After a restart, take the running pool back instead of reloading it."""
        state = self.saved_state
        if state is None or self.control_lock.locked():
            return
        route = [StageAssignment.model_validate(item) for item in state["route"]]
        request = LoadModelRequest.model_validate(state["request"])
        if self.installed_epoch == state["epoch"] and self.route:
            self.saved_state = None  # already following this route
            return
        serving = []
        for stage in route:
            record = self.workers.get(stage.worker_id)
            health = (record.health or {}) if record and record.live else {}
            serving.append(health.get("model") == request.model_id and health.get("epoch") == state["epoch"])
        if all(serving):
            async with self.control_lock:
                await self._install(state)
                self.saved_state = None
            self.add_event("recovery", "Pool recovered", f"{request.model_id} kept running through a restart")
            return
        if time.monotonic() - self.saved_state_seen_at < ADOPTION_GRACE_SECONDS or not self.candidates():
            return
        self.saved_state = None
        self.add_event("recovery", "Reloading", f"Its workers did not all return, so {request.model_id} is loading again")
        self.job = LoadJob(job_id=uuid.uuid4().hex, model_id=request.model_id)
        async with self.control_lock:
            await self.run_job(self.job, self.load(request, self.job))

    def _absent_from_route(self) -> set[str]:
        absent = set()
        for stage in self.route:
            record = self.workers.get(stage.worker_id)
            if record is None or not record.live or record.quarantined or self._cordoned(stage.worker_id):
                absent.add(stage.worker_id)
                continue
            # Restarted workers come back live but empty, having lost their stage.
            health = record.health or {}
            if health.get("status") == "ok" and health.get("epoch") != self.epoch:
                self.need_reconfigure.add(stage.worker_id)
            else:
                self.need_reconfigure.discard(stage.worker_id)
        return absent

    async def _repair_route(self) -> None:
        """Replace a lost stage while idle, and take back a node that recovered.

        Requests repair the pipeline when they trip over a failure. Doing it here
        as well means the next person to ask is not the one who pays for it.
        """
        if self.active is None or not self.route:
            return
        now = time.monotonic()
        missing = self._absent_from_route()
        self.absent_since = {worker: self.absent_since.get(worker, now) for worker in missing}
        absent = {worker for worker, since in self.absent_since.items() if now - since >= REPAIR_GRACE_SECONDS}
        self.lost_from_route |= absent
        returned = {
            worker_id
            for worker_id in self.lost_from_route
            if (record := self.workers.get(worker_id)) and record.live and not record.quarantined and not self._cordoned(worker_id)
        }
        if not absent and not returned and not self.need_reconfigure:
            return
        waited = time.monotonic() - self.last_repair_at
        # A reply in progress waits out a move, except one that cannot finish: with
        # a stage gone or restarted empty it needs this repair, and they must not
        # wait for each other.
        broken = absent or self.need_reconfigure
        if waited < REPAIR_COOLDOWN_SECONDS or self.control_lock.locked() or (self.replies_in_flight and not broken):
            return
        self.last_repair_at = time.monotonic()
        if absent:
            reason = f"{', '.join(sorted(absent))} lost"
        elif returned:
            reason = f"{', '.join(sorted(returned))} returned"
        else:
            reason = f"{', '.join(sorted(self.need_reconfigure))} restarted without its stage"
        self.add_event("recovery", "Repairing the route", reason)
        try:
            async with self.control_lock:
                await self.rebalance(generation_lock_held=False)
        except HTTPException:
            # Nothing could take the model yet. A worker may be on its way back,
            # and the replies waiting on it should not sit out a full cooldown.
            self.last_repair_at = time.monotonic() - REPAIR_COOLDOWN_SECONDS + REPAIR_RETRY_SECONDS
            raise
        self.lost_from_route -= {stage.worker_id for stage in self.route} | returned
        self.need_reconfigure.clear()

    def register(self, registration: WorkerRegistration) -> None:
        previous = self.workers.get(registration.worker_id)
        if previous is not None and previous.live:
            previous.registration = registration
            previous.last_seen = time.monotonic()
        else:
            record = WorkerRecord(registration=registration, last_seen=time.monotonic())
            if previous is not None:
                # Going quiet and coming back is no way out of quarantine.
                record.quarantined, record.quarantine_reason = previous.quarantined, previous.quarantine_reason
                record.quarantine_count = previous.quarantine_count
            self.workers[registration.worker_id] = record
        if previous is None:
            hardware = registration.hardware
            label = hardware.accelerator_name or hardware.backend.upper()
            kinds = " + ".join(pool.kind for pool in hardware.memory_pools)
            detail = f"{label} via {registration.advertised_url}"
            self.add_event("join", f"{registration.worker_id} joined", f"{detail} ({kinds})" if kinds else detail)

    def heartbeat(self, worker_id: str) -> bool:
        record = self.workers.get(worker_id)
        if not record:
            return False
        record.last_seen = time.monotonic()
        return True

    @staticmethod
    def _live_pools(record: WorkerRecord) -> list[MemoryPool]:
        """A worker's pools with current free space.

        Memory already holding this pool's weights counts as free, because a
        new placement replaces those weights.
        """
        pools = record.registration.hardware.memory_pools
        reported = {item["pool_id"]: item for item in (record.health or {}).get("memory_pools", [])}
        live: list[MemoryPool] = []
        for pool in pools:
            status = reported.get(pool.pool_id)
            if status is None:
                live.append(pool)
            else:
                free = int(status["free_bytes"]) + int(status.get("used_bytes", 0))
                live.append(pool.model_copy(update={"free_bytes": free}))
        return live

    def _candidate(self, worker_id: str, record: WorkerRecord) -> Candidate:
        return Candidate(
            worker_id=worker_id,
            url=str(record.registration.advertised_url),
            pools=self._live_pools(record),
        )

    def _cordoned(self, worker_id: str) -> bool:
        return worker_id in self.access.cordoned or node_of(worker_id) in self.access.cordoned

    def cost_of_losing(self, name: str) -> int | None:
        """Bytes the pool must transfer if this node leaves, or None if it cannot cope."""
        if self.active is None or not any(node_of(stage.worker_id) == name for stage in self.route):
            return 0
        held = self.access.cordoned
        self.access.cordoned = held | {name}
        try:
            return int(describe_plan(self.active, self.plan(self.active))["download_bytes"])
        except PlacementError:
            return None
        finally:
            self.access.cordoned = held

    async def move_off(self) -> None:
        """Carry out a cordon: replies finish first, and only that node's share moves."""
        async with self.control_lock:
            await self.rebalance(generation_lock_held=False)

    def candidate_groups(self) -> tuple[list[Candidate], list[Candidate]]:
        """Healthy inference workers, split into preferred and standby."""
        preferred: list[Candidate] = []
        standby: list[Candidate] = []
        for worker_id, record in self.workers.items():
            contribution = record.registration.hardware.contribution
            usable = record.live and not record.quarantined and not self._cordoned(worker_id)
            if usable and "inference" in contribution.roles and self._hosts(contribution):
                group = standby if contribution.standby else preferred
                group.append(self._candidate(worker_id, record))
        return preferred, standby

    def _hosts(self, contribution: Any) -> bool:
        model = self.planning_model or self.model_id
        return not model or hosts_model(contribution.host_models, model)

    def candidates(self) -> list[Candidate]:
        preferred, standby = self.candidate_groups()
        return preferred or standby

    def choose_cache_url(self, route: list[StageAssignment]) -> str | None:
        route_workers = {stage.worker_id for stage in route}
        candidates = [
            (worker_id, record)
            for worker_id, record in self.workers.items()
            if record.live
            and not record.quarantined
            and "kv-cache" in record.registration.hardware.contribution.roles
            and record.registration.hardware.contribution.ram_budget_bytes > 0
        ]
        if not candidates:
            return None
        worker_id, record = max(
            candidates,
            key=lambda item: (
                item[0] not in route_workers,
                item[1].registration.hardware.contribution.ram_budget_bytes,
            ),
        )
        return str(record.registration.advertised_url).rstrip("/")

    async def refresh_holdings(self, target: ModelTarget) -> None:
        """Ask every worker which of this model's units are on its disk."""

        async def ask(worker_id: str, record: WorkerRecord) -> tuple[str, set[str], bool]:
            try:
                response = await self.client.get(
                    str(record.registration.advertised_url).rstrip("/") + "/internal/shards/index",
                    params={"model_id": target.model_id, "revision": target.manifest.revision},
                    headers=auth_headers(self.pool_token),
                    timeout=10.0,
                )
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError):
                return worker_id, set(), False
            # A smaller file than the unit's data cannot be a finished copy.
            finished = {
                name
                for name, size in body.get("units", {}).items()
                if name in target.units and size >= target.units[name].nbytes
            }
            return worker_id, finished, bool(body.get("serves_peers"))

        answers = await asyncio.gather(
            *(ask(worker_id, record) for worker_id, record in self.workers.items() if record.live)
        )
        target.held = {worker_id: finished for worker_id, finished, _ in answers}
        target.serving = {worker_id for worker_id, _, serves in answers if serves}

    def _peers_for(self, target: ModelTarget, worker_id: str, units: list[WeightUnit]) -> dict[str, list[str]]:
        """Other workers that can hand over each unit, nearest guess first."""
        if not target.peer_fetch:
            return {}
        peers: dict[str, list[str]] = {}
        for unit in units:
            for other, finished in target.held.items():
                record = self.workers.get(other)
                if other != worker_id and other in target.serving and unit.name in finished and record and record.live:
                    peers.setdefault(unit.name, []).append(str(record.registration.advertised_url).rstrip("/"))
        return peers

    def _with_holdings(self, group: list[Candidate], target: ModelTarget) -> list[Candidate]:
        """Tell the planner what each worker has, and what it could get cheaply."""
        if not target.reuse_cached:
            return group
        updated: list[Candidate] = []
        for candidate in group:
            shared: set[str] = set()
            if target.peer_fetch:
                for other, finished in target.held.items():
                    if other != candidate.worker_id and other in target.serving:
                        shared |= finished
            record = self.workers.get(candidate.worker_id)
            updated.append(
                replace(
                    candidate,
                    cached=frozenset(target.held.get(candidate.worker_id, set())),
                    peer_units=frozenset(shared),
                    fetch_bytes_per_second=record.fetch_bytes_per_second if record else 0.0,
                )
            )
        return updated

    def plan(self, target: ModelTarget, *, honor_pins_strictly: bool = False) -> list[StageAssignment]:
        """Plan a route. A pin on a lost worker is dropped, not fatal.

        Recovery must not fail because a preferred machine died, so only an
        explicit request treats an unavailable pin as an error.
        """
        preferred, standby = self.candidate_groups()
        available = {candidate.worker_id for candidate in preferred + standby}
        pins: dict[str, str | None] = {"entry_worker": target.entry_worker, "exit_worker": target.exit_worker}
        if not honor_pins_strictly:
            for role, worker_id in pins.items():
                if worker_id and worker_id not in available:
                    self.add_event("failure", f"{worker_id} is unavailable", f"Planning without its {role} pin")
                    pins[role] = None
        return plan_placement(
            self._with_holdings(preferred, target),
            self._with_holdings(standby, target),
            target.profile,
            spread=target.spread,
            **pins,
        )

    def _quarantine(self, worker_id: str, reason: str) -> None:
        record = self.workers.get(worker_id)
        if record is not None:
            record.quarantined = True
            record.quarantine_reason = reason
            record.quarantine_count += 1
            record.probation_passes = 0
        self.add_event("failure", f"{worker_id} quarantined", reason)

    async def _prefetch(
        self,
        target: ModelTarget,
        route: list[StageAssignment],
        job: LoadJob,
    ) -> dict[str, str]:
        """Have every stage download its allowlist. Returns workers that failed.

        Runs without any lock: the current model keeps serving meanwhile.
        """
        headers = auth_headers(self.pool_token)
        failed: dict[str, str] = {}
        pending: dict[str, StageAssignment] = {}
        for stage in route:
            # A zero-byte placement shares the head's tensors: nothing to fetch.
            units = [target.units[item.unit] for item in stage.placements if item.bytes]
            request = PrepareRequest(
                job_id=job.job_id,
                model_id=target.model_id,
                source=target.source_for(stage.worker_id),
                revision=target.manifest.revision,
                source_revision=target.source_revision,
                units=units,
                peers=self._peers_for(target, stage.worker_id, units),
                file_sizes=target.file_sizes,
                file_sources=target.file_sources,
            )
            try:
                response = await self.client.post(
                    stage.url.rstrip("/") + "/internal/prepare",
                    json=request.model_dump(),
                    headers=headers,
                    timeout=60.0,
                )
                response.raise_for_status()
                job.workers[stage.worker_id] = response.json()
                pending[stage.worker_id] = stage
            except httpx.HTTPError as exc:
                failed[stage.worker_id] = f"could not start download: {exc}"

        missed = dict.fromkeys(pending, 0)
        while pending and not failed and not job.cancelled:
            await asyncio.sleep(PROGRESS_POLL_SECONDS)
            for worker_id, stage in list(pending.items()):
                try:
                    response = await self.client.get(
                        stage.url.rstrip("/") + f"/internal/prepare/{job.job_id}",
                        headers=headers,
                        timeout=10.0,
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    missed[worker_id] += 1
                    if missed[worker_id] >= MAX_MISSED_POLLS:
                        failed[worker_id] = f"stopped reporting download progress: {exc}"
                        pending.pop(worker_id)
                    continue
                missed[worker_id] = 0
                progress = response.json()
                job.workers[worker_id] = progress
                if progress["phase"] == "ready":
                    pending.pop(worker_id)
                    # Only large, direct downloads say anything about the link.
                    if progress["bytes_done"] > 256 * 1024**2 and not progress.get("bytes_from_peers"):
                        self.workers[worker_id].fetch_bytes_per_second = progress["bytes_per_second"]
                elif progress["phase"] == "failed":
                    failed[worker_id] = progress["error"] or "download failed"
                    pending.pop(worker_id)

        if job.cancelled or failed:
            # Units already finished stay cached, so a retry resumes from them.
            await asyncio.gather(
                *(
                    self.client.delete(
                        stage.url.rstrip("/") + f"/internal/prepare/{job.job_id}",
                        headers=headers,
                        timeout=10.0,
                    )
                    for stage in pending.values()
                ),
                return_exceptions=True,
            )
        return failed

    async def _activate(
        self,
        target: ModelTarget,
        route: list[StageAssignment],
    ) -> dict[str, str]:
        """Move fetched units into memory and switch to the new route."""
        # KV checkpoints append to a snapshot. Recurrent state changes whole
        # every token, so hybrid models recover by replaying the prompt.
        cache_url = self.choose_cache_url(route) if target.adapter.supports_kv_snapshots else None
        new_epoch = self.epoch + 1
        await self._drain_entries(route)
        self.route = []  # nothing may run while weights are moving

        async def configure(stage: StageAssignment) -> None:
            request = ConfigureRequest(
                model_id=target.model_id,
                revision=target.manifest.revision,
                architecture=target.manifest.config,
                epoch=new_epoch,
                assignment=stage,
                route=route,
                cache_url=cache_url,
                context_tokens=target.profile.context_tokens,
                speculate_tokens=target.speculate_tokens,
                draft=target.draft,
                term=self.mesh.term if self.mesh else 0,
                slots=target.slots,
                merge_passes=target.merge_passes,
                unit_hashes={
                    item.unit: target.request.model_hashes[item.unit]
                    for item in stage.placements
                    if item.unit in (target.request.model_hashes or {})
                },
            )
            response = await self.client.post(
                stage.url.rstrip("/") + "/internal/configure",
                json=request.model_dump(),
                headers=auth_headers(self.pool_token),
            )
            if response.status_code >= 400:
                raise RuntimeError(response.text[:400])

        results = await asyncio.gather(*(configure(stage) for stage in route), return_exceptions=True)
        await self._release_entries()
        failed = {
            stage.worker_id: str(result)
            for stage, result in zip(route, results)
            if isinstance(result, BaseException)
        }
        if failed:
            return failed

        self.active = target
        self.model_id = target.model_id
        if target.tokenizer is not None:
            self.tokenizer = target.tokenizer
        self.route = route
        self.active_cache_url = cache_url
        self.epoch = new_epoch
        self.installed_epoch = new_epoch
        self.saved_state = None
        self.save_state(target)
        if self.mesh is not None:
            self.spawn(self._gossip_round())  # peers should not wait for the next tick
        summary = " → ".join(f"{stage.worker_id} L{stage.start_layer}–{stage.end_layer - 1}" for stage in route)
        self.add_event("rebalance", f"Pipeline epoch {new_epoch}", summary)
        return {}

    async def _drain_entries(self, route: list[StageAssignment]) -> None:
        """Let replies finish before anything moves.

        This node's own replies wait on the generation lock, but other nodes run
        replies too, and all of them pass through the entry worker.
        """
        # Asking a worker nobody can reach would only hold up the repair it needs.
        entries = {
            stage.url
            for stage in (route[:1] + self.route[:1])
            if (record := self.workers.get(stage.worker_id)) is not None and record.live
        }
        self.drained = entries

        async def drain(url: str) -> None:
            try:
                await self.client.post(
                    url.rstrip("/") + "/internal/drain",
                    json={"seconds": DRAIN_WAIT_SECONDS},
                    headers=auth_headers(self.pool_token),
                    timeout=DRAIN_WAIT_SECONDS + 5,
                )
            except httpx.HTTPError:
                pass  # a dead entry has nothing left to protect

        await asyncio.gather(*(drain(url) for url in entries))

    async def _release_entries(self) -> None:
        """Let drained entry workers go, once the route has moved on.

        A worker leaving the route gets no configure to release it, and replies
        queued there would otherwise sit until its timer ran out: seen as 178 s of
        silence. Released, they find their epoch is gone and move to the new route.
        """

        async def release(url: str) -> None:
            try:
                await self.client.post(url.rstrip("/") + "/internal/release", headers=auth_headers(self.pool_token), timeout=5.0)
            except httpx.HTTPError:
                pass  # its own timer still covers it

        drained, self.drained = self.drained, set()
        await asyncio.gather(*(release(url) for url in drained))

    async def wait_for_route(self) -> None:
        """A reply arriving during a move waits for it, rather than being turned away."""
        deadline = time.monotonic() + MOVE_WAIT_SECONDS
        while not self.route and self.active is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

    async def deploy(
        self,
        target: ModelTarget,
        job: LoadJob,
        *,
        generation_lock_held: bool,
    ) -> list[StageAssignment]:
        """Plan, fetch, and activate, replanning around workers that fail."""
        job.total_bytes = target.profile.total_bytes
        for _ in range(MAX_DEPLOY_ATTEMPTS):
            await self.refresh_worker_metrics()
            await self.refresh_holdings(target)
            try:
                route = self.plan(target)
            except PlacementError as exc:
                raise HTTPException(status_code=507, detail=str(exc)) from exc
            job.route = route
            job.workers = {}
            job.phase = "fetching"
            failed = await self._prefetch(target, route, job)
            if job.cancelled:
                raise HTTPException(status_code=409, detail="load cancelled")
            if not failed:
                job.phase = "loading"
                # Waiting here puts the switch at a request boundary.
                gate = nullcontext() if generation_lock_held else self.generation_lock
                async with gate:
                    failed = await self._activate(target, route)
                    for _ in range(ACTIVATE_RETRIES):
                        # Often a passing shortage, such as memory not yet returned.
                        if not failed or job.cancelled:
                            break
                        self.add_event("failure", "Activation failed, retrying", str(failed)[:300])
                        await asyncio.sleep(ACTIVATE_RETRY_SECONDS)
                        failed = await self._activate(target, route)
            if not failed:
                return route
            await self._refuse_big_move(target, failed)
            for worker_id, reason in failed.items():
                self._quarantine(worker_id, reason)
        raise HTTPException(status_code=503, detail=f"gave up after {MAX_DEPLOY_ATTEMPTS} attempts: {failed}")

    async def _refuse_big_move(self, target: ModelTarget, failed: dict[str, str]) -> None:
        """Weights move only when they must.

        A node that died leaves no choice. A node that is alive but failed to load
        is different: moving its share elsewhere can take far longer than fixing
        it, and can land the model somewhere slow. So that takes a person.
        """
        await self.refresh_worker_metrics()
        alive = [
            worker_id
            for worker_id in failed
            if (record := self.workers.get(worker_id)) and record.live and (record.health or {}).get("status") == "ok"
        ]
        if not alive:
            return
        held = {worker_id: self.workers[worker_id].quarantined for worker_id in alive}
        for worker_id in alive:
            self.workers[worker_id].quarantined = True
        try:
            route = self.plan(target)
            moved = describe_plan(target, route)["download_bytes"]
        except PlacementError:
            moved = None
        finally:
            for worker_id, state in held.items():
                self.workers[worker_id].quarantined = state
        if moved is None or moved > REPLAN_MOVE_LIMIT_BYTES:
            size = "the model could not fit elsewhere" if moved is None else f"replanning would move {moved / 1024**3:.1f} GiB"
            raise HTTPException(status_code=503, detail=f"{', '.join(alive)} failed to load, and {size}: {failed}")

    async def run_job(self, job: LoadJob, work: Any) -> list[StageAssignment]:
        """Run a load to completion, recording how it ended."""
        try:
            route = await work
            job.phase = "ready"
            return route
        except HTTPException as exc:
            job.phase = "cancelled" if job.cancelled else "failed"
            job.error = str(exc.detail)
            if not job.cancelled:
                self.add_event("failure", "Model load failed", job.error)
            raise
        except Exception as exc:
            # A background job has no caller to report to, so an unexpected
            # failure has to land on the job or it would look stuck forever.
            job.phase = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self.add_event("failure", "Model load failed", job.error)
            raise HTTPException(status_code=500, detail=job.error) from exc
        finally:
            job.finished_at = time.time()

    async def load(self, request: LoadModelRequest, job: LoadJob) -> list[StageAssignment]:
        self.planning_model = request.model_id
        try:
            return await self._load(request, job)
        finally:
            self.planning_model = None

    async def _load(self, request: LoadModelRequest, job: LoadJob) -> list[StageAssignment]:
        try:
            target = await self._resolve(request)
        except (UnsupportedModel, WeightSourceError, OSError, httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"could not load model: {exc}") from exc
        try:
            await self.refresh_worker_metrics()
            await self.refresh_holdings(target)
            self.plan(target, honor_pins_strictly=True)  # a bad pin should fail now, loudly
        except PlacementError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        route = await self.deploy(target, job, generation_lock_held=False)
        self.add_event("model", "Model loaded", f"{request.model_id} across {len(route)} stage(s)")
        return route

    async def rebalance(
        self,
        include_quarantined: bool = False,
        generation_lock_held: bool = True,
    ) -> list[StageAssignment]:
        if self.active is None:
            raise HTTPException(status_code=409, detail="load a model before rebalancing")
        if include_quarantined:
            for record in self.workers.values():
                if record.live:
                    record.quarantined = False
                    record.quarantine_reason = None
        self.job = LoadJob(job_id=uuid.uuid4().hex, model_id=self.active.model_id)
        return await self.run_job(
            self.job,
            self.deploy(self.active, self.job, generation_lock_held=generation_lock_held),
        )

    async def restore_session_from_cache(
        self,
        session_id: str,
        source_epoch: int,
        cache_url: str,
        expected_length: int,
    ) -> bool:
        request = CacheRestoreRequest(
            session_id=session_id,
            source_epoch=source_epoch,
            target_epoch=self.epoch,
            cache_url=cache_url,
            expected_length=expected_length,
        )

        async def restore(stage: StageAssignment) -> None:
            response = await self.client.post(
                stage.url.rstrip("/") + "/internal/restore",
                json=request.model_dump(),
                headers=auth_headers(self.pool_token),
                timeout=120.0,
            )
            response.raise_for_status()

        results = await asyncio.gather(*(restore(stage) for stage in self.route), return_exceptions=True)
        restored = all(not isinstance(result, BaseException) for result in results)
        if restored:
            self.add_event(
                "cache",
                "KV cache restored",
                f"Session {session_id[:8]} recovered from epoch {source_epoch} into epoch {self.epoch}",
            )
        return restored

    async def _await_route_change(self, epoch: int) -> None:
        patience = COORDINATOR_WAIT_SECONDS if self.coordinating else ROUTE_WAIT_SECONDS
        deadline = time.monotonic() + patience
        while self.epoch == epoch and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

    async def _report_failure(self, entry_worker: str, response: httpx.Response | None) -> None:
        worker = self._failed_worker(response, entry_worker) if response is not None else entry_worker
        await self._report_worker(worker, f"HTTP {response.status_code if response is not None else 'no answer'}")

    async def _report_worker(self, worker_id: str, reason: str) -> None:
        url = self.mesh.leader_url if self.mesh else None
        if url is None:
            return
        try:
            await self.client.post(
                url + "/internal/report",
                json={"worker_id": worker_id, "reason": f"reported by {self.mesh.node_id}: {reason}"},
                headers=auth_headers(self.pool_token),
                timeout=5.0,
            )
        except httpx.HTTPError:
            pass  # if the coordinator is gone too, the next one finds this for itself

    async def _answers(self, worker_id: str) -> bool:
        record = self.workers.get(worker_id)
        if record is None:
            return False
        try:
            url = str(record.registration.advertised_url).rstrip("/") + "/health"
            response = await self.client.get(url, headers=auth_headers(self.pool_token), timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def handle_report(self, worker_id: str, reason: str) -> None:
        """A follower's reply hit a dead stage. Repair now, as a local failure would.

        Any node can report any other, so this checks first. A worker that answers
        here keeps its place: the problem is the reporter's link, or the report.
        """
        if await self._answers(worker_id):
            self.add_event("failure", f"{worker_id} reported, but it answers here", reason[:160])
            return
        self._quarantine(worker_id, reason)
        async with self.control_lock:
            await self.rebalance(generation_lock_held=False)

    @staticmethod
    def _failed_worker(response: httpx.Response, entry_worker: str) -> str:
        try:
            detail = response.json().get("detail")
            if isinstance(detail, dict) and detail.get("failed_worker"):
                return str(detail["failed_worker"])
        except (ValueError, AttributeError):
            pass
        return entry_worker

    @staticmethod
    def _asks_for_replay(response: httpx.Response) -> bool:
        if response.status_code != 503:
            return False
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            return False
        return isinstance(detail, dict) and bool(detail.get("retry"))

    def _refused_request(self, response: httpx.Response | None) -> bool:
        """A stage is up and could not process this request.

        Only losing a stage may cost a worker its place. If a request's own
        content could do that, any user could empty the pool one reply at a time.
        """
        if response is None or response.status_code not in {400, 422, 500}:
            return False
        self.stage_errors += 1
        self.add_event("failure", "A stage refused a request", f"HTTP {response.status_code}: {response.text[:160]}")
        return True

    async def _mark_inference_failure(self, response: httpx.Response, entry_worker: str) -> None:
        failed_worker = self._failed_worker(response, entry_worker)
        if failed_worker in self.workers:
            self._quarantine(failed_worker, f"inference request failed: HTTP {response.status_code}")

    async def next_token(
        self,
        session: SessionState,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
    ) -> int:
        started = time.perf_counter()
        attempt = 0
        moves = 0
        # Real failures are limited to two. A route moving on is not a failure,
        # and several changes in a row must not use those attempts up.
        while attempt < 2 and moves < MAX_ROUTE_MOVES:
            await self.wait_for_route()
            if not self.route:
                raise HTTPException(status_code=409, detail="no active model pipeline")
            must_replay = session.needs_reset or session.epoch != self.epoch
            input_ids = session.tokens if must_replay else [session.tokens[-1]]
            request = InferRequest(
                session_id=session.session_id,
                epoch=self.epoch,
                input_ids=input_ids,
                reset=must_replay,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
            entry = self.route[0]
            try:
                response = await self.client.post(
                    entry.url.rstrip("/") + "/v1/infer",
                    json=request.model_dump(),
                    headers=auth_headers(self.pool_token),
                    timeout=httpx.Timeout(pass_seconds(len(request.input_ids), len(self.route) + 1), connect=10.0),
                )
            except httpx.HTTPError as exc:
                self._quarantine(entry.worker_id, str(exc))
                response = None
            if response is not None and response.is_success:
                session.epoch = self.epoch
                session.needs_reset = False
                elapsed_ms = (time.perf_counter() - started) * 1000
                # A prompt pass is not a token; it would skew the latency metric.
                if len(input_ids) == 1:
                    self.token_latency_ms_ema = self._ema(self.token_latency_ms_ema, elapsed_ms)
                self.tokens_generated += 1
                return int(response.json()["token_id"])
            if response is not None and self._asks_for_replay(response):
                # A stage restarted in place: the worker is fine, the state is not.
                self.stage_restarts += 1
                session.needs_reset = True
                attempt += 1
                continue
            if response is not None and response.status_code == 409:
                # The route moved on. Nothing failed, so nobody takes the blame.
                await self._await_route_change(session.epoch)
                session.needs_reset = True
                moves += 1
                continue
            if self._refused_request(response):
                raise HTTPException(status_code=502, detail="a stage could not process this request")
            if not self.coordinating:
                # Repairs belong to the coordinator. The reply waits and carries on.
                await self._report_failure(entry.worker_id, response)
                await self._await_route_change(session.epoch)
                session.needs_reset = True
                attempt += 1
                continue
            if response is not None:
                await self._mark_inference_failure(response, entry.worker_id)
            if attempt == 0:
                self.recoveries_total += 1
                source_epoch = self.epoch
                source_cache_url = self.active_cache_url
                await self.rebalance()
                restored = bool(source_cache_url) and await self.restore_session_from_cache(
                    session.session_id,
                    source_epoch,
                    source_cache_url,
                    len(session.tokens) - 1,  # the newest token has not run yet
                )
                session.epoch = self.epoch
                session.needs_reset = not restored
            attempt += 1
        raise HTTPException(status_code=502, detail="inference failed after pipeline recovery")

    async def token_stream(
        self,
        session: SessionState,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        boundary: int | None = None,
    ) -> AsyncIterator[int]:
        """Yield generated tokens. The caller adds each one to ``session.tokens``.

        The entry worker runs the loop while the route holds. If its stream ends
        early for any reason, the rest goes one token at a time through
        ``next_token``, which replays the session and recovers the pipeline.
        """
        produced = 0
        stops = self.tokenizer.stop_token_ids if self.tokenizer else set()
        for _ in range(STREAM_ATTEMPTS):
            if not self.route:
                break
            entry = self.route[0]
            epoch = self.epoch
            request = GenerateRequest(
                session_id=session.session_id,
                epoch=epoch,
                input_ids=session.tokens,
                reset=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                max_tokens=max_tokens - produced,
                stop_token_ids=sorted(stops),
                checkpoint_at=None if produced else boundary,
            )
            started = time.perf_counter()
            failure: Any = None
            try:
                async with self.client.stream(
                    "POST",
                    entry.url.rstrip("/") + "/v1/generate",
                    json=request.model_dump(),
                    headers=auth_headers(self.pool_token),
                ) as response:
                    if response.status_code == 429:
                        raise HTTPException(status_code=429, detail="the pool is busy; try again shortly")
                    if response.status_code != 200:
                        failure = "epoch" if response.status_code == 409 else f"HTTP {response.status_code}"
                    else:
                        async with aclosing(self._lines_from(response, entry.worker_id)) as lines:
                            async for line in lines:
                                event = json.loads(line)
                                if "token_id" not in event:
                                    failure = event.get("error")
                                    self.add_event("failure", "Generation stream stopped", str(failure)[:200])
                                    break
                                now = time.perf_counter()
                                if produced:
                                    self.token_latency_ms_ema = self._ema(self.token_latency_ms_ema, (now - started) * 1000)
                                started = now
                                produced += 1
                                self.tokens_generated += 1
                                session.needs_reset = False
                                token = int(event["token_id"])
                                yield token
                                if token in stops or produced >= max_tokens:
                                    return
            except WorkerSilent as exc:
                failure = {"failed_worker": entry.worker_id, "message": str(exc)}
                self.add_event("failure", "Generation stream lost", str(exc)[:200])
            except (httpx.HTTPError, ValueError) as exc:
                failure = f"lost: {exc}"
                self.add_event("failure", "Generation stream lost", str(exc)[:200])
            # Workers may hold more or less than the session does now.
            session.needs_reset = True
            self.stream_fallbacks += 1
            if not await self._recover_stream(entry.worker_id, failure, epoch):
                break
        for _ in range(max_tokens - produced):
            yield await self.next_token(session, temperature, top_p, top_k, seed)

    async def _lines_from(self, response: httpx.Response, worker_id: str) -> AsyncIterator[str]:
        """A worker's stream, which ends if the worker stops checking in.

        A reply can wait a long time for its first token, so silence alone means
        nothing. Silence from a worker nobody can hear from any more means the
        link is gone, and nothing else would ever say so.
        """
        arrived: asyncio.Queue[Any] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for line in response.aiter_lines():
                    arrived.put_nowait(line)
                arrived.put_nowait(None)
            except (httpx.HTTPError, ValueError) as exc:
                arrived.put_nowait(exc)

        reading = asyncio.ensure_future(pump())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(arrived.get(), STREAM_CHECK_SECONDS)
                except asyncio.TimeoutError:
                    record = self.workers.get(worker_id)
                    if record is not None and not record.live:
                        raise WorkerSilent(f"{worker_id} went silent") from None
                    continue
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            reading.cancel()

    async def _recover_stream(self, entry_worker: str, failure: Any, epoch: int) -> bool:
        """Put things right after a stream stopped. True if a new stream can follow.

        Replaying through a new stream keeps the reply on a sequence of its own.
        Anything this cannot classify goes the slow way, through ``next_token``.
        """
        if isinstance(failure, dict) and failure.get("retry"):
            self.stage_restarts += 1
            return True
        if failure == "epoch" or (isinstance(failure, str) and "epoch mismatch" in failure):
            if self.coordinating and self.epoch == epoch:
                return False  # the coordinator sets the epoch, so waiting would be for itself
            await self._await_route_change(epoch)
            return True
        if isinstance(failure, dict) and failure.get("failed_worker"):
            worker = str(failure["failed_worker"])
            if self.coordinating:
                self._quarantine(worker, f"reply failed: {str(failure.get('message'))[:120]}")
                self.recoveries_total += 1
                await self.rebalance()
            else:
                await self._report_worker(worker, "reply failed")
                await self._await_route_change(epoch)
            return True
        return False

    def metrics_text(self) -> str:
        """This node's numbers in Prometheus format. Every node reports for itself."""
        lines: list[str] = []

        def add(name: str, value: float | None, help_text: str, kind: str = "gauge", **labels: str) -> None:
            if f"# HELP shardmesh_{name} " not in "\n".join(lines):
                lines.append(f"# HELP shardmesh_{name} {help_text}")
                lines.append(f"# TYPE shardmesh_{name} {kind}")
            tags = ",".join(f'{key}="{text}"' for key, text in labels.items())
            lines.append(f"shardmesh_{name}{{{tags}}} {0 if value is None else value}" if tags else f"shardmesh_{name} {0 if value is None else value}")

        add("requests_total", self.requests_total, "Replies started on this node", "counter")
        add("tokens_generated_total", self.tokens_generated, "Tokens produced for this node's clients", "counter")
        add("stream_fallbacks_total", self.stream_fallbacks, "Replies that fell back to replay; should stay near zero", "counter")
        add("stage_errors_total", self.stage_errors, "Requests a stage refused; nobody is quarantined for these", "counter")
        add("recoveries_total", self.recoveries_total, "Pipeline repairs during a reply", "counter")
        add("stage_restarts_total", self.stage_restarts, "Replays after a stage restarted in place", "counter")
        add("token_latency_ms", self.token_latency_ms_ema, "Moving average of time per token")
        add("epoch", self.epoch, "Current pipeline epoch")
        add("serving", 1 if self.route else 0, "Whether this node can serve replies")
        add("replies_in_flight", self.replies_in_flight, "Replies this node is running now")
        add("coordinator", 1 if self.mesh and self.mesh.is_leader else 0, "Whether this node coordinates the pool")
        if self.mesh is not None:
            add("mesh_term", self.mesh.term, "Leadership term")
            add("mesh_members", len(self.mesh.alive()), "Control planes reachable from here")
        live = [record for record in self.workers.values() if record.live]
        add("workers_live", len(live), "Workers heard from recently")
        add("workers_quarantined", sum(record.quarantined for record in self.workers.values()), "Workers held out of the route")
        for worker_id, record in self.workers.items():
            telemetry = (record.health or {}).get("telemetry") or {}
            add("worker_live", 1 if record.live else 0, "Whether a worker is reachable", worker=worker_id)
            add("worker_compute_ms", telemetry.get("compute_ms_ema"), "Moving average of compute per pass", worker=worker_id)
            add("worker_stage_restarts_total", telemetry.get("stage_restarts"), "Stage processes restarted", "counter", worker=worker_id)
            add("worker_stages_kept_total", telemetry.get("stages_kept"), "Reconfigures that moved no weights", "counter", worker=worker_id)
            if telemetry.get("draft_tokens_per_pass") is not None:
                add("worker_draft_tokens_per_pass", telemetry["draft_tokens_per_pass"], "Tokens per speculative pass", worker=worker_id)
        return "\n".join(lines) + "\n"

    def _fit_context(self, prompt_tokens: int, max_tokens: int) -> int:
        """Keep a GGUF request inside the context its stages were built for.

        llama.cpp fails a pass that overruns its context, and a failed pass looks
        like a dead worker. Too much text must not cost a healthy node its place.
        Guesses need room too, since a pass runs them all before checking any.
        """
        if self.active is None or not isinstance(self.active.adapter, GgufAdapter):
            return max_tokens
        limit = self.active.profile.context_tokens
        room = limit - prompt_tokens - self.active.speculate_tokens - 1
        if room < 1:
            raise HTTPException(
                status_code=400,
                detail=f"the prompt is {prompt_tokens} tokens, and this model was loaded with room for {limit}; "
                "start a new chat or load it with a larger context",
            )
        return min(max_tokens, room)

    def _history_boundary(self, prompt: list[int], history: str | None) -> int | None:
        """Where a chat's settled history ends, which is worth checkpointing.

        The rest of the prompt opens the reply, and chat templates rewrite that
        part next turn, so a checkpoint past it could never be reused.
        """
        if history is None or self.tokenizer is None or not isinstance(getattr(self.active, "adapter", None), GgufAdapter):
            return None
        settled = self.tokenizer.encode(history)
        shared = 0
        for ours, theirs in zip(prompt, settled):
            if ours != theirs:
                break
            shared += 1
        return shared if 0 < shared < len(prompt) else None

    async def stream_tokens(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        stop: str | list[str] | None,
        history: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield text as it is generated, then one final summary.

        Text is held back while it could still be the start of a stop string,
        or while it ends inside a multi-byte character.
        """
        if not self.tokenizer or not self.model_id:
            raise HTTPException(status_code=409, detail="no model loaded")
        self.requests_total += 1
        tokenizer = self.tokenizer
        prompt_tokens = await asyncio.to_thread(tokenizer.encode, prompt)
        if not prompt_tokens:
            if not tokenizer.stop_token_ids:
                raise HTTPException(status_code=400, detail="prompt must not be empty")
            prompt_tokens = [min(tokenizer.stop_token_ids)]
        max_tokens = self._fit_context(len(prompt_tokens), max_tokens)
        session = SessionState(tokens=list(prompt_tokens), epoch=self.epoch, needs_reset=True)
        boundary = await asyncio.to_thread(self._history_boundary, prompt_tokens, history)
        markers = [marker for marker in ([stop] if isinstance(stop, str) else (stop or [])) if marker]
        holdback = max((len(marker) for marker in markers), default=1) - 1
        generated: list[int] = []
        sent = ""
        finish_reason = "length"
        started = time.perf_counter()
        first_token_at: float | None = None

        decoder = IncrementalText(tokenizer)
        self.replies_in_flight += 1
        try:
            source = self.token_stream(session, max_tokens, temperature, top_p, top_k, seed, boundary)
            async with aclosing(source) as tokens:
                async for token in tokens:
                    first_token_at = first_token_at or time.perf_counter()
                    if token in tokenizer.stop_token_ids:
                        finish_reason = "stop"
                        break
                    generated.append(token)
                    session.tokens.append(token)
                    text = decoder.add(token)
                    hits = [text.index(marker) for marker in markers if marker in text]
                    if hits:
                        text = text[: min(hits)]
                        finish_reason = "stop"
                    ready = text if hits else text[: len(text) - holdback]
                    if ready.endswith("�") and not hits:
                        continue
                    if len(ready) > len(sent):
                        delta, sent = ready[len(sent) :], ready
                        yield {"text": delta}
                    if hits:
                        break
        finally:
            self.replies_in_flight -= 1
            # This generator may be closing because the client left, and
            # awaiting here would be cancelled along with it.
            self.spawn(self.clear_session(session.session_id))

        text = tokenizer.decode(generated)
        hits = [text.index(marker) for marker in markers if marker in text]
        if hits:
            text = text[: min(hits)]
        if len(text) > len(sent):
            yield {"text": text[len(sent) :]}
        finished = time.perf_counter()
        decode_seconds = finished - (first_token_at or finished)
        yield {
            "finish_reason": finish_reason,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generated),
            "time_to_first_token_ms": round(((first_token_at or finished) - started) * 1000, 1),
            "tokens_per_second": (
                round((len(generated) - 1) / decode_seconds, 2) if len(generated) > 1 and decode_seconds else None
            ),
        }

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        stop: str | list[str] | None,
        history: str | None = None,
    ) -> tuple[str, int, int, str]:
        pieces: list[str] = []
        summary: dict[str, Any] = {}
        async for item in self.stream_tokens(prompt, max_tokens, temperature, top_p, top_k, seed, stop, history):
            if "text" in item:
                pieces.append(item["text"])
            else:
                summary = item
        return "".join(pieces), summary["prompt_tokens"], summary["completion_tokens"], summary["finish_reason"]

    async def clear_session(self, session_id: str) -> None:
        async def clear(url: str) -> None:
            try:
                await self.client.delete(
                    url.rstrip("/") + f"/internal/sessions/{session_id}",
                    headers=auth_headers(self.pool_token),
                    timeout=5.0,
                )
            except httpx.HTTPError:
                pass

        urls = {str(record.registration.advertised_url) for record in self.workers.values() if record.live}
        await asyncio.gather(*(clear(url) for url in urls))

    async def node_inventory(self) -> dict[str, Any]:
        """Each node's own report, from one of its workers, with how far away it is."""
        chosen: dict[str, str] = {}
        for worker_id, record in self.workers.items():
            if record.live:
                chosen.setdefault(node_of(worker_id), str(record.registration.advertised_url).rstrip("/"))

        async def ask(url: str) -> dict[str, Any]:
            try:
                response = await self.client.get(url + "/internal/inventory", headers=auth_headers(self.pool_token), timeout=10.0)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError):
                return {"error": "no answer"}

        reports = await asyncio.gather(*(ask(url) for url in chosen.values()))
        nodes = dict(zip(chosen, reports))
        for node, report in nodes.items():
            if node in self.round_trip_ms:
                report["round_trip_ms"] = self.round_trip_ms[node]
        return {"nodes": nodes}

    def network_view(self) -> dict[str, Any] | None:
        """This node's own view of a network nobody owns. Another node's may differ."""
        if not self.access.ownerless:
            return None
        commons = self.access.authority
        members = commons.members(self.access.signed)
        return {
            "this_node": commons.me,
            "joining": commons.policy["joining"],
            "quorum": commons.policy["quorum"],
            "accepted": sorted(members) if members is not None else None,
            "blocked": sorted(self.access.revoked),
            "peers": sorted(self.mesh.alive()) if self.mesh else [],
        }

    def memory_summary(self) -> dict[str, dict[str, int]]:
        """Pool totals by kind, counting each physical resource once."""
        groups: dict[str, dict[str, int | str]] = {}
        for record in self.workers.values():
            if not record.live:
                continue
            reported = {item["pool_id"]: item for item in (record.health or {}).get("memory_pools", [])}
            for pool in record.registration.hardware.memory_pools:
                group = groups.setdefault(pool.capacity_group, {"kind": pool.kind, "budget": 0, "used": 0})
                group["budget"] = max(int(group["budget"]), pool.budget_bytes)
                group["used"] = int(group["used"]) + int(reported.get(pool.pool_id, {}).get("used_bytes", 0))
        summary: dict[str, dict[str, int]] = {}
        for group in groups.values():
            kind = summary.setdefault(str(group["kind"]), {"budget_bytes": 0, "used_bytes": 0})
            kind["budget_bytes"] += int(group["budget"])
            kind["used_bytes"] += int(group["used"])
        return summary

    def status(self) -> dict[str, Any]:
        loading = self.job is not None and self.job.finished_at is None
        live_records = [record for record in self.workers.values() if record.live]
        hosts: dict[str, Any] = {}
        accelerators: dict[tuple[str, str, str | None], Any] = {}
        for record in live_records:
            hardware = record.registration.hardware
            hosts.setdefault(hardware.host_id, hardware)
            if hardware.accelerator_memory_bytes:
                key = (hardware.host_id, hardware.device, hardware.accelerator_name)
                accelerators.setdefault(key, hardware)
        ram = sum(hardware.ram_total_bytes for hardware in hosts.values())
        accelerator_memory = sum(hardware.accelerator_memory_bytes for hardware in accelerators.values())
        contributed_ram = sum(
            record.registration.hardware.contribution.ram_budget_bytes for record in live_records
        )
        contributed_cpu_threads = sum(
            record.registration.hardware.contribution.cpu_threads for record in live_records
        )
        contributed_accelerator_memory = sum(
            max(
                record.registration.hardware.contribution.accelerator_memory_budget_bytes
                for record in live_records
                if (
                    record.registration.hardware.host_id,
                    record.registration.hardware.device,
                    record.registration.hardware.accelerator_name,
                ) == key
            )
            for key in accelerators
        )
        return {
            "status": "ready" if self.route and self.tokenizer else "loading" if loading else "recovering" if self.saved_state else "waiting",
            "mesh": (
                {"node": self.mesh.node_id, "leader": self.mesh.leader_id, "term": self.mesh.term, "members": self.mesh.alive()}
                if self.mesh
                else None
            ),
            "cordoned_nodes": sorted(self.access.cordoned),
            "lanes": self.lanes.status(),
            "ownerless": self.network_view(),
            "model": self.model_id,
            "load_job": self.job.snapshot() if self.job else None,
            "active_plan": describe_plan(self.active, self.route) if self.active and self.route else None,
            "model_profile": (
                "protocol-test"
                if self.model_id and Path(self.model_id).name.lower() == ".smoke-model"
                else "trained"
                if self.model_id
                else "none"
            ),
            "epoch": self.epoch,
            "uptime_seconds": int(time.time() - self.started_at),
            "route": [stage.model_dump() for stage in self.route],
            "cache_url": self.active_cache_url,
            "metrics": {
                "requests_total": self.requests_total,
                "tokens_generated": self.tokens_generated,
                "recoveries_total": self.recoveries_total,
                "token_latency_ms_ema": self.token_latency_ms_ema,
                "tokens_per_second_ema": (
                    1000 / self.token_latency_ms_ema if self.token_latency_ms_ema else None
                ),
            },
            "virtual_hardware": {
                "workers": sum(record.live for record in self.workers.values()),
                "hosts": len(hosts),
                "accelerators": len(accelerators),
                "ram_total_bytes": ram,
                "accelerator_memory_bytes": accelerator_memory,
                "cpu_cores": sum(hardware.cpu_cores for hardware in hosts.values()),
                "contributed_ram_bytes": contributed_ram,
                "contributed_cpu_threads": contributed_cpu_threads,
                "contributed_accelerator_memory_bytes": contributed_accelerator_memory,
                "memory_by_kind": self.memory_summary(),
                "note": "capacities are physically distributed, not uniform shared memory",
            },
            "workers": {
                worker_id: {
                    **record.registration.model_dump(mode="json"),
                    "live": record.live,
                    "quarantined": record.quarantined,
                    "quarantine_reason": record.quarantine_reason,
                    "active": any(stage.worker_id == worker_id for stage in self.route),
                    "state": (
                        "offline"
                        if not record.live
                        else "quarantined"
                        if record.quarantined
                        else "active"
                        if any(stage.worker_id == worker_id for stage in self.route)
                        else "service"
                        if "inference" not in record.registration.hardware.contribution.roles
                        else "standby"
                    ),
                    "control_latency_ms": record.control_latency_ms,
                    "runtime": record.health,
                }
                for worker_id, record in self.workers.items()
            },
            "events": list(self.events),
        }

    async def refresh_worker_metrics(self) -> None:
        async def probe(worker_id: str, record: WorkerRecord) -> None:
            started = time.perf_counter()
            try:
                response = await self.client.get(
                    str(record.registration.advertised_url).rstrip("/") + "/health",
                    headers=auth_headers(self.pool_token),
                    timeout=3.0,
                )
                response.raise_for_status()
                record.control_latency_ms = round((time.perf_counter() - started) * 1000, 2)
                record.health = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                record.control_latency_ms = None
                record.health = {"status": "unreachable", "error": str(exc)}

        await asyncio.gather(*(probe(worker_id, record) for worker_id, record in self.workers.items()))


runtime = Orchestrator()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start_maintenance()
    try:
        yield
    finally:
        await runtime.close()


app = FastAPI(title="ShardMesh orchestrator", version="0.2.0", lifespan=lifespan)


def require(authorization: str | None, role: str) -> Caller:
    behalf = BEHALF.get()
    caller = Caller(*behalf) if behalf else runtime.access.identify(authorization, PEER.get())
    if caller is None:
        raise HTTPException(status_code=401, detail="a valid key is required")
    if not runtime.access.permits(caller, role):
        raise HTTPException(status_code=403, detail=f"this needs the {role} role")
    return caller


def credentials(authorization: str | None, api_key: str | None) -> str | None:
    """Anthropic's clients send a key in x-api-key, not as a bearer token."""
    return authorization or (f"Bearer {api_key}" if api_key else None)


def network_caller(authorization: str | None) -> Caller:
    """Whoever asks, as the network sees them.

    Where nobody owns the network, no key commands it, the coordinator's own
    operator's included. Coordinating orders a route and carries no other rights.
    """
    caller = require(authorization, ROLE_MEMBER)
    if runtime.access.ownerless and caller.role == ROLE_ADMIN:
        assert runtime.mesh is not None
        return Caller(f"{runtime.mesh.node_id}/{caller.name}", ROLE_MEMBER)
    return caller


def require_own_node(authorization: str | None, name: str) -> Caller:
    """An admin, or a member acting on their own node."""
    caller = network_caller(authorization)
    if caller.role == ROLE_MEMBER and caller.name.partition("/")[0] != name:
        raise HTTPException(status_code=403, detail="a member can only take their own node out")
    return caller


def verify_token(authorization: str | None) -> None:
    """Node to node. API keys do not reach this far."""
    require(authorization, ROLE_NODE)


def verify_worker_name(worker_id: str) -> None:
    """A node speaks for its own workers, and a revoked one for none."""
    node, peer = node_of(worker_id), PEER.get()
    if node in runtime.access.revoked:
        raise HTTPException(status_code=403, detail=f"{node} has been revoked")
    if peer is not None and node != peer:
        raise HTTPException(status_code=403, detail=f"{peer} cannot speak for {worker_id}")


def verify_admin(authorization: str | None) -> None:
    require(authorization, ROLE_ADMIN)


def verify_user(authorization: str | None) -> Caller:
    caller = require(authorization, ROLE_USER)
    wait = runtime.access.spend(caller)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit reached for {caller.name}",
            headers={"Retry-After": str(max(int(wait) + 1, 1))},
        )
    return caller


FORWARDED_PATHS = ("/v1/", "/api/", "/admin/")
SAFE_NODE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# Connection details belong to each hop, and httpx sets the length itself.
HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "content-encoding"}


@app.middleware("http")
async def forward_to_coordinator(request: Request, call_next: Any) -> Any:
    """Serve the API from any node.

    Replies run on the node that was asked, straight to the entry worker, so they
    do not depend on the coordinator. Only management goes there, along with
    replies from a node that has not installed the route yet.
    """
    if FORWARD_MARK in request.headers:
        # This connection proves which node passed the request on, not who sent it.
        # With an authority, that node checked the key and kept it, and says who asked.
        vouched = request.headers.get(BEHALF_HEADER, "")
        role, _, name = vouched.partition(":")
        if PEER.get() is not None and runtime.access.authority is not None and role in {ROLE_USER, ROLE_MEMBER, ROLE_ADMIN}:
            if runtime.access.ownerless:
                # An operator commands their own node. Here they are only a member,
                # and the node that passed this on is the one that answers for them.
                name, role = f"{PEER.get()}/{name.rpartition('/')[2]}", ROLE_USER if role == ROLE_USER else ROLE_MEMBER
            BEHALF.set((name, role))
        PEER.set(None)
    mesh = runtime.mesh
    path = request.url.path
    if any(part in {".", ".."} for part in path.replace("\\", "/").split("/")):
        # httpx would resolve these, turning a forwarded path into any other.
        return JSONResponse(status_code=400, content={"detail": "bad path"})
    serves_locally = path.startswith(("/v1/", "/api/")) and bool(runtime.route)
    own_choice = path in {"/admin/invites", "/admin/network"} or (runtime.access.ownerless and (path.startswith("/admin/keys") or path.endswith("/revoke")))
    if mesh is None or mesh.is_leader or serves_locally or own_choice or not path.startswith(FORWARDED_PATHS):
        return await call_next(request)
    target = mesh.leader_url
    if target is None or FORWARD_MARK in request.headers:
        return JSONResponse(status_code=503, content={"detail": "no coordinator yet; try again in a few seconds"})
    headers = {name: value for name, value in request.headers.items() if name.lower() not in HOP_HEADERS}
    headers[FORWARD_MARK] = mesh.node_id
    headers.pop(BEHALF_HEADER, None)
    if runtime.access.authority is not None:
        # Anyone can claim to coordinate, so a person's key never leaves the node
        # they gave it to. It is checked here, and only the result travels.
        caller = runtime.access.identify(credentials(request.headers.get("authorization"), request.headers.get("x-api-key")), None)
        if caller is None:
            return JSONResponse(status_code=401, content={"detail": "a valid key is required"})
        wait = runtime.access.spend(caller)
        if wait:
            return JSONResponse(status_code=429, content={"detail": f"rate limit reached for {caller.name}"})
        headers.pop("authorization", None)
        headers[BEHALF_HEADER] = f"{caller.role}:{caller.name}"
    try:
        upstream = await runtime.client.send(
            runtime.client.build_request(
                request.method,
                target + request.url.path,
                params=request.query_params,
                content=await request.body(),
                headers=headers,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=503, content={"detail": f"the coordinator did not answer: {exc}"})
    passed = {name: value for name, value in upstream.headers.items() if name.lower() not in HOP_HEADERS}
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code, headers=passed, background=BackgroundTask(upstream.aclose)
    )


def page(name: str) -> HTMLResponse:
    content = Path(__file__).with_name(name).read_text(encoding="utf-8")
    return HTMLResponse(content, headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def allow_mesh_pages(request: Request, call_next: Any) -> Any:
    """A chat page may fail over to another node, and only to another node.

    Browsers treat each node as a separate site. Only pages served by a member of
    the mesh are allowed across, and they still need a key of their own. The page
    that most needs this is on a node that has just died, so that node still counts.
    """
    origin = request.headers.get("origin")
    known = origin is not None and runtime.mesh is not None and origin.rstrip("/") in runtime.mesh.known_urls()
    if known and request.method == "OPTIONS":
        response: Any = Response(status_code=204)
        response.headers["Access-Control-Allow-Methods"] = "GET, POST"
        response.headers["Access-Control-Allow-Headers"] = "authorization, content-type"
    else:
        response = await call_next(request)
    if known:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response


@app.middleware("http")
async def audit_requests(request: Request, call_next: Any) -> Any:
    """Record who changed things and who asked for replies. Prompts stay out of it."""
    response = await call_next(request)
    path = request.url.path
    if request.method != "GET" and path.startswith(FORWARDED_PATHS):
        # Every node that handles a request records it: anyone can send the forward mark.
        marked = FORWARD_MARK in request.headers
        peer = None if marked else PEER.get()
        caller = runtime.access.identify(request.headers.get("authorization"), peer)
        vouched = request.headers.get(BEHALF_HEADER) if marked and PEER.get() else None
        client = request.client.host if request.client else None
        actor = f"{vouched} via {PEER.get()}" if vouched else caller.name if caller else "unknown"
        runtime.audit.record(actor, request.method, path, response.status_code, client)
    return response


# Added last, so it runs first: nothing oversized is read, forwarded or parsed.
app.add_middleware(TicketIdentity)
app.add_middleware(BodyLimit)


@app.get("/ready")
async def ready() -> JSONResponse:
    """For load balancers: can this node serve a reply right now?"""
    serving = bool(runtime.route and runtime.tokenizer)
    return JSONResponse(status_code=200 if serving else 503, content={"ready": serving, "model": runtime.model_id})


@app.get("/metrics")
async def metrics(authorization: str | None = Header(default=None)) -> Response:
    verify_admin(authorization)
    return Response(runtime.metrics_text(), media_type="text/plain; version=0.0.4")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/chat", status_code=307)


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return page("dashboard.html")


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat() -> HTMLResponse:
    return page("chat.html")


@app.post("/internal/register")
async def register(
    body: WorkerRegistration,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_token(authorization)
    verify_worker_name(body.worker_id)
    if body.protocol_version != PROTOCOL_VERSION:
        detail = (
            f"{body.worker_id} speaks protocol {body.protocol_version} (code {body.code_version}), and this "
            f"orchestrator speaks {PROTOCOL_VERSION} (code {__version__}); update the older one"
        )
        runtime.add_event("failure", f"{body.worker_id} refused", detail)
        raise HTTPException(status_code=426, detail=detail)
    runtime.register(body)
    return {
        "status": "registered",
        "active_epoch": runtime.epoch,
        "controls": control_urls(),
        "revoked_nodes": sorted(runtime.access.revoked),
        "signed_revocations": runtime.access.signed_revocations(),
    }


@app.post("/internal/heartbeat")
async def heartbeat(body: Heartbeat, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    verify_worker_name(body.worker_id)
    if not runtime.heartbeat(body.worker_id):
        raise HTTPException(status_code=404, detail="worker must register")
    return {
        "status": "ok",
        "controls": control_urls(),
        "revoked_nodes": sorted(runtime.access.revoked),
        "signed_revocations": runtime.access.signed_revocations(),
    }


def control_urls() -> list[str]:
    return sorted(runtime.mesh.alive().values()) if runtime.mesh else []


@app.post("/internal/gossip")
async def gossip(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    if runtime.mesh is None:
        raise HTTPException(status_code=404, detail="this control plane is not part of a mesh")
    peer = PEER.get()
    if peer is not None and body.get("node_id") != peer:
        raise HTTPException(status_code=403, detail=f"{peer} cannot speak for {body.get('node_id')}")
    try:
        runtime.receive_gossip(body)
        reply = runtime.gossip_message()
        tickets = runtime.client.auth if isinstance(runtime.client.auth, NodeTickets) else None
        if tickets is not None:
            reply["proof"] = tickets.maker.ticket(reply_audience(clean_url(body["url"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"bad gossip: {exc}") from exc
    return reply


@app.get("/internal/diag/tasks")
async def diag_tasks(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_token(authorization)
    return where_everything_is()


@app.post("/internal/report")
async def report(body: dict[str, str], authorization: str | None = Header(default=None)) -> dict[str, str]:
    """A follower's reply found a dead stage."""
    verify_token(authorization)
    if not runtime.coordinating:
        raise HTTPException(status_code=409, detail="this node does not coordinate")
    runtime.spawn(runtime.handle_report(body["worker_id"], body.get("reason", "reported by a peer")))
    return {"status": "repairing"}


@app.get("/internal/mesh")
async def mesh_view(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """This node's own view of the mesh, never forwarded, for comparing nodes."""
    verify_token(authorization)
    mesh = runtime.mesh
    if mesh is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "node": mesh.node_id,
        "term": mesh.term,
        "leader": mesh.leader_id,
        "members": mesh.alive(),
        "has_state": mesh.replica is not None,
        "serving_epoch": runtime.installed_epoch if runtime.route else None,
    }


async def _run_load(body: LoadModelRequest, job: LoadJob) -> list[StageAssignment]:
    async with runtime.control_lock:
        return await runtime.run_job(job, runtime.load(body, job))


@app.post("/admin/load")
async def load(body: LoadModelRequest, authorization: str | None = Header(default=None)) -> Any:
    caller = network_caller(authorization)
    if caller.role == ROLE_MEMBER and not body.lane and runtime.route:
        raise HTTPException(status_code=403, detail="nobody can replace what this network runs; load beside it with a lane")
    if body.lane:
        return JSONResponse(status_code=202, content=await runtime.lanes.load(body, caller))
    if runtime.control_lock.locked():
        raise HTTPException(status_code=409, detail="another load is already running")
    job = LoadJob(job_id=uuid.uuid4().hex, model_id=body.model_id)
    runtime.job = job
    if not body.wait:
        task = runtime.spawn(_run_load(body, job))
        # The outcome is on the job, where /admin/status reports it.
        task.add_done_callback(lambda finished: finished.cancelled() or finished.exception())
        return JSONResponse(status_code=202, content={"status": "started", "job": job.snapshot()})
    route = await _run_load(body, job)
    return {"status": "loaded", "model": runtime.model_id, "epoch": runtime.epoch, "route": route}


@app.post("/admin/load/cancel")
async def cancel_load(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_admin(authorization)
    job = runtime.job
    if job is None or job.phase not in {"resolving", "fetching"}:
        raise HTTPException(status_code=409, detail="no load can be cancelled right now")
    job.cancelled = True
    return {"status": "cancelling", "job_id": job.job_id}


@app.post("/admin/plan")
async def preview_plan(body: LoadModelRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Show where a model would go, without moving a byte of weights."""
    verify_admin(authorization)
    try:
        # Bare file names resolve through the nodes holding them, as in a real load.
        sources = await runtime.gguf_sources()
        target = await asyncio.to_thread(build_target, body, runtime.reuse_cached, runtime.peer_fetch, sources, runtime.pool_token)
    except (UnsupportedModel, WeightSourceError, OSError, httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await runtime.refresh_worker_metrics()
    await runtime.refresh_holdings(target)
    try:
        return describe_plan(target, runtime.plan(target, honor_pins_strictly=True))
    except PlacementError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc


@app.get("/admin/keys")
async def list_keys(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Names and roles. A key itself is shown only once, when it is created."""
    verify_admin(authorization)
    document = runtime.access.document()
    return {
        "keys": [{key: item[key] for key in ("name", "role", "requests_per_minute")} for item in document["keys"]],
        "revoked_nodes": document["revoked_nodes"],
        "version": document["version"],
    }


@app.post("/admin/keys")
async def create_api_key(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_admin(authorization)
    try:
        key = runtime.access.add_key(str(body["name"]), str(body.get("role", ROLE_USER)), int(body.get("requests_per_minute", 0)))
    except NeedsAuthority as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runtime.add_event("security", "API key created", f"{body['name']} ({body.get('role', ROLE_USER)})")
    return {"name": str(body["name"]), "key": key}


@app.delete("/admin/keys/{name}")
async def delete_api_key(name: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    verify_admin(authorization)
    try:
        removed = runtime.access.remove_key(name)
    except NeedsAuthority as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"no key named {name!r}")
    runtime.add_event("security", "API key revoked", name)
    return {"status": "revoked"}


@app.post("/admin/nodes/{name}/revoke")
async def revoke_node(name: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Stop trusting a node's certificate and take its workers out of the route."""
    verify_admin(authorization)
    try:
        runtime.access.revoke_node(name)
    except NeedsAuthority as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    runtime.add_event("security", "Node revoked", name)
    runtime.apply_revocations()
    return {"status": "revoked"}


@app.delete("/admin/nodes/{name}/revoke")
async def unblock_node(name: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Where nobody owns the network, a block is this node's own choice to undo."""
    verify_admin(authorization)
    try:
        runtime.access.unblock_node(name)
    except NeedsAuthority as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "unblocked"}


@app.post("/join/requests")
async def request_to_join(body: dict[str, Any]) -> dict[str, str]:
    """Anyone may ask. Asking grants nothing until the owner signs an answer."""
    if runtime.access.authority is None or runtime.access.ownerless:
        raise HTTPException(status_code=404, detail="this pool has no owner to ask")
    try:
        details = read_membership_request(str(body["request"]))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"bad request: {exc}") from exc
    if not SAFE_NODE_NAME.fullmatch(details["name"]):
        raise HTTPException(status_code=400, detail="node names are letters, digits, dots, dashes, and underscores")
    invite = {key: str(body[key])[:128] for key in INVITE_FIELDS if key in body}
    state = runtime.access.add_join_request(details, str(body["request"]), invite)
    if state == "waiting":
        runtime.add_event("security", "A node asked to join", f"{details['name']} from {details['addresses'][:2]}, key {details['fingerprint']}")
    return {"status": state, "fingerprint": details["fingerprint"]}


@app.get("/join/requests")
async def join_requests(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_admin(authorization)
    return {"requests": list(runtime.access.join_requests.values())}


@app.get("/join/authority")
async def join_authority() -> dict[str, str]:
    """The mesh's public certificate, for a new machine to check against its invite."""
    if runtime.access.authority is None or runtime.access.ownerless:
        raise HTTPException(status_code=404, detail="this pool has no authority")
    return {"authority": Path(os.environ["SHARDMESH_CA_FILE"]).read_text(encoding="utf-8")}


@app.post("/admin/invites")
async def make_invite(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """A code that adds a machine. Only the machine holding the authority can honor one."""
    verify_admin(authorization)
    home = os.getenv("SHARDMESH_UP_HOME")
    if not home or not Path(home, "ca.key").is_file():
        raise HTTPException(status_code=409, detail="only the machine that made this mesh can invite; open this page there")
    from .up import create_invite

    assert runtime.mesh is not None
    urls = [runtime.mesh.url, *sorted(peer.url for peer in runtime.mesh.peers.values())]
    return create_invite(Path(home), urls, int(body.get("uses", 1)), float(body.get("hours", 24)))


@app.get("/join/certificate/{name}")
async def join_certificate(name: str) -> dict[str, str]:
    """A certificate is public. Only the node holding the matching key can use it."""
    certificate = runtime.access.certificates.get(name)
    if certificate is None:
        raise HTTPException(status_code=404, detail="not approved yet")
    return {"certificate": certificate, "authority": Path(os.environ["SHARDMESH_CA_FILE"]).read_text(encoding="utf-8")}


@app.post("/access/entries")
async def add_access_entry(body: dict[str, Any]) -> dict[str, str]:
    """Take a change signed by the pool authority. The signature is the credential."""
    if runtime.access.authority is None:
        raise HTTPException(status_code=404, detail="this pool has no authority")
    if not runtime.access.add_signed(body):
        raise HTTPException(status_code=403, detail="this entry is not signed by the pool authority")
    runtime.add_event("security", "Access list changed", str(body["entry"].get("type")))
    runtime.apply_revocations()
    return {"status": "accepted"}


@app.post("/admin/nodes/{name}/cordon")
async def cordon_node(
    name: str,
    force: bool = False,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    """Take a node out of service for maintenance, without dropping a reply."""
    caller = require_own_node(authorization, name)
    cost = runtime.cost_of_losing(name)
    if cost is None:
        raise HTTPException(status_code=409, detail=f"the rest of the pool cannot hold this model without {name}")
    if cost > REPLAN_MOVE_LIMIT_BYTES and not force:
        detail = f"cordoning {name} would transfer {cost / 1024**3:.1f} GiB between nodes; use force if that is what you want"
        raise HTTPException(status_code=409, detail=detail)
    runtime.access.cordon(name)
    runtime.add_event("recovery", "Node cordoned", f"Moving {name}'s share elsewhere")
    if runtime.active is not None:
        runtime.spawn(runtime.move_off())
    runtime.spawn(runtime.lanes.tell_all(f"/admin/nodes/{name}/cordon?force=true", caller))
    return {"status": "cordoned"}


@app.post("/admin/nodes/{name}/uncordon")
async def uncordon_node(name: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    require_own_node(authorization, name)
    if name not in runtime.access.cordoned:
        return {"status": "uncordoned"}  # nothing changed, so nothing should move
    runtime.access.uncordon(name)
    runtime.add_event("recovery", "Node back in service", name)
    if runtime.job is not None and runtime.job.phase in {"resolving", "fetching"}:
        runtime.job.cancelled = True  # a move still under way is no longer wanted
    if runtime.active is not None:
        runtime.spawn(runtime.move_off())
    return {"status": "uncordoned"}


@app.get("/admin/catalog")
async def catalog(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(authorization)
    return {"models": CATALOG, "supported_types": sorted(SUPPORTED_TYPES)}


@app.post("/admin/rebalance")
async def rebalance(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not runtime.access.permits(network_caller(authorization), ROLE_ADMIN):
        raise HTTPException(status_code=403, detail="this needs the admin role")
    # Fetching happens with generation still running; the switch itself waits
    # for a request boundary. Failure-triggered rebalances happen inside the
    # generation lock and call the runtime method directly.
    async with runtime.control_lock:
        route = await runtime.rebalance(include_quarantined=True, generation_lock_held=False)
    return {"status": "rebalanced", "epoch": runtime.epoch, "route": route}


@app.delete("/admin/lanes/{name}")
async def remove_lane(name: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    runtime.lanes.remove(name, network_caller(authorization))
    return {"status": "removed"}


@app.get("/internal/lane")
async def lane_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """What a front door needs to know about the lane it sits beside."""
    verify_token(authorization)
    return {
        "lane": runtime.lane_name,
        "model": runtime.model_id,
        "ready": bool(runtime.tokenizer and runtime.route),
        "leader_url": runtime.mesh.leader_url if runtime.mesh else None,
        "workers": sum(1 for record in runtime.workers.values() if record.live),
        "route": [{"worker_id": stage.worker_id, "start_layer": stage.start_layer, "end_layer": stage.end_layer} for stage in runtime.route],
        "load_job": runtime.job.snapshot() if runtime.job else None,
    }


@app.post("/internal/lane/access")
async def lane_access(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Signed access changes, passed in by this node's front door. Signatures are checked in merge."""
    verify_token(authorization)
    runtime.access.merge(body)
    runtime.apply_revocations()
    return {"status": "merged"}


@app.get("/admin/status")
async def status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(authorization)
    await runtime.refresh_worker_metrics()
    return runtime.status()


@app.get("/admin/network")
async def network_view(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """This node's own view of a network nobody owns, which /admin/status, from the coordinator, is not."""
    verify_user(authorization)
    view = runtime.network_view()
    if view is None:
        raise HTTPException(status_code=404, detail="this network has an owner; see /admin/status")
    return view


@app.get("/admin/inventory")
async def node_inventory(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(authorization)
    return await runtime.node_inventory()


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(credentials(authorization, x_api_key))
    # Both OpenAI's and Anthropic's clients read this, so each entry suits either.
    data = [{"id": name, "object": "model", "type": "model", "display_name": name, "owned_by": "shardmesh"} for name in running_models()]
    return {"object": "list", "data": data, "has_more": False}


def running_models() -> list[str]:
    names = [runtime.model_id] if runtime.model_id else []
    return names + [name for name in runtime.lanes.models() if name not in names]


async def check_request(model: str | None) -> None:
    if model and runtime.model_id and model != runtime.model_id:
        raise HTTPException(status_code=404, detail=f"model {model!r} is not loaded")
    await runtime.wait_for_route()
    if not runtime.tokenizer or not runtime.route:
        raise HTTPException(status_code=409, detail="no model is ready; load one first")


def server_sent(payload: dict[str, Any] | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


async def event_stream(
    tokens: AsyncIterator[dict[str, Any]],
    envelope: dict[str, Any],
    to_choice: Any,
) -> AsyncIterator[str]:
    """Wrap generated text in OpenAI-style server-sent events."""
    try:
        async for item in tokens:
            if "text" in item:
                yield server_sent({**envelope, "choices": [to_choice(item["text"], None)]})
                continue
            usage = {
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "total_tokens": item["prompt_tokens"] + item["completion_tokens"],
            }
            timing = {
                "time_to_first_token_ms": item["time_to_first_token_ms"],
                "tokens_per_second": item["tokens_per_second"],
                "epoch": runtime.epoch,
                "stages": len(runtime.route),
            }
            yield server_sent(
                {
                    **envelope,
                    "choices": [to_choice("", item["finish_reason"])],
                    "usage": usage,
                    "shardmesh": timing,
                }
            )
    except HTTPException as exc:
        # Headers are long gone, so the failure has to travel in the stream.
        yield server_sent({"error": {"message": str(exc.detail), "code": exc.status_code}})
    yield server_sent("[DONE]")


@app.post("/v1/completions")
async def completions(
    body: CompletionRequest,
    authorization: str | None = Header(default=None),
) -> Any:
    caller = verify_user(authorization)
    lane = await runtime.lanes.locate(body.model)
    if lane:
        return await runtime.lanes.forward(lane, "/v1/completions", body.model_dump(exclude_none=True), caller)
    await check_request(body.model)
    if isinstance(body.prompt, list):
        if len(body.prompt) != 1:
            raise HTTPException(status_code=400, detail="prototype supports one prompt per request")
        prompt = body.prompt[0]
    else:
        prompt = body.prompt
    options = (body.max_tokens, body.temperature, body.top_p, body.top_k, body.seed, body.stop)
    identifier = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if body.stream:
        envelope = {"id": identifier, "object": "text_completion", "created": created, "model": runtime.model_id}
        return StreamingResponse(
            event_stream(
                runtime.stream_tokens(prompt, *options),
                envelope,
                lambda text, reason: {"index": 0, "text": text, "finish_reason": reason},
            ),
            media_type="text/event-stream",
        )
    text, prompt_count, completion_count, finish_reason = await runtime.generate(prompt, *options)
    return {
        "id": identifier,
        "object": "text_completion",
        "created": created,
        "model": runtime.model_id,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_count,
            "completion_tokens": completion_count,
            "total_tokens": prompt_count + completion_count,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> Any:
    caller = verify_user(authorization)
    lane = await runtime.lanes.locate(body.model)
    if lane:
        return await runtime.lanes.forward(lane, "/v1/chat/completions", body.model_dump(exclude_none=True), caller)
    await check_request(body.model)
    messages = [message.model_dump() for message in body.messages]
    assert runtime.tokenizer is not None
    prompt = await asyncio.to_thread(runtime.tokenizer.render_chat, messages, body.enable_thinking)
    history = await asyncio.to_thread(runtime.tokenizer.render_chat, messages, body.enable_thinking, add_generation_prompt=False)
    options = (body.max_tokens, body.temperature, body.top_p, body.top_k, body.seed, body.stop, history)
    identifier = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if body.stream:
        envelope = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": runtime.model_id}
        return StreamingResponse(
            event_stream(
                runtime.stream_tokens(prompt, *options),
                envelope,
                lambda text, reason: {
                    "index": 0,
                    "delta": {"content": text} if text else {},
                    "finish_reason": reason,
                },
            ),
            media_type="text/event-stream",
        )
    text, prompt_count, completion_count, finish_reason = await runtime.generate(prompt, *options)
    return {
        "id": identifier,
        "object": "chat.completion",
        "created": created,
        "model": runtime.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_count,
            "completion_tokens": completion_count,
            "total_tokens": prompt_count + completion_count,
        },
    }


async def read_format(request: Request, translate: Any) -> tuple[dict[str, Any], ChatCompletionRequest]:
    """A request in another client's shape, as a chat request."""
    try:
        body = await request.json()
        return body, translate(body)
    except formats.Unsupported as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"could not read this request: {exc}") from exc


async def reply_pieces(chat: ChatCompletionRequest, authorization: str | None) -> AsyncIterator[dict[str, Any]]:
    """Run a chat request the normal way, so failover, lanes, keys, and limits all apply."""
    response = await chat_completions(chat, authorization)
    if response.status_code >= 400:
        content = b"".join([piece if isinstance(piece, bytes) else piece.encode() async for piece in response.body_iterator])
        raise HTTPException(status_code=response.status_code, detail=content.decode("utf-8", "replace")[:300])
    return formats.pieces(response.body_iterator)


async def whole_reply(chat: ChatCompletionRequest, authorization: str | None) -> dict[str, Any]:
    try:
        return await formats.gather(await reply_pieces(chat, authorization))
    except formats.StreamFailed as exc:
        raise HTTPException(status_code=exc.code, detail=str(exc)) from exc


def model_named(chat: ChatCompletionRequest) -> str:
    return chat.model or runtime.model_id or "unknown"


@app.post("/v1/messages")
async def anthropic_messages(request: Request, authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)) -> Any:
    body, chat = await read_format(request, formats.from_anthropic)
    key = credentials(authorization, x_api_key)
    if body.get("stream"):
        stream = formats.anthropic_stream(await reply_pieces(chat, key), model_named(chat))
        return StreamingResponse(stream, media_type="text/event-stream")
    return formats.anthropic_message(await whole_reply(chat, key), model_named(chat))


@app.post("/v1/responses")
async def openai_responses(request: Request, authorization: str | None = Header(default=None)) -> Any:
    body, chat = await read_format(request, formats.from_responses)
    if body.get("stream"):
        stream = formats.responses_stream(await reply_pieces(chat, authorization), model_named(chat))
        return StreamingResponse(stream, media_type="text/event-stream")
    return formats.responses_object(await whole_reply(chat, authorization), model_named(chat))


async def ollama_answer(request: Request, translate: Any, is_chat: bool, authorization: str | None) -> Any:
    started = time.monotonic()
    body, chat = await read_format(request, translate)
    if body.get("stream", True):  # Ollama streams unless told not to
        stream = formats.ollama_stream(await reply_pieces(chat, authorization), model_named(chat), is_chat, started)
        return StreamingResponse(stream, media_type="application/x-ndjson")
    return formats.ollama_reply(await whole_reply(chat, authorization), model_named(chat), is_chat, started)


@app.post("/api/chat")
async def ollama_chat(request: Request, authorization: str | None = Header(default=None)) -> Any:
    return await ollama_answer(request, formats.from_ollama_chat, True, authorization)


@app.post("/api/generate")
async def ollama_generate(request: Request, authorization: str | None = Header(default=None)) -> Any:
    return await ollama_answer(request, formats.from_ollama_generate, False, authorization)


@app.get("/api/tags")
async def ollama_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(authorization)
    return formats.ollama_tags(running_models())


@app.get("/api/version")
async def ollama_version() -> dict[str, str]:
    return {"version": app.version}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ShardMesh orchestrator")
    parser.add_argument("--host", default=os.getenv("SHARDMESH_HOST", "0.0.0.0"), help="address to listen on")
    parser.add_argument("--port", type=int, default=int(os.getenv("SHARDMESH_PORT", "8000")), help="port to listen on")
    parser.add_argument("--no-reuse-cached", action="store_true", help="plan as if no worker held any shards")
    parser.add_argument("--no-peer-fetch", action="store_true", help="always download from the model host")
    parser.add_argument("--state-file", default=os.getenv("SHARDMESH_STATE_FILE"), help="where the running route is saved")
    parser.add_argument("--node-id", default=os.getenv("SHARDMESH_NODE_ID"), help="this node's name in the mesh")
    parser.add_argument("--advertise-url", default=os.getenv("SHARDMESH_NODE_URL"), help="how other nodes reach this one; joins the mesh")
    parser.add_argument("--join", default=os.getenv("SHARDMESH_JOIN", ""), help="any existing nodes, comma separated")
    parser.add_argument("--keys-file", default=os.getenv("SHARDMESH_KEYS_FILE"), help="where this node keeps the pool's API keys")
    parser.add_argument("--trusted-network", action="store_true", help="send pool credentials unencrypted, because nobody else can read this network")
    parser.add_argument("--open", action="store_true", help="serve a network with no credentials, trusting everyone on it")
    parser.add_argument("--audit-file", default=os.getenv("SHARDMESH_AUDIT_FILE"), help="where to record who did what")
    parser.add_argument("--lane", default=None, help="run as this lane, beside a node's main control plane")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["SHARDMESH_KEYS_FILE"] = args.keys_file or str(pool_home() / "access.json")
    os.environ["SHARDMESH_NODE_ID"] = args.node_id or ""
    os.environ["SHARDMESH_NODE_URL"] = args.advertise_url or ""
    os.environ["SHARDMESH_JOIN"] = args.join
    os.environ["SHARDMESH_LANE"] = args.lane or ""
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "ShardMesh"
    os.environ["SHARDMESH_STATE_FILE"] = args.state_file or str(base / "orchestrator-state.json")
    os.environ["SHARDMESH_AUDIT_FILE"] = args.audit_file or str(base / "audit.jsonl")
    if args.no_reuse_cached:
        os.environ["SHARDMESH_REUSE_CACHED"] = "0"
    if args.no_peer_fetch:
        os.environ["SHARDMESH_PEER_FETCH"] = "0"
    check_exposure(args.host, load_token(), os.getenv("SHARDMESH_CA_FILE"), args.open, args.advertise_url, args.trusted_network)
    global runtime
    runtime = Orchestrator()
    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
