from __future__ import annotations

import argparse
import asyncio
import base64
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
from urllib.parse import urlencode, urlsplit

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from jinja2.exceptions import TemplateError
from starlette.background import BackgroundTask

from . import __version__
from .access import INVITE_FIELDS, load_authority, NeedsAuthority, node_of, ROLE_ADMIN, ROLE_MEMBER, ROLE_NODE, ROLE_USER, AccessControl, AuditLog, Caller
from .embedding import POOLED_TYPES, dimensions_problem, encoded, inputs_problem, length_problem, passes, shaped, smallest_dimensions
from .gguf import GgufHeader, parse_source, split_parts
from .llamacpp import KV_TYPES, PINNED_BUILD
from .statestore import stage_namespace
from . import audit
from .discovery import start_beacon
from .lanes import LaneDoor
from .mcp_server import INVALID_REQUEST, McpHandler, MeshTools, RpcError, answer_http
from .mesh import GOSSIP_SECONDS, ControlMesh
from .models import Adapter, GgufAdapter, SUPPORTED_TYPES, UnsupportedModel, create_adapter
from .planner import AMORTIZE_TOKENS, HOP_MS, Candidate, ModelProfile, PlacementError, keep_layers, memory_report, plan_placement
from .protocol import (
    hosts_model,
    pass_seconds,
    CacheRestoreRequest,
    ChatCompletionRequest,
    CompletionRequest,
    ConfigureRequest,
    PROTOCOL_VERSION,
    DraftModel,
    EmbeddingRequest,
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
from .pool import LOOPBACK, check_exposure, home as pool_home, load_token, read_membership_request
from .serving import MAX_STRANGER_BYTES, AddressLimit, BodyLimit, ChunkedResponses, client_address, serve, where_everything_is
from .mesh import clean_url
from .statements import Refused, Statements, make_statement
from .tickets import PREFIX as TICKET_PREFIX, NodeTickets, TicketChecker, TicketIdentity, client_auth, make_checker, reply_audience
from .tls import BEHALF, BEHALF_HEADER, FORWARD_MARK, INTERCEPTED, KEY_HELD, STATEMENT, intercepted, PEER, client_verify, node_transport, stay_on_this_machine_async
from .tokenizer import ChatTokenizer, IncrementalText, load_tokenizer
from .toolcalls import (
    MARKER_TOKENS,
    ReplyScanner,
    ToolRequestError,
    forced_opening,
    opens_in_thought,
    plan_tools,
    reply_limit,
    template_messages,
    tool_deltas,
    uses_tools,
)
from .vision import ImageRefused, chat_images, expand_prompt, image_id, projector_marker
from .weights import apply_model_hashes, Manifest, WeightSourceError, build_manifest, has_model_host

# Replans allowed when workers fail mid-load before the job gives up.
MAX_DEPLOY_ATTEMPTS = 4
PROGRESS_POLL_SECONDS = 0.5
# Missed progress polls in a row before a fetching worker counts as gone.
MAX_MISSED_POLLS = 6

# Least time a layer takes on a GPU, measured at 0.5 to 0.75 ms on an RX 7900 XT.
LAYER_FLOOR_MS = 0.5
# Small models that share a vocabulary with a family, for speculative decoding.
AUTO_DRAFTS = {
    "qwen3": "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf",
    "qwen3moe": "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf",
    "qwen35": "lmstudio-community/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q8_0.gguf",
    "qwen35moe": "lmstudio-community/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q8_0.gguf",
}
# llama.cpp's compute buffers on a GPU, on top of weights and context.
GGUF_STAGE_WORKSPACE_BYTES = 768 * 1024**2
DRAFT_WORKSPACE_BYTES = 512 * 1024**2
# The image encoder's process and buffers, beside its weights. With a 198 MiB
# projector, it peaked at 887 MiB in all on a 3000 by 3000 image.
PROJECTOR_WORKSPACE_BYTES = 768 * 1024**2
# How long to wait for an entry worker to read a request's images.
IMAGE_SECONDS = 120.0
# How often the orchestrator looks at the pool when nothing is asking it to.
MAINTENANCE_SECONDS = 10.0
LOGGED_EVENTS = {"failure", "recovery", "mesh", "rebalance", "security"}
# A live worker that failed to activate gets this many more tries, this far apart.
ACTIVATE_RETRIES = 2
ACTIVATE_RETRY_SECONDS = 5.0
# How long a measured link is believed, how many tries make one measurement, and the
# most a load waits for measuring. A link nobody could cross counts as this slow.
LINK_TTL_SECONDS = 600.0
LINK_PROBE_ROUNDS = 3
LINK_PROBE_SECONDS = 30.0
UNREACHABLE_LINK_MS = 10_000.0
# Replanning around a worker that is alive but failed may move this much at most.
REPLAN_MOVE_LIMIT_BYTES = 2 * 1024**3
# A restarted worker is back in seconds with its weights on disk, while moving them
# takes minutes. Replacing a lost worker waits this long, if replacing it would move much.
WORKER_RETURN_SECONDS = 75.0
WORKER_RETURN_CHECK_SECONDS = 5.0
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
# Anyone can ask to join, so this is how much of it one address may do. A real machine
# asks once, and one address cannot fill the queue before its first requests expire.
JOIN_BURST = 10
JOINS_PER_MINUTE = 1.0
# A good entry is free, or a stranger's bad ones could lock out an owner sharing their
# address. This is for entries that cost a signature check and changed nothing.
WASTED_ENTRY_BURST = 20
WASTED_ENTRIES_PER_MINUTE = 6.0
# Nodes a stage asks for copies of saved states. Each answers with a short list.
MAX_STATE_PEERS = 16
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
    embedding: bool = False
    projector: DraftModel | None = None
    image_marker: str | None = None

    @property
    def file_sizes(self) -> dict[str, int]:
        config = self.manifest.config
        return config.get("gguf_parts") or ({config["gguf_file"]: config["gguf_file_size"]} if "gguf_file" in config else {})

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
    check_pooling(request, adapter)
    draft = resolve_draft(request, manifest, adapter, sources, token)
    projector, image_marker = resolve_projector(request, adapter, sources, token)
    speculate_tokens = request.draft_tokens if draft else 0
    slots = request.parallel if isinstance(adapter, GgufAdapter) and not request.embedding else 1
    kv_share = KV_TYPES[request.kv_type if isinstance(adapter, GgufAdapter) else "f16"][1]
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
        kv_bytes_per_token=[adapter.kv_bytes_per_token(i, 2) * slots * kv_share // KV_TYPES["f16"][1] for i in range(adapter.num_layers)],
        # Rewinding needs a snapshot of recurrent state for each guess.
        state_bytes=[adapter.recurrent_state_bytes(i) * (1 + speculate_tokens) * slots for i in range(adapter.num_layers)],
        context_tokens=request.context_tokens,
        # Only a lone embedding table can be shared: GPT-2's unit also holds
        # position embeddings, which the head does not have.
        tied_embeddings=len(embedding) == 1 and embedding[0].name in head_tensors,
        amortize_tokens=AMORTIZE_TOKENS if reuse_cached else 0,
        stage_workspace_bytes=GGUF_STAGE_WORKSPACE_BYTES if isinstance(adapter, GgufAdapter) else 0,
        entry_extra_bytes=(draft.file_size + DRAFT_WORKSPACE_BYTES if draft else 0)
        + (projector.file_size + PROJECTOR_WORKSPACE_BYTES if projector else 0),
        layer_step=max(getattr(adapter, "interval", 1), 1),
    )
    hosted_copy = manifest.revision.startswith("local-") and control_source != request.model_id
    return ModelTarget(
        model_id=request.model_id,
        manifest=manifest,
        adapter=adapter,
        units={unit.name: unit for unit in units},
        profile=profile,
        spread=request.placement == "spread" and not request.embedding,
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
        embedding=request.embedding,
        projector=projector,
        image_marker=image_marker,
    )


def check_pooling(request: LoadModelRequest, adapter: Adapter) -> None:
    """An embedding model has to be loaded as one, and nothing else can be.

    A GGUF says how it pools. Left to chat, an embedding model only makes noise.
    """
    pooling = adapter.header.arch_key("pooling_type") if isinstance(adapter, GgufAdapter) else None
    if request.embedding and not isinstance(adapter, GgufAdapter):
        raise UnsupportedModel("embedding models run from GGUF files, and this is not one")
    if request.embedding and pooling not in POOLED_TYPES:
        raise UnsupportedModel(f"{request.model_id} declares no pooling, so it is not an embedding model; load it without --embedding")
    if not request.embedding and pooling in POOLED_TYPES:
        raise UnsupportedModel(f"{request.model_id} is an embedding model; load it with --embedding")


def holders_of(sources: dict[str, list[str]], model: str) -> list[str]:
    located = parse_source(model)
    if not located:
        return []
    # A node holding only some parts of a split file cannot serve the rest.
    parts = split_parts(located[1])
    return [holder for holder in sources.get(parts[0], []) if all(holder in sources.get(part, []) for part in parts)]


def resolve_draft(
    request: LoadModelRequest,
    manifest: Manifest,
    adapter: Adapter,
    sources: dict[str, list[str]] | None = None,
    token: str | None = None,
) -> DraftModel | None:
    """Pick a draft model, or None. "auto" never fails a load; a named draft can."""
    named = request.draft_model not in {"auto", "none"}
    if request.embedding and named:
        raise UnsupportedModel("an embedding model generates nothing, so it cannot use a draft model")
    if request.draft_model == "none" or request.embedding or not isinstance(adapter, GgufAdapter):
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


def resolve_projector(
    request: LoadModelRequest,
    adapter: Adapter,
    sources: dict[str, list[str]] | None = None,
    token: str | None = None,
) -> tuple[DraftModel | None, str | None]:
    """The projector a load names, with the marker its chat template writes for an image."""
    if not request.mmproj:
        return None, None
    if request.embedding or not isinstance(adapter, GgufAdapter):
        raise UnsupportedModel("a projector goes with a GGUF chat model")
    located = parse_source(request.mmproj)
    if located is None:
        raise UnsupportedModel(f"{request.mmproj} is not a GGUF file")
    holders = holders_of(sources or {}, request.mmproj)
    manifest = build_manifest(request.mmproj, "main", holders, token)
    try:
        marker = projector_marker(GgufHeader.from_skeleton(manifest.config["gguf"]), adapter.header.hidden_size, located[1])
    except ImageRefused as exc:
        raise UnsupportedModel(str(exc)) from exc
    projector = DraftModel(
        source=request.mmproj,
        file=located[1],
        file_size=manifest.config["gguf_file_size"],
        revision=manifest.revision,
        file_sources=holders,
    )
    return projector, marker


def check_image_marker(target: ModelTarget) -> None:
    """A template that never writes the marker would leave images with nowhere to go."""
    tokenizer, marker = target.tokenizer, target.image_marker
    assert tokenizer is not None and marker is not None
    if marker not in (tokenizer.chat_template or "") or len(tokenizer.encode(marker)) != 1:
        raise UnsupportedModel(f"this model's chat template has no {marker} for images, so {target.projector.file} cannot be used with it")


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
        "embedding": target.embedding,
        "mmproj": target.projector.source if target.projector else None,
        "download_bytes": download,
        "route": [stage.model_dump() for stage in route],
        "entry_worker": route[0].worker_id,
        "exit_worker": route[-1].worker_id,
        "estimated_tokens_per_second": round(1000 / milliseconds, 1) if milliseconds else None,
    }


class Orchestrator:
    def __init__(self) -> None:
        self.pool_token = load_token()
        self.reuse_cached = os.getenv("ISHARDED_REUSE_CACHED", "1") != "0"
        self.peer_fetch = os.getenv("ISHARDED_PEER_FETCH", "1") != "0"
        self.workers: dict[str, WorkerRecord] = {}
        self.links: dict[tuple[str, str], float] = {}  # round trip in ms, from one worker to another
        self.links_checked: dict[tuple[str, str], float] = {}
        self.model_id: str | None = None
        self.planning_model: str | None = None
        self.active: ModelTarget | None = None
        self.tokenizer: ChatTokenizer | None = None
        self.route: list[StageAssignment] = []
        self.active_cache_url: str | None = None
        self.epoch = 0
        self.job: LoadJob | None = None
        self.client = httpx.AsyncClient(transport=node_transport(), event_hooks={"request": [stay_on_this_machine_async]}, timeout=httpx.Timeout(3600.0, connect=10.0), verify=client_verify(), auth=client_auth())
        authority = load_authority()
        self.access = AccessControl.load(
            self.pool_token,
            os.getenv("ISHARDED_KEYS_FILE"),
            identities=authority is not None,
            authority=authority,
        )
        self.audit = AuditLog(os.getenv("ISHARDED_AUDIT_FILE"), os.getenv("ISHARDED_NODE_ID") or socket.gethostname())
        self.control_lock = asyncio.Lock()
        self.generation_lock = asyncio.Lock()
        # Survives a restart: what was running, so the pool can be picked back up.
        self.state_path = Path(os.environ["ISHARDED_STATE_FILE"]) if os.getenv("ISHARDED_STATE_FILE") else None
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
        seen_as = os.getenv("ISHARDED_NODE_URL")
        if seen_as:
            seeds = tuple(url.strip() for url in os.getenv("ISHARDED_JOIN", "").split(",") if url.strip())
            self.mesh = ControlMesh(os.getenv("ISHARDED_NODE_ID") or socket.gethostname(), seen_as, seeds)
        self.background: set[asyncio.Task[Any]] = set()
        self.lanes = LaneDoor(self)
        self.statements = Statements()
        self.join_limit = AddressLimit(JOIN_BURST, JOINS_PER_MINUTE)
        self.entry_limit = AddressLimit(WASTED_ENTRY_BURST, WASTED_ENTRIES_PER_MINUTE)
        self.round_trip_ms: dict[str, float] = {}
        self.join_answers: dict[str, set[str]] = {}
        self.last_audit: dict[str, Any] | None = None
        self.beacon: Any = None
        self.lane_name = os.getenv("ISHARDED_LANE") or None
        self.started_at = time.time()
        self.requests_total = 0
        self.tokens_generated = 0
        self.embedding_tokens = 0
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

    def control_name(self, node: str | None = None) -> str:
        """A control plane, as a statement names it. A lane shares its node's name."""
        name = node or (self.mesh.node_id if self.mesh else "")
        return f"{name}/{self.lane_name}" if self.lane_name else name

    def speak_for(self, caller: Caller, target: str, method: str, path: str, body: bytes = b"") -> str:
        """What tells the next control plane who asked. The person's key does not go with it."""
        if self.access.ownerless:
            return f"{caller.role}:{caller.name}"  # keys never travel here, so nobody could check it
        statement, key = STATEMENT.get(), KEY_HELD.get()
        if statement or not key:
            return statement or ""
        assert self.mesh is not None
        return make_statement(key, self.mesh.node_id, target, method, path, body)

    async def close(self) -> None:
        for task in (self.maintenance_task, self.gossip_task):
            if task is not None:
                task.cancel()
        if self.beacon is not None:
            self.beacon.close()
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
        if target.projector is not None:
            check_image_marker(target)
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
        if os.getenv("ISHARDED_DISCOVERABLE") and not self.lane_name and self.mesh is not None:
            self.spawn(self._start_beacon())

    async def _start_beacon(self) -> None:
        from .up import authority_hash

        assert self.mesh is not None
        pem = Path(os.environ["ISHARDED_CA_FILE"]).read_text(encoding="utf-8")
        mesh = self.mesh
        self.beacon = await start_beacon(lambda: {"url": mesh.url, "authority": authority_hash(pem), "name": mesh.node_id})

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
            if usable and "inference" in contribution.roles and self._hosts(contribution) and self._runs_on(record.registration.hardware):
                group = standby if contribution.standby else preferred
                group.append(self._candidate(worker_id, record))
        return preferred, standby

    def _hosts(self, contribution: Any) -> bool:
        model = self.planning_model or self.model_id
        return not model or hosts_model(contribution.host_models, model)

    def _runs_on(self, hardware: Any) -> bool:
        """A GPU reached through Vulkan runs llama.cpp, so it can only take GGUF models."""
        model = self.planning_model or self.model_id
        return hardware.backend != "vulkan" or not model or ".gguf" in model.lower()

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

    def copy_holders(self, route: list[StageAssignment]) -> dict[str, str]:
        """One node to keep a copy of each stage's saved states.

        A stage's KV is for its own layers, so one copy on one other machine is enough
        to survive losing this one. A node offering memory for this comes first, then
        the next stage along the route, which already sees what this stage sees.
        """
        live = {name: record for name, record in self.workers.items() if record.live and not record.quarantined}
        offered = [name for name, record in live.items() if "kv-cache" in record.registration.hardware.contribution.roles]
        order = [stage.worker_id for stage in route]
        holders: dict[str, str] = {}
        for index, stage in enumerate(route):
            record = live.get(stage.worker_id)
            if record is None:
                continue
            own_machine = record.registration.hardware.host_id
            others = [name for name in live if name not in order]
            for name in [*offered, *order[index + 1 :], *order[:index], *others]:
                holder = live.get(name)
                if holder is not None and holder.registration.hardware.host_id != own_machine:
                    holders[stage.worker_id] = str(holder.registration.advertised_url).rstrip("/")
                    break
        return holders

    def state_peers(self, worker_id: str) -> list[str]:
        """Who to ask for a copy, which may be any node, as the holder may have changed."""
        urls = [
            str(record.registration.advertised_url).rstrip("/")
            for name, record in self.workers.items()
            if record.live and not record.quarantined and name != worker_id
        ]
        return urls[:MAX_STATE_PEERS]

    async def _clear_old_copies(self, target: ModelTarget, route: list[StageAssignment]) -> None:
        """Copies for layers nothing runs any more are no use, on any node."""
        kv_type = target.request.kv_type if target.request else "f16"
        wanted = [
            stage_namespace(PINNED_BUILD, target.model_id, target.manifest.revision, stage.start_layer, stage.end_layer, target.speculate_tokens, kv_type)
            for stage in route
        ]

        async def tell(record: WorkerRecord) -> None:
            try:
                await self.client.post(
                    str(record.registration.advertised_url).rstrip("/") + "/internal/state/keep",
                    json={"namespaces": wanted},
                    headers=auth_headers(self.pool_token),
                    timeout=5.0,
                )
            except httpx.HTTPError:
                pass  # eviction gets to them in the end

        await asyncio.gather(*(tell(record) for record in self.workers.values() if record.live))

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
                # A worker that is slow to answer has not lost its shards, and
                # planning as if it had would move them all.
                return worker_id, target.held.get(worker_id, set()), worker_id in target.serving
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

    async def measure_links(self) -> None:
        """Time the links between workers, which the planner routes by.

        Every token crosses each link on its route and comes back, so on a mesh that
        spans cities these round trips set the speed. Each is measured from the end
        that would send, and a link that fails stays on record so that no route needs it.
        """
        urls = {
            worker_id: str(record.registration.advertised_url).rstrip("/")
            for worker_id, record in self.workers.items()
            if record.live and not record.quarantined
        }
        now = time.monotonic()

        async def probe(source: str, target: str) -> None:
            try:
                response = await self.client.post(
                    urls[source] + "/internal/diag/probe",
                    json={"url": urls[target], "sizes": [100], "rounds": LINK_PROBE_ROUNDS},
                    headers=auth_headers(self.pool_token),
                    timeout=10.0,
                )
                response.raise_for_status()
                self.links[source, target] = float(response.json()["100"]["median_ms"])
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                self.links[source, target] = UNREACHABLE_LINK_MS
            self.links_checked[source, target] = now

        async def from_one(source: str) -> None:
            for target in urls:
                if target != source and now - self.links_checked.get((source, target), -LINK_TTL_SECONDS) >= LINK_TTL_SECONDS:
                    await probe(source, target)

        try:
            await asyncio.wait_for(asyncio.gather(*map(from_one, urls)), LINK_PROBE_SECONDS)
        except asyncio.TimeoutError:
            self.add_event("mesh", "Link measuring cut short", f"Planning with what {LINK_PROBE_SECONDS:.0f} s could measure")

    def plan_memory(self, target: ModelTarget, route: list[StageAssignment]) -> dict[str, Any]:
        """A route's use of the memory on offer, with what each stage spills.

        A running route can sit on a worker that planning would now pass over,
        such as a cordoned one, and its spill still needs showing.
        """
        preferred, standby = self.candidate_groups()
        candidates = preferred + standby
        offered = {candidate.worker_id for candidate in candidates}
        running = {stage.worker_id for stage in route} - offered
        candidates += [self._candidate(name, self.workers[name]) for name in sorted(running) if name in self.workers]
        return memory_report(route, candidates, target.profile)

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
        preferred, standby = self._with_holdings(preferred, target), self._with_holdings(standby, target)
        kept = None if honor_pins_strictly else self._keep_stages(target, preferred, standby, available, pins)
        route = kept or plan_placement(preferred, standby, target.profile, spread=target.spread, **pins)
        if target.embedding and len(route) > 1:
            # Pooling happens at the end of the model, in llama.cpp, for whole
            # sequences, so a stage boundary has nothing correct to pass along.
            raise PlacementError(f"{target.model_id} would take {len(route)} stages here, and an embedding model cannot be split; it needs one worker that can hold it all")
        return route

    def _keep_stages(
        self, target: ModelTarget, preferred: list[Candidate], standby: list[Candidate], available: set[str], pins: dict[str, str | None]
    ) -> list[StageAssignment] | None:
        """Replace only what was lost, if something can take its layers.

        A standby worker is only for when a model does not fit without it, so it steps
        in only if the workers still running could not hold the model between them.
        """
        if target is not self.active or all(stage.worker_id in available for stage in self.route):
            return None
        route = keep_layers(self.route, preferred, [], target.profile)
        if route is not None or not standby:
            return route
        try:
            plan_placement(preferred, [], target.profile, spread=target.spread, **pins)
            return None
        except PlacementError:
            return keep_layers(self.route, preferred, standby, target.profile)

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
                file_sources=target.file_sources if target.peer_fetch or not has_model_host(target.source_for(stage.worker_id)) else [],
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
                failed[stage.worker_id] = f"could not start download: {str(exc) or type(exc).__name__}"

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
        saved_states = isinstance(target.adapter, GgufAdapter)  # see statecopy.py
        holders = self.copy_holders(route) if saved_states else {}
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
                prompt_cache=target.request is None or target.request.prompt_cache,
                kv_type=target.request.kv_type if target.request else "f16",
                copy_url=holders.get(stage.worker_id),
                state_peers=self.state_peers(stage.worker_id) if saved_states else [],
                embedding=target.embedding,
                projector=target.projector,
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
        if saved_states:
            self.spawn(self._clear_old_copies(target, route))
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
            await self.measure_links()
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
        else:
            await self._await_lost_workers()
        self.job = LoadJob(job_id=uuid.uuid4().hex, model_id=self.active.model_id)
        return await self.run_job(
            self.job,
            self.deploy(self.active, self.job, generation_lock_held=generation_lock_held),
        )

    async def _await_lost_workers(self) -> None:
        """Give a lost worker time to come back, when replacing it would move many weights.

        Nothing is lost by waiting, since the replies on this route would wait for the
        move anyway. A worker that does not return is replaced as before.
        """
        deadline = time.monotonic() + WORKER_RETURN_SECONDS
        assert self.active is not None
        while self._absent_from_route() and time.monotonic() < deadline:
            await self.refresh_holdings(self.active)
            try:
                moved = describe_plan(self.active, self.plan(self.active))["download_bytes"]
            except PlacementError:
                moved = None  # nothing could take the model without it
            if moved is not None and moved <= REPLAN_MOVE_LIMIT_BYTES:
                return
            await asyncio.sleep(WORKER_RETURN_CHECK_SECONDS)
            await self.refresh_worker_metrics()
            self._review_quarantine()

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
                images=session.images if must_replay else [],
                prompt_rows=must_replay,
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
        # Whatever opens the reply opens every other turn too, so it marks the cuts.
        cut_token = None if boundary is None else session.tokens[boundary]
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
                cut_token=cut_token,
                images=session.images,
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

    def model_kind(self) -> dict[str, Any]:
        """What the model on this route is for, as /v1/models reports it."""
        active = self.active
        if active is not None and active.embedding:
            assert isinstance(active.adapter, GgufAdapter)
            return {"kind": "embedding", "vision": False, "dimensions": active.adapter.output_width}
        return {"kind": "chat", "vision": bool(active and active.projector)}

    async def ask_entry(self, path: str, payload: dict[str, Any], seconds: float) -> dict[str, Any]:
        """One request to the entry worker, following the route through any failure.

        The epoch goes in on each try, because a retry can land on a new route.
        """
        problem: Any = "no entry worker answered"
        for _ in range(STREAM_ATTEMPTS):
            await self.wait_for_route()
            if not self.route:
                raise HTTPException(status_code=409, detail="no active model pipeline")
            entry, epoch = self.route[0], self.epoch
            try:
                response = await self.client.post(
                    entry.url.rstrip("/") + path,
                    json={**payload, "epoch": epoch},
                    headers=auth_headers(self.pool_token),
                    timeout=httpx.Timeout(seconds, connect=10.0),
                )
            except httpx.HTTPError as exc:
                problem = {"failed_worker": entry.worker_id, "message": str(exc)}
            else:
                if response.is_success:
                    return response.json()
                if response.status_code in {400, 429}:
                    raise HTTPException(status_code=response.status_code, detail=reply_detail(response))
                problem = "epoch" if response.status_code == 409 else reply_detail(response)
            if not await self._recover_stream(entry.worker_id, problem, epoch):
                break
        raise HTTPException(status_code=502, detail=f"the entry worker could not answer: {problem}")

    async def place_images(self, tokens: list[int], boundary: int | None, images: list[bytes]) -> tuple[list[int], int | None]:
        """Give each image in a prompt its rows, which only the entry worker can count."""
        assert self.active is not None and self.active.image_marker is not None and self.tokenizer is not None
        marker = self.tokenizer.encode(self.active.image_marker)[0]
        if tokens.count(marker) != len(images):
            raise HTTPException(status_code=400, detail=f"this prompt holds {tokens.count(marker)} image markers for {len(images)} images; take {self.active.image_marker} out of the text")
        encoded = [base64.b64encode(data).decode("ascii") for data in images]
        answer = await self.ask_entry("/internal/images/measure", {"images": encoded}, IMAGE_SECONDS)
        sizes = [(image_id(data), int(item["rows"])) for data, item in zip(images, answer["images"])]
        return expand_prompt(tokens, boundary, marker, sizes)

    async def embed(self, texts: list[str], dimensions: int | None) -> tuple[list[np.ndarray], int]:
        """A unit vector for each text, in order, and the tokens they cost."""
        assert self.active is not None and self.tokenizer is not None
        tokenizer = self.tokenizer
        inputs = await asyncio.to_thread(lambda: [tokenizer.encode_input(text) for text in texts])
        lengths = [len(tokens) for tokens in inputs]
        problem = length_problem(lengths, self.active.profile.context_tokens)
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        self.requests_total += 1
        vectors: list[np.ndarray] = [np.empty(0)] * len(inputs)
        for group in passes(lengths):
            pack = [inputs[index] for index in group]
            answer = await self.ask_entry("/v1/embed", {"inputs": pack}, pass_seconds(sum(map(len, pack))))
            rows = np.frombuffer(base64.b64decode(answer["vectors"]), dtype=np.float32).reshape(len(group), -1)
            for index, row in zip(group, rows):
                vectors[index] = shaped(row, dimensions)
            self.embedding_tokens += sum(map(len, pack))
        return vectors, sum(lengths)

    def cache_summary(self) -> dict[str, Any]:
        """What reuse is saving. Every stage runs the same rows, so the entry speaks for the route.

        Adding the stages up would count each token once per stage. Bytes are real on
        each node, so those do add up. RAM and disk are mostly the same states twice.
        """
        caches = {name: ((record.health or {}).get("telemetry") or {}).get("cache") or {} for name, record in self.workers.items()}
        entry = caches.get(self.route[0].worker_id, {}) if self.route else {}
        computed = entry.get("prompt_tokens_computed", 0)
        sources = entry.get("tokens_reused") or {}
        reused = sum(sources.values())
        return {
            "prompt_tokens_computed": computed,
            "prompt_tokens_reused": reused,
            "tokens_reused": sources,
            "saved_percent": round(100 * reused / (computed + reused), 1) if computed + reused else None,
            "ram_bytes": sum(item.get("ram_bytes", 0) for item in caches.values()),
            "disk_bytes": sum(item.get("disk_bytes", 0) for item in caches.values()),
            "held_bytes": sum(item.get("held_bytes", 0) for item in caches.values()),
        }

    def metrics_text(self) -> str:
        """This node's numbers in Prometheus format. Every node reports for itself."""
        lines: list[str] = []

        def add(name: str, value: float | None, help_text: str, kind: str = "gauge", **labels: str) -> None:
            if f"# HELP isharded_{name} " not in "\n".join(lines):
                lines.append(f"# HELP isharded_{name} {help_text}")
                lines.append(f"# TYPE isharded_{name} {kind}")
            tags = ",".join(f'{key}="{text}"' for key, text in labels.items())
            lines.append(f"isharded_{name}{{{tags}}} {0 if value is None else value}" if tags else f"isharded_{name} {0 if value is None else value}")

        add("requests_total", self.requests_total, "Replies started on this node", "counter")
        add("tokens_generated_total", self.tokens_generated, "Tokens produced for this node's clients", "counter")
        add("embedding_tokens_total", self.embedding_tokens, "Tokens embedded for this node's clients", "counter")
        saving = self.cache_summary()
        add("prompt_tokens_computed_total", saving["prompt_tokens_computed"], "Prompt tokens the route computed", "counter")
        add("prompt_tokens_reused_total", saving["prompt_tokens_reused"], "Prompt tokens the route restored instead of computing", "counter")
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
            cache = telemetry.get("cache") or {}
            add("worker_prompt_tokens_computed_total", cache.get("prompt_tokens_computed"), "Prompt tokens this stage computed", "counter", worker=worker_id)
            for source, count in (cache.get("tokens_reused") or {}).items():
                add("worker_tokens_reused_total", count, "Prompt tokens this stage restored instead of computing", "counter", worker=worker_id, source=source)
            add("worker_state_misses_total", cache.get("state_misses"), "Saved states a stage did not hold when asked", "counter", worker=worker_id)
            for tier in ("ram", "disk", "held"):
                add("worker_cache_bytes", cache.get(f"{tier}_bytes"), "Bytes of saved state held", worker=worker_id, tier=tier)
            add("worker_state_copy_bytes_total", cache.get("copy_bytes_sent"), "Bytes of state copied to another node", "counter", worker=worker_id)
            add("worker_state_copies_failed_total", cache.get("copies_failed"), "State copies that did not arrive", "counter", worker=worker_id)
            add("worker_state_copies_dropped_total", cache.get("copies_dropped"), "State copies dropped behind a newer one or a full queue", "counter", worker=worker_id)
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
        images: list[bytes] | None = None,
        keep_markers: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield text as it is generated, then one final summary.

        Text is held back while it could still be the start of a stop string,
        or while it ends inside a multi-byte character. ``keep_markers`` keeps
        the special tokens a tool call is written with.
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
        boundary = await asyncio.to_thread(self._history_boundary, prompt_tokens, history)
        if images:
            prompt_tokens, boundary = await self.place_images(prompt_tokens, boundary, images)
        max_tokens = self._fit_context(len(prompt_tokens), max_tokens)
        session = SessionState(tokens=list(prompt_tokens), epoch=self.epoch, needs_reset=True)
        session.images = [base64.b64encode(data).decode("ascii") for data in images or []]
        markers = [marker for marker in ([stop] if isinstance(stop, str) else (stop or [])) if marker]
        holdback = max((len(marker) for marker in markers), default=1) - 1
        generated: list[int] = []
        sent = ""
        finish_reason = "length"
        started = time.perf_counter()
        first_token_at: float | None = None

        reader = tokenizer.keeping(MARKER_TOKENS) if keep_markers else tokenizer
        decoder = IncrementalText(reader)
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

        text = reader.decode(generated)
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
        images: list[bytes] | None = None,
    ) -> tuple[str, int, int, str]:
        pieces: list[str] = []
        summary: dict[str, Any] = {}
        async for item in self.stream_tokens(prompt, max_tokens, temperature, top_p, top_k, seed, stop, history, images):
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
            "last_audit": self.last_audit,
            "model": self.model_id,
            **self.model_kind(),
            "load_job": self.job.snapshot() if self.job else None,
            "active_plan": (
                {**describe_plan(self.active, self.route), "memory": self.plan_memory(self.active, self.route)}
                if self.active and self.route
                else None
            ),
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
                "cache": self.cache_summary(),
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


app = FastAPI(title="iSharded orchestrator", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def require(authorization: str | None, role: str) -> Caller:
    behalf = BEHALF.get()
    caller = Caller(*behalf) if behalf else runtime.access.identify(authorization, PEER.get())
    if caller is None:
        raise HTTPException(status_code=401, detail="a valid key is required")
    if not runtime.access.permits(caller, role):
        raise HTTPException(status_code=403, detail=f"this needs the {role} role")
    return caller


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


FORWARDED_PATHS = ("/v1/", "/admin/")
SAFE_NODE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# Connection details belong to each hop, and httpx sets the length itself.
REPEATABLE_METHODS = {"GET", "HEAD"}
HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "content-encoding"}


def asked_path(request: Request) -> str:
    """A request's path and query, written the same way by every node that handles it."""
    query = urlencode(sorted(request.query_params.multi_items()))
    return f"{request.url.path}?{query}" if query else request.url.path


async def believe_forward(request: Request) -> str | None:
    """Take from a forwarded request only what it can prove. Says why, if it proved nothing."""
    access, peer = runtime.access, PEER.get()
    claim = request.headers.get(BEHALF_HEADER)
    if peer is None or not claim or access.authority is None or peer in access.revoked:
        return None
    if access.ownerless:
        # An operator commands their own node. Here they are only a member,
        # and the node that passed this on is the one that answers for them.
        role, _, name = claim.partition(":")
        if role not in {ROLE_USER, ROLE_MEMBER, ROLE_ADMIN}:
            return None
        proven = f"{peer}/{name.rpartition('/')[2]}", ROLE_USER if role == ROLE_USER else ROLE_MEMBER
    else:
        try:
            proven = runtime.statements.check(access, claim, peer, runtime.control_name(), request.method, asked_path(request), await request.body(), bool(runtime.lane_name))
        except Refused as exc:
            return str(exc)
        STATEMENT.set(claim)
    BEHALF.set(proven)
    request.state.vouched = f"{proven[1]}:{proven[0]}"
    return None


@app.middleware("http")
async def forward_to_coordinator(request: Request, call_next: Any) -> Any:
    """Serve the API from any node.

    Replies run on the node that was asked, straight to the entry worker, so they
    do not depend on the coordinator. Only management goes there, along with
    replies from a node that has not installed the route yet.
    """
    KEY_HELD.set(runtime.access.bearer(request.headers.get("authorization")))
    if FORWARD_MARK in request.headers:
        # This connection proves which node passed the request on, not who sent it.
        refused = await believe_forward(request)
        PEER.set(None)
        if refused:
            return JSONResponse(status_code=401, content={"detail": refused})
    mesh = runtime.mesh
    path = request.url.path
    if any(part in {".", ".."} for part in path.replace("\\", "/").split("/")):
        # httpx would resolve these, turning a forwarded path into any other.
        return JSONResponse(status_code=400, content={"detail": "bad path"})
    serves_locally = path.startswith("/v1/") and bool(runtime.route)
    own_choice = path in {"/admin/invites", "/admin/network", "/admin/audit"} or path.startswith("/admin/join/") or (runtime.access.ownerless and (path.startswith("/admin/keys") or path.endswith("/revoke")))
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
        # they gave it to. It is checked here, and only a statement signed with it travels.
        caller = runtime.access.identify(request.headers.get("authorization"), None)
        if caller is None:
            return JSONResponse(status_code=401, content={"detail": "a valid key is required"})
        wait = runtime.access.spend(caller)
        if wait:
            return JSONResponse(status_code=429, content={"detail": f"rate limit reached for {caller.name}"})
        headers.pop("authorization", None)
    else:
        caller = None

    async def ask_coordinator() -> httpx.Response:
        if caller is not None:  # a statement is good once, so each try makes its own
            headers[BEHALF_HEADER] = runtime.speak_for(caller, runtime.control_name(mesh.leader_id), request.method, asked_path(request), await request.body())
        return await runtime.client.send(
            runtime.client.build_request(
                request.method,
                target + request.url.path,
                params=request.query_params,
                content=await request.body(),
                headers=headers,
            ),
            stream=True,
        )

    try:
        try:
            upstream = await ask_coordinator()
        except httpx.RemoteProtocolError:
            # A keep-alive connection can die while idle, seen only on its next use.
            if request.method not in REPEATABLE_METHODS:
                raise
            upstream = await ask_coordinator()
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=503, content={"detail": f"the coordinator did not answer: {exc}"})
    passed = {name: value for name, value in upstream.headers.items() if name.lower() not in HOP_HEADERS}
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code, headers=passed, background=BackgroundTask(upstream.aclose)
    )


def page(name: str) -> HTMLResponse:
    folder = Path(__file__).parent
    styles = (folder / "ui.css").read_text(encoding="utf-8")
    content = (folder / name).read_text(encoding="utf-8").replace("/* isharded:ui.css */", styles)
    content = content.replace("<!-- isharded:flags.svg -->", (folder / "flags.svg").read_text(encoding="utf-8"))
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
        vouched = getattr(request.state, "vouched", None)  # what was proven, and never what was claimed
        client = request.client.host if request.client else None
        actor = f"{vouched} via {PEER.get()}" if vouched else caller.name if caller else "unknown"
        runtime.audit.record(actor, request.method, path, response.status_code, client)
    return response


# Added last, so it runs first: nothing oversized is read, forwarded or parsed.
app.add_middleware(TicketIdentity)
app.add_middleware(BodyLimit, paths={"/join/requests": MAX_STRANGER_BYTES, "/access/entries": MAX_STRANGER_BYTES})
app.add_middleware(ChunkedResponses, path="/mcp")


@app.get("/ready")
async def ready() -> JSONResponse:
    """For load balancers: can this node serve a reply right now?"""
    serving = bool(runtime.route and runtime.tokenizer)
    return JSONResponse(status_code=200 if serving else 503, content={"ready": serving})


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
    await runtime.measure_links()
    try:
        route = runtime.plan(target, honor_pins_strictly=True)
    except PlacementError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    return {**describe_plan(target, route), "memory": runtime.plan_memory(target, route)}


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


def hold_back(limit: AddressLimit, address: str) -> None:
    wait = limit.wait(address)
    if wait:
        raise HTTPException(status_code=429, detail="too many requests from this address; try again soon", headers={"Retry-After": str(int(wait) + 1)})


def address_of(request: Request) -> str:
    return client_address(request.client.host if request.client else None)


@app.post("/join/requests")
async def request_to_join(request: Request, body: dict[str, Any]) -> dict[str, str]:
    """Anyone may ask. Asking grants nothing until the owner signs an answer."""
    if runtime.access.authority is None or runtime.access.ownerless:
        raise HTTPException(status_code=404, detail="this pool has no owner to ask")
    address = address_of(request)
    hold_back(runtime.join_limit, address)
    runtime.join_limit.charge(address)  # before the parsing, which is the work a stranger is asking for
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
    runtime.access.expire_join_requests()
    # What a person decided stays on this machine, for the signer beside it to act on.
    return {"requests": [{**held, "approved": name in runtime.join_answers.get("approve", ()), "denied": name in runtime.join_answers.get("deny", ())} for name, held in runtime.access.join_requests.items()]}


@app.post("/admin/join/{name}/{answer}")
async def answer_join(name: str, answer: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    """A person's decision. Only the machine holding the authority can act on it."""
    verify_admin(authorization)
    home = os.getenv("ISHARDED_UP_HOME")
    if answer not in {"approve", "deny"}:
        raise HTTPException(status_code=404, detail="the answer is approve or deny")
    if not home or not Path(home, "ca.key").is_file():
        raise HTTPException(status_code=409, detail="only the machine that made this mesh can answer; open this page there")
    if name not in runtime.access.join_requests:
        raise HTTPException(status_code=404, detail=f"{name} is not asking to join")
    runtime.join_answers.setdefault(answer, set()).add(name)
    return {"status": answer}


@app.get("/join/authority")
async def join_authority() -> dict[str, str]:
    """The mesh's public certificate, for a new machine to check against its invite."""
    if runtime.access.authority is None or runtime.access.ownerless:
        raise HTTPException(status_code=404, detail="this pool has no authority")
    return {"authority": Path(os.environ["ISHARDED_CA_FILE"]).read_text(encoding="utf-8")}


@app.post("/admin/invites")
async def make_invite(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """A code that adds a machine. Only the machine holding the authority can honor one."""
    verify_admin(authorization)
    home = os.getenv("ISHARDED_UP_HOME")
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
    return {"certificate": certificate, "authority": Path(os.environ["ISHARDED_CA_FILE"]).read_text(encoding="utf-8")}


@app.post("/access/entries")
async def add_access_entry(request: Request, body: dict[str, Any]) -> dict[str, str]:
    """Take a change signed by the pool authority. The signature is the credential."""
    if runtime.access.authority is None:
        raise HTTPException(status_code=404, detail="this pool has no authority")
    address = address_of(request)
    hold_back(runtime.entry_limit, address)
    held = len(runtime.access.signed)
    if not runtime.access.add_signed(body):
        runtime.entry_limit.charge(address)
        raise HTTPException(status_code=403, detail="this entry is not signed by the pool authority")
    if len(runtime.access.signed) == held:
        runtime.entry_limit.charge(address)  # a repeat costs the same to check and changes nothing
        return {"status": "accepted"}
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
    # A node that has not installed the route yet still knows it from gossip.
    state = (runtime.mesh.replica if runtime.mesh else None) or {}
    route = [stage.model_dump() for stage in runtime.route] or state.get("route", [])
    return {
        "lane": runtime.lane_name,
        "model": runtime.model_id or state.get("request", {}).get("model_id"),
        "ready": bool(runtime.tokenizer and runtime.route),
        "leader": runtime.mesh.leader_id if runtime.mesh else None,
        "leader_url": runtime.mesh.leader_url if runtime.mesh else None,
        "entries_held": runtime.access.entries_held,
        "workers": sum(1 for record in runtime.workers.values() if record.live),
        "route": [{key: stage[key] for key in ("worker_id", "start_layer", "end_layer")} for stage in route],
        "load_job": runtime.job.snapshot() if runtime.job else None,
        **runtime.model_kind(),
    }


@app.post("/internal/lane/access")
async def lane_access(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Signed access changes, passed in by this node's front door. Signatures are checked in merge."""
    verify_token(authorization)
    runtime.access.merge(body)
    if runtime.lane_name and runtime.mesh and PEER.get() == runtime.mesh.node_id and not runtime.access.ownerless:
        runtime.access.take_node_keys(body)
    runtime.apply_revocations()
    return {"status": "merged"}


@app.get("/admin/status")
async def status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(authorization)
    await runtime.refresh_worker_metrics()
    return runtime.status()


@app.post("/admin/audit")
async def audit_routes(body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Ask every route serving a model the same made-up questions, and compare."""
    caller = network_caller(authorization)
    model = str(body.get("model") or runtime.model_id or "")
    samples = min(max(int(body.get("samples", audit.SAMPLES)), 1), audit.MOST_SAMPLES)
    lanes = runtime.lanes.routes_for(model)
    on_main = model == runtime.model_id and bool(runtime.route)
    if len(lanes) + on_main < 2:
        raise HTTPException(status_code=409, detail=f"checking needs two routes for {model or 'a model'}; load another with --lane NAME --lane-apart")
    questions = audit.canaries(samples)

    def request(question: str) -> dict[str, Any]:
        return {"model": model, "messages": [{"role": "user", "content": question}], "max_tokens": audit.REPLY_TOKENS, "temperature": 0.0, "seed": 1, "enable_thinking": False}

    async def ask_main() -> list[str]:
        answers = []
        for question in questions:
            reply = await chat_completions(ChatCompletionRequest(**request(question)), authorization)
            answers.append(reply["choices"][0]["message"]["content"])
        return answers

    async def ask_lane(name: str) -> list[str]:
        return [await runtime.lanes.ask(name, request(question), caller) for question in questions]

    names = (["main"] if on_main else []) + lanes
    work = ([ask_main()] if on_main else []) + [ask_lane(name) for name in lanes]
    try:
        answers = await asyncio.gather(*work)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"a route did not answer: {exc}") from exc
    result = audit.judge(dict(zip(names, answers)))
    workers = {"main": [stage.worker_id for stage in runtime.route]} if on_main else {}
    workers |= {name: [str(stage["worker_id"]) for stage in runtime.lanes.seen.get(name, {}).get("route", [])] for name in lanes}
    result |= {"model": model, "samples": samples, "routes": workers, "at": int(time.time())}
    runtime.last_audit = result
    if result["verdict"] != "agree":
        detail = f"{model}: suspects {result['suspects']}" if result["suspects"] else f"{model}: two routes differ, so a third is needed to say which"
        runtime.add_event("failure", "Routes disagree", detail)
    return result


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
async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    verify_user(authorization)
    rows = {runtime.model_id: runtime.model_kind()} if runtime.model_id else {}
    rows |= {name: kind for name, kind in runtime.lanes.model_kinds().items() if name not in rows}
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "isharded", **kind} for name, kind in rows.items()]}


async def check_request(model: str | None, embedding: bool = False) -> None:
    if model and runtime.model_id and model != runtime.model_id:
        raise HTTPException(status_code=404, detail=f"model {model!r} is not loaded")
    await runtime.wait_for_route()
    if not runtime.tokenizer or not runtime.route:
        raise HTTPException(status_code=409, detail="no model is ready; load one first")
    runs_embeddings = bool(runtime.active and runtime.active.embedding)
    if runs_embeddings and not embedding:
        raise HTTPException(status_code=400, detail=f"{runtime.model_id} is an embedding model and cannot chat; send text to POST /v1/embeddings")
    if embedding and not runs_embeddings:
        others = [name for name, kind in runtime.lanes.model_kinds().items() if kind["kind"] == "embedding"]
        hint = f"name one of these in model: {', '.join(others)}" if others else "load one with --embedding"
        raise HTTPException(status_code=400, detail=f"{runtime.model_id} is a chat model and gives no embeddings; {hint}")


def reply_detail(response: httpx.Response) -> Any:
    """What a worker said went wrong, which may not be JSON if something between us answered."""
    try:
        return response.json().get("detail")
    except (ValueError, AttributeError):
        return f"HTTP {response.status_code}"


def server_sent(payload: dict[str, Any] | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


async def event_stream(
    tokens: AsyncIterator[dict[str, Any]],
    envelope: dict[str, Any],
    to_choice: Any,
    first_delta: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Wrap generated text in OpenAI-style server-sent events."""

    def delta_event(delta: dict[str, Any]) -> str:
        return server_sent({**envelope, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

    calls = 0
    try:
        if first_delta:
            yield delta_event(first_delta)
        async for item in tokens:
            if "text" in item:
                yield server_sent({**envelope, "choices": [to_choice(item["text"], None)]})
                continue
            if "tool_call" in item:
                for delta in tool_deltas(item["tool_call"], calls):
                    yield delta_event(delta)
                calls += 1
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
                    "isharded": timing,
                }
            )
    except HTTPException as exc:
        # Headers are long gone, so the failure has to travel in the stream.
        yield server_sent({"error": {"message": str(exc.detail), "code": exc.status_code}})
    yield server_sent("[DONE]")


@app.post("/v1/embeddings")
async def embeddings(
    body: EmbeddingRequest,
    authorization: str | None = Header(default=None),
) -> Any:
    caller = verify_user(authorization)
    lane = await runtime.lanes.locate(body.model)
    if lane:
        return await runtime.lanes.forward(lane, "/v1/embeddings", body.model_dump(exclude_none=True), caller)
    await check_request(body.model, embedding=True)
    assert runtime.active is not None
    adapter = runtime.active.adapter
    assert isinstance(adapter, GgufAdapter)
    texts = [body.input] if isinstance(body.input, str) else body.input
    names = (runtime.model_id, adapter.header.get("general.basename"), adapter.header.get("general.name"))
    problem = inputs_problem(texts) or dimensions_problem(body.dimensions, adapter.output_width, smallest_dimensions(filter(None, names)))
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    vectors, tokens = await runtime.embed(texts, body.dimensions)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": encoded(vector, body.encoding_format)}
            for index, vector in enumerate(vectors)
        ],
        "model": runtime.model_id,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


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


def render_prompts(
    tokenizer: Any, messages: list[dict[str, Any]], thinking: bool | None, tools: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """The prompt, and the settled history inside it, through the model's own template."""
    options = {"tools": tools} if tools else {}
    try:
        return (
            tokenizer.render_chat(messages, thinking, **options),
            tokenizer.render_chat(messages, thinking, add_generation_prompt=False, **options),
        )
    except (TemplateError, TypeError) as exc:
        raise ToolRequestError(f"this conversation does not fit the model's chat template: {exc}") from exc


async def scanned(tokens: AsyncIterator[dict[str, Any]], scanner: ReplyScanner) -> AsyncIterator[dict[str, Any]]:
    """A reply with its tool calls taken out of the text."""
    async with aclosing(tokens) as source:
        async for item in source:
            if "text" in item:
                for found in scanner.feed(item["text"]):
                    yield found
                continue
            for found in scanner.finish(item["finish_reason"] == "length"):
                yield found
            yield {**item, "finish_reason": "tool_calls"} if scanner.calls else item


async def whole_reply(items: AsyncIterator[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The assistant message a reply comes to, and its summary."""
    pieces: list[str] = []
    calls: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    async for item in items:
        if "text" in item:
            pieces.append(item["text"])
        elif "tool_call" in item:
            calls.append(item["tool_call"])
        else:
            summary = item
    text = "".join(pieces)
    if calls:
        return {"role": "assistant", "content": text or None, "tool_calls": calls}, summary
    return {"role": "assistant", "content": text}, summary


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
    assert runtime.tokenizer is not None
    try:
        plain, images = chat_images([message.model_dump() for message in body.messages])
    except ImageRefused as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if images and not (runtime.active and runtime.active.projector):
        raise HTTPException(status_code=400, detail=f"{runtime.model_id} has no image projector, so it cannot see images; load it with --mmproj FILE")
    try:
        plan = plan_tools(body.tools, body.tool_choice, body.parallel_tool_calls, runtime.tokenizer.tool_template)
        messages = template_messages([message.model_copy(update={"content": item["content"]}) for message, item in zip(body.messages, plain)])
        # A forced call starts straight away. Told nothing, Qwen3 took one for its
        # reasoning, closed that, and wrote the call a second time.
        thinking = False if plan and plan.forced is not None else body.enable_thinking
        prompt, history = await asyncio.to_thread(render_prompts, runtime.tokenizer, messages, thinking, plan.tools if plan else None)
    except ToolRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    limit = reply_limit(body.max_tokens, body.max_completion_tokens, uses_tools(body.tools, body.messages))
    scanner = None
    if plan:
        opening = forced_opening(plan)
        in_thought = opens_in_thought(prompt)
        # Some templates open reasoning whatever they are told.
        prompt += ("\n</think>\n\n" if opening and in_thought else "") + opening
        scanner = ReplyScanner(plan, in_thought and not opening, opening)
    options = (limit, body.temperature, body.top_p, body.top_k, body.seed, body.stop, history, images)
    items = runtime.stream_tokens(prompt, *options, keep_markers=scanner is not None)
    if scanner:
        items = scanned(items, scanner)
    identifier = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if body.stream:
        envelope = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": runtime.model_id}
        return StreamingResponse(
            event_stream(
                items,
                envelope,
                lambda text, reason: {
                    "index": 0,
                    "delta": {"content": text} if text else {},
                    "finish_reason": reason,
                },
                {"role": "assistant"},
            ),
            media_type="text/event-stream",
        )
    message, summary = await whole_reply(items)
    return {
        "id": identifier,
        "object": "chat.completion",
        "created": created,
        "model": runtime.model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": summary["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": summary["prompt_tokens"],
            "completion_tokens": summary["completion_tokens"],
            "total_tokens": summary["prompt_tokens"] + summary["completion_tokens"],
        },
    }


def origin_is_ours(origin: str | None) -> bool:
    """Whether a browser page may reach /mcp. No Origin means no browser.

    Matching the Host header would prove nothing: under DNS rebinding the
    attacker's page and its Host are the same name.
    """
    if origin is None:
        return True
    name = urlsplit(origin).hostname
    return name in LOOPBACK or (runtime.mesh is not None and origin.rstrip("/") in runtime.mesh.known_urls())


def own_api(request: Request, authorization: str | None) -> httpx.AsyncClient:
    """This node's own API, as the caller reaches it.

    Going back in at the top means MCP inherits everything /v1 does for lanes,
    followers, keys, limits, and the audit log, and cannot drift from it.
    """
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False, client=(request.client.host, request.client.port) if request.client else ("unknown", 0))
    headers = {"Authorization": authorization} if authorization else {}
    return httpx.AsyncClient(transport=transport, base_url="http://this-node", headers=headers, timeout=None)


@app.post("/mcp")
async def mcp_endpoint(request: Request, authorization: str | None = Header(default=None)) -> Response:
    """MCP over Streamable HTTP. Stateless: each request stands alone, on any node."""
    if not origin_is_ours(request.headers.get("origin")):
        return JSONResponse(status_code=403, content=RpcError(INVALID_REQUEST, "This Origin may not use this endpoint").body(None))
    try:
        require(authorization, ROLE_USER)
    except HTTPException as exc:
        if exc.status_code == 401:
            raise HTTPException(status_code=401, detail=exc.detail, headers={"WWW-Authenticate": "Bearer"}) from exc
        raise

    def charge(method: str) -> None:
        # tools/call pays when it reaches /v1, which is enough.
        if method != "tools/call":
            verify_user(authorization)

    async with own_api(request, authorization) as client:
        status, reply = await answer_http(McpHandler(MeshTools(client), stateless=True), request.headers, await request.body(), charge)
    return Response(status_code=status) if reply is None else JSONResponse(status_code=status, content=reply)


@app.get("/mcp")
@app.delete("/mcp")
async def mcp_elsewhere() -> Response:
    """There is no stream to open and no session to end."""
    return Response(status_code=405, headers={"Allow": "POST"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the iSharded orchestrator")
    parser.add_argument("--host", default=os.getenv("ISHARDED_HOST", "0.0.0.0"), help="address to listen on")
    parser.add_argument("--port", type=int, default=int(os.getenv("ISHARDED_PORT", "8000")), help="port to listen on")
    parser.add_argument("--no-reuse-cached", action="store_true", help="plan as if no worker held any shards")
    parser.add_argument("--no-peer-fetch", action="store_true", help="always download from the model host")
    parser.add_argument("--state-file", default=os.getenv("ISHARDED_STATE_FILE"), help="where the running route is saved")
    parser.add_argument("--node-id", default=os.getenv("ISHARDED_NODE_ID"), help="this node's name in the mesh")
    parser.add_argument("--advertise-url", default=os.getenv("ISHARDED_NODE_URL"), help="how other nodes reach this one; joins the mesh")
    parser.add_argument("--join", default=os.getenv("ISHARDED_JOIN", ""), help="any existing nodes, comma separated")
    parser.add_argument("--keys-file", default=os.getenv("ISHARDED_KEYS_FILE"), help="where this node keeps the pool's API keys")
    parser.add_argument("--trusted-network", action="store_true", help="send pool credentials unencrypted, because nobody else can read this network")
    parser.add_argument("--open", action="store_true", help="serve a network with no credentials, trusting everyone on it")
    parser.add_argument("--audit-file", default=os.getenv("ISHARDED_AUDIT_FILE"), help="where to record who did what")
    parser.add_argument("--lane", default=None, help="run as this lane, beside a node's main control plane")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["ISHARDED_KEYS_FILE"] = args.keys_file or str(pool_home() / "access.json")
    os.environ["ISHARDED_NODE_ID"] = args.node_id or ""
    os.environ["ISHARDED_NODE_URL"] = args.advertise_url or ""
    os.environ["ISHARDED_JOIN"] = args.join
    os.environ["ISHARDED_LANE"] = args.lane or ""
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "iSharded"
    os.environ["ISHARDED_STATE_FILE"] = args.state_file or str(base / "orchestrator-state.json")
    os.environ["ISHARDED_AUDIT_FILE"] = args.audit_file or str(base / "audit.jsonl")
    if args.no_reuse_cached:
        os.environ["ISHARDED_REUSE_CACHED"] = "0"
    if args.no_peer_fetch:
        os.environ["ISHARDED_PEER_FETCH"] = "0"
    check_exposure(args.host, load_token(), os.getenv("ISHARDED_CA_FILE"), args.open, args.advertise_url, args.trusted_network)
    global runtime
    runtime = Orchestrator()
    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
