from __future__ import annotations

import fnmatch

import base64
import json
import re
import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Annotated, Any, Literal
import uuid

import torch
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, HttpUrl
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


META_HEADER = "x-isharded-meta"
# Raised whenever workers and the orchestrator stop being able to talk safely.
# A node on another version is refused at the door, not found out mid-request.
PROTOCOL_VERSION = 4


class ResourceContribution(BaseModel):
    roles: list[str] = Field(default_factory=lambda: ["inference"])
    cpu_threads: int = 0
    ram_budget_bytes: int = 0
    accelerator_memory_budget_bytes: int = 0
    standby: bool = False
    # Patterns naming the only models this node will host. Empty means any.
    host_models: list[str] = Field(default_factory=list)


def host_model_patterns() -> list[str]:
    return [item.strip() for item in os.getenv("ISHARDED_HOST_MODELS", "").split(",") if item.strip()]


def hosts_model(patterns: list[str], model_id: str) -> bool:
    """Whether a node's owner agreed to host this model."""
    if not patterns:
        return True
    wanted = model_id.lower()
    return any(fnmatch.fnmatchcase(wanted, pattern.lower()) for pattern in patterns)


class MemoryPool(BaseModel):
    """One kind of memory a worker can hold model weights in.

    Pools differ per node: a laptop may offer VRAM + RAM + NVMe, a Mac one
    unified pool, a server several GPUs. ``capacity_group`` names the physical
    resource behind the pool, so that two workers on one host (or a GPU and the
    CPU on unified memory) cannot both spend the same bytes.
    """

    pool_id: str
    kind: str  # vram | unified | ram | disk
    device: str  # torch device that executes weights held in this pool
    label: str = ""
    capacity_group: str
    total_bytes: int = 0
    budget_bytes: int = 0
    free_bytes: int = 0
    bandwidth_gbps: float = 1.0
    dtype: str = "bfloat16"  # bfloat16 | float32 | float16 | checkpoint (disk)


class HardwareInfo(BaseModel):
    host_id: str = "unknown"
    device: str
    backend: str
    runtime_version: str | None = None
    supported_dtypes: list[str] = Field(default_factory=list)
    cpu_cores: int
    ram_total_bytes: int
    accelerator_name: str | None = None
    accelerator_memory_bytes: int = 0
    contribution: ResourceContribution = Field(default_factory=ResourceContribution)
    memory_pools: list[MemoryPool] = Field(default_factory=list)


class WorkerRegistration(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    advertised_url: HttpUrl
    hardware: HardwareInfo
    protocol_version: int = 0
    code_version: str = "unknown"


class Heartbeat(BaseModel):
    worker_id: str


class UnitPlacement(BaseModel):
    """Which memory pool holds one weight unit of a stage."""

    unit: str
    pool_id: str
    bytes: int = 0
    # Keep FP8 weights as stored and expand them on each use, instead of
    # expanding once at load. Half the memory, several times slower.
    compact: bool = False


class StageAssignment(BaseModel):
    worker_id: str
    url: str
    start_layer: int
    end_layer: int
    placements: list[UnitPlacement] = Field(default_factory=list)
    estimated_ms_per_token: float = 0.0


def _plain_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"not a plain name: {value[:80]!r}")
    return value


def _file_in_folder(value: str) -> str:
    """A file inside a model's folder. These become paths, and arrive from peers."""
    parts = re.split(r"[\\/]", value)
    if not value or ":" in value or "\0" in value or not parts[0] or ".." in parts:
        raise ValueError(f"not a file inside a model: {value[:80]!r}")
    return value


def _gguf_file(value: str) -> str:
    if not _file_in_folder(value).lower().endswith(".gguf"):
        raise ValueError(f"not a GGUF file: {value[:80]!r}")
    return value


# Far beyond any real pass, and far short of leaving a reply hung on a dead link.
PASS_SECONDS = 30.0
PASS_SECONDS_PER_TOKEN = 1.0

PlainName = Annotated[str, AfterValidator(_plain_name)]
ModelFile = Annotated[str, AfterValidator(_file_in_folder)]
GgufFile = Annotated[str, AfterValidator(_gguf_file)]
# torch and numpy refuse anything wider, and a refused seed must not look like a dead node.
Seed = Annotated[int, Field(ge=0, lt=2**63)]
# Below this, sampling overflows. Nobody can tell it from greedy anyway.
Temperature = Annotated[float, Field(ge=0.0, le=5.0), AfterValidator(lambda value: 0.0 if value < 1e-3 else value)]


class TensorRef(BaseModel):
    """Where one tensor's bytes live inside the upstream checkpoint."""

    name: str
    file: ModelFile
    dtype: str
    shape: list[int]
    start: int
    end: int


class WeightUnit(BaseModel):
    """Smallest piece of a model that is fetched and placed as a whole."""

    name: str
    kind: str  # embed | head | dense | experts
    layer: int | None = None
    tensors: list[TensorRef]
    # When set, what this unit's file must hash to, wherever it came from.
    sha256: str | None = None

    @property
    def nbytes(self) -> int:
        return sum(tensor.end - tensor.start for tensor in self.tensors)

    @property
    def raw(self) -> bool:
        """True for GGUF units: opaque bytes that llama.cpp reads as they are."""
        return bool(self.tensors) and all(tensor.dtype == "U8" for tensor in self.tensors)


class PrepareRequest(BaseModel):
    """Download this worker's allowlist of tensors. Touches disk only."""

    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    model_id: str
    source: str
    # Cache key, pinned when the manifest was read.
    revision: PlainName
    # Revision to ask the host for. Differs from ``revision`` only when the
    # manifest came from a local copy of a hosted model.
    source_revision: str
    units: list[WeightUnit]
    # Other workers already holding a unit, tried before the model host.
    peers: dict[str, list[str]] = Field(default_factory=dict)
    # Full sizes of source files, so a worker can recognize its own local copy.
    file_sizes: dict[str, int] = Field(default_factory=dict)
    # Workers holding the whole source file, for pools with no model host.
    file_sources: list[str] = Field(default_factory=list)


class DraftModel(BaseModel):
    """A small model the entry worker runs to guess the next tokens."""

    source: str
    file: GgufFile
    file_size: int
    revision: PlainName
    file_sources: list[str] = Field(default_factory=list)


class ConfigureRequest(BaseModel):
    """Move prepared units into their pools and join a pipeline epoch."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    revision: PlainName
    architecture: dict[str, Any]
    epoch: int
    assignment: StageAssignment
    route: list[StageAssignment]
    cache_url: str | None = None
    context_tokens: int = 4096
    # Guesses checked per pass, which every stage must be able to rewind.
    speculate_tokens: int = 0
    draft: DraftModel | None = None
    # Leadership generation. Workers refuse orders older than the newest they saw.
    term: int = 0
    slots: int = 1
    # Let conversations share a pass through each stage. See worker._can_merge.
    merge_passes: bool = True
    # What this stage's units must hash to, when the model was loaded with hashes.
    unit_hashes: dict[str, str] = Field(default_factory=dict)
    # Run the stage in llama.cpp's embedding mode, for /v1/embeddings.
    embedding: bool = False
    # The image projector, which only the entry worker loads.
    projector: DraftModel | None = None
    # Save and reuse prompt states. Off, prompts are cut the same way and read in full.
    prompt_cache: bool = True
    # The one other node that keeps a copy of this stage's states, and the nodes to
    # ask when taking over layers from a stage that was lost. See statecopy.py.
    copy_url: str | None = None
    state_peers: list[str] = Field(default_factory=list)


class CacheRestoreRequest(BaseModel):
    session_id: str
    source_epoch: int
    target_epoch: int
    cache_url: str
    # Tokens the session has committed. Snapshots are sent in the background, so
    # a store can be behind: a shorter cache must be refused, not resumed from.
    expected_length: int


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = "distilbert/distilgpt2"
    # Load beside what is running, in this lane, rather than replacing it.
    lane: str | None = None
    lane_gpu_gb: float | None = None
    lane_ram_gb: float | None = None
    # Keep this lane off every node another route uses, so routes can check each other.
    lane_apart: bool = False
    control_model_id: str | None = None
    worker_model_ids: dict[str, str] = Field(default_factory=dict)
    revision: str = "main"
    # spread: use every preferred worker. compact: fewest stages that fit.
    placement: Literal["spread", "compact"] = "spread"
    context_tokens: int = Field(default=4096, ge=128, le=1_048_576)
    # Which machines hold the two ends. Left unset, the planner picks the fastest
    # order. The entry sees prompt tokens and the exit sees generated tokens.
    entry_worker: str | None = None
    exit_worker: str | None = None
    # None uses the orchestrator's default. reuse_cached weighs shards workers
    # already hold when planning; peer_fetch copies shards between workers.
    reuse_cached: bool | None = None
    peer_fetch: bool | None = None
    # A small GGUF that guesses tokens for the big one to check in one pass.
    # "auto" picks one for the model's family, and "none" turns this off.
    draft_model: str = "auto"
    draft_tokens: int = Field(default=6, ge=1, le=16)  # the most guesses per pass
    # Replies the pipeline runs at once. Each costs its own context memory.
    parallel: int = Field(default=1, ge=1, le=32)
    merge_passes: bool = True
    # Reuse saved prompt states. Replies are the same either way at temperature 0.
    prompt_cache: bool = True
    # From `isharded model hash`, run on a copy you trust. Every node then checks
    # every unit against it, so no peer or mirror has to be believed.
    model_hashes: dict[str, str] | None = None
    # An embedding model: one stage, answering /v1/embeddings and not chat.
    embedding: bool = False
    # A projector GGUF for images, hosted on the entry stage.
    mmproj: str | None = None
    # False returns as soon as the job starts; poll /admin/status for progress.
    wait: bool = True


class SavePoint(BaseModel):
    """Where in a pass to save a sequence's state, and what to call it."""

    at: int  # rows into this pass
    length: int  # rows of the prompt the state holds
    key: str
    kind: str
    # A state this one makes useless, an earlier turn of the same conversation.
    replaces: str | None = None


class InferRequest(BaseModel):
    session_id: str
    epoch: int
    input_ids: list[int]
    reset: bool = False
    temperature: Temperature = 0.0
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: Seed | None = None
    # Speculative pass: forget positions from here on, then return the model's
    # choice after every input token instead of one sampled token.
    rollback_to: int | None = None
    verify: bool = False
    # Reuse: restore the state named by ``checkpoint_key``, which leaves the sequence
    # at ``restore_length``. The key is a digest of the tokens, so two conversations
    # of the same length can never be confused. ``reused_rows`` and ``restore_kind``
    # are only for each stage's counters.
    restore_length: int | None = None
    checkpoint_key: str | None = None
    reused_rows: int = 0
    restore_kind: str | None = None
    # Which of a stage's sequences this reply uses, given out by the entry worker.
    slot: int = 0
    # This pass is rows of a prompt, which cross between stages the same way however
    # many of them a restored state left to run.
    prompt_rows: bool = False
    # Rows into this pass where every stage must stop and start a new piece. See
    # statestore.py for why these must not depend on what is cached.
    cuts: list[int] = Field(default_factory=list)
    saves: list[SavePoint] = Field(default_factory=list)
    # Images in this prompt, in base64, for the entry worker alone. Their rows are in
    # ``input_ids`` as negative numbers. See vision.py.
    images: list[str] = Field(default_factory=list)
    # Where the sequence stands after this pass, for a pass holding image rows.
    next_position: int | None = None


class GenerateRequest(InferRequest):
    """Generate on the entry worker, so that tokens need no orchestrator round trip.

    ``input_ids`` is always the whole prompt, ``checkpoint_at`` is where its settled
    history ends, and ``cut_token`` is the token that opens a turn, which says where
    to cut it. The entry worker works out what it can reuse by itself, so it does not
    matter which node's control plane is asking.
    """

    max_tokens: int = Field(default=32, ge=1)
    stop_token_ids: list[int] = Field(default_factory=list)
    checkpoint_at: int | None = None
    cut_token: int | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(default=32, ge=1, le=32768)
    temperature: Temperature = 0.0
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: Seed | None = None
    stream: bool = False
    stop: str | list[str] | None = None


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None  # OpenAI clients send this. Nothing here uses it.


class EmbedPass(BaseModel):
    """Inputs for one embedding pass, tokenized and packed by the control plane."""

    epoch: int
    inputs: list[list[int]]


class ImagesMeasure(BaseModel):
    """Ask the entry worker what images will cost, before they go in a prompt."""

    epoch: int
    images: list[str]


class ChatMessage(BaseModel):
    role: str
    # A string, or a list of text and image parts. An assistant message that only calls tools has none.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    # Left unset, the limit depends on whether tools are in play: see toolcalls.reply_limit.
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: Temperature = 0.0
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: Seed | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    # Passed to chat templates that support a thinking switch (Qwen3 onward).
    enable_thinking: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


def pass_seconds(tokens: int, hops: int = 1) -> float:
    """How long a pass may take before the stage after this one counts as silent.

    A lost link often gives no error at all: packets just stop. Each hop waits
    longer than the next one in, so the failure is blamed on the right stage.
    """
    return (PASS_SECONDS + tokens * PASS_SECONDS_PER_TOKEN) * max(hops, 1)


def auth_headers(token: str | None) -> dict[str, str]:
    # With a pool authority, nodes prove themselves by certificate or ticket. The
    # token stays home, since any address gossip mentions would otherwise receive it.
    if not token or os.getenv("ISHARDED_CA_FILE"):
        return {}
    return {"Authorization": f"Bearer {token}"}


def encode_meta(meta: dict[str, Any]) -> str:
    raw = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_meta(value: str) -> dict[str, Any]:
    return json.loads(base64.urlsafe_b64decode(value.encode("ascii")))


def encode_activation(tensor: torch.Tensor, places: torch.Tensor | None = None) -> bytes:
    # Safetensors avoids accepting executable pickle data from another node.
    tensors = {"hidden": tensor.detach().contiguous().cpu()}
    if places is not None:
        tensors["places"] = places.contiguous()
    return save_safetensors(tensors)


def decode_activation(payload: bytes) -> torch.Tensor:
    tensors = load_safetensors(payload)
    if set(tensors) != {"hidden"}:
        raise ValueError("activation payload must contain exactly one 'hidden' tensor")
    return tensors["hidden"]


def decode_pass(payload: bytes) -> tuple[torch.Tensor, torch.Tensor | None]:
    """A stage's input, with the positions of a pass holding image rows.

    Every stage applies M-RoPE in its own attention layers, so positions have to
    travel with the rows. In the body, because a header could not hold many images.
    """
    tensors = load_safetensors(payload)
    if not {"hidden"} <= set(tensors) <= {"hidden", "places"}:
        raise ValueError("activation payload must hold a 'hidden' tensor and at most 'places' besides")
    return tensors["hidden"], tensors.get("places")


def encode_tensors(tensors: dict[str, torch.Tensor]) -> bytes:
    return save_safetensors(
        {name: tensor.detach().contiguous().cpu() for name, tensor in tensors.items()}
    )


def decode_tensors(payload: bytes) -> dict[str, torch.Tensor]:
    return load_safetensors(payload)


@dataclass(slots=True)
class SessionState:
    tokens: list[int]
    epoch: int
    needs_reset: bool = True
    session_id: str = dataclass_field(default_factory=lambda: uuid.uuid4().hex)
    images: list[str] = dataclass_field(default_factory=list)

