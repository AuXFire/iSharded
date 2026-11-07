"""Decide which worker runs which layers, and which memory holds each unit.

A token passes through the layers in order, so a layer's cost is the time to
read its active weights from wherever they sit. Two consequences shape the
plan. Fast memory should hold whatever is read most: every token reads a
layer's dense weights, but only ``top_k / num_experts`` of its experts, so
experts are the first thing to move to slower memory. And a worker's layers
must be contiguous, so that activations cross the network once per stage.

Layer boundaries come from a dynamic program over contiguous ranges. Within a
stage, units then fill the worker's pools fastest first.

VRAM that nodes contribute comes first, ahead of speed and of what is already
downloaded: weights go to RAM only once no worker has room left for another
layer. A stage can be larger than its card for that reason, because its layers
stay whole for llama.cpp while its routed experts sit in RAM. A model that can
only be cut at coarse boundaries needs this most, since a card then holds whole
blocks, so those models get a search over every worker order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from itertools import pairwise, permutations
from typing import Any

from .gguf import holds_both_kinds
from .protocol import MemoryPool, StageAssignment, StageReads, UnitPlacement, WeightUnit

# Cost of one activation hop, in the same units as the rest of the estimate, for
# two workers with no measured link. A stage waits for the next stage's reply, so
# a hop costs its full round trip: on a mesh across countries, 713 of 794 ms a token.
HOP_MS = 25.0
# What a link that failed its measurement counts as. A route avoids one if it can.
UNREACHABLE_LINK_MS = 10_000.0
# A route that needs an unreachable link loses to any that does not, even one that
# leaves VRAM open, so this outweighs every open layer a route could have.
DEAD_LINK_PENALTY = 1_000_000
# Up to this many workers, the shortest way through their links is worked out.
MAX_LINKED_WORKERS = 12
# On a ring a stage hands a pass forward without waiting, and the last stage sends
# the answer straight to the entry, so a token costs one trip around the loop: half
# of each round trip, plus this for the software at each hop. A placeholder until
# the native ring has been measured across a real mesh.
RING_HOP_MS = 1.0
# Up to this many workers, automatic placement tries every subset of them, and past
# that it grows one from each of the first seeds.
MAX_SUBSET_WORKERS = 12
AUTO_SEEDS = 200
# At each step a seed weighs this many of its nearest workers. Long loops cost more
# per step, so seeds share this much work between them, which bounds planning time.
AUTO_NEIGHBOURS = 48
AUTO_WORK = 1_000_000
# Subsets worth planning in full, since their estimates are only estimates.
AUTO_SUBSETS = 3
# Spare steps of room to ask for in turn. A subset can fit on paper and still not
# plan, since the two ends of a model go on top of its layers.
AUTO_SLACK = (0, 2, 6)
# Kept back on accelerators for activations and allocator overhead.
WORKSPACE_BYTES = 256 * 1024**2
WORKSPACE_SHARE = 0.1
# Reading the embedding touches one row per token, which is effectively free.
EMBEDDING_ACTIVE_FRACTION = 0.0
# Rate at which FP8 weights can be expanded per use, in GB of FP8 per second.
# PyTorch has no fused kernel for this, and neither RTX 30 nor RDNA3 GPUs do FP8
# natively, so this, not memory bandwidth, limits how fast compact units run.
# Measured on an RTX 3060 and a laptop CPU with 8 threads.
EXPAND_RATE_GPU_GBPS = 20.0
EXPAND_RATE_CPU_GBPS = 0.65
# Shares of a stage's FP8 units to expand at load, hottest first, when choosing
# between speed and capacity. Whatever is left stays compact.
EXPANSION_SHARES = (1.0, 0.75, 0.5, 0.25, 0.0)
# A download is a one-time cost and token time is paid forever, so a plan is
# charged its download time spread over this many tokens. Near-ties stay where
# the shards already are, while a real speedup still earns its download.
AMORTIZE_TOKENS = 20_000
# Used until a worker has shown its real rate.
DEFAULT_FETCH_BYTES_PER_SECOND = 40e6
PEER_FETCH_BYTES_PER_SECOND = 100e6
# Up to this many workers, every order through them is tried.
MAX_PERMUTED_WORKERS = 4
# The same for models with a block rule, which have few places to cut. The work
# more than doubles with each worker, since a hop depends on who came before. A
# loop's last hop depends on who came first as well, so it reaches fewer workers.
MAX_SEARCHED_WORKERS = 8
MAX_LOOP_SEARCHED_WORKERS = 6
# When a plan must use RAM and fast memory is as full either way, how much slower
# the whole route may get to speed up its slowest stage. A judgment, not a
# measurement.
EVEN_STAGE_ALLOWANCE = 0.1
# Breaks ties toward even stages without outweighing real latency differences.
BALANCE_WEIGHT = 1e-4
# Whole-unit packing wastes some of what the aggregate estimate assumed.
PACKING_RETRIES = (1.0, 0.95, 0.9, 0.8, 0.65)
# Memory kinds in the order they are used. Weights below a worker's best are spill.
MEMORY_TIERS = {"vram": 0, "unified": 0, "ram": 1, "disk": 2}
FAST_MEMORY = {"vram", "unified"}
RAM_MEMORY = ("ram", "unified")
INFEASIBLE = float("inf")


class PlacementError(ValueError):
    """The model does not fit the pool, with the numbers that show why."""


@dataclass(slots=True)
class Candidate:
    worker_id: str
    url: str
    pools: list[MemoryPool] = field(default_factory=list)
    # Units on this worker's disk, and units another worker could copy to it.
    cached: frozenset[str] = frozenset()
    peer_units: frozenset[str] = frozenset()
    fetch_bytes_per_second: float = 0.0
    # Measured round trip in ms from this worker to others, by worker id.
    links: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelProfile:
    """What the planner needs to know about a model."""

    units: list[WeightUnit]
    num_layers: int
    expert_active_fraction: float
    # Per layer: bytes per token of context, and fixed per-session state.
    kv_bytes_per_token: list[int]
    state_bytes: list[int]
    context_tokens: int = 4096
    tied_embeddings: bool = False
    # Fast memory a stage needs beyond weights and context, for compute buffers.
    stage_workspace_bytes: int = 0
    # Stages may only start on a multiple of this.
    layer_step: int = 1
    # Every this many layers is full attention, and each stage needs one of those and a recurrent one.
    attention_interval: int = 0
    # What the entry stage must also hold, such as a draft model and its cache.
    entry_extra_bytes: int = 0
    # Per layer, the RAM a stage keeps for rewinding a recurrent model. Empty if it keeps none.
    snapshot_bytes: list[int] = field(default_factory=list)
    # Tokens a plan has to earn back its download over. Zero ignores downloads.
    amortize_tokens: int = 0
    # Whether stages pass tokens around a closed loop, instead of each waiting for
    # the next stage's reply. It sets what a route's network time is.
    ring: bool = False

    def __post_init__(self) -> None:
        # The layer search asks for the same sizes thousands of times, and a
        # MoE unit can hold hundreds of tensors.
        self._sizes: dict[tuple[str, str, bool], int] = {}
        self._quantized: dict[str, bool] = {}
        self.dense = {unit.layer: unit for unit in self.units if unit.kind == "dense"}
        self.experts = {unit.layer: unit for unit in self.units if unit.kind == "experts"}
        self.embed = next(unit for unit in self.units if unit.kind == "embed")
        self.head = next(unit for unit in self.units if unit.kind == "head")

    @property
    def total_bytes(self) -> int:
        return sum(unit.nbytes for unit in self.units)

    def context_bytes(self, layer: int) -> int:
        return self.kv_bytes_per_token[layer] * self.context_tokens + self.state_bytes[layer]

    def layer_sizes(self) -> list[tuple[int, int, int]]:
        """Each layer's dense bytes, expert bytes, and context bytes."""
        return [
            (
                self.dense[layer].nbytes,
                self.experts[layer].nbytes if layer in self.experts else 0,
                self.context_bytes(layer),
            )
            for layer in range(self.num_layers)
        ]

    def size_in(self, unit: WeightUnit, pool: MemoryPool, compact: bool = False) -> int:
        key = (unit.name, pool.dtype, compact)
        if key not in self._sizes:
            self._sizes[key] = unit.nbytes if compact else stored_bytes(unit, pool)
        return self._sizes[key]

    def quantized(self, unit: WeightUnit) -> bool:
        if unit.name not in self._quantized:
            self._quantized[unit.name] = any(tensor.dtype == "F8_E4M3" for tensor in unit.tensors)
        return self._quantized[unit.name]


def stored_bytes(unit: WeightUnit, pool: MemoryPool) -> int:
    """Size of a unit once held in a pool.

    Pools convert to their dtype, so an FP32 pool stores a BF16 checkpoint at
    twice its download size, and a BF16 pool expands FP8 to twice its size. The
    disk tier maps the file as it is.
    """
    if pool.dtype == "checkpoint" or unit.raw:
        return unit.nbytes  # GGUF units stay quantized wherever they sit
    target = 4 if pool.dtype == "float32" else 2
    return sum(
        (tensor.end - tensor.start) * target // _source_itemsize(tensor.dtype)
        for tensor in unit.tensors
    )


def _source_itemsize(dtype: str) -> int:
    return {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}.get(dtype, 1)


@dataclass(slots=True)
class _Item:
    """One unit waiting for a pool."""

    unit: WeightUnit
    active: float  # share of its bytes one token reads
    extra_bytes: int = 0  # KV cache or recurrent state that must sit alongside
    # Held apart from the unit, so that a unit pushed to slower memory cannot take them with it.
    entry_bytes: int = 0
    snapshot_bytes: int = 0


def _speed(pool: MemoryPool) -> float:
    return max(pool.bandwidth_gbps, 0.001)


def _use_order(pool: MemoryPool) -> tuple[bool, float]:
    """Fast memory first, even if a probe reads RAM as faster, then by speed."""
    return pool.kind not in FAST_MEMORY, -_speed(pool)


def _stage_items(profile: ModelProfile, start: int, end: int) -> list[_Item]:
    """A stage's units, most frequently read first."""
    items: list[_Item] = []
    if end == profile.num_layers:
        items.append(_Item(profile.head, 1.0))
    items.extend(
        _Item(
            profile.dense[layer],
            1.0,
            profile.context_bytes(layer),
            profile.entry_extra_bytes if layer == 0 else 0,
            profile.snapshot_bytes[layer] if profile.snapshot_bytes else 0,
        )
        for layer in range(start, end)
    )
    items.extend(
        _Item(profile.experts[layer], profile.expert_active_fraction)
        for layer in range(start, end)
        if layer in profile.experts
    )
    tied_with_head = profile.tied_embeddings and end == profile.num_layers
    if start == 0 and not tied_with_head:
        items.append(_Item(profile.embed, EMBEDDING_ACTIVE_FRACTION))
    return items


def _read_ms(size: int, active: float, pool: MemoryPool, compact: bool) -> float:
    """Time for one token to use a unit held in a pool."""
    rate = _speed(pool)
    if compact:
        expand = EXPAND_RATE_CPU_GBPS if pool.device == "cpu" else EXPAND_RATE_GPU_GBPS
        rate = min(rate, expand)
    return size * active / (rate * 1e6)


def _state_bytes(item: _Item, pool: MemoryPool) -> int:
    """Context state a unit brings into a pool.

    It is sized for BF16. For a disk-mapped layer it lives in RAM, uncounted
    here, and is small beside the layer's weights.
    """
    if pool.kind == "disk":
        return 0
    return item.extra_bytes * (2 if pool.dtype == "float32" else 1)


def _room_left(items: list[_Item], ordered: list[MemoryPool], capacity: _Shares) -> dict[str, int] | None:
    """What a stage's units have to work with, or None if what it holds beside them does not fit.

    This comes first because VRAM comes first: a draft runs on the GPU, so the entry's weights
    spill before it does. Snapshots live in the stage's own process, which is to say in RAM. A
    worker offering none has no budget to charge them to, and a disk pool holds neither.
    """
    remaining = dict(capacity)
    entry = sum(item.entry_bytes for item in items)
    if entry:
        for group, size in capacity.set_aside.items():
            remaining[group] += size
    held = (
        (entry, [pool for pool in ordered if pool.kind != "disk"]),
        (sum(item.snapshot_bytes for item in items), [pool for pool in ordered if pool.kind in RAM_MEMORY]),
    )
    for size, pools in held:
        if size and pools:
            pool = next((pool for pool in pools if size <= remaining[pool.capacity_group]), None)
            if pool is None:
                return None
            remaining[pool.capacity_group] -= size
    return remaining


def _pack(
    profile: ModelProfile,
    items: list[_Item],
    pools: list[MemoryPool],
    capacity: _Shares,
    expandable: int,
    compact_first: bool,
) -> tuple[list[UnitPlacement], float] | None:
    """Place whole units, fastest pool first. Returns None if they do not fit.

    ``capacity`` is keyed by capacity group, because pools can share one: a GPU
    on unified memory spends the same bytes as the RAM pool beside it.
    ``expandable`` is how many FP8 units may expand, taken from the hottest.
    ``compact_first`` gives compact units first claim on fast memory: expanding
    FP8 on a GPU beats reading BF16 from RAM, so that order can win overall.
    """
    ordered = sorted(pools, key=_use_order)
    remaining = _room_left(items, ordered, capacity)
    if remaining is None:
        return None
    placements: list[UnitPlacement] = []
    milliseconds = 0.0
    queue: list[tuple[_Item, bool, bool]] = []
    quantized_seen = 0
    for item in items:
        quantized = profile.quantized(item.unit)
        queue.append((item, quantized, quantized and quantized_seen >= expandable))
        quantized_seen += quantized
    if compact_first:
        queue.sort(key=lambda entry: not entry[2])  # stable: heat order is kept
    for item, quantized, stays_compact in queue:
        for pool in ordered:
            # Mapped files cannot be converted, so disk always stays compact.
            compact = stays_compact or (quantized and pool.kind == "disk")
            size = profile.size_in(item.unit, pool, compact)
            needed = size + _state_bytes(item, pool)
            if needed <= remaining[pool.capacity_group]:
                remaining[pool.capacity_group] -= needed
                placements.append(
                    UnitPlacement(unit=item.unit.name, pool_id=pool.pool_id, bytes=size, compact=compact)
                )
                milliseconds += _read_ms(size, item.active, pool, compact)
                break
        else:
            return None
    return placements, milliseconds


def _fill(
    profile: ModelProfile,
    items: list[_Item],
    pools: list[MemoryPool],
    capacity: _Shares,
) -> tuple[list[UnitPlacement], float] | None:
    """Pack a stage, trading FP8 expansion against capacity for the best time.

    Expanded units run at full speed but take twice the memory, so expanding
    everything can push a cold unit onto disk. Packing is cheap, so the
    candidates are simply tried.
    """
    quantized = sum(profile.quantized(item.unit) for item in items)
    shares = EXPANSION_SHARES if quantized else (1.0,)
    attempts = [
        _pack(profile, items, pools, capacity, int(quantized * share), compact_first)
        for share in shares
        for compact_first in ((False, True) if quantized else (False,))
    ]
    packed = [attempt for attempt in attempts if attempt is not None]
    best = min(packed, key=lambda attempt: attempt[1]) if packed else None
    if best is not None and profile.layer_step > 1 and not quantized:
        return _spill_less(profile, items, pools, capacity, best)
    return best


def _fill_stage(
    profile: ModelProfile, candidate: Candidate, start: int, end: int, capacities: dict[str, dict[str, int]]
) -> tuple[list[UnitPlacement], float] | None:
    return _fill(profile, _stage_items(profile, start, end), candidate.pools, capacities[candidate.worker_id])


def _spill_less(
    profile: ModelProfile,
    items: list[_Item],
    pools: list[MemoryPool],
    capacity: _Shares,
    filled: tuple[list[UnitPlacement], float],
) -> tuple[list[UnitPlacement], float]:
    """Spill the fewest expert bytes that make a stage fit.

    Packing keeps experts in layer order until the fast pool is full, so it
    spills whichever came last. They are all read alike but differ in size, so
    the better choice is the smallest unit that covers what will not fit, or
    failing that the largest, and then the same again for what is left.
    """
    fastest = min(pools, key=_use_order)
    # The embedding is cold and leaves first, so it never pushes an expert out.
    wanted = [item for item in items if item.unit.kind != "embed"]
    excess = sum(profile.size_in(item.unit, fastest) + _state_bytes(item, fastest) for item in wanted)
    left = _room_left(items, sorted(pools, key=_use_order), capacity)
    assert left is not None  # this stage was packed, so it fits
    excess -= left[fastest.capacity_group]
    waiting = sorted(
        (item for item in items if item.unit.kind == "experts"), key=lambda item: profile.size_in(item.unit, fastest)
    )
    chosen: list[_Item] = []
    while excess > 0 and waiting:
        sizes = [profile.size_in(item.unit, fastest) for item in waiting]
        pick = next((index for index, size in enumerate(sizes) if size >= excess), len(waiting) - 1)
        chosen.append(waiting.pop(pick))
        excess -= sizes[pick]
    if excess > 0 or not chosen:
        return filled  # nothing has to leave, or dense weights must leave as well
    names = {item.unit.name for item in chosen}
    queue = [item for item in items if item.unit.name not in names] + chosen
    packed = _pack(profile, queue, pools, capacity, 0, False)
    return filled if packed is None else min(filled, packed, key=lambda attempt: attempt[1])


def _spilled_bytes(placements: list[UnitPlacement], pools: list[MemoryPool]) -> int:
    """Bytes held below the best kind of memory the worker offers."""
    best = min(MEMORY_TIERS[pool.kind] for pool in pools)
    below = {pool.pool_id for pool in pools if MEMORY_TIERS[pool.kind] > best}
    return sum(item.bytes for item in placements if item.pool_id in below)


def _download_ms(candidate: Candidate, profile: ModelProfile, start: int, end: int) -> float:
    """A stage's download time, spread across the tokens it must pay back over."""
    if not profile.amortize_tokens:
        return 0.0
    rate = candidate.fetch_bytes_per_second or DEFAULT_FETCH_BYTES_PER_SECOND
    peer_rate = max(rate, PEER_FETCH_BYTES_PER_SECOND)
    seconds = 0.0
    for item in _stage_items(profile, start, end):
        if item.unit.name not in candidate.cached:
            seconds += item.unit.nbytes / (peer_rate if item.unit.name in candidate.peer_units else rate)
    return seconds * 1000 / profile.amortize_tokens


def _weight_room(pool: MemoryPool, workspace: int) -> int:
    """What a pool can give to weights and context, limited by what is free right now.

    ``workspace`` is a stage's own working memory. It covers what the general
    allowance for accelerators is for, so a pool keeps back the larger of the
    two and not both. A 20-layer stage on an A40 measured 1.4 GiB over its weights
    in all. At 8192 tokens and 2 slots this holds back 1.375 GiB with its context.

    Whatever reads a pool's free space reads its size too, so a pool of no known size
    was never measured. Its 0 means nobody asked, where a measured 0 means it is full.
    """
    measured = pool.free_bytes or pool.total_bytes
    available = min(pool.budget_bytes, pool.free_bytes) if measured else pool.budget_bytes
    if pool.kind in FAST_MEMORY:
        available -= max(workspace, min(WORKSPACE_BYTES, int(available * WORKSPACE_SHARE)))
    return max(available, 0)


def _all_pools(candidates: list[Candidate]) -> list[MemoryPool]:
    return [pool for candidate in candidates for pool in candidate.pools]


def _group_rooms(pools: Iterable[MemoryPool], workspace: int = 0) -> dict[str, int]:
    """What each physical memory can give, going by the pool that offers it most."""
    rooms: dict[str, int] = {}
    for pool in pools:
        rooms[pool.capacity_group] = max(rooms.get(pool.capacity_group, 0), _weight_room(pool, workspace))
    return rooms


class _Shares(dict[str, int]):
    """A worker's share of each memory, and what only an entry stage may add to that."""

    def __init__(self, shares: dict[str, int], set_aside: dict[str, int]) -> None:
        super().__init__(shares)
        self.set_aside = set_aside


def _capacities(candidates: list[Candidate], shrink: float, workspace: int = 0, entry_bytes: int = 0) -> dict[str, _Shares]:
    """Give each worker its share of every capacity group it can reach.

    Workers on one host share RAM and disk. Each is planned independently, so
    a shared group is split evenly, which can never oversubscribe it.

    Only one stage holds ``entry_bytes``, so that comes off a shared group before the split
    and the entry gets it back. Split evenly with the rest, a draft larger than one share
    stopped a load on memory with room for all of it.
    """
    sharers: dict[str, set[str]] = {}
    for candidate in candidates:
        for pool in candidate.pools:
            sharers.setdefault(pool.capacity_group, set()).add(candidate.worker_id)
    limits = _group_rooms(_all_pools(candidates), workspace)
    aside: dict[str, int] = {}
    for candidate in candidates if entry_bytes else ():
        holders = sorted((pool for pool in candidate.pools if pool.kind != "disk"), key=_use_order)
        holder = next((pool.capacity_group for pool in holders if entry_bytes <= limits[pool.capacity_group]), None)
        if holder is not None and len(sharers[holder]) > 1:
            aside[holder] = entry_bytes
    return {
        candidate.worker_id: _Shares(
            {
                pool.capacity_group: int(
                    (limits[pool.capacity_group] - aside.get(pool.capacity_group, 0)) * shrink / len(sharers[pool.capacity_group])
                )
                for pool in candidate.pools
            },
            {pool.capacity_group: aside[pool.capacity_group] for pool in candidate.pools if pool.capacity_group in aside},
        )
        for candidate in candidates
    }


def link_ms(links: dict[tuple[str, str], float], source: str, target: str) -> float:
    """A hop's round trip, measured either way, or ``HOP_MS`` if nobody knows."""
    known = links.get((source, target), links.get((target, source)))
    return HOP_MS if known is None else known


def _hop_ms(left: Candidate, right: Candidate) -> float:
    """The round trip from one stage's worker to the next one's."""
    known = left.links.get(right.worker_id, right.links.get(left.worker_id))
    return HOP_MS if known is None else known


def _one_way_ms(round_trip: float, ring: bool) -> float:
    """What a token spends crossing one stage boundary forward."""
    return round_trip / 2 + RING_HOP_MS if ring else round_trip


def _closing_ms(round_trip: float, walked: float) -> float:
    """What the answer spends getting from the last stage back to the entry.

    A dead link here is not fatal: the answer walks back along the route, which
    costs the same again as the way out.
    """
    return walked if round_trip >= UNREACHABLE_LINK_MS else _one_way_ms(round_trip, True)


def _with_stage(total: float, cost: float, hop: float) -> float:
    """A route's total after one more stage. The square leans toward even stages."""
    return total + cost + BALANCE_WEIGHT * cost * cost + hop


def _network_ms(forward: list[float], closing: float, ring: bool) -> float:
    """A token's time on the network, from a route's round trips in ms."""
    walked = sum(_one_way_ms(round_trip, ring) for round_trip in forward)
    return walked + _closing_ms(closing, walked) if ring and forward else walked


def route_network_ms(route: list[StageAssignment], links: dict[tuple[str, str], float], ring: bool = False) -> float:
    """A planned route's network time per token, from measured round trips."""
    names = [stage.worker_id for stage in route]
    forward = [link_ms(links, left, right) for left, right in pairwise(names)]
    return _network_ms(forward, link_ms(links, names[-1], names[0]) if names else 0.0, ring)


class _Packer:
    """Packs stages for one set of capacities, for the searches to compare.

    A long model has thousands of ranges and most hold the same sizes. Packing
    them was most of planning: 6.5 s for 94 layers on 7 workers. So stages of GGUF
    units, whose size is the same in every pool, share a packing when their layers
    match. Its times and sizes hold for all of them, but its unit names are only
    right for the first, so a route is built by packing each stage afresh.
    """

    def __init__(self, profile: ModelProfile, capacities: dict[str, _Shares]) -> None:
        self.profile = profile
        self.capacities = capacities
        self.shared: dict[Any, tuple[list[UnitPlacement], float] | None] = {}
        self.shapes: list[int] | None = None
        if all(unit.raw for unit in profile.units):
            seen: dict[tuple[int, int, int], int] = {}
            self.shapes = [seen.setdefault(size, len(seen)) for size in profile.layer_sizes()]

    def pack(self, candidate: Candidate, start: int, end: int) -> tuple[list[UnitPlacement], float] | None:
        if not holds_both_kinds(start, end, self.profile.attention_interval):
            return None
        if self.shapes is None:
            return _fill_stage(self.profile, candidate, start, end, self.capacities)
        key = (candidate.worker_id, start == 0, end == self.profile.num_layers, tuple(self.shapes[start:end]))
        if key not in self.shared:
            self.shared[key] = _fill_stage(self.profile, candidate, start, end, self.capacities)
        return self.shared[key]


def _partition(
    candidates: list[Candidate],
    profile: ModelProfile,
    packer: _Packer,
    spread: bool,
    required: set[str],
    costs: dict[tuple[str, int, int], float],
) -> list[tuple[Candidate, int, int]] | None:
    """Choose contiguous layer ranges with a dynamic program.

    A token crosses the stages one after another, so total latency is what a
    user feels, and that is what both modes minimize. ``spread`` must use every
    candidate; otherwise workers outside ``required`` may be skipped, which
    favors fewer stages. ``costs`` is shared between orders, because a stage's
    cost depends on its worker and layers and not on who comes before it.
    """
    layers = profile.num_layers
    count = len(candidates)

    def table(fill: Any) -> list[list[Any]]:
        return [[fill] * (layers + 1) for _ in range(count + 1)]

    def stage_cost(index: int, start: int, end: int) -> float:
        candidate = candidates[index]
        key = (candidate.worker_id, start, end)
        if key not in costs:
            filled = packer.pack(candidate, start, end)
            costs[key] = INFEASIBLE if filled is None else filled[1] + _download_ms(candidate, profile, start, end)
        return costs[key]

    # best[i][j]: lowest cost covering layers [0, j) using candidates [0, i).
    best = table(INFEASIBLE)
    choice: list[list[int | None]] = table(None)
    # The workers that best path starts and ends on, since the next hop starts at one
    # and a loop closes at the other, and what its hops have cost so far.
    ending: list[list[Candidate | None]] = table(None)
    opening: list[list[Candidate | None]] = table(None)
    walked = table(0.0)
    best[0][0] = 0.0
    for index in range(1, count + 1):
        candidate = candidates[index - 1]
        skippable = not spread and candidate.worker_id not in required
        for end in range(layers + 1):
            if skippable and best[index - 1][end] < best[index][end]:
                best[index][end] = best[index - 1][end]
                choice[index][end] = end  # this worker is skipped
                ending[index][end] = ending[index - 1][end]
                opening[index][end] = opening[index - 1][end]
                walked[index][end] = walked[index - 1][end]
            # A stage too long to pack stays too long if it grows, so with no RAM to
            # fall back on most ranges need not be packed at all. A hybrid stage can
            # also be too short, which says nothing about a longer one.
            earliest = 0
            for start in reversed(range(0, end, profile.layer_step)):
                allowed = holds_both_kinds(start, end, profile.attention_interval)
                if allowed and best[index - 1][start] != INFEASIBLE and stage_cost(index - 1, start, end) == INFEASIBLE:
                    earliest = start + profile.layer_step
                    break
            for start in range(earliest, end, profile.layer_step):
                previous = best[index - 1][start]
                if previous == INFEASIBLE:
                    continue
                cost = stage_cost(index - 1, start, end)
                if cost == INFEASIBLE:
                    continue
                before = ending[index - 1][start]
                hop = _one_way_ms(_hop_ms(before, candidate), profile.ring) if start and before else 0.0
                total = _with_stage(previous, cost, hop)
                first = opening[index - 1][start] or candidate
                gone = walked[index - 1][start] + hop
                if profile.ring and end == layers and first is not candidate:
                    total += _closing_ms(_hop_ms(candidate, first), gone)
                if total < best[index][end]:
                    best[index][end] = total
                    choice[index][end] = start
                    ending[index][end] = candidate
                    opening[index][end] = first
                    walked[index][end] = gone
    if best[count][layers] == INFEASIBLE:
        return None

    ranges: list[tuple[Candidate, int, int]] = []
    end = layers
    for index in range(count, 0, -1):
        start = choice[index][end]
        if start is None:
            return None
        if start != end:
            ranges.append((candidates[index - 1], start, end))
        end = start
    ranges.reverse()
    return ranges


# Layers of fast memory left open, bytes outside it, read time, and a stage's cost
# or a route's total. A route's score ends with what its hops have cost so far.
_Score = tuple[Any, ...]


def _extended(
    score: _Score, found: _Score, before: Candidate | None, candidate: Candidate, first: Candidate | None, ring: bool
) -> _Score:
    """A route's score after one more stage.

    ``before`` is the stage this one follows. ``first`` is the entry, given when this
    stage ends the route, since on a ring that is where the answer goes next. A dead
    forward link outranks everything. A dead closing link is only time.
    """
    open_layers, slow, read, cost = found
    round_trip = 0.0 if before is None else _hop_ms(before, candidate)
    hop = 0.0 if before is None else _one_way_ms(round_trip, ring)
    walked = score[4] + hop
    total = _with_stage(score[3], cost, hop)
    if ring and first is not None and first is not candidate:
        total += _closing_ms(_hop_ms(candidate, first), walked)
    dead = DEAD_LINK_PENALTY if round_trip >= UNREACHABLE_LINK_MS else 0
    return score[0] + open_layers + dead, score[1] + slow, max(score[2], read), total, walked


def _stage_scorer(profile: ModelProfile, packer: _Packer) -> Callable[[Candidate, int, int], _Score | None]:
    """Score stages by how much fast memory they leave open.

    What counts is whole steps a stage could still have taken, so a card that is
    full to within one layer, or one block, is full. Counting bytes instead would
    move dense weights onto a CPU to gain a few hundred MiB of VRAM. A worker
    without a stage leaves all its room open, so a stage scores what it fills:
    zero for a card it cannot use, and lower the more of a card it takes.
    """
    sizes = [sum(size) for size in profile.layer_sizes()]
    step = profile.layer_step
    largest = max(sum(sizes[start : start + step]) for start in range(0, profile.num_layers, step))
    units = {unit.name: unit for unit in profile.units}
    known: dict[tuple[str, int, int], _Score | None] = {}

    def fast_bytes(placements: list[UnitPlacement], fast: dict[str, MemoryPool]) -> int:
        """Weights and context in fast memory. A shared packing names other
        layers, but they are the same size as this stage's, context included."""
        used = 0
        for item in placements:
            pool = fast.get(item.pool_id)
            if pool is not None:
                used += item.bytes
                unit = units[item.unit]
                if unit.kind == "dense":
                    used += _state_bytes(_Item(unit, 1.0, profile.context_bytes(unit.layer)), pool)
                    used += profile.entry_extra_bytes if unit.layer == 0 else 0
        return used

    def stage(candidate: Candidate, start: int, end: int) -> _Score | None:
        key = (candidate.worker_id, start, end)
        if key not in known:
            filled = packer.pack(candidate, start, end)
            known[key] = None
            if filled is not None:
                placements, milliseconds = filled
                fast = {pool.pool_id: pool for pool in candidate.pools if pool.kind in FAST_MEMORY}
                shares = packer.capacities[candidate.worker_id]
                room = sum(shares[group] for group in {pool.capacity_group for pool in fast.values()})
                open_layers = max(room - fast_bytes(placements, fast), 0) // largest - room // largest
                slow = sum(item.bytes for item in placements if item.pool_id not in fast)
                cost = milliseconds + _download_ms(candidate, profile, start, end)
                known[key] = (open_layers, slow, milliseconds, cost)
        return known[key]

    return stage


def _cuts(profile: ModelProfile) -> list[int]:
    """Where a stage may start or end."""
    return [*range(0, profile.num_layers, profile.layer_step), profile.num_layers]


def _search_order(
    order: list[Candidate],
    profile: ModelProfile,
    stage: Callable[[Candidate, int, int], _Score | None],
    spread: bool,
    required: set[str],
    rank: Callable[[_Score], tuple[Any, ...]],
    slowest_allowed: float,
) -> tuple[_Score, list[tuple[Candidate, int, int]]] | None:
    """Choose stages through one order of workers, when every order is too many."""
    cuts = _cuts(profile)
    last = len(cuts) - 1
    # best[i][cut]: the best way to reach this cut with the first i workers.
    best: list[list[_Score | None]] = [[None] * len(cuts) for _ in range(len(order) + 1)]
    came: dict[tuple[int, int], int] = {}
    # The workers each best path starts and ends on, for the hops that follow.
    ending: dict[tuple[int, int], Candidate] = {}
    opening: dict[tuple[int, int], Candidate] = {}
    best[0][0] = (0, 0, 0.0, 0.0, 0.0)
    for index, candidate in enumerate(order):
        skippable = not spread and candidate.worker_id not in required
        for here in range(last + 1):
            score = best[index][here]
            if score is None:
                continue
            before = ending.get((index, here))
            first = opening.get((index, here), candidate)
            held = best[index + 1][here]
            if skippable and (held is None or rank(score) < rank(held)):
                best[index + 1][here] = score
                came[index + 1, here] = here
                if before is not None:
                    ending[index + 1, here] = before
                    opening[index + 1, here] = first
            for there in range(here + 1, last + 1):
                found = stage(candidate, cuts[here], cuts[there])
                if found is None or found[2] > slowest_allowed:
                    continue
                reached = _extended(score, found, before, candidate, first if there == last else None, profile.ring)
                held = best[index + 1][there]
                if held is None or rank(reached) < rank(held):
                    best[index + 1][there] = reached
                    came[index + 1, there] = here
                    ending[index + 1, there] = candidate
                    opening[index + 1, there] = first
    final = best[len(order)][last]
    if final is None:
        return None
    ranges: list[tuple[Candidate, int, int]] = []
    here = last
    for index in range(len(order), 0, -1):
        previous = came[index, here]
        if previous != here:
            ranges.append((order[index - 1], cuts[previous], cuts[here]))
        here = previous
    ranges.reverse()
    return final, ranges


def _search_blocks(
    candidates: list[Candidate],
    profile: ModelProfile,
    stage: Callable[[Candidate, int, int], _Score | None],
    spread: bool,
    entry: str | None,
    exit_: str | None,
    rank: Callable[[_Score], tuple[Any, ...]],
    slowest_allowed: float = INFEASIBLE,
) -> list[tuple[Candidate, int, int]] | None:
    """Choose every stage at once, for a model with few places to cut.

    A stage's cost does not depend on who comes before it, so walking over which
    workers are spent and how far they reach covers every order through them.
    Trying orders one by one could only sample them past a few workers. ``stage``
    gives a stage's score, and ``rank`` compares what they add up to.
    """
    cuts = _cuts(profile)
    last = len(cuts) - 1
    everyone = (1 << len(candidates)) - 1
    # best[spent, opening, ending, cut]: the best way to reach this cut with exactly
    # these workers, the last being ``ending``, since the next hop starts there. A
    # loop also closes at ``opening``. Without a loop that stays -1, costing nothing.
    best: dict[tuple[int, int, int, int], _Score] = {(0, -1, -1, 0): (0, 0, 0.0, 0.0, 0.0)}
    came: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    ring = profile.ring
    for spent in range(everyone + 1):  # adding a worker only ever raises this number
        for opening in range(-1, len(candidates) if ring else 0):
            for ending in range(-1, len(candidates)):
                for here in range(last):
                    position = (spent, opening, ending, here)
                    score = best.get(position)
                    if score is None:
                        continue
                    for index, candidate in enumerate(candidates):
                        if spent >> index & 1 or (entry and not spent and candidate.worker_id != entry):
                            continue
                        before = candidates[ending] if spent else None
                        first = (candidates[opening] if spent else candidate) if ring else None
                        for there in range(here + 1, last + 1):
                            if exit_ and (candidate.worker_id == exit_) != (there == last):
                                continue
                            found = stage(candidate, cuts[here], cuts[there])
                            if found is None or found[2] > slowest_allowed:
                                continue
                            reached = _extended(score, found, before, candidate, first if there == last else None, ring)
                            target = (spent | 1 << index, (opening if spent else index) if ring else -1, index, there)
                            held = best.get(target)
                            if held is None or rank(reached) < rank(held):
                                best[target] = reached
                                came[target] = position

    endings = [item for item in best if item[3] == last and (item[0] == everyone or not spread)]
    if not endings:
        return None
    position = min(endings, key=lambda item: rank(best[item]))
    ranges: list[tuple[Candidate, int, int]] = []
    while position in came:
        previous = came[position]
        ranges.append((candidates[position[2]], cuts[previous[3]], cuts[position[3]]))
        position = previous
    ranges.reverse()
    return ranges


def _slow_bytes(route: list[StageAssignment], candidates: list[Candidate]) -> int:
    """Bytes of a route held outside VRAM and unified memory."""
    fast = {
        (candidate.worker_id, pool.pool_id)
        for candidate in candidates
        for pool in candidate.pools
        if pool.kind in FAST_MEMORY
    }
    return sum(item.bytes for stage in route for item in stage.placements if (stage.worker_id, item.pool_id) not in fast)


def _fill_fast_memory(
    fastest: list[StageAssignment],
    search: Callable[..., list[StageAssignment] | None],
    candidates: list[Candidate],
    profile: ModelProfile,
) -> list[StageAssignment]:
    """Put a route's weights in RAM only when no worker has VRAM open.

    This comes before speed and downloads. A plan that needs no RAM stays as the
    fastest plan, which already weighs load time against speed. Otherwise open
    layers are compared first, and that same weighing decides between plans that
    leave the same room, so a layout that nodes hold stays unless it leaves a card
    with room for a layer. That only overrides speed if it takes weights out of
    RAM: a spread can force a stage onto a worker with nothing but RAM, however
    much VRAM is open. Then, as long as it costs little of the route's time and
    moves nothing more, comes less in RAM and an evener slowest stage.
    """
    slow = _slow_bytes(fastest, candidates)
    if not slow or not any(pool.kind in FAST_MEMORY for pool in _all_pools(candidates)):
        return fastest
    fullest = search(lambda score: (score[0], score[3])) or fastest
    if _slow_bytes(fullest, candidates) >= slow:
        fullest = fastest
    reads = [stage.estimated_ms_per_token for stage in fullest]
    leaner = search(lambda score: score, max(reads) * (1 + EVEN_STAGE_ALLOWANCE))
    if leaner is None:
        return fullest
    gain = (_slow_bytes(leaner, candidates), max(stage.estimated_ms_per_token for stage in leaner))
    if gain >= (_slow_bytes(fullest, candidates), max(reads)):
        return fullest
    slower = _route_cost(leaner, candidates, profile) - _route_cost(fullest, candidates, profile)
    moves_more = _moved_ms(leaner, candidates, profile) > _moved_ms(fullest, candidates, profile)
    hops_longer = _hops_ms(leaner, candidates, profile) > _hops_ms(fullest, candidates, profile)
    return fullest if moves_more or hops_longer or slower > sum(reads) * EVEN_STAGE_ALLOWANCE else leaner


def _best_speed(candidate: Candidate) -> float:
    return max((_speed(pool) for pool in candidate.pools), default=0.0)


def _held_order(candidates: list[Candidate], profile: ModelProfile) -> list[Candidate]:
    """Workers in the order of the layers they already hold, those with none last."""
    layer_of = {unit.name: unit.layer for unit in profile.units if unit.layer is not None}

    def place(candidate: Candidate) -> tuple[bool, int]:
        held = sorted(layer_of[name] for name in candidate.cached if name in layer_of)
        return not held, held[len(held) // 2] if held else 0

    return sorted(candidates, key=place)


def _shortest_order(
    first: list[Candidate], middle: list[Candidate], last: list[Candidate], ring: bool = False
) -> list[Candidate]:
    """``middle`` in the order with the fewest unreachable links, then the shortest hops.

    On a mesh across countries the hops are most of a token's time, so this is the
    order to start from, whatever the speeds say. Held-Karp: for each set of workers
    and the one it ends on, the best way to have visited them. On a ring the way home
    counts too, and any worker can start a loop, so one is taken as the start.
    """
    if ring and not first and len(middle) > 1:
        return [middle[0], *_shortest_order(middle[:1], middle[1:], last, True)]
    count = len(middle)

    def step(left: Candidate, right: Candidate) -> tuple[int, float]:
        hop = _hop_ms(left, right)
        return int(hop >= UNREACHABLE_LINK_MS), _one_way_ms(hop, ring)

    def add(cost: tuple[int, float], more: tuple[int, float]) -> tuple[int, float]:
        return cost[0] + more[0], cost[1] + more[1]

    best = {(1 << index, index): step(first[0], item) if first else (0, 0.0) for index, item in enumerate(middle)}
    came: dict[tuple[int, int], int] = {}
    for spent in range(1, 1 << count):
        for ending in range(count):
            cost = best.get((spent, ending))
            if cost is None:
                continue
            for index in range(count):
                if not spent >> index & 1:
                    reached = add(cost, step(middle[ending], middle[index]))
                    target = (spent | 1 << index, index)
                    if target not in best or reached < best[target]:
                        best[target] = reached
                        came[target] = ending
    everyone = (1 << count) - 1

    def finish(ending: int) -> tuple[int, float]:
        cost = best[everyone, ending]
        end = middle[ending]
        if last:
            cost = add(cost, step(end, last[0]))
            end = last[0]
        if ring and first and end is not first[0]:
            cost = cost[0], cost[1] + _closing_ms(_hop_ms(end, first[0]), cost[1])
        return cost

    ending = min(range(count), key=finish)
    order = [ending]
    spent = everyone
    while (spent, order[-1]) in came:
        previous = came[spent, order[-1]]
        spent ^= 1 << order[-1]
        order.append(previous)
    return [middle[index] for index in reversed(order)]


def _orders(
    candidates: list[Candidate],
    profile: ModelProfile,
    entry: str | None,
    exit_: str | None,
) -> list[list[Candidate]]:
    """Worker sequences worth planning, honoring any pinned ends.

    Direction matters because the ends differ: the output head is read in full
    for every token, while the embedding costs one row. A fixed order can hand
    the expensive end to the slowest machine. Past a few workers these are only a
    sample, and it follows measured speed, which shifts a little on every restart.
    So the order nodes already hold always comes first: without it, speeds that
    moved by 2% reordered seven workers and moved 74 to 116 GiB of 180.
    """
    first = [item for item in candidates if item.worker_id == entry]
    last = [item for item in candidates if item.worker_id == exit_ and item.worker_id != entry]
    middle = [item for item in candidates if item not in first and item not in last]
    if len(middle) <= MAX_PERMUTED_WORKERS:
        arrangements = [list(order) for order in permutations(middle)]
    else:
        # Too many to enumerate: try each worker at the exit, and both directions.
        arrangements = [middle, middle[::-1]]
        arrangements += [[*middle[:index], *middle[index + 1 :], middle[index]] for index in range(len(middle))]
        if len(middle) <= MAX_LINKED_WORKERS and any(item.links for item in candidates):
            shortest = _shortest_order(first, middle, last, profile.ring)
            if profile.ring and not first and not last:
                # Any turn of a loop costs the same to cross, but the ends of the
                # model differ, and speeds say nothing a measured loop does not.
                turns = [shortest[index:] + shortest[:index] for index in range(len(shortest))]
                arrangements = [*turns, *(turn[::-1] for turn in turns)]
            else:
                arrangements[:0] = [shortest, shortest[::-1]]
        if any(item.cached for item in middle):
            arrangements.insert(0, _held_order(middle, profile))
    return [[*first, *order, *last] for order in arrangements]


def _moved_ms(route: list[StageAssignment], candidates: list[Candidate], profile: ModelProfile) -> float:
    workers = {candidate.worker_id: candidate for candidate in candidates}
    return sum(_download_ms(workers[stage.worker_id], profile, stage.start_layer, stage.end_layer) for stage in route)


def _round_trips(route: list[StageAssignment], candidates: list[Candidate]) -> tuple[list[float], float]:
    """Each stage boundary of a route, and the link that closes it on a ring."""
    workers = {candidate.worker_id: candidate for candidate in candidates}
    chain = [workers[stage.worker_id] for stage in route]
    forward = [_hop_ms(left, right) for left, right in pairwise(chain)]
    return forward, _hop_ms(chain[-1], chain[0]) if forward else 0.0


def _hops_ms(route: list[StageAssignment], candidates: list[Candidate], profile: ModelProfile) -> float:
    """What a token spends crossing a route's stage boundaries, and coming home on a ring."""
    forward, closing = _round_trips(route, candidates)
    return _network_ms(forward, closing, profile.ring)


def _dead_links(route: list[StageAssignment], candidates: list[Candidate]) -> int:
    """Forward links that failed. A closing link that failed only costs time."""
    return sum(hop >= UNREACHABLE_LINK_MS for hop in _round_trips(route, candidates)[0])


def _route_cost(route: list[StageAssignment], candidates: list[Candidate], profile: ModelProfile) -> float:
    reads = sum(stage.estimated_ms_per_token for stage in route)
    return reads + _hops_ms(route, candidates, profile) + _moved_ms(route, candidates, profile)


def _sampled(
    candidates: list[Candidate],
    profile: ModelProfile,
    spread: bool,
    required: set[str],
    shrinks: tuple[float, ...],
    orders: list[list[Candidate]],
) -> tuple[list[StageAssignment] | None, _Packer]:
    """The fastest route through the orders worth trying, with the packer that found it."""
    for shrink in shrinks:
        packer = _Packer(profile, _capacities(candidates, shrink, profile.stage_workspace_bytes, profile.entry_extra_bytes))
        costs: dict[tuple[str, int, int], float] = {}
        best: list[StageAssignment] | None = None
        best_cost = INFEASIBLE
        for order in orders:
            ranges = _partition(order, profile, packer, spread, required, costs)
            if ranges is None:
                continue
            route = _build_route(ranges, profile, packer.capacities)
            if route is None:
                continue
            cost = _route_cost(route, candidates, profile)
            if best is None or cost < best_cost:
                best, best_cost = route, cost
        if best is not None:
            return best, packer
    return None, packer


def _attempt(
    candidates: list[Candidate],
    profile: ModelProfile,
    spread: bool,
    entry: str | None = None,
    exit_: str | None = None,
    orders: list[list[Candidate]] | None = None,
) -> list[StageAssignment] | None:
    """Plan one group of workers, in one placement mode.

    A model with a block rule has few places to cut, so its search covers every
    order. That search can find a way to use every worker where sampled orders
    find none, and such a plan puts a stage on whichever worker was left over,
    however slow. Up to 12.7 times slower was measured on random pools, so a spread
    still needs a sampled order to stand, and the plan otherwise falls back to
    skipping workers, as it always has past a few of them.
    """
    searched = MAX_LOOP_SEARCHED_WORKERS if profile.ring else MAX_SEARCHED_WORKERS
    every_order = profile.layer_step > 1 and len(candidates) <= searched
    shrinks = (1.0,) if every_order or orders else PACKING_RETRIES
    orders = orders or _orders(candidates, profile, entry, exit_)
    required = {name for name in (entry, exit_) if name}
    sampled, packer = _sampled(candidates, profile, spread, required, shrinks, orders)
    if sampled is None and (spread or not every_order):
        return None
    stage = _stage_scorer(profile, packer)

    def search(rank: Callable[[_Score], tuple[Any, ...]], slowest: float = INFEASIBLE) -> list[StageAssignment] | None:
        if every_order:
            ranges = _search_blocks(candidates, profile, stage, spread, entry, exit_, rank, slowest)
        else:
            found = (_search_order(order, profile, stage, spread, required, rank, slowest) for order in orders)
            best = min((item for item in found if item), key=lambda item: rank(item[0]), default=None)
            ranges = best[1] if best else None
        return None if ranges is None else _build_route(ranges, profile, packer.capacities)

    fastest = search(lambda score: (score[3], score[0], score[1], score[2])) if every_order else sampled
    return None if fastest is None else _fill_fast_memory(fastest, search, candidates, profile)


def _build_route(
    ranges: list[tuple[Candidate, int, int]],
    profile: ModelProfile,
    capacities: dict[str, dict[str, int]],
) -> list[StageAssignment] | None:
    route: list[StageAssignment] = []
    for candidate, start, end in ranges:
        filled = _fill_stage(profile, candidate, start, end, capacities)
        if filled is None:
            return None
        placements, milliseconds = filled
        if start == 0 and profile.tied_embeddings and end == profile.num_layers:
            # One worker holds both ends, so head and embedding are one tensor.
            head_pool = next(item.pool_id for item in placements if item.unit == profile.head.name)
            placements.append(UnitPlacement(unit=profile.embed.name, pool_id=head_pool, bytes=0))
        route.append(
            StageAssignment(
                worker_id=candidate.worker_id,
                url=candidate.url.rstrip("/"),
                start_layer=start,
                end_layer=end,
                placements=placements,
                estimated_ms_per_token=round(milliseconds, 2),
                read_bytes=_stage_reads(profile, start, end, placements),
            )
        )
    return route


def _stage_reads(profile: ModelProfile, start: int, end: int, placements: list[UnitPlacement]) -> StageReads:
    """What a stage reads per token, for anyone modelling a route from outside."""
    held = {item.unit: item.bytes for item in placements}
    layers = range(start, end)
    experts = sum(held[profile.experts[layer].name] for layer in layers if layer in profile.experts)
    return StageReads(
        dense=sum(held[profile.dense[layer].name] for layer in layers),
        experts=round(experts * profile.expert_active_fraction),
        embedding=held.get(profile.embed.name, 0),
        head=held.get(profile.head.name, 0),
    )


def with_links(group: list[Candidate], links: dict[tuple[str, str], float] | None) -> list[Candidate]:
    """Give each worker its measured links, going through the table once."""
    if not links:
        return group
    by_source: dict[str, dict[str, float]] = {}
    for (source, target), milliseconds in links.items():
        by_source.setdefault(source, {})[target] = milliseconds
    return [replace(item, links=by_source.get(item.worker_id, {})) for item in group]


def _fast_room(candidate: Candidate, workspace: int) -> int:
    """What a worker's VRAM can give to weights and context."""
    return sum(_group_rooms((pool for pool in candidate.pools if pool.kind in FAST_MEMORY), workspace).values())


def _loop_lengths(hops: list[list[float]], ring: bool) -> list[float]:
    """The shortest way through every subset of workers, by bitmask of its members.

    Held-Karp. A loop can start anywhere, so it starts at its lowest member, and
    with no loop to close an open path can start at any of them.
    """
    count = len(hops)
    shortest = [INFEASIBLE] * (1 << count)
    for start in range(count) if ring else (0,):
        table: list[dict[int, float] | None] = [None] * (1 << count)
        for seed in (start,) if ring else range(count):
            table[1 << seed] = {seed: 0.0}
        lowest = start + 1 if ring else 0
        for spent in range(1 << start, 1 << count):
            ends = table[spent]
            if ends is None:
                continue
            for ending, cost in ends.items():
                row = hops[ending]
                for index in range(lowest, count):
                    if not spent >> index & 1:
                        reached = cost + row[index]
                        target = table[spent | 1 << index]
                        if target is None:
                            table[spent | 1 << index] = {index: reached}
                        elif reached < target.get(index, INFEASIBLE):
                            target[index] = reached
            closed = (cost + (hops[ending][start] if ring and ending != start else 0.0) for ending, cost in ends.items())
            shortest[spent] = min(shortest[spent], *closed, INFEASIBLE)
    return shortest


def _grown_loops(
    hops: list[list[float]], rooms: list[int], steps: list[int], need: int, steps_needed: int, required: list[int]
) -> list[list[int]]:
    """Subsets of a pool too big to try them all: grow a loop from each seed.

    A seed takes on whichever nearby worker adds least to the loop for the room it
    brings, until the model fits, then gives back any worker it no longer needs.
    """
    count = len(hops)
    seeds = sorted(range(count), key=lambda index: -rooms[index])[:AUTO_SEEDS]
    # Small cards mean long loops, and each step costs more on a long loop, so fewer
    # seeds get the same time. The largest cards still go first.
    typical = sorted(rooms)[count // 2] or 1
    length = max(need // typical, 1)
    seeds = seeds[: max(8, min(len(seeds), AUTO_WORK // (AUTO_NEIGHBOURS * length * length + 1)))]
    found: dict[frozenset[int], list[int]] = {}
    for seed in seeds:
        loop = [seed, *(index for index in required if index != seed)]
        have, held = sum(rooms[index] for index in loop), sum(steps[index] for index in loop)
        # Nearest first. Only the nearest few are weighed at each step, but a model
        # that needs more workers than that can keep reaching further out.
        near = sorted(range(count), key=lambda index: hops[seed][index] + hops[index][seed])
        left = [index for index in near if index not in loop and steps[index]]
        while (have < need or held < steps_needed) and left:
            edges = list(pairwise([*loop, loop[0]]))
            best: tuple[float, int, int] | None = None
            for index in left[:AUTO_NEIGHBOURS]:
                row = hops[index]
                added, place = min((hops[a][index] + row[b] - hops[a][b], place) for place, (a, b) in enumerate(edges))
                score = (added + 1.0) / max(min(rooms[index], need - have), 1)
                if best is None or score < best[0]:
                    best = (score, index, place)
            assert best is not None
            _, index, place = best
            loop.insert(place + 1, index)
            left.remove(index)
            have += rooms[index]
            held += steps[index]
        if have < need or held < steps_needed:
            continue
        for index in sorted(loop, key=lambda item: rooms[item]):
            spare = have - rooms[index] >= need and held - steps[index] >= steps_needed
            if len(loop) > 1 and spare and index not in required:
                loop.remove(index)
                have -= rooms[index]
                held -= steps[index]
        found.setdefault(frozenset(loop), _untangled(loop, hops))
    return list(found.values())


def _untangled(loop: list[int], hops: list[list[float]]) -> list[int]:
    """Uncross a loop: reverse any stretch that makes it shorter (2-opt)."""
    size = len(loop)
    improved = True
    while improved:
        improved = False
        for left in range(size - 1):
            for right in range(left + 2, size):
                a, b, c, d = loop[left], loop[left + 1], loop[right], loop[(right + 1) % size]
                if a != d and hops[a][c] + hops[b][d] < hops[a][b] + hops[c][d] - 1e-9:
                    loop[left + 1 : right + 1] = reversed(loop[left + 1 : right + 1])
                    improved = True
    return loop


def _closest_subsets(
    group: list[Candidate], profile: ModelProfile, required: set[str], slack: int
) -> list[list[Candidate]]:
    """The subsets that could hold a model in VRAM, by estimated time per token.

    Every worker on a route costs a trip across its links, so the estimate is the
    shortest way through a subset plus reading the model from its fastest VRAM. That
    is what makes few, close workers win. These are only estimates, so the best few
    get planned in full, and ``slack`` asks for that many spare steps of room, for
    when one that only just fits turns out not to. Empty if nothing here can hold the
    whole model in VRAM.
    """
    workspace = profile.stage_workspace_bytes
    sizes = profile.layer_sizes()
    layers = sum(sum(size) for size in sizes)
    need = layers + profile.embed.nbytes + profile.head.nbytes + profile.entry_extra_bytes
    active = sum(size[0] + size[1] * profile.expert_active_fraction for size in sizes) + profile.head.nbytes
    active_share = active / max(need, 1)
    steps_needed = -(-profile.num_layers // profile.layer_step)
    step = max(layers // steps_needed, 1)
    steps_needed += slack
    fastest_first = sorted(group, key=lambda item: -_best_speed(item))
    workers = [item for item in fastest_first if _fast_room(item, workspace) >= step or item.worker_id in required]
    if not workers:
        return []
    rooms = [_fast_room(item, workspace) for item in workers]
    steps = [room // step for room in rooms]
    speeds = [_best_speed(item) for item in workers]
    hops = [
        [0.0 if left is right else _one_way_ms(_hop_ms(left, right), profile.ring) for right in workers] for left in workers
    ]
    indices = [index for index, item in enumerate(workers) if item.worker_id in required]

    def reads_ms(members: Iterable[int]) -> float:
        milliseconds, left = 0.0, need
        for index in members:  # fastest first, as packing would fill them
            taken = min(rooms[index], left)
            milliseconds += taken * active_share / (speeds[index] * 1e6)
            left -= taken
        return milliseconds

    def around(members: list[int]) -> float:
        total = sum(hops[left][right] for left, right in pairwise([*members, members[0]]))
        return total if profile.ring else total - max(hops[members[-1]][members[0]], 0.0)

    scored: list[tuple[float, int, list[int]]] = []
    if len(workers) <= MAX_SUBSET_WORKERS:
        lengths = _loop_lengths(hops, profile.ring)
        pinned = sum(1 << index for index in indices)
        for spent in range(1, 1 << len(workers)):
            if spent & pinned != pinned:
                continue
            members = [index for index in range(len(workers)) if spent >> index & 1]
            if sum(rooms[index] for index in members) >= need and sum(steps[index] for index in members) >= steps_needed:
                scored.append((reads_ms(members) + lengths[spent], len(members), members))
    else:
        # A long loop of small cards is the likeliest to fit only on paper, and trying
        # again is slow, so it leaves room for the two ends of the model from the start.
        ends = -(-(need - layers) // step) + 1
        for loop in _grown_loops(hops, rooms, steps, need, steps_needed + ends, indices):
            scored.append((reads_ms(sorted(loop)) + around(loop), len(loop), loop))  # kept in loop order
    scored.sort(key=lambda item: item[:2])
    return [[workers[index] for index in members] for _, _, members in scored[:AUTO_SUBSETS]]


def _plan_auto(
    group: list[Candidate], profile: ModelProfile, entry: str | None, exit_: str | None
) -> list[StageAssignment] | None:
    """The fewest, closest workers that hold a model in VRAM. The rest stay spare.

    The subset comes first, then its order. Every hop costs every token, so on a
    mesh across countries this is worth more than anything else a plan decides. VRAM
    still ranks first, so a plan that keeps weights in RAM only stands if no subset
    does better, and a layout nodes already hold is planned too, so that a move has
    to pay for itself in the same load time against speed as any other.
    """
    required = {name for name in (entry, exit_) if name}
    held = [item for item in group if item.cached or item.worker_id in required]

    def worth(route: list[StageAssignment]) -> tuple[int, int, float]:
        return _dead_links(route, group), _slow_bytes(route, group), _route_cost(route, group, profile)

    best: list[StageAssignment] | None = None
    tried: set[frozenset[str]] = set()
    for slack in AUTO_SLACK:
        subsets = _closest_subsets(group, profile, required, slack)
        if any(item.cached for item in held) and len(held) <= MAX_SUBSET_WORKERS:
            subsets.append(held)
        for subset in subsets:
            names = frozenset(item.worker_id for item in subset)
            if names in tried:
                continue
            tried.add(names)
            # A subset this long came out of growing a loop, so it is already in order.
            orders = [subset, subset[::-1]] if len(subset) > MAX_LINKED_WORKERS and not required else None
            route = _attempt(subset, profile, True, entry, exit_, orders)
            if route is None and orders is None:
                route = _attempt(subset, profile, False, entry, exit_)
            if route is not None and (best is None or worth(route) < worth(best)):
                best = route
            if orders and best is not None and not _slow_bytes(best, group):
                return best  # a loop this long is slow to plan, and it was ranked first for a reason
        if best is not None and not _slow_bytes(best, group):
            return best
    # Nothing holds it all in VRAM, so every worker that can help is in play, in the
    # same speed order as any other plan.
    room = profile.stage_workspace_bytes
    pins = [item for item in group if item.worker_id in required]
    others = [item for item in group if item.worker_id not in required]
    widest = sorted(others, key=lambda item: -_fast_room(item, room))[: MAX_SUBSET_WORKERS - len(pins)]
    chosen = [item for item in group if item in pins or item in widest]
    route = _attempt(chosen, profile, False, entry, exit_)
    return route if route is not None and (best is None or worth(route) < worth(best)) else best


def keep_layers(
    current: list[StageAssignment],
    preferred: list[Candidate],
    standby: list[Candidate],
    profile: ModelProfile,
) -> list[StageAssignment] | None:
    """The same stages, with a spare standing in for each worker that is gone.

    Planning from scratch can move every stage to get a slightly better split. That
    moves weights that could have stayed, and a saved state is only good for the
    layers it came from, so the lost stage's copy would be wasted too. Of the spares
    that fit, the one that adds least to the route takes over, counting what it would
    have to fetch. None if there are not enough spares, or one that fits.
    """
    everyone = {candidate.worker_id: candidate for candidate in [*preferred, *standby] if candidate.pools}
    staying = {stage.worker_id for stage in current if stage.worker_id in everyone}
    spares = [candidate for name, candidate in everyone.items() if name not in staying]

    def added(spare: Candidate, index: int, chosen: list[Candidate]) -> tuple[int, float]:
        """Dead links, then ms, that this spare brings to this place in the route."""
        before = chosen[-1] if chosen else None
        after = everyone.get(current[index + 1].worker_id) if index + 1 < len(current) else None
        stage = current[index]
        forward = [(left, right) for left, right in ((before, spare), (spare, after)) if left and right]
        dead = sum(_hop_ms(left, right) >= UNREACHABLE_LINK_MS for left, right in forward)
        milliseconds = sum(_one_way_ms(_hop_ms(left, right), profile.ring) for left, right in forward)
        if profile.ring and len(current) > 1:
            # The first and last stages are neighbours too, around the loop.
            if index == 0 and current[-1].worker_id in everyone:
                milliseconds += _one_way_ms(_hop_ms(everyone[current[-1].worker_id], spare), True)
            elif index == len(current) - 1 and chosen:
                milliseconds += _one_way_ms(_hop_ms(spare, chosen[0]), True)
        return dead, milliseconds + _download_ms(spare, profile, stage.start_layer, stage.end_layer)

    for shrink in PACKING_RETRIES:
        free = list(spares)
        ranges: list[tuple[Candidate, int, int]] = []
        for index, stage in enumerate(current):
            candidate = everyone.get(stage.worker_id)
            if candidate is None:
                items = _stage_items(profile, stage.start_layer, stage.end_layer)
                room = _capacities(free, shrink, profile.stage_workspace_bytes, profile.entry_extra_bytes)
                fitting = [spare for spare in free if _fill(profile, items, spare.pools, room[spare.worker_id])]
                if not fitting:
                    return None
                chosen = [item[0] for item in ranges]
                candidate = min(fitting, key=lambda spare: added(spare, index, chosen))
                free.remove(candidate)
            ranges.append((candidate, stage.start_layer, stage.end_layer))
        capacities = _capacities([candidate for candidate, _, _ in ranges], shrink, profile.stage_workspace_bytes, profile.entry_extra_bytes)
        route = _build_route(ranges, profile, capacities)
        if route is not None:
            return route
    return None


def _no_fit(everyone: list[Candidate], profile: ModelProfile, pinned: list[str]) -> PlacementError:
    """Say why nothing fits, with the numbers."""
    context = sum(profile.context_bytes(layer) for layer in range(profile.num_layers))
    # Report what the pools would really hold: FP32 pools double a BF16 checkpoint.
    pools = _all_pools(everyone)
    needed = min(sum(profile.size_in(unit, pool, profile.quantized(unit)) for unit in profile.units) for pool in pools)
    workspace = profile.stage_workspace_bytes
    offered = sum(_group_rooms(pools, workspace).values())
    hint = "remove a pin, " if pinned else ""
    extra = f" and {profile.entry_extra_bytes / 1024**3:.1f} GiB for a draft model or projector" if profile.entry_extra_bytes else ""
    snapshots = sum(profile.snapshot_bytes)
    extra += f" and {snapshots / 1024**3:.2f} GiB of RAM for snapshots" if snapshots else ""
    blocks = ""
    if needed + context + profile.entry_extra_bytes + snapshots <= offered:
        # Added up across workers and kinds of memory, the numbers above look like it fits.
        shares = _capacities(everyone, 1.0, workspace, profile.entry_extra_bytes)
        largest = max(sum(share.values()) + sum(share.set_aside.values()) for share in shares.values())
        blocks = (
            f". That is all the memory together, but a stage has to fit in its own worker, where the most "
            f"any one can use is {largest / 1024**3:.1f} GiB, and workers on one machine share what it has"
        )
    if profile.layer_step > 1:
        # Without this the numbers above can look like the model fits.
        size = needed / -(-profile.num_layers // profile.layer_step)
        blocks += (
            f". Stages of this model start on a multiple of {profile.layer_step} layers, so a worker takes whole "
            f"blocks of about {size / 1024**3:.1f} GiB, and room left on a card goes unused unless that worker "
            "also offers RAM for the part of a block that does not fit"
        )
    return PlacementError(
        f"This model needs {needed / 1024**3:.1f} GiB of weights plus "
        f"{context / 1024**3:.2f} GiB of context state{extra}, but the pool offers "
        f"{offered / 1024**3:.1f} GiB across {len(everyone)} worker(s); "
        f"{hint}add a worker, raise a RAM or disk budget, or lower the context length{blocks}"
    )


def plan_placement(
    preferred: list[Candidate],
    standby: list[Candidate],
    profile: ModelProfile,
    *,
    spread: bool | None = True,
    entry_worker: str | None = None,
    exit_worker: str | None = None,
    links: dict[tuple[str, str], float] | None = None,
    ring: bool | None = None,
) -> list[StageAssignment]:
    """Plan a route, adding standby workers only if the model needs them.

    ``spread`` uses every worker, False the fewest that fit, and None the fewest and
    closest that hold the model in VRAM. ``entry_worker`` and ``exit_worker`` pin which
    machines hold the two ends. A worker holds one stage, so pinning one worker to
    both ends means it runs the whole model. ``links`` holds measured round trips in
    ms, from one worker to another, and a hop that has none costs ``HOP_MS``. ``ring``
    costs a route as a closed loop, and otherwise it comes from the profile.
    """
    if ring is not None and ring != profile.ring:
        profile = replace(profile, ring=ring)
    preferred, standby = with_links(preferred, links), with_links(standby, links)
    usable = [candidate for candidate in preferred if candidate.pools]
    reserve = [candidate for candidate in standby if candidate.pools]
    if not usable and not reserve:
        raise PlacementError("no worker offers any memory for weights")
    everyone = usable + reserve
    pinned = [name for name in (entry_worker, exit_worker) if name]
    missing = [name for name in pinned if not any(item.worker_id == name for item in everyone)]
    if missing:
        raise PlacementError(f"{missing[0]} cannot be pinned: it is not an available worker with memory for weights")

    def by_speed(group: list[Candidate]) -> list[Candidate]:
        ordered = sorted(group, key=lambda item: (-_best_speed(item), item.worker_id))
        chosen = [item for item in ordered if item.worker_id in pinned]
        others = [item for item in ordered if item.worker_id not in pinned]
        return (chosen + others)[: profile.num_layers] if spread else chosen + others

    if entry_worker and entry_worker == exit_worker:
        attempts = [[item for item in everyone if item.worker_id == entry_worker]]
    else:
        pinned_standby = [item for item in reserve if item.worker_id in pinned]
        attempts = [by_speed(usable + pinned_standby)] if usable or pinned_standby else []
        if reserve:
            attempts.append(by_speed(everyone))

    def rank(route: list[StageAssignment]) -> tuple[int, int]:
        return _dead_links(route, everyone), _slow_bytes(route, everyone)

    # A standby worker with VRAM comes before RAM, like any other VRAM.
    standby_vram = any(pool.kind in FAST_MEMORY for pool in _all_pools(reserve))
    best: list[StageAssignment] | None = None
    for group in attempts:
        if spread is None:
            route = _plan_auto(group, profile, entry_worker, exit_worker)
        else:
            route = _attempt(group, profile, spread, entry_worker, exit_worker)
        if spread and (route is None or _dead_links(route, everyone)):
            # Spreading demands a layer on every worker, which a worker with little
            # memory, or one nobody can reach, can make impossible on its own.
            fewer = _attempt(group, profile, False, entry_worker, exit_worker)
            if fewer is not None and (route is None or rank(fewer) < rank(route)):
                route = fewer
        if route is not None and (best is None or rank(route) < rank(best)):
            best = route
        if best is not None and (not standby_vram or not any(rank(best))):
            return best
    if best is None:
        raise _no_fit(everyone, profile, pinned)
    return best


def memory_report(route: list[StageAssignment], candidates: list[Candidate], profile: ModelProfile) -> dict[str, Any]:
    """What a plan does with the memory it was offered.

    One row per physical memory, so RAM two workers share counts once. A pool's
    room is what weights and context could have taken, after working space and
    anything not free right now. Its unused bytes are what is left of that room,
    which on a worker with no stage is all of it. Spilled bytes are weights held
    below the best kind of memory their worker offers, and ``stages`` says how
    much of that each stage holds, for showing beside it.
    """
    rows: dict[str, dict[str, Any]] = {}
    room = _group_rooms(_all_pools(candidates), profile.stage_workspace_bytes)
    shares = _Shares(room, {})
    for candidate in candidates:
        for pool in candidate.pools:
            row = rows.setdefault(
                pool.capacity_group,
                {"workers": [], "pools": [], "kind": pool.kind, "offered_bytes": 0, "used_bytes": 0, "spilled_bytes": 0},
            )
            if candidate.worker_id not in row["workers"]:
                row["workers"].append(candidate.worker_id)
            if pool.pool_id not in row["pools"]:
                row["pools"].append(pool.pool_id)
            row["offered_bytes"] = max(row["offered_bytes"], pool.budget_bytes)

    workers = {candidate.worker_id: candidate for candidate in candidates}
    stages: list[dict[str, Any]] = []
    for stage in route:
        candidate = workers.get(stage.worker_id)
        if candidate is None or not candidate.pools:
            continue  # a worker that left while this route still runs
        by_id = {pool.pool_id: pool for pool in candidate.pools}
        best = min(MEMORY_TIERS[pool.kind] for pool in candidate.pools)
        placed = {item.unit: item for item in stage.placements}
        items = _stage_items(profile, stage.start_layer, stage.end_layer)
        for group, left in (_room_left(items, sorted(candidate.pools, key=_use_order), shares) or room).items():
            if group in rows:
                rows[group]["used_bytes"] += room[group] - left
        for item in items:
            placement = placed.get(item.unit.name)
            pool = by_id.get(placement.pool_id) if placement else None
            if placement is None or pool is None:
                continue  # status reads this on every poll, so a stale route must not break it
            row = rows[pool.capacity_group]
            row["used_bytes"] += placement.bytes + _state_bytes(item, pool)
            if MEMORY_TIERS[pool.kind] > best:
                row["spilled_bytes"] += placement.bytes
        stages.append(
            {
                "worker_id": stage.worker_id,
                "start_layer": stage.start_layer,
                "end_layer": stage.end_layer,
                "spilled_bytes": _spilled_bytes(stage.placements, candidate.pools),
            }
        )

    for group, row in rows.items():
        row["reserved_bytes"] = row["offered_bytes"] - room[group]
        row["unused_bytes"] = max(room[group] - row["used_bytes"], 0)
    return {
        "spilled_bytes": sum(row["spilled_bytes"] for row in rows.values()),
        "unused_bytes": sum(row["unused_bytes"] for row in rows.values()),
        "pools": list(rows.values()),
        "stages": stages,
    }
