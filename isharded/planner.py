"""Decide which worker runs which layers, and which memory holds each unit.

A token passes through the layers in order, so a layer's cost is the time to
read its active weights from wherever they sit. Two consequences shape the
plan. Fast memory should hold whatever is read most: every token reads a
layer's dense weights, but only ``top_k / num_experts`` of its experts, so
experts are the first thing to move to slower memory. And a worker's layers
must be contiguous, so that activations cross the network once per stage.

Layer boundaries come from a dynamic program over contiguous ranges. Within a
stage, units then fill the worker's pools fastest first.

A model that can only be cut at coarse boundaries strands memory: a card holds
whole blocks, and what is left of it holds nothing. Such a stage can still be
larger than its card, because its layers stay whole for llama.cpp while its
routed experts sit in RAM. So those models get a search over every worker order,
which keeps that spill small and off the slowest stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Callable

from .protocol import MemoryPool, StageAssignment, UnitPlacement, WeightUnit

# Cost of one activation hop, in the same units as the rest of the estimate.
# Measured round trip between two machines on Wi-Fi; a wired LAN is far lower.
HOP_MS = 25.0
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
# doubles with each worker: 10 took about 0.3 s, and 12 about a second.
MAX_SEARCHED_WORKERS = 10
# How much slower than the fastest plan the plan with the least spill may be, on
# the whole route and on its slowest stage. A judgment, not a measurement: the
# estimate counts a spilled expert's read but not the CPU's work on it or the
# trips between GPU and CPU, so it flatters spill and near ties should avoid it.
SPILL_ALLOWANCE = 0.1
# Breaks ties toward even stages without outweighing real latency differences.
BALANCE_WEIGHT = 1e-4
# Whole-unit packing wastes some of what the aggregate estimate assumed.
PACKING_RETRIES = (1.0, 0.95, 0.9, 0.8, 0.65)
# Memory kinds from fastest to slowest. Weights below a worker's best are spill.
MEMORY_TIERS = {"vram": 0, "unified": 0, "ram": 1, "disk": 2}
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
    # Stages may only start on a multiple of this, for hybrid attention patterns.
    layer_step: int = 1
    # What the entry stage must also hold, such as a draft model.
    entry_extra_bytes: int = 0
    # Tokens a plan has to earn back its download over. Zero ignores downloads.
    amortize_tokens: int = 0

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


def usable_bytes(pool: MemoryPool) -> int:
    """A pool's budget, limited by what is actually free right now."""
    available = min(pool.budget_bytes, pool.free_bytes or pool.budget_bytes)
    if pool.kind in {"vram", "unified"}:
        available -= min(WORKSPACE_BYTES, int(available * WORKSPACE_SHARE))
    return max(available, 0)


@dataclass(slots=True)
class _Item:
    """One unit waiting for a pool."""

    unit: WeightUnit
    active: float  # share of its bytes one token reads
    extra_bytes: int = 0  # KV cache or recurrent state that must sit alongside


def _speed(pool: MemoryPool) -> float:
    return max(pool.bandwidth_gbps, 0.001)


def _stage_items(profile: ModelProfile, start: int, end: int) -> list[_Item]:
    """A stage's units, most frequently read first."""
    items: list[_Item] = []
    if end == profile.num_layers:
        items.append(_Item(profile.head, 1.0))
    items.extend(
        _Item(profile.dense[layer], 1.0, profile.context_bytes(layer) + (profile.entry_extra_bytes if layer == 0 else 0))
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


def _pack(
    profile: ModelProfile,
    items: list[_Item],
    pools: list[MemoryPool],
    capacity: dict[str, int],
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
    remaining = dict(capacity)
    ordered = sorted(pools, key=_speed, reverse=True)
    placements: list[UnitPlacement] = []
    milliseconds = 0.0
    queue: list[tuple[_Item, bool]] = []
    quantized_seen = 0
    for item in items:
        quantized = profile.quantized(item.unit)
        queue.append((item, quantized and quantized_seen >= expandable))
        quantized_seen += quantized
    if compact_first:
        queue.sort(key=lambda entry: not entry[1])  # stable: heat order is kept
    for item, stays_compact in queue:
        quantized = profile.quantized(item.unit)
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
    capacity: dict[str, int],
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


def _spill_less(
    profile: ModelProfile,
    items: list[_Item],
    pools: list[MemoryPool],
    capacity: dict[str, int],
    filled: tuple[list[UnitPlacement], float],
) -> tuple[list[UnitPlacement], float]:
    """Spill the fewest expert bytes that make a stage fit.

    Packing keeps experts in layer order until the fast pool is full, so it
    spills whichever came last. They are all read alike but differ in size, so
    the better choice is the smallest unit that covers what will not fit, or
    failing that the largest, and then the same again for what is left.
    """
    fastest = max(pools, key=_speed)
    # The embedding is cold and leaves first, so it never pushes an expert out.
    wanted = [item for item in items if item.unit.kind != "embed"]
    excess = sum(profile.size_in(item.unit, fastest) + _state_bytes(item, fastest) for item in wanted)
    excess -= capacity[fastest.capacity_group]
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
    seconds = 0.0
    for item in _stage_items(profile, start, end):
        if item.unit.name in candidate.cached:
            continue
        rate = candidate.fetch_bytes_per_second or DEFAULT_FETCH_BYTES_PER_SECOND
        if item.unit.name in candidate.peer_units:
            rate = max(rate, PEER_FETCH_BYTES_PER_SECOND)
        seconds += item.unit.nbytes / rate
    return seconds * 1000 / profile.amortize_tokens


def _weight_room(pool: MemoryPool, workspace: int) -> int:
    """What a pool can give to weights and context."""
    available = usable_bytes(pool)
    if pool.kind in {"vram", "unified"}:
        available = max(available - workspace, 0)
    return available


def _capacities(candidates: list[Candidate], shrink: float, workspace: int = 0) -> dict[str, dict[str, int]]:
    """Give each worker its share of every capacity group it can reach.

    Workers on one host share RAM and disk. Each is planned independently, so
    a shared group is split evenly, which can never oversubscribe it.
    """
    sharers: dict[str, set[str]] = {}
    limits: dict[str, int] = {}
    for candidate in candidates:
        for pool in candidate.pools:
            sharers.setdefault(pool.capacity_group, set()).add(candidate.worker_id)
            limits[pool.capacity_group] = max(limits.get(pool.capacity_group, 0), _weight_room(pool, workspace))
    return {
        candidate.worker_id: {
            pool.capacity_group: int(
                limits[pool.capacity_group] * shrink / len(sharers[pool.capacity_group])
            )
            for pool in candidate.pools
        }
        for candidate in candidates
    }


def _partition(
    candidates: list[Candidate],
    profile: ModelProfile,
    capacities: dict[str, dict[str, int]],
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

    def stage_cost(index: int, start: int, end: int) -> float:
        candidate = candidates[index]
        key = (candidate.worker_id, start, end)
        if key not in costs:
            filled = _fill(
                profile,
                _stage_items(profile, start, end),
                candidate.pools,
                capacities[candidate.worker_id],
            )
            costs[key] = INFEASIBLE if filled is None else filled[1] + _download_ms(candidate, profile, start, end)
        return costs[key]

    # best[i][j]: lowest cost covering layers [0, j) using candidates [0, i).
    best = [[INFEASIBLE] * (layers + 1) for _ in range(count + 1)]
    choice: list[list[int | None]] = [[None] * (layers + 1) for _ in range(count + 1)]
    best[0][0] = 0.0
    for index in range(1, count + 1):
        for end in range(layers + 1):
            skippable = not spread and candidates[index - 1].worker_id not in required
            if skippable and best[index - 1][end] < best[index][end]:
                best[index][end] = best[index - 1][end]
                choice[index][end] = end  # this worker is skipped
            for start in range(0, end, profile.layer_step):
                previous = best[index - 1][start]
                if previous == INFEASIBLE:
                    continue
                cost = stage_cost(index - 1, start, end)
                if cost == INFEASIBLE:
                    continue
                total = previous + cost + BALANCE_WEIGHT * cost * cost + (HOP_MS if start else 0.0)
                if total < best[index][end]:
                    best[index][end] = total
                    choice[index][end] = start
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


# Spilled bytes, read time, and a third number: a stage's cost, or a route's total.
_Score = tuple[int, float, float]


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
    gives a stage's spilled bytes, read time, and cost, and a route's score is
    (spilled bytes, slowest read, total cost), compared through ``rank``.
    """
    cuts = [*range(0, profile.num_layers, profile.layer_step), profile.num_layers]
    last = len(cuts) - 1
    everyone = (1 << len(candidates)) - 1
    # best[spent][cut]: the best way to reach this cut with exactly these workers.
    best: list[list[_Score | None]] = [[None] * len(cuts) for _ in range(everyone + 1)]
    came: dict[tuple[int, int], tuple[int, int, int]] = {}
    best[0][0] = (0, 0.0, 0.0)
    for spent in range(everyone + 1):  # adding a worker only ever raises this number
        for here in range(last):
            score = best[spent][here]
            if score is None:
                continue
            for index, candidate in enumerate(candidates):
                if spent >> index & 1 or (entry and not spent and candidate.worker_id != entry):
                    continue
                for there in range(here + 1, last + 1):
                    if exit_ and (candidate.worker_id == exit_) != (there == last):
                        continue
                    found = stage(candidate, cuts[here], cuts[there])
                    if found is None or found[1] > slowest_allowed:
                        continue
                    spilled, read, cost = found
                    total = score[2] + cost + BALANCE_WEIGHT * cost * cost + (HOP_MS if here else 0.0)
                    reached = (score[0] + spilled, max(score[1], read), total)
                    held = best[spent | 1 << index][there]
                    if held is None or rank(reached) < rank(held):
                        best[spent | 1 << index][there] = reached
                        came[spent | 1 << index, there] = (spent, here, index)

    endings = [spent for spent in range(everyone + 1) if best[spent][last] and (spent == everyone or not spread)]
    if not endings:
        return None
    position = min(endings, key=lambda spent: rank(best[spent][last])), last  # type: ignore[arg-type]
    ranges: list[tuple[Candidate, int, int]] = []
    while position in came:
        spent, here, index = came[position]
        ranges.append((candidates[index], cuts[here], cuts[position[1]]))
        position = spent, here
    ranges.reverse()
    return ranges


def _plan_blocks(
    candidates: list[Candidate],
    profile: ModelProfile,
    spread: bool,
    entry: str | None,
    exit_: str | None,
) -> list[StageAssignment] | None:
    """Plan a model with a block rule, keeping spill small without losing speed.

    The fastest plan comes first, as for any model. Spill counts toward its
    estimate, but a plan with less spill is taken if it is nearly as fast,
    because the estimate flatters spill. Least spill alone would be wrong: a
    worker with only RAM has no spill by definition, so it would draw dense
    weights off a GPU. A route's cost counts its hops and downloads, so a plan
    does not add a stage or move shards just to spill a little less.
    """
    capacities = _capacities(candidates, 1.0, profile.stage_workspace_bytes)
    known: dict[tuple[str, int, int], _Score | None] = {}

    def stage(candidate: Candidate, start: int, end: int) -> _Score | None:
        key = (candidate.worker_id, start, end)
        if key not in known:
            filled = _fill(profile, _stage_items(profile, start, end), candidate.pools, capacities[candidate.worker_id])
            known[key] = None
            if filled is not None:
                placements, milliseconds = filled
                cost = milliseconds + _download_ms(candidate, profile, start, end)
                known[key] = (_spilled_bytes(placements, candidate.pools), milliseconds, cost)
        return known[key]

    def route(rank: Callable[[_Score], tuple[Any, ...]], slowest: float = INFEASIBLE) -> list[StageAssignment] | None:
        ranges = _search_blocks(candidates, profile, stage, spread, entry, exit_, rank, slowest)
        return None if ranges is None else _build_route(ranges, profile, capacities)

    fastest = route(lambda score: (score[2], score[0], score[1]))
    workers = {candidate.worker_id: candidate for candidate in candidates}
    if fastest is None or not any(_spilled_bytes(item.placements, workers[item.worker_id].pools) for item in fastest):
        return fastest
    reads = [item.estimated_ms_per_token for item in fastest]
    leanest = route(lambda score: score, max(reads) * (1 + SPILL_ALLOWANCE))
    if leanest is None:
        return fastest
    slower = _route_cost(leanest, candidates, profile) - _route_cost(fastest, candidates, profile)
    return leanest if slower <= sum(reads) * SPILL_ALLOWANCE else fastest


def _best_speed(candidate: Candidate) -> float:
    return max((_speed(pool) for pool in candidate.pools), default=0.0)


def _orders(
    candidates: list[Candidate],
    entry: str | None,
    exit_: str | None,
) -> list[list[Candidate]]:
    """Worker sequences worth planning, honoring any pinned ends.

    Direction matters because the ends differ: the output head is read in full
    for every token, while the embedding costs one row. A fixed order can hand
    the expensive end to the slowest machine.
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
    return [[*first, *order, *last] for order in arrangements]


def _route_cost(route: list[StageAssignment], candidates: list[Candidate], profile: ModelProfile) -> float:
    workers = {candidate.worker_id: candidate for candidate in candidates}
    downloads = sum(
        _download_ms(workers[stage.worker_id], profile, stage.start_layer, stage.end_layer) for stage in route
    )
    return sum(stage.estimated_ms_per_token for stage in route) + HOP_MS * (len(route) - 1) + downloads


def _attempt(
    candidates: list[Candidate],
    profile: ModelProfile,
    spread: bool,
    entry: str | None = None,
    exit_: str | None = None,
) -> list[StageAssignment] | None:
    if profile.layer_step > 1 and len(candidates) <= MAX_SEARCHED_WORKERS:
        return _plan_blocks(candidates, profile, spread, entry, exit_)
    required = {name for name in (entry, exit_) if name}
    for shrink in PACKING_RETRIES:
        capacities = _capacities(candidates, shrink, profile.stage_workspace_bytes)
        costs: dict[tuple[str, int, int], float] = {}
        best: list[StageAssignment] | None = None
        for order in _orders(candidates, entry, exit_):
            ranges = _partition(order, profile, capacities, spread, required, costs)
            if ranges is None:
                continue
            route = _build_route(ranges, profile, capacities)
            if route is None:
                continue
            if best is None or _route_cost(route, candidates, profile) < _route_cost(best, candidates, profile):
                best = route
        if best is not None:
            return best
    return None


def _build_route(
    ranges: list[tuple[Candidate, int, int]],
    profile: ModelProfile,
    capacities: dict[str, dict[str, int]],
) -> list[StageAssignment] | None:
    route: list[StageAssignment] = []
    for candidate, start, end in ranges:
        filled = _fill(
            profile,
            _stage_items(profile, start, end),
            candidate.pools,
            capacities[candidate.worker_id],
        )
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
            )
        )
    return route


def plan_placement(
    preferred: list[Candidate],
    standby: list[Candidate],
    profile: ModelProfile,
    *,
    spread: bool = True,
    entry_worker: str | None = None,
    exit_worker: str | None = None,
) -> list[StageAssignment]:
    """Plan a route, adding standby workers only if the model needs them.

    ``entry_worker`` and ``exit_worker`` pin which machines hold the two ends.
    A worker holds one stage, so pinning one worker to both ends means it runs
    the whole model.
    """
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
    for group in attempts:
        route = _attempt(group, profile, spread, entry_worker, exit_worker)
        if route is not None:
            return route
        if spread:
            # Spreading demands a layer on every worker, which a worker with
            # little memory can make impossible on its own.
            route = _attempt(group, profile, False, entry_worker, exit_worker)
            if route is not None:
                return route

    offered: dict[str, int] = {}
    for candidate in everyone:
        for pool in candidate.pools:
            offered[pool.capacity_group] = max(offered.get(pool.capacity_group, 0), usable_bytes(pool))
    context = sum(profile.context_bytes(layer) for layer in range(profile.num_layers))
    # Report what the pools would really hold: FP32 pools double a BF16 checkpoint.
    pools = [pool for candidate in everyone for pool in candidate.pools]
    needed = min(sum(profile.size_in(unit, pool, profile.quantized(unit)) for unit in profile.units) for pool in pools)
    hint = "remove a pin, " if pinned else ""
    extra = f" and {profile.entry_extra_bytes / 1024**3:.1f} GiB for a draft model or projector" if profile.entry_extra_bytes else ""
    blocks = ""
    if profile.layer_step > 1:
        # Without this the numbers above can look like the model fits.
        size = needed / -(-profile.num_layers // profile.layer_step)
        blocks = (
            f". Stages of this model start on a multiple of {profile.layer_step} layers, so a worker takes whole "
            f"blocks of about {size / 1024**3:.1f} GiB, and room left on a card goes unused unless that worker "
            "also offers RAM for the part of a block that does not fit"
        )
    raise PlacementError(
        f"This model needs {needed / 1024**3:.1f} GiB of weights plus "
        f"{context / 1024**3:.2f} GiB of context state{extra}, but the pool offers "
        f"{sum(offered.values()) / 1024**3:.1f} GiB across {len(everyone)} worker(s); "
        f"{hint}add a worker, raise a RAM or disk budget, or lower the context length{blocks}"
    )


def memory_report(route: list[StageAssignment], candidates: list[Candidate], profile: ModelProfile) -> dict[str, Any]:
    """What a plan does with the memory it was offered.

    One row per physical memory, so RAM two workers share counts once. A pool's
    room is what weights and context could have taken, after working space and
    anything not free right now. Its unused bytes are what is left of that room,
    which on a worker with no stage is all of it. Spilled bytes are weights held
    below the best kind of memory their worker offers.
    """
    rows: dict[str, dict[str, Any]] = {}
    room: dict[str, int] = {}
    for candidate in candidates:
        for pool in candidate.pools:
            group = pool.capacity_group
            row = rows.setdefault(
                group,
                {"workers": [], "pools": [], "kind": pool.kind, "offered_bytes": 0, "used_bytes": 0, "spilled_bytes": 0},
            )
            if candidate.worker_id not in row["workers"]:
                row["workers"].append(candidate.worker_id)
            if pool.pool_id not in row["pools"]:
                row["pools"].append(pool.pool_id)
            row["offered_bytes"] = max(row["offered_bytes"], pool.budget_bytes)
            room[group] = max(room.get(group, 0), _weight_room(pool, profile.stage_workspace_bytes))

    workers = {candidate.worker_id: candidate for candidate in candidates}
    for stage in route:
        pools = workers[stage.worker_id].pools
        by_id = {pool.pool_id: pool for pool in pools}
        best = min(MEMORY_TIERS[pool.kind] for pool in pools)
        placed = {item.unit: item for item in stage.placements}
        for item in _stage_items(profile, stage.start_layer, stage.end_layer):
            placement = placed[item.unit.name]
            pool = by_id[placement.pool_id]
            row = rows[pool.capacity_group]
            row["used_bytes"] += placement.bytes + _state_bytes(item, pool)
            if MEMORY_TIERS[pool.kind] > best:
                row["spilled_bytes"] += placement.bytes

    for group, row in rows.items():
        row["reserved_bytes"] = row["offered_bytes"] - room[group]
        row["unused_bytes"] = max(room[group] - row["used_bytes"], 0)
    return {
        "spilled_bytes": sum(row["spilled_bytes"] for row in rows.values()),
        "unused_bytes": sum(row["unused_bytes"] for row in rows.values()),
        "pools": list(rows.values()),
    }
