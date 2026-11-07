"""One worker's slice of a GGUF model: its stage file, loaded in llama.cpp."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Sequence

import numpy as np

from .gguf import GgufHeader, TensorInfo, plan_stage, write_stage
from .llama_host import Channel, StageHost
from .llamacpp import LlamaError, StageLost
from .models.gguf import GgufAdapter
from .protocol import ConfigureRequest, MemoryPool, SavePoint, UnitPlacement
from .statestore import HISTORY, PREFIX, PrefixIndex, cut_points, keys_at, pass_ends
from .vision import SECTIONS, EncodedImage, image_id
from .weights import file_sha256, UnitCache, _read_unit_header, iter_unit_tensor, unit_tensor_span

FAST_POOLS = {"vram", "unified"}
INITIAL_GUESSES = 3
# Ways of making tokens once a draft is loaded. See DraftChoice.
PIPELINED, GUESSED, PLAIN = "pipelined", "guessed", "plain"
# How many tokens a pipeline runs before the choice is looked at again.
PIPELINE_STRETCH = 24
# How many tokens go by between trials of the slower way, and how fast its rate adapts.
DRAFT_TRIAL_TOKENS = 96
DRAFT_MEMORY = 0.9
FEWEST_GUESSES = 2


def draft_apart() -> bool:
    """Whether a draft keeps a process of its own, so the two ways can be compared on a mesh."""
    return os.environ.get("ISHARDED_DRAFT_APART") == "1"


def load_on(host: Channel, operation: str, settings: dict[str, Any]) -> None:
    """Has a host do this load. One that failed it does not stay."""
    try:
        host.call(operation, **settings)
    except LlamaError:
        host.close()
        raise


def start_host(operation: str, settings: dict[str, Any]) -> StageHost:
    """A stage process that has done this load."""
    host = StageHost()
    load_on(host, operation, settings)
    return host


def as_payload(hidden: np.ndarray) -> bytes:
    return np.ascontiguousarray(hidden, dtype=np.float32).tobytes()


def cpu_tensor_patterns(placements: list[UnitPlacement], pools: dict[str, MemoryPool], start: int) -> list[str]:
    """Regexes for tensors the plan keeps out of GPU memory.

    Layer numbers are the stage file's own, which start at zero. Disk placements
    count as CPU too: llama.cpp maps the file, so the OS pages those in on use.
    """
    dense: list[str] = []
    experts: list[str] = []
    patterns: list[str] = []
    for item in placements:
        pool = pools.get(item.pool_id)
        if not item.bytes or pool is None or pool.kind in FAST_POOLS:
            continue
        if item.unit == "head":
            patterns.append(r"^output(_norm)?\.weight$")
            continue
        if not item.unit.startswith("layer-"):
            continue
        layer = int(item.unit[6:11]) - start
        (experts if item.unit.endswith(".experts") else dense).append(str(layer))
    if dense:
        patterns.append(rf"^blk\.({'|'.join(dense)})\.")
    if experts:
        patterns.append(rf"^blk\.({'|'.join(experts)})\.ffn_[a-z_]*_exps\.")
    return patterns


def build_stage_file(
    cache: UnitCache,
    request: ConfigureRequest,
    header: GgufHeader,
    units: list[str],
    start: int,
    end: int,
    last: bool,
) -> Path:
    """Assemble a stage file from cached units, reusing a finished one."""
    folder = cache.directory(request.model_id, request.revision) / "stages"
    folder.mkdir(parents=True, exist_ok=True)
    # With hashes, the name says what the file was built from, so a stage file
    # left by an earlier load cannot stand in for one built from checked units.
    label = ""
    if request.unit_hashes:
        listing = json.dumps(sorted(request.unit_hashes.items()))
        label = "-" + hashlib.sha256(listing.encode()).hexdigest()[:16]
    target = folder / f"stage-{start}-{end}{label}.gguf"
    for old in folder.glob("stage-*.gguf"):
        if old != target:
            old.unlink(missing_ok=True)  # a new split makes the old file useless
    if target.is_file():
        return target

    # A stage file is a second copy of its units, and a disk that held the units may not hold both.
    paths = {unit: cache.path(request.model_id, request.revision, unit) for unit in units}
    cache.make_room(sum(path.stat().st_size for path in paths.values()), set(paths.values()))
    owners: dict[str, Path] = {}
    for unit, path in paths.items():
        expected = request.unit_hashes.get(unit)
        if request.unit_hashes and (expected is None or file_sha256(path) != expected):
            raise ValueError(f"{unit} does not match its hash, so this stage cannot be built from it")
        owners.update(dict.fromkeys(_read_unit_header(path), path))

    def read_tensor(tensor: TensorInfo) -> Iterable[bytes]:
        return iter_unit_tensor(owners[tensor.name], tensor.name)

    plan = plan_stage(header, start, end, last=last)
    partial = target.with_suffix(".part")
    try:
        with partial.open("wb") as handle:
            write_stage(header, plan, handle, read_tensor)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def needs_mapping(placements: list[UnitPlacement], pools: dict[str, MemoryPool]) -> bool:
    """Disk-tier weights only work mapped, because the OS pages them in on use."""
    return any(item.bytes and item.pool_id in pools and pools[item.pool_id].kind == "disk" for item in placements)


class GgufStage:
    def __init__(
        self,
        folder: str,
        extra_paths: tuple[str, ...],
        cache: UnitCache,
        request: ConfigureRequest,
        adapter: GgufAdapter,
        pools: dict[str, MemoryPool],
        threads: int,
        state_folder: Path | None = None,
        state_ram_bytes: int = 0,
        state_disk_bytes: int = 0,
        wire_bf16: bool = True,
    ) -> None:
        assignment = request.assignment
        self.first = assignment.start_layer == 0
        self.last = assignment.end_layer == adapter.num_layers
        self.width = adapter.header.hidden_size
        self.embedding = request.embedding
        placements = assignment.placements
        units = [item.unit for item in placements if item.bytes]
        self.path = build_stage_file(
            cache, request, adapter.header, units, assignment.start_layer, assignment.end_layer, self.last
        )
        self.uses_gpu = any(pools[item.pool_id].kind in FAST_POOLS for item in placements if item.pool_id in pools)
        table = None
        if self.first:
            holder = "embed" if "embed" in units else "head"
            table_path = cache.path(request.model_id, request.revision, holder)
            offset, length = unit_tensor_span(table_path, "token_embd.weight")
            table = {
                "path": str(table_path),
                "offset": offset,
                "length": length,
                "ggml_type": adapter.header.tensors["token_embd.weight"].ggml_type,
            }
        self._settings = {
            "folder": folder,
            "extra_paths": list(extra_paths),
            "path": str(self.path),
            "last": self.last,
            "context_tokens": request.context_tokens,
            "gpu_layers": -1 if self.uses_gpu else 0,
            "threads": threads,
            "cpu_tensors": cpu_tensor_patterns(placements, pools, assignment.start_layer),
            "table": table,
            "rollback_tokens": request.speculate_tokens,
            "slots": request.slots,
            "map_file": needs_mapping(placements, pools),
            "embedding": request.embedding,
            "kv_type": request.kv_type,
            "wire_bf16": wire_bf16,
            "state_dir": None if state_folder is None else str(state_folder),
            "state_ram_bytes": state_ram_bytes,
            "state_disk_bytes": state_disk_bytes,
        }
        # The store's last report. Asking would wait behind whatever the stage is running.
        self.states: dict[str, int] = {}
        self.host = start_host("load", self._settings)

    def respawn(self) -> None:
        """Replace a dead stage process. Its sequence and checkpoint are gone."""
        self.host.close()
        self.host = start_host("load", self._settings)

    def embed(self, tokens: list[int]) -> np.ndarray:
        _, data = self.host.call("embed", tokens=tokens)
        return np.frombuffer(data, dtype=np.float32).reshape(-1, self.width)

    def pool(self, inputs: list[list[int]]) -> np.ndarray:
        """One pooled vector per input, in one pass, as the model made them."""
        answer, data = self.host.call("pool", inputs=inputs)
        return np.frombuffer(data, dtype=np.float32).reshape(len(inputs), answer["width"])

    def forward(
        self,
        hidden: np.ndarray,
        position: int,
        slot: int = 0,
        places: np.ndarray | None = None,
        cuts: list[int] | None = None,
        saves: list[SavePoint] | None = None,
    ) -> np.ndarray:
        """Hidden states [tokens, width] in; hidden states out, or logits at the last stage."""
        where: dict[str, Any] = {} if places is None else {"places": places.tolist()}
        if cuts:
            where["cuts"] = cuts
        if saves:
            where["saves"] = [save.model_dump() for save in saves]
        answer, data = self.host.call("forward", as_payload(hidden), position=position, slot=slot, **where)
        self.states = answer.get("states", self.states)
        output = np.frombuffer(data, dtype=np.float32)
        return output if self.last else output.reshape(-1, self.width)

    def forward_many(self, hidden: np.ndarray, positions: list[int], slots: list[int]) -> np.ndarray:
        """One row per conversation, in slot order. Rows out, or logits at the last stage."""
        _, data = self.host.call("forward_many", as_payload(hidden), positions=positions, slots=slots)
        return np.frombuffer(data, dtype=np.float32).reshape(len(slots), -1)

    def verify(self, hidden: np.ndarray, position: int, **sampling: Any) -> list[int]:
        return self.host.call("verify", as_payload(hidden), position=position, **sampling)[0]["tokens"]

    def truncate(self, keep: int, slot: int = 0) -> None:
        self.host.call("truncate", keep=keep, slot=slot)

    def restore(self, key: str, slot: int = 0) -> str:
        """Put a saved state on a sequence, and say which tier it came from."""
        answer, _ = self.host.call("restore", key=key, slot=slot)
        self.states = answer.get("states", self.states)
        return answer["tier"]

    def clear(self, slot: int = 0) -> None:
        self.host.call("clear", slot=slot)

    def ring_call(self, operation: str, **meta: Any) -> dict[str, Any]:
        return self.host.call(operation, **meta)[0]

    def ring_stats(self) -> dict[str, Any] | None:
        """How the engine is doing, unless asking would wait behind a pass."""
        try:
            return self.host.ask_if_idle("ring_stats")
        except LlamaError:
            return None

    def refresh_states(self) -> None:
        """Bring the store's report up to date, unless that would mean waiting."""
        try:
            answer = self.host.ask_if_idle("state_stats")
        except LlamaError:
            return  # whatever is wrong, the next reply finds it and deals with it
        if answer is not None:
            self.states = answer["states"]

    def close(self) -> None:
        self.host.close()


class ImageEncoder:
    """A projector on the entry worker, in a process of its own.

    An encode takes seconds, and the stage has other replies to serve meanwhile.
    Images also come from strangers and native code decodes them, so one that
    stops the decoder costs this process and nothing else.
    """

    def __init__(
        self, folder: str, extra_paths: tuple[str, ...], stage: Path, projector: Path, width: int, threads: int, use_gpu: bool
    ) -> None:
        self.width = width
        self._settings = {
            "folder": folder,
            "extra_paths": list(extra_paths),
            "stage": str(stage),
            "projector": str(projector),
            "width": width,
            "threads": threads,
            "use_gpu": use_gpu,
        }
        self.lock = threading.Lock()
        self.host = self._start()

    def _start(self) -> StageHost:
        return start_host("load_projector", self._settings)

    def _restart(self) -> None:
        self.host.close()
        self.host = self._start()

    def _ask(self, operation: str, image: bytes) -> tuple[dict[str, Any], bytes]:
        with self.lock:
            if self.host.process.poll() is not None:
                self._restart()  # it was gone before this image arrived, so this one is not to blame
            try:
                return self.host.call(operation, image, digest=hashlib.sha256(image).hexdigest())
            except StageLost as exc:
                self._restart()
                raise LlamaError("this image could not be read; it stopped the decoder") from exc

    def measure(self, image: bytes) -> dict[str, int]:
        """The rows and positions an image will take, without encoding it."""
        return self._ask("measure_image", image)[0]

    def encode(self, image: bytes) -> EncodedImage:
        answer, data = self._ask("encode_image", image)
        rows = answer["rows"]
        split = rows * SECTIONS * 4
        places = np.frombuffer(data[:split], dtype=np.int32).reshape(SECTIONS, rows)
        vectors = np.frombuffer(data[split:], dtype=np.float32).reshape(rows, self.width)
        return EncodedImage(image_id(image), answer["span"], places, vectors)

    def close(self) -> None:
        self.host.close()


@dataclass(slots=True)
class PromptPlan:
    """What of a prompt can be restored, where it is cut, and what to save for next time.

    ``cuts`` and each save's ``at`` count rows from ``start``, as a pass does.
    """

    start: int = 0
    key: str | None = None
    kind: str | None = None
    cuts: list[int] = field(default_factory=list)
    saves: list[SavePoint] = field(default_factory=list)


def plan_prompt(
    index: PrefixIndex, prompt: list[int], cut_token: int | None, settled: int | None, caching: bool = True
) -> PromptPlan:
    """Decide what a prompt reuses and saves. Its cuts are the same in any case.

    Three places are worth a state. The first cut is where a chat's system prompt
    and tool list end, which other conversations are likely to share. The last cut
    in the settled history is where this conversation picks up next turn. Without a
    chat template to go by, a place an earlier prompt also reached is shared text.
    """
    cuts = cut_points(prompt, cut_token)
    plan = PromptPlan()
    found = index.longest(prompt) if caching else None
    if found is not None:
        plan.start, plan.key, plan.kind = found
    plan.cuts = [cut - plan.start for cut in cuts if cut > plan.start]
    if not caching:
        return plan
    ends = pass_ends(cuts, len(prompt))
    keys = keys_at(prompt, ends)
    wanted: dict[int, str] = {}
    history = [cut for cut in cuts if settled is not None and cut <= settled]
    if history:
        wanted[history[-1]] = HISTORY
    shared = [end for end in ends if index.repeated(keys[end])]
    if shared:
        wanted[shared[-1]] = PREFIX
    if cuts:
        wanted[cuts[0]] = PREFIX
    for at, kind in sorted(wanted.items()):
        if at > plan.start and keys[at] not in index:
            replaces = plan.key if kind == HISTORY and plan.kind == HISTORY else None
            plan.saves.append(SavePoint(at=at - plan.start, length=at, key=keys[at], kind=kind, replaces=replaces))
    return plan


@dataclass(slots=True)
class DraftChoice:
    """Keeps to whichever way of making tokens has been faster lately.

    Guessing is not always a win. On a mixture of experts each extra row reads more experts,
    so inside one site 4 rows cost about 3.4 times one row, and a draft ran at 70% of plain
    decoding. Across countries the network dominates, and the same draft ran 26% faster. So
    this is measured and not configured, and the other ways get a short trial now and then,
    because a route or a reply can change which one wins.
    """

    # Recent tokens and seconds, kept apart. Averaging each pass's rate would overrate
    # guessing, because the passes where every guess landed would count like the rest.
    tokens: dict[str, float] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    # When each way last had a turn, so trials go round.
    turns: dict[str, int] = field(default_factory=dict)
    since_trial: int = 0

    def rate(self, way: str) -> float:
        return self.tokens[way] / self.seconds[way] if self.seconds.get(way) else 0.0

    def best(self, ways: Sequence[str]) -> str:
        return max(ways, key=self.rate)

    def pick(self, ways: Sequence[str]) -> str:
        """The way to make the next tokens. ``ways`` are what can run now, the likeliest first."""
        for way in ways:
            if not self.seconds.get(way):
                return way
        best = self.best(ways)
        others = [way for way in ways if way != best]
        if self.since_trial < DRAFT_TRIAL_TOKENS or not others:
            return best
        return min(others, key=lambda way: self.turns.get(way, 0))

    def note(self, way: str, tokens: int, seconds: float) -> None:
        self.tokens[way] = self.tokens.get(way, 0.0) * DRAFT_MEMORY + tokens
        self.seconds[way] = self.seconds.get(way, 0.0) * DRAFT_MEMORY + max(seconds, 1e-6)
        self.turns[way] = max(self.turns.values(), default=0) + 1
        self.since_trial = 0 if way != self.best(list(self.seconds)) else self.since_trial + tokens


def next_guess_count(current: int, right: int, limit: int) -> int:
    """Guess further when guesses land, and less when they do not.

    Code and lists predict well and reward long guesses. Open-ended text does
    not, and every wasted guess costs draft time and a bigger verify pass.
    """
    if right >= current:
        return min(current + 1, limit)
    if right == 0:
        return max(current - 1, min(FEWEST_GUESSES, limit))
    return current


@dataclass(slots=True)
class DraftText:
    """What a draft holds for one slot."""

    # Tokens on the slot's sequence, which only ever holds confirmed text.
    committed: int = 0
    # Tokens after those. They run on the scratch sequence, and may still prove wrong.
    chain: list[int] = field(default_factory=list)
    # The scratch sequence has more than ``chain``, so it must start again from the slot.
    stale: bool = True


class DraftModel:
    """A small whole model on the entry worker, kept in step with the real text.

    It runs in the entry stage's process when given ``beside``, and in one of its own if not.
    """

    def __init__(
        self,
        folder: str,
        extra_paths: tuple[str, ...],
        path: Path,
        context_tokens: int,
        threads: int,
        slots: int = 1,
        beside: StageHost | None = None,
    ) -> None:
        self.slots = slots
        self.text: dict[int, DraftText] = {}
        self.host: Channel = StageHost() if beside is None else beside.channel()
        load_on(
            self.host,
            "load_draft",
            {
                "folder": folder,
                "extra_paths": list(extra_paths),
                "path": str(path),
                "context_tokens": context_tokens,
                "gpu_layers": -1,
                "threads": threads,
                "slots": slots,
            },
        )

    def _commit(self, tokens: list[int], slot: int) -> None:
        if tokens:
            text = self.text[slot]
            self.host.call("commit", tokens=tokens, position=text.committed, slot=slot)
            text.committed += len(tokens)

    def prepare(self, prompt: list[int], plan: PromptPlan, slot: int = 0) -> None:
        """Catch up with a prompt, reusing and saving checkpoints like the stages do."""
        text = self.text[slot] = DraftText()
        self.host.call("clear", slot=self.slots + slot)  # the last reply's guesses hold cells this one may need
        try:
            if plan.start:
                self.host.call("restore", key=plan.key, slot=slot)
                text.committed = plan.start
        except StageLost:
            raise
        except LlamaError:
            pass  # this one is only a guesser: it can read the whole prompt instead
        if not text.committed:
            self.host.call("clear", slot=slot)
        for save in plan.saves:
            if save.length > text.committed:
                self._commit(prompt[text.committed : save.length], slot)
                self.host.call("save", slot=slot, **save.model_dump())
        self._commit(prompt[text.committed :], slot)

    def seen(self, slot: int = 0) -> int:
        """How much of the text this draft has read, guesses included."""
        text = self.text[slot]
        return text.committed + len(text.chain)

    def rewind(self, keep: int, slot: int = 0) -> None:
        """Forget text from ``keep`` on, because the guesses it held did not land.

        What came before is confirmed now. The next guess does the work, so a miss costs one
        trip to the draft and not two, and it relies on what a rewind means: the model has
        said what goes at ``keep``, so the next token after a rewind is confirmed too.
        """
        text = self.text[slot]
        right = max(keep - text.committed, 0)
        if right < len(text.chain):
            del text.chain[right:]
            text.stale = True

    def guess(self, fresh: list[int], count: int, slot: int = 0) -> list[int]:
        """Guess what follows the text so far and ``fresh``, the tokens this draft has not seen."""
        text = self.text[slot]
        if text.stale:
            commit, fresh = text.chain + fresh[:1], fresh[1:]
            answer, _ = self.host.call("guess", commit=commit, position=text.committed, count=count, slot=slot, fresh=fresh)
            text.committed += len(commit)
            text.chain = []
            text.stale = False
        else:
            answer, _ = self.host.call("guess", commit=[], position=self.seen(slot), count=count, slot=slot, fresh=fresh, restart=False)
        # The newest guess has not run yet. If it lands, it comes back as fresh text.
        text.chain += fresh + answer["tokens"][:-1]
        return answer["tokens"]

    def close(self) -> None:
        self.host.close()
