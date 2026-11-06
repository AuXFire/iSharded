"""One worker's slice of a GGUF model: its stage file, loaded in llama.cpp."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable

import numpy as np

from .gguf import GgufHeader, TensorInfo, plan_stage, write_stage
from .llama_host import StageHost
from .llamacpp import LlamaError, StageLost
from .models.gguf import GgufAdapter
from .protocol import ConfigureRequest, MemoryPool, SavePoint, UnitPlacement
from .statestore import HISTORY, PREFIX, PrefixIndex, cut_points, keys_at, pass_ends
from .vision import SECTIONS, EncodedImage, image_id
from .weights import file_sha256, UnitCache, _read_unit_header, iter_unit_tensor, unit_tensor_span

FAST_POOLS = {"vram", "unified"}
INITIAL_GUESSES = 3
FEWEST_GUESSES = 2


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

    owners: dict[str, Path] = {}
    for unit in units:
        path = cache.path(request.model_id, request.revision, unit)
        expected = request.unit_hashes.get(unit)
        if request.unit_hashes and (expected is None or file_sha256(path) != expected):
            raise ValueError(f"{unit} does not match its hash, so this stage cannot be built from it")
        for name in _read_unit_header(path):
            if name != "__metadata__":
                owners[name] = path

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
    ) -> None:
        assignment = request.assignment
        self.first = assignment.start_layer == 0
        self.last = assignment.end_layer == adapter.num_layers
        self.width = adapter.header.hidden_size
        self.embedding = request.embedding
        placements = assignment.placements
        units = [item.unit for item in placements if item.bytes]
        started = time.perf_counter()
        path = build_stage_file(
            cache, request, adapter.header, units, assignment.start_layer, assignment.end_layer, self.last
        )
        self.assemble_seconds = time.perf_counter() - started
        uses_gpu = any(pools[item.pool_id].kind in FAST_POOLS for item in placements if item.pool_id in pools)
        self.path = path
        self.uses_gpu = uses_gpu
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
            "path": str(path),
            "last": self.last,
            "context_tokens": request.context_tokens,
            "gpu_layers": -1 if uses_gpu else 0,
            "threads": threads,
            "cpu_tensors": cpu_tensor_patterns(placements, pools, assignment.start_layer),
            "table": table,
            "rollback_tokens": request.speculate_tokens,
            "slots": request.slots,
            "map_file": needs_mapping(placements, pools),
            "embedding": request.embedding,
            "kv_type": request.kv_type,
            "state_dir": None if state_folder is None else str(state_folder),
            "state_ram_bytes": state_ram_bytes,
            "state_disk_bytes": state_disk_bytes,
        }
        # The store's last report. Asking would wait behind whatever the stage is running.
        self.states: dict[str, int] = {}
        self.host = self._start()

    def _start(self) -> StageHost:
        host = StageHost()
        try:
            host.call("load", **self._settings)
        except LlamaError:
            host.close()
            raise
        return host

    def respawn(self) -> None:
        """Replace a dead stage process. Its sequence and checkpoint are gone."""
        self.host.close()
        self.host = self._start()

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
        payload = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        where: dict[str, Any] = {} if places is None else {"places": places.tolist()}
        if cuts:
            where["cuts"] = cuts
        if saves:
            where["saves"] = [save.model_dump() for save in saves]
        answer, data = self.host.call("forward", payload, position=position, slot=slot, **where)
        self.states = answer.get("states", self.states)
        output = np.frombuffer(data, dtype=np.float32)
        return output if self.last else output.reshape(-1, self.width)

    def forward_many(self, hidden: np.ndarray, positions: list[int], slots: list[int]) -> np.ndarray:
        """One row per conversation, in slot order. Rows out, or logits at the last stage."""
        payload = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        _, data = self.host.call("forward_many", payload, positions=positions, slots=slots)
        return np.frombuffer(data, dtype=np.float32).reshape(len(slots), -1)

    def verify(self, hidden: np.ndarray, position: int, **sampling: Any) -> list[int]:
        payload = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        return self.host.call("verify", payload, position=position, **sampling)[0]["tokens"]

    def truncate(self, keep: int, slot: int = 0) -> None:
        self.host.call("truncate", keep=keep, slot=slot)

    def restore(self, key: str, slot: int = 0) -> str:
        """Put a saved state on a sequence, and say which tier it came from."""
        answer, _ = self.host.call("restore", key=key, slot=slot)
        self.states = answer.get("states", self.states)
        return answer["tier"]

    def clear(self, slot: int = 0) -> None:
        self.host.call("clear", slot=slot)

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
        host = StageHost()
        try:
            host.call("load_projector", **self._settings)
        except LlamaError:
            host.close()
            raise
        return host

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


class DraftModel:
    """A small whole model on the entry worker, kept in step with the real text."""

    def __init__(
        self, folder: str, extra_paths: tuple[str, ...], path: Path, context_tokens: int, threads: int, slots: int = 1
    ) -> None:
        self.host = StageHost()
        self.committed: dict[int, int] = {}
        try:
            self.host.call(
                "load_draft",
                folder=folder,
                extra_paths=list(extra_paths),
                path=str(path),
                context_tokens=context_tokens,
                gpu_layers=-1,
                threads=threads,
                slots=slots,
            )
        except LlamaError:
            self.host.close()
            raise

    def _commit(self, tokens: list[int], slot: int) -> None:
        if tokens:
            self.host.call("commit", tokens=tokens, position=self.committed[slot], slot=slot)
            self.committed[slot] += len(tokens)

    def prepare(self, prompt: list[int], plan: "PromptPlan", slot: int = 0) -> None:
        """Catch up with a prompt, reusing and saving checkpoints like the stages do."""
        self.committed[slot] = 0
        try:
            if plan.start:
                self.host.call("restore", key=plan.key, slot=slot)
                self.committed[slot] = plan.start
        except StageLost:
            raise
        except LlamaError:
            pass  # this one is only a guesser: it can read the whole prompt instead
        if not self.committed[slot]:
            self.host.call("clear", slot=slot)
        for save in plan.saves:
            if save.length > self.committed[slot]:
                self._commit(prompt[self.committed[slot] : save.length], slot)
                self.host.call("save", slot=slot, **save.model_dump())
        self._commit(prompt[self.committed[slot] :], slot)

    def guess(self, history: list[int], count: int, slot: int = 0) -> list[int]:
        """Guess what follows ``history``, feeding it only the tokens it has not seen."""
        seen = self.committed[slot]
        answer, _ = self.host.call("guess", commit=history[seen:], position=seen, count=count, slot=slot)
        self.committed[slot] = len(history)
        return answer["tokens"]

    def close(self) -> None:
        self.host.close()
