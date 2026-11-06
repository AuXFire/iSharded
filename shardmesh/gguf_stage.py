"""One worker's slice of a GGUF model: its stage file, loaded in llama.cpp."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .gguf import GgufHeader, TensorInfo, plan_stage, write_stage
from .llama_host import StageHost
from .llamacpp import LlamaError, StageLost
from .models.gguf import GgufAdapter
from .protocol import ConfigureRequest, MemoryPool, UnitPlacement
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
        }
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

    def forward(self, hidden: np.ndarray, position: int, slot: int = 0) -> np.ndarray:
        """Hidden states [tokens, width] in; hidden states out, or logits at the last stage."""
        payload = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        _, data = self.host.call("forward", payload, position=position, slot=slot)
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

    def save(self, key: str, slot: int = 0) -> None:
        self.host.call("save", key=key, slot=slot)

    def restore(self, key: str, slot: int = 0) -> None:
        self.host.call("restore", key=key, slot=slot)

    def clear(self, slot: int = 0) -> None:
        self.host.call("clear", slot=slot)

    def close(self) -> None:
        self.host.close()


def checkpoint_key(tokens: list[int]) -> str:
    return hashlib.sha1(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()[:20]


@dataclass(slots=True)
class PromptPlan:
    """What of a prompt can be restored, and where to save for next time."""

    start: int = 0
    key: str | None = None
    mark: int | None = None
    new_key: str | None = None


def plan_prompt(held: tuple[str, list[int]] | None, prompt: list[int], mark: int | None) -> PromptPlan:
    plan = PromptPlan()
    if held is not None and len(held[1]) < len(prompt) and prompt[: len(held[1])] == held[1]:
        plan.start, plan.key = len(held[1]), held[0]
    if mark is not None and mark > plan.start:
        plan.mark, plan.new_key = mark, checkpoint_key(prompt[:mark])
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
        if plan.mark is not None and plan.mark > self.committed[slot]:
            self._commit(prompt[self.committed[slot] : plan.mark], slot)
            self.host.call("save", key=plan.new_key, slot=slot)
        self._commit(prompt[self.committed[slot] :], slot)

    def guess(self, history: list[int], count: int, slot: int = 0) -> list[int]:
        """Guess what follows ``history``, feeding it only the tokens it has not seen."""
        seen = self.committed[slot]
        answer, _ = self.host.call("guess", commit=history[seen:], position=seen, count=count, slot=slot)
        self.committed[slot] = len(history)
        return answer["tokens"]

    def close(self) -> None:
        self.host.close()
