"""Saved sequence states: a bounded store on each stage, in RAM and then on disk.

llama.cpp hands over what a stage holds for a sequence as one block of bytes, and
takes it back the same way. That block covers attention and recurrent state alike,
so this store works for hybrid models, and never needs to look inside one.

A state is named by the exact tokens it holds, so two prompts share one only if
they match token for token. An image's rows count as tokens: they carry a digest of
the image. Stores are per stage and per model, because a state means nothing to
any other layers.

Where a prompt is cut into passes changes the logits by up to 0.5 here, since the
kernels add up in another order, and that is enough to flip a near-tie. A restored
state continues bit for bit only if every run cuts in the same places. So cuts come
from the prompt alone, whether or not anything is cached, and a state is only ever
saved at one. A cut must also stay put as a conversation grows, which is why they
are found from the start of the prompt, each from the one before.

Every state is written to disk behind the reply, which is what lets one outlive
the stage process that made it. The file is a copy of what is in RAM, so losing the
disk tier costs recovery and nothing else.

This module must not import torch: stage processes use it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import threading
from typing import Any, Iterable
import zlib

import numpy as np

MAGIC = b"ISHSTATE"
FORMAT = 1
SUFFIX = ".state"
PARTIAL = ".part"
# How long closing waits for writes still queued. A state not written is only a replay.
FLUSH_SECONDS = 10.0
# States waiting for a slow disk stay in RAM, past its budget, up to this much.
WRITE_BACKLOG_BYTES = 1024**3
PREFIX = "prefix"
HISTORY = "history"
# Rows in one pass. Every stage has to use the same number, as cuts depend on it.
PASS_ROWS = 512
# Rows a piece must have. A state costs more to restore than this many rows to read.
CUT_LEAST = 64
# What an entry worker remembers. Forgetting one only costs a replay.
INDEX_LIMIT = 4096
SEEN_LIMIT = 16384


def prefix_key(tokens: Iterable[int]) -> str:
    """The name of an exact run of tokens."""
    return hashlib.sha256(np.asarray(list(tokens), dtype="<i8").tobytes()).hexdigest()


def keys_at(tokens: list[int], lengths: Iterable[int]) -> dict[int, str]:
    """The name of each of these prefixes, reading the tokens once."""
    data = np.asarray(tokens, dtype="<i8")
    digest = hashlib.sha256()
    found: dict[int, str] = {}
    done = 0
    for length in sorted(set(lengths)):
        if 0 < length <= len(tokens):
            digest.update(data[done:length].tobytes())
            done = length
            found[length] = digest.hexdigest()
    return found


def cut_points(tokens: list[int], token: int | None) -> list[int]:
    """Where a prompt is cut: before ``token``, the one that opens a turn in a chat.

    The result for any start of a prompt is the start of this result, which is what
    keeps a saved state exact as a conversation grows.
    """
    if token is None:
        return []
    cuts: list[int] = []
    last = 0
    for place in np.flatnonzero(np.asarray(tokens, dtype=np.int64) == token).tolist():
        if place - last >= CUT_LEAST:
            cuts.append(place)
            last = place
    return cuts


def pass_ends(cuts: list[int], total: int) -> list[int]:
    """Every place a pass stops before the end of a prompt, where a state can be saved."""
    ends: list[int] = []
    edges = [0, *cuts, total]
    for start, stop in zip(edges, edges[1:]):
        ends.extend(range(start + PASS_ROWS, stop, PASS_ROWS))
        ends.append(stop)
    return [end for end in ends if end < total]


def stage_namespace(build: str, model_id: str, revision: str, start: int, end: int, rollback_tokens: int, kv_type: str) -> str:
    """What a stage's states belong to. Anything that changes their meaning must be in here.

    A node works this out for any stage, which is how a copy finds its way to whoever
    takes over those layers. For a GGUF, the revision is a fingerprint of the file.
    """
    parts = [FORMAT, PASS_ROWS, CUT_LEAST, build, model_id, revision, start, end, rollback_tokens, kv_type]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:24]


@dataclass(slots=True)
class Entry:
    length: int  # prompt rows it holds
    size: int
    kind: str
    data: np.ndarray | None = None  # in RAM
    on_disk: bool = False


class StateStore:
    """States by key, least recently used out first, in two tiers.

    ``folder`` is this store's own: anything else in it is treated as left over
    from before. Without a folder or a disk budget, the store is RAM only.
    """

    def __init__(self, folder: Path | str | None, ram_bytes: int, disk_bytes: int) -> None:
        self.folder = Path(folder) if folder and disk_bytes > 0 else None
        self.ram_bytes = max(0, ram_bytes)
        self.disk_bytes = max(0, disk_bytes) if self.folder else 0
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.leftovers: set[Path] = set()
        self.lock = threading.RLock()
        self.saved = 0
        self.saved_bytes = 0
        self.evicted = 0
        self.refused = 0
        self.damaged = 0
        self.write_failures = 0
        self.writes: queue.Queue[str | None] = queue.Queue()
        self.writer: threading.Thread | None = None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            os.chmod(self.folder, 0o700)  # a state is a person's conversation
            self._scan()
            self.writer = threading.Thread(target=self._write_loop, daemon=True)
            self.writer.start()

    def _file(self, key: str) -> Path:
        assert self.folder is not None
        return self.folder / (key + SUFFIX)

    def _remove(self, path: Path) -> None:
        """Delete a file, now or later. This must never fail the reply that asked.

        Windows will not delete a file another process has open, and a copy on its
        way to another node keeps one open for as long as the link takes.
        """
        try:
            path.unlink(missing_ok=True)
        except OSError:
            self.leftovers.add(path)

    def _clear_leftovers(self) -> None:
        for path in list(self.leftovers):
            self.leftovers.discard(path)
            self._remove(path)

    def _scan(self) -> None:
        """Pick up what an earlier process left, oldest first so age order survives."""
        assert self.folder is not None
        found: list[tuple[float, str, Entry]] = []
        for path in self.folder.iterdir():
            if path.suffix == PARTIAL:
                self._remove(path)  # that process stopped in the middle of writing it
                continue
            if path.suffix != SUFFIX:
                continue
            try:
                with path.open("rb") as handle:
                    header = self._read_header(handle)
                found.append((path.stat().st_mtime, path.stem, Entry(header["length"], header["size"], header["kind"], on_disk=True)))
            except (OSError, ValueError, KeyError):
                self.damaged += 1
                self._remove(path)
        for _, key, entry in sorted(found, key=lambda item: item[0]):
            self.entries[key] = entry
        self._shed_disk()

    @staticmethod
    def _read_header(handle: Any) -> dict[str, Any]:
        start = handle.read(len(MAGIC) + 8)
        if len(start) != len(MAGIC) + 8 or start[: len(MAGIC)] != MAGIC:
            raise ValueError("not a saved state")
        version, length = struct.unpack("<II", start[len(MAGIC) :])
        if version != FORMAT:
            raise ValueError(f"saved state format {version}")
        header = json.loads(handle.read(length))
        if not isinstance(header, dict):
            raise ValueError("saved state header")
        return header

    def _write_loop(self) -> None:
        while True:
            key = self.writes.get()
            try:
                if key is None:
                    return
                self._write(key)
            except Exception:  # noqa: BLE001 - this thread is all that writes, so it has to outlive one bad state
                self.write_failures += 1
            finally:
                self.writes.task_done()

    def _write(self, key: str) -> None:
        with self.lock:
            entry = self.entries.get(key)
            if entry is None or entry.on_disk or entry.data is None:
                return
            data, kind, length = entry.data, entry.kind, entry.length
        target = self._file(key)
        partial = target.with_suffix(PARTIAL)
        header = json.dumps({"length": length, "size": int(data.size), "kind": kind, "crc": zlib.crc32(data)}).encode()
        try:
            with partial.open("wb") as handle:
                os.chmod(partial, 0o600)
                handle.write(MAGIC + struct.pack("<II", FORMAT, len(header)) + header)
                handle.write(memoryview(data))
            partial.replace(target)
        except OSError:
            self._remove(partial)
            return  # still in RAM, and a full disk must not stop anything
        with self.lock:
            current = self.entries.get(key)
            if current is None:
                self._remove(target)  # dropped while it was being written
                return
            current.on_disk = True
            self._shed_disk()
            self._shed_ram()

    def _shed_ram(self) -> None:
        held = [(key, entry) for key, entry in self.entries.items() if entry.data is not None]
        used = sum(entry.size for _, entry in held)
        waiting = sum(entry.size for _, entry in held if not entry.on_disk)
        for key, entry in held:
            if used <= self.ram_bytes:
                return
            if entry.on_disk:
                entry.data = None
            elif self.disk_bytes and (waiting <= WRITE_BACKLOG_BYTES or key == held[-1][0]):
                continue  # its write is still queued, and then it can leave RAM
            else:
                # With no disk, or one that cannot keep up, this state has nowhere to go.
                waiting -= entry.size
                del self.entries[key]
                self.evicted += 1
            used -= entry.size

    def _shed_disk(self) -> None:
        if self.folder is None:
            return
        self._clear_leftovers()
        used = sum(entry.size for entry in self.entries.values() if entry.on_disk)
        for key, entry in list(self.entries.items()):
            if used <= self.disk_bytes:
                return
            if entry.on_disk:
                self._remove(self._file(key))
                entry.on_disk = False
                used -= entry.size
                if entry.data is None:
                    del self.entries[key]
                    self.evicted += 1

    def put(self, key: str, data: np.ndarray, length: int, kind: str = HISTORY) -> bool:
        """Keep a state. False if it could not fit in either tier."""
        if data.size > max(self.ram_bytes, self.disk_bytes):
            self.refused += 1
            return False
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return True
            self.entries[key] = Entry(length, int(data.size), kind, data=data)
            self.saved += 1
            self.saved_bytes += int(data.size)
            if self.folder is not None and data.size <= self.disk_bytes:
                self.leftovers.discard(self._file(key))  # that name is about to mean this state
                self.writes.put(key)
            self._shed_ram()
        return True

    def get(self, key: str) -> tuple[np.ndarray, str] | None:
        """A state and the tier it came from, or None. A damaged file counts as gone."""
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            self.entries.move_to_end(key)
            if entry.data is not None:
                return entry.data, "ram"
        data = self._read(key, entry)
        if data is None:
            self.drop(key)
            self.damaged += 1
            return None
        with self.lock:
            current = self.entries.get(key)
            if current is not None and data.size <= self.ram_bytes:
                current.data = data
                self._shed_ram()
        return data, "disk"

    def _read(self, key: str, entry: Entry) -> np.ndarray | None:
        try:
            with self._file(key).open("rb") as handle:
                header = self._read_header(handle)
                data = np.fromfile(handle, dtype=np.uint8)
        except (OSError, ValueError, KeyError):
            return None
        if data.size != entry.size or zlib.crc32(data) != header.get("crc"):
            return None
        return data

    def adopt(self, key: str, path: Path) -> bool:
        """Take over a state file that arrived from another node. False if it is no good."""
        if self.folder is None:
            self._remove(path)
            return False
        try:
            with path.open("rb") as handle:
                header = self._read_header(handle)
                data = np.fromfile(handle, dtype=np.uint8)
            entry = Entry(int(header["length"]), int(header["size"]), str(header["kind"]), on_disk=True)
            if data.size != entry.size or zlib.crc32(data) != header.get("crc"):
                raise ValueError("a saved state that does not match its checksum")
            with self.lock:
                path.replace(self._file(key))
                self.entries[key] = entry
                self.entries.move_to_end(key)
                self._shed_disk()
            return True
        except (OSError, ValueError, KeyError, TypeError):
            self.damaged += 1
            self._remove(path)
            return False

    def file_of(self, key: str) -> Path | None:
        """Where a state's file is, once it has been written."""
        with self.lock:
            entry = self.entries.get(key)
            return self._file(key) if entry is not None and entry.on_disk else None

    def has(self, key: str) -> bool:
        with self.lock:
            return key in self.entries

    def drop(self, key: str) -> None:
        with self.lock:
            entry = self.entries.pop(key, None)
            if entry is not None and entry.on_disk:
                self._remove(self._file(key))

    def wipe(self) -> None:
        """Forget everything, which is what a change of model calls for."""
        with self.lock:
            for key in list(self.entries):
                self.drop(key)

    def listing(self) -> list[dict[str, Any]]:
        with self.lock:
            return [{"key": key, "length": entry.length, "kind": entry.kind} for key, entry in self.entries.items()]

    def stats(self) -> dict[str, int]:
        with self.lock:
            entries = list(self.entries.values())
        return {
            "states": len(entries),
            "ram_bytes": sum(entry.size for entry in entries if entry.data is not None),
            "disk_bytes": sum(entry.size for entry in entries if entry.on_disk),
            "ram_budget_bytes": self.ram_bytes,
            "disk_budget_bytes": self.disk_bytes,
            "saved": self.saved,
            "saved_bytes": self.saved_bytes,
            "evicted": self.evicted,
            "refused": self.refused,
            "damaged": self.damaged,
            "write_failures": self.write_failures,
        }

    def flush(self, seconds: float = FLUSH_SECONDS) -> bool:
        """Wait for queued writes. False if they did not finish in time."""
        if self.writer is None:
            return True
        done = threading.Event()

        def wait() -> None:
            self.writes.join()
            done.set()

        threading.Thread(target=wait, daemon=True).start()
        return done.wait(seconds)

    def close(self) -> None:
        if self.writer is not None:
            self.flush()
            self.writes.put(None)
            self.writer.join(FLUSH_SECONDS)
            self.writer = None


class PrefixIndex:
    """What the stages of a route hold, as far as the entry worker knows.

    It may be wrong, because each stage evicts for itself. A stage that turns out
    not to hold a state says so, and the entry worker plans again without it.
    """

    def __init__(self) -> None:
        self.entries: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self.seen: OrderedDict[str, None] = OrderedDict()
        # States a stage may first have to fetch from whichever node holds its copy.
        self.elsewhere: set[str] = set()

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def add(self, key: str, length: int, kind: str, elsewhere: bool = False) -> None:
        self.entries[key] = (length, kind)
        self.entries.move_to_end(key)
        (self.elsewhere.add if elsewhere else self.elsewhere.discard)(key)
        while len(self.entries) > INDEX_LIMIT:
            self.elsewhere.discard(self.entries.popitem(last=False)[0])

    def drop(self, key: str | None) -> None:
        self.entries.pop(key, None)
        self.elsewhere.discard(key)

    def clear(self) -> None:
        self.entries.clear()
        self.elsewhere.clear()

    def longest(self, prompt: list[int]) -> tuple[int, str, str] | None:
        """The longest state this prompt starts with. Something must be left to run."""
        lengths = {length for length, _ in self.entries.values() if length < len(prompt)}
        found = keys_at(prompt, lengths)
        for length in sorted(found, reverse=True):
            key = found[length]
            if key in self.entries:
                self.entries.move_to_end(key)
                return length, key, self.entries[key][1]
        return None

    def repeated(self, key: str) -> bool:
        """Whether an earlier prompt reached this same place."""
        before = key in self.seen
        self.seen[key] = None
        self.seen.move_to_end(key)
        while len(self.seen) > SEEN_LIMIT:
            self.seen.popitem(last=False)
        return before


def clear_other_stores(root: Path | str, keep: set[str]) -> None:
    """Delete stores for other models or layers. They can never be used again."""
    root = Path(root)
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path, ignore_errors=True)
