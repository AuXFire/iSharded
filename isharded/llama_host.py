"""A separate process that runs one llama.cpp stage.

PyTorch and llama.cpp each load their own OpenMP runtime, and the two abort when
they meet in a process. Keeping them apart also means a native crash cannot take
the worker with it, and that unloading is a process exit that frees everything.

A draft model shares its stage's process, on a connection and a thread of its own. Two
processes on one GPU take turns on it, and this way both run in one CUDA context.

This module must not import torch, directly or through other iSharded modules.
"""

from __future__ import annotations

from collections import OrderedDict
import contextlib
import ctypes as C
import json
from multiprocessing.connection import Client, Connection, Listener
import mmap
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import psutil

from .deadlines import PASS_SECONDS, PASS_SECONDS_PER_TOKEN, pass_seconds
from .llamacpp import LlamaError, LlamaLibrary, LlamaStage, StageLost, TokenEmbedding
from .statestore import HISTORY, PARTIAL, StateStore

# What every operation answers with: fields for the reply, and any raw bytes.
Reply = tuple[dict[str, Any], bytes]

KEY_VARIABLE = "ISHARDED_STAGE_KEY"
CONNECT_SECONDS = 60.0
EXIT_SECONDS = 15.0
# What a kill gets to take effect, each time. A healthy process is gone long before.
KILL_SECONDS = 5.0
# A frozen process cannot say so, and a long prompt's deadline, or a pass on the engine that no
# call here waits on, would leave it frozen for as long as a real reply can wait. So a thread
# in the process answers this all the time, whatever its passes are doing.
PULSE_EVERY = 2.0
PULSE_SECONDS = 10.0
# How often a call that is waiting looks up to see whether its process has been stopped.
ANSWER_LOOK_SECONDS = 0.25
# Past this, a write can stick until the far end reads. A Windows pipe holds 8 KiB, which
# one row of a wide model is over, and a call sends two messages.
PIPE_BYTES = 4 * 1024
# Encoded images kept for the next turn. A conversation resends its images every time.
ENCODED_IMAGE_BYTES = 128 * 1024 * 1024
# States a draft model keeps. It is small, and a draft that loses one just reads more.
DRAFT_STATE_BYTES = 256 * 1024 * 1024
# Operations that touch only the engine's links, and so need not wait for a running pass.
# A flip waits for the old route's passes, which need the gate, so it must not hold it.
UNGATED = {
    "load", "prime", "ring_open", "ring_connect", "ring_close", "ring_stats", "exactness",
    "ring_stage", "ring_connect_staged", "ring_slots", "ring_flip", "ring_close_epoch",
}  # fmt: skip
# These put a sequence on another history, which the engine's snapshots cannot follow.
NEW_HISTORY = {"clear", "restore", "truncate"}
# A load takes as long as the disk takes, so nothing here can say it has hung.
LOADS = {"load", "load_draft", "load_projector", "prime"}
# These have no rows to go by and can be slow when healthy: a state can be gigabytes on a
# slow disk, and an image a long encode on a CPU. They get the time of a long prompt.
SLOW = {"save", "restore", "state_adopt", "state_list", "measure_image", "encode_image"}
SLOW_ROWS = 600
BELOW_NORMAL = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS, for a spare's start on Windows
PASS_MS, ROW_MS = round(PASS_SECONDS * 1000), round(PASS_SECONDS_PER_TOKEN * 1000)


def seconds_for(operation: str, rows: int) -> float | None:
    """How long a stage may take over a call, or None if nothing can say."""
    if operation in LOADS:
        return None
    if operation == "pulse":
        return PULSE_SECONDS
    return pass_seconds(max(rows, SLOW_ROWS) if operation in SLOW else rows)


class _Host:
    def __init__(self) -> None:
        self.library: LlamaLibrary | None = None
        self.stage: LlamaStage | None = None
        self.embedding: TokenEmbedding | None = None
        self.table: mmap.mmap | None = None
        self.table_view: memoryview | None = None
        self.states = StateStore(None, 0, 0)
        self.scratch = 1
        self.projector: Any = None
        self.encoded: OrderedDict[str, tuple[dict[str, Any], bytes]] = OrderedDict()
        # What the load's probe found about merged passes. None means nothing was probed.
        self.found: dict[str, Any] | None = None

    def prime(self, meta: dict[str, Any], _: bytes) -> Reply:
        """Brings the library and its devices up with no model, so a load later skips that.

        A primed process waits as a spare: its start, its imports and its device contexts are
        paid in an idle moment, not in the middle of a change. See ``SpareHost``.
        """
        self.library = LlamaLibrary.load(meta["folder"], tuple(meta["extra_paths"]))
        devices = self.library.wake_devices()
        return {"folder": self.library.folder, "devices": devices}, b""

    def load(self, meta: dict[str, Any], _: bytes) -> Reply:
        # A primed process already has the library. It is one per process, so it is that one.
        self.library = LlamaLibrary.load(meta["folder"], tuple(meta["extra_paths"]))
        self.stage = LlamaStage(
            self.library,
            meta["path"],
            last=meta["last"],
            context_tokens=meta["context_tokens"],
            gpu_layers=meta["gpu_layers"],
            threads=meta["threads"],
            cpu_tensors=meta["cpu_tensors"],
            rollback_tokens=meta.get("rollback_tokens", 0),
            sequences=meta.get("slots", 1),
            map_file=meta.get("map_file", True),
            embedding=meta.get("embedding", False),
            kv_type=meta.get("kv_type", "f16"),
            native=not meta.get("embedding", False),
        )
        self.states = StateStore(meta.get("state_dir"), meta.get("state_ram_bytes", 0), meta.get("state_disk_bytes", 0))
        table = meta.get("table")
        if table:
            with open(table["path"], "rb") as handle:
                self.table = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self.table_view = memoryview(self.table)[table["offset"] : table["offset"] + table["length"]]
            self.embedding = TokenEmbedding(self.library, self.table_view, table["ggml_type"], self.stage.hidden_size)
        if self.stage.engine is not None:
            found = self.embedding
            self.stage.engine.attach(
                self.stage,
                meta.get("wire_bf16", True),
                self.table_view,
                len(self.table_view) // found.row_bytes if found else 0,
                found.row_bytes if found else 0,
                C.cast(found.to_float, C.c_void_p).value if found and found.to_float else None,
                meta.get("rollback_tokens", 0) + 2,
            )
            # Only exact replies use what it finds, and on a CPU stage with many slots it takes seconds.
            self.found = self._probe(meta.get("slots", 1)) if meta.get("exact") else None
        return {"hidden_size": self.stage.hidden_size, "vocab_size": self.stage.vocab_size, "engine": self.stage.engine is not None}, b""

    def _probe(self, slots: int) -> dict[str, Any]:
        """Runs before the worker can send this stage anything, which the probe needs."""
        try:
            return self.stage.engine.probe(slots)
        except OSError as exc:
            return {"exact_merge_limit": 0, "verify_exact": False, "error": str(exc)}

    def exactness(self, _: dict[str, Any], __: bytes) -> Reply:
        return {"found": self.found}, b""

    def embed(self, meta: dict[str, Any], _: bytes) -> Reply:
        if self.embedding is None:
            raise LlamaError("this stage has no embedding table")
        return {}, self.embedding.rows(meta["tokens"]).tobytes()

    def pool(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.stage is not None and self.embedding is not None
        inputs = meta["inputs"]
        rows = self.embedding.rows([token for item in inputs for token in item])
        pooled = self.stage.pool(rows, [len(item) for item in inputs])
        return {"width": self.stage.output_width}, pooled.tobytes()

    def forward(self, meta: dict[str, Any], payload: bytes) -> Reply:
        assert self.stage is not None
        hidden = np.frombuffer(payload, dtype=np.float32).reshape(-1, self.stage.hidden_size)
        places = meta.get("places")
        places = None if places is None else np.array(places, dtype=np.int32)
        slot = meta.get("slot", 0)
        saves = {item["at"]: item for item in meta.get("saves", [])}

        def reached(row: int) -> None:
            if row in saves:
                self._keep(saves[row], slot)

        output = self.stage.forward(hidden, meta["position"], slot, places, meta.get("cuts", ()), reached if saves else None)
        return ({"states": self.states.stats()} if saves else {}), output.tobytes()

    def _keep(self, save: dict[str, Any], slot: int) -> None:
        assert self.stage is not None
        kept = self.states.put(save["key"], self.stage.save_state(slot), save["length"], save.get("kind", HISTORY))
        if kept and save.get("replaces"):
            self.states.drop(save["replaces"])

    def forward_many(self, meta: dict[str, Any], payload: bytes) -> Reply:
        assert self.stage is not None
        hidden = np.frombuffer(payload, dtype=np.float32).reshape(-1, self.stage.hidden_size)
        return {}, self.stage.forward_many(hidden, meta["positions"], meta["slots"]).tobytes()

    def clear(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.stage is not None
        self.stage.clear(meta.get("slot"))
        return {}, b""

    def save(self, meta: dict[str, Any], _: bytes) -> Reply:
        self._keep(meta, meta.get("slot", 0))
        return {"states": self.states.stats()}, b""

    def restore(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.stage is not None
        held = self.states.get(meta["key"])
        if held is None:
            raise LlamaError(f"no saved state {meta['key']} here")
        self.stage.restore_state(held[0], meta.get("slot", 0))
        return {"tier": held[1], "states": self.states.stats()}, b""

    def state_stats(self, _: dict[str, Any], __: bytes) -> Reply:
        return {"states": self.states.stats()}, b""

    def state_list(self, _: dict[str, Any], __: bytes) -> Reply:
        return {"states": self.states.listing()}, b""

    def state_has(self, meta: dict[str, Any], _: bytes) -> Reply:
        return {"held": self.states.has(meta["key"])}, b""

    def state_adopt(self, meta: dict[str, Any], _: bytes) -> Reply:
        """Take a state the worker fetched into this store's folder from another node."""
        folder = self.states.folder
        if folder is None:
            return {"adopted": False}, b""
        return {"adopted": self.states.adopt(meta["key"], folder / (meta["key"] + PARTIAL))}, b""

    def commit(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.stage is not None
        self.stage.decode_tokens(meta["tokens"], meta["position"], sequence_id=meta.get("slot", 0))
        return {}, b""

    def truncate(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.stage is not None
        self.stage.truncate(meta["keep"], meta.get("slot", 0))
        return {}, b""

    def verify(self, meta: dict[str, Any], payload: bytes) -> Reply:
        assert self.stage is not None
        hidden = np.frombuffer(payload, dtype=np.float32).reshape(-1, self.stage.hidden_size)
        sampling = {key: meta[key] for key in ("guesses", "temperature", "top_p", "top_k", "seed")}
        sampling["slot"] = meta.get("slot", 0)
        return {"tokens": self.stage.verify(hidden, meta["position"], **sampling)}, b""

    def load_draft(self, meta: dict[str, Any], _: bytes) -> Reply:
        slots = meta.get("slots", 1)
        self.library = LlamaLibrary.load(meta["folder"], tuple(meta["extra_paths"]))
        self.stage = LlamaStage(
            self.library,
            meta["path"],
            last=True,
            # Every slot has a scratch sequence, but a copy shares its cells, so scratch
            # needs only room for guesses and the shared cache stays the size it was.
            context_tokens=meta["context_tokens"] * (slots + 1) // (2 * slots),
            gpu_layers=meta["gpu_layers"],
            threads=meta["threads"],
            sequences=2 * slots,
            unified_cache=True,
            map_file=False,
        )
        self.scratch = slots
        self.states = StateStore(None, DRAFT_STATE_BYTES, 0)
        return {"rewinds": not self.stage.recurrent}, b""

    def load_projector(self, meta: dict[str, Any], _: bytes) -> Reply:
        from .mtmd import Projector

        self.library = LlamaLibrary.load(meta["folder"], tuple(meta["extra_paths"]))
        self.projector = Projector(self.library, meta["projector"], meta["stage"], meta["width"], meta["threads"], meta["use_gpu"])
        return {}, b""

    def measure_image(self, _: dict[str, Any], payload: bytes) -> Reply:
        shape = self.projector.measure(payload)
        return {"rows": shape.rows, "span": shape.span}, b""

    def encode_image(self, meta: dict[str, Any], payload: bytes) -> Reply:
        digest = meta["digest"]
        if digest not in self.encoded:
            shape, rows = self.projector.encode(payload)
            self.encoded[digest] = {"rows": shape.rows, "span": shape.span}, shape.places.tobytes() + rows.tobytes()
        self.encoded.move_to_end(digest)
        while len(self.encoded) > 1 and sum(len(data) for _, data in self.encoded.values()) > ENCODED_IMAGE_BYTES:
            self.encoded.popitem(last=False)
        return self.encoded[digest]

    def guess(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.stage is not None
        slot = meta.get("slot", 0)
        before = self.stage.decodes, self.stage.decoded_rows
        tokens = self.stage.guess(
            meta["commit"],
            meta["position"],
            meta["count"],
            slot,
            self.scratch + slot,
            meta.get("fresh", ()),
            meta.get("restart", True),
            meta.get("catch_up"),
            meta.get("sampling"),
        )
        return {"tokens": tokens, "decodes": self.stage.decodes - before[0], "decoded_rows": self.stage.decoded_rows - before[1]}, b""

    @property
    def engine(self) -> Any:
        return None if self.stage is None else self.stage.engine

    def pulse(self, _: dict[str, Any], __: bytes) -> Reply:
        return {}, b""

    def ring_open(self, meta: dict[str, Any], _: bytes) -> Reply:
        if self.engine is None:
            return {"engine": False}, b""
        door = self.engine.listen(meta["host"], meta["port"])
        # The engine's watchdog holds a pass to what this process holds a call to.
        limits = (meta.get("pass_ms", PASS_MS), meta.get("row_ms", ROW_MS))
        keys = {name: bytes.fromhex(key) for name, key in meta["keys"].items() if key is not None}
        self.engine.open(meta["epoch"], meta["index"], meta["count"], keys, meta["ack_timeout_ms"], meta["silence_timeout_ms"], *limits, merge_passes=meta.get("merge_passes", True), merge_most=meta.get("merge_most"))
        return {"engine": True, **door}, b""

    def ring_connect(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.engine is not None
        self.engine.connect(meta["links"])
        return {}, b""

    def ring_stage(self, meta: dict[str, Any], _: bytes) -> Reply:
        """Takes this stage's place in the next route, beside the one it serves. Same door."""
        if self.engine is None or not self.engine.stages:
            return {"engine": False}, b""
        limits = (meta.get("pass_ms", PASS_MS), meta.get("row_ms", ROW_MS))
        keys = {name: bytes.fromhex(key) for name, key in meta["keys"].items() if key is not None}
        self.engine.stage(meta["epoch"], meta["index"], meta["count"], keys, meta["ack_timeout_ms"], meta["silence_timeout_ms"], *limits, merge_passes=meta.get("merge_passes", True), merge_most=meta.get("merge_most"))
        return {"engine": True}, b""

    def ring_connect_staged(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.engine is not None
        self.engine.connect_staged(meta["links"])
        return {}, b""

    def ring_slots(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.engine is not None
        self.engine.give_slots(meta["epoch"], meta["slots"])
        return {}, b""

    def ring_flip(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.engine is not None
        return {"quiet": self.engine.flip(meta["epoch"], meta["slots"], meta.get("wait_ms", 0))}, b""

    def ring_close_epoch(self, meta: dict[str, Any], _: bytes) -> Reply:
        closed = self.engine is not None and self.engine.stages and self.engine.close_epoch(meta["epoch"])
        return {"closed": bool(closed)}, b""

    def ring_close(self, _: dict[str, Any], __: bytes) -> Reply:
        if self.engine is not None:
            self.engine.close_ring()
        return {}, b""

    def ring_stats(self, _: dict[str, Any], __: bytes) -> Reply:
        return ({} if self.engine is None else self.engine.stats()), b""

    def shut(self) -> None:
        self.states.close()


class _Beside(_Host):
    """A second model in a stage's process. That process outlives it, so it frees what it holds."""

    def shut(self) -> None:
        super().shut()
        if self.stage is not None:
            self.stage.close()
            self.stage = None


def beside(address: Any, key: bytes) -> None:
    host = _Beside()
    try:
        serve(Client(address, authkey=key), host, address, key)
    finally:
        host.shut()


def run(address: Any, key: bytes) -> None:
    serve(Client(address, authkey=key), _Host(), address, key)


def serve(connection: Connection, host: _Host, address: Any, key: bytes) -> None:
    while True:
        try:
            meta = json.loads(connection.recv_bytes())
            payload = connection.recv_bytes()
        except EOFError:
            return  # the worker went away, so this stage has no purpose left
        operation = meta.pop("op")
        if operation == "close":
            host.shut()
            connection.send_bytes(b'{"ok":true}')
            connection.send_bytes(b"")
            return
        if operation == "channel":
            # It connects from a thread, because the worker only accepts once it has this answer.
            threading.Thread(target=beside, args=(address, key), daemon=True).start()
            connection.send_bytes(b'{"ok":true}')
            connection.send_bytes(b"")
            continue
        try:
            # The engine runs passes on its own threads, and llama.cpp allows one at a time.
            engine = None if operation in UNGATED else host.engine
            if engine is not None:
                engine.lock()
            try:
                if engine is not None and operation in NEW_HISTORY:
                    engine.forget(meta.get("slot"))
                    if meta.get("generation") and meta.get("slot") is not None:
                        # The slot is a new reply's now. Passes of the one before it that are
                        # still on their way here are dropped, not run on this reply's state.
                        engine.begin(int(meta["slot"]), int(meta["generation"]))
                answer, data = getattr(host, operation)(meta, payload)
            finally:
                if engine is not None:
                    engine.unlock()
            connection.send_bytes(json.dumps({"ok": True, **answer}).encode())
            connection.send_bytes(data)
        except (LlamaError, OSError, ValueError, AttributeError, AssertionError) as exc:
            connection.send_bytes(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}).encode())
            connection.send_bytes(b"")


class Channel:
    """One connection to a stage process, which runs one exchange at a time."""

    def __init__(self, process: subprocess.Popen[bytes], connection: Connection, stopped: threading.Event) -> None:
        self.process = process
        self.connection = connection
        self.lock = threading.Lock()
        # Shared by every connection to this process. A kill can take a while to land, and
        # until it does the pipe stays open and each call would sit out a deadline of its own.
        self.stopped = stopped

    def _send(self, payload: bytes) -> None:
        """Sends part of a call. Much of it can stick in the pipe with no way to time it out, so
        it goes from a thread of its own, which a stop can leave behind.
        """
        if len(payload) <= PIPE_BYTES:
            self.connection.send_bytes(payload)
            return
        failed: list[OSError] = []

        def send() -> None:
            try:
                self.connection.send_bytes(payload)
            except OSError as exc:
                failed.append(exc)

        sender = threading.Thread(target=send, daemon=True)
        sender.start()
        while sender.is_alive():
            sender.join(ANSWER_LOOK_SECONDS)
            if self.stopped.is_set():
                raise StageLost("the stage process was stopped", self)
        if failed:
            raise failed[0]

    def _answered(self, seconds: float | None) -> bool:
        """Waits for an answer, and says whether it came in time.

        It looks up now and then. A process that takes no kill leaves this pipe open, so a
        stop made through another connection would otherwise reach nobody waiting here. It
        counts those looks and reads no clock: a laptop that slept has run for none of that time.
        """
        left = seconds
        while not self.connection.poll(ANSWER_LOOK_SECONDS):
            if self.stopped.is_set():
                raise StageLost("the stage process was stopped", self)
            if left is not None:
                left -= ANSWER_LOOK_SECONDS
                if left <= 0:
                    return False
        return True

    def _exchange(self, operation: str, payload: bytes, meta: dict[str, Any], seconds: float | None) -> Reply:
        if self.stopped.is_set():
            raise StageLost("the stage process was stopped", self)
        try:
            self._send(json.dumps({"op": operation, **meta}).encode())
            self._send(payload)
            if not self._answered(seconds):
                self.stopped.set()
                self.process.kill()
                raise StageLost(f"the stage process did not answer {operation} in {seconds:.0f} s, so it was stopped", self)
            answer = json.loads(self.connection.recv_bytes())
            data = self.connection.recv_bytes()
        except (EOFError, OSError) as exc:
            raise StageLost(f"the stage process died (exit code {self.process.poll()})", self) from exc
        if not answer.pop("ok"):
            raise LlamaError(answer.get("error", "stage call failed"))
        return answer, data

    def call(self, operation: str, payload: bytes = b"", rows: int = 1, **meta: Any) -> Reply:
        """Runs an operation on the stage. ``rows`` says how long it may take; see ``seconds_for``."""
        with self.lock:
            return self._exchange(operation, payload, meta, seconds_for(operation, rows))

    def ask_if_idle(self, operation: str, **meta: Any) -> dict[str, Any] | None:
        """A quick question, or None if the stage is busy. For what must never wait on a pass."""
        if not self.lock.acquire(blocking=False):
            return None
        try:
            return self._exchange(operation, b"", meta, seconds_for(operation, 1))[0]
        finally:
            self.lock.release()

    def watch(self) -> None:
        """A second connection needs no watching: the host of its process does that."""

    def _say_goodbye(self) -> None:
        """Asks the far end to finish, unless that would mean waiting on a stage that has hung."""
        if self.stopped.is_set() or not self.lock.acquire(timeout=EXIT_SECONDS):
            return
        try:
            self._exchange("close", b"", {}, EXIT_SECONDS)
        except LlamaError:
            pass
        finally:
            self.lock.release()

    def close(self) -> None:
        self._say_goodbye()
        self.connection.close()


class StageHost(Channel):
    """The worker's handle on a stage process.

    ``low_priority`` starts it below normal, for a spare started beside a stage that serves:
    its start and imports then give way to the live stage's passes. ``serve_normally`` puts it
    back once it is to serve. Only Windows can do both; elsewhere a process cannot take back
    a priority it gave up, so a spare there starts at normal priority in an idle moment.
    """

    def __init__(self, low_priority: bool = False) -> None:
        key = secrets.token_bytes(32)
        self.quiet = threading.Event()
        self.listener = Listener(authkey=key)
        environment = {**os.environ, KEY_VARIABLE: key.hex()}
        package_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, environment.get("PYTHONPATH")]))
        lowered = low_priority and sys.platform == "win32"
        self.process = subprocess.Popen(
            [sys.executable, "-m", "isharded.llama_host", json.dumps(self.listener.address)],
            env=environment,
            stdin=subprocess.DEVNULL,
            creationflags=BELOW_NORMAL if lowered else 0,
        )
        self.lowered = lowered
        try:
            super().__init__(self.process, self._accept(), threading.Event())
        except LlamaError:
            self.process.kill()
            raise

    def serve_normally(self) -> None:
        """Back to normal priority, for a spare that is about to serve."""
        if self.lowered:
            with contextlib.suppress(psutil.Error, OSError):
                psutil.Process(self.process.pid).nice(psutil.NORMAL_PRIORITY_CLASS)
            self.lowered = False

    def channel(self) -> Channel:
        """Another connection to this process, with a thread and a handler of its own.

        A pass runs on the engine's thread and ctypes lets go of the GIL, so a model on a
        channel works beside the stage. The stage's gate does not cover it and need not:
        they are separate models and contexts. A crash in either still takes both.
        """
        with self.lock:
            self._exchange("channel", b"", {}, seconds_for("channel", 1))
            return Channel(self.process, self._accept(), self.stopped)

    def watch(self) -> None:
        """Stops the process if it ever goes quiet. Only after a load: a load on a machine short
        of memory can stall everything for a long time, and that is no reason to start again.
        """
        threading.Thread(target=self._listen, args=(self.channel(),), daemon=True).start()

    def _listen(self, channel: Channel) -> None:
        try:
            while not self.quiet.wait(PULSE_EVERY):
                channel.call("pulse")
        except LlamaError:
            pass  # gone, or stopped by that call. Whoever calls next hears it.
        channel.connection.close()

    def _accept(self) -> Connection:
        accepted: list[Connection] = []
        thread = threading.Thread(target=lambda: accepted.append(self.listener.accept()), daemon=True)
        thread.start()
        thread.join(CONNECT_SECONDS)
        if not accepted:
            raise LlamaError(f"the stage process did not start (exit code {self.process.poll()})")
        return accepted[0]

    def _kill_harder(self) -> None:
        """For a process a kill did not end. What it may have started goes first."""
        try:
            found = psutil.Process(self.process.pid)
            for item in [*found.children(recursive=True), found]:
                item.kill()
            self.process.wait(KILL_SECONDS)
        except (psutil.Error, OSError, subprocess.TimeoutExpired):
            pass

    def _end_process(self) -> None:
        """A process stuck in a driver can outlast any kill, and a reply must not wait for it.

        It holds its memory until the system lets it go. If that leaves no room, the new stage
        fails to load, and a stage that will not restart is how this node leaves the route.
        """
        try:
            self.process.kill()
            self.process.wait(KILL_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            self._kill_harder()

    def close(self) -> None:
        self.quiet.set()
        self._say_goodbye()
        try:
            self.process.wait(0 if self.stopped.is_set() else EXIT_SECONDS)
        except subprocess.TimeoutExpired:
            self._end_process()
        self.connection.close()
        self.listener.close()


class Spare:
    """One primed stage process a worker keeps ready, with its library and devices up and no model.

    Starting a stage process costs a Python start, imports, the library and a device context:
    0.65-1.5 s measured in the handover prototypes, and a live stage beside it stalled about
    100 ms while the child imported. A spare pays that once, in an idle moment and below normal
    priority, so a stage that changes or a stage process that died loads straight away.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.host: StageHost | None = None
        self.folder: str | None = None
        self.starting = False
        self.started = 0
        self.taken = 0
        # How long the last spare took to come up, start to primed.
        self.prime_seconds: float | None = None

    def ready(self) -> bool:
        with self.lock:
            return self.host is not None and self.host.process.poll() is None

    def prime(self, folder: str, extra_paths: list[str]) -> bool:
        """Starts a spare and brings it up. Blocks; a worker runs this off its loop. Says whether
        a new one came up: one already there, or one already starting, is left alone.
        """
        with self.lock:
            if self.starting or (self.host is not None and self.host.process.poll() is None):
                return False
            self.starting = True
        began = time.perf_counter()
        host: StageHost | None = None
        try:
            host = StageHost(low_priority=True)
            answer, _ = host.call("prime", folder=folder, extra_paths=list(extra_paths))
        except LlamaError:
            if host is not None:
                host.close()
            with self.lock:
                self.starting = False
            return False
        with self.lock:
            self.starting = False
            old, self.host, self.folder = self.host, host, answer.get("folder", folder)
            self.started += 1
            self.prime_seconds = round(time.perf_counter() - began, 3)
        if old is not None:
            old.close()
        return True

    def take(self, folder: str) -> StageHost | None:
        """The spare, if it is alive and has this folder's library, now at normal priority."""
        with self.lock:
            host, self.host = self.host, None
            same = self.folder is not None and Path(self.folder).resolve() == Path(folder).resolve()
        if host is None:
            return None
        if host.process.poll() is not None or not same:
            host.close()
            return None
        host.serve_normally()
        with self.lock:
            self.taken += 1
        return host

    def report(self) -> dict[str, Any]:
        with self.lock:
            return {"ready": self.host is not None and self.host.process.poll() is None, "started": self.started, "taken": self.taken, "prime_seconds": self.prime_seconds}

    def close(self) -> None:
        with self.lock:
            host, self.host = self.host, None
        if host is not None:
            host.close()


SPARE = Spare()


if __name__ == "__main__":
    target = json.loads(sys.argv[1])
    run(tuple(target) if isinstance(target, list) else target, bytes.fromhex(os.environ[KEY_VARIABLE]))
