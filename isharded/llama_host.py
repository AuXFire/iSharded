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
from typing import Any

import numpy as np

from .llamacpp import LlamaError, LlamaLibrary, LlamaStage, StageLost, TokenEmbedding
from .statestore import HISTORY, PARTIAL, StateStore

# What every operation answers with: fields for the reply, and any raw bytes.
Reply = tuple[dict[str, Any], bytes]

KEY_VARIABLE = "ISHARDED_STAGE_KEY"
CONNECT_SECONDS = 60.0
EXIT_SECONDS = 15.0
# Encoded images kept for the next turn. A conversation resends its images every time.
ENCODED_IMAGE_BYTES = 128 * 1024 * 1024
# States a draft model keeps. It is small, and a draft that loses one just reads more.
DRAFT_STATE_BYTES = 256 * 1024 * 1024
# Operations that touch only the engine's links, and so need not wait for a running pass.
UNGATED = {"load", "ring_open", "ring_connect", "ring_close", "ring_stats"}


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

    def load(self, meta: dict[str, Any], _: bytes) -> Reply:
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
            )
        return {"hidden_size": self.stage.hidden_size, "vocab_size": self.stage.vocab_size, "engine": self.stage.engine is not None}, b""

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
        return {}, b""

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
        tokens = self.stage.guess(
            meta["commit"], meta["position"], meta["count"], slot, self.scratch + slot, meta.get("fresh", ()), meta.get("restart", True)
        )
        return {"tokens": tokens}, b""

    @property
    def engine(self) -> Any:
        return None if self.stage is None else self.stage.engine

    def ring_open(self, meta: dict[str, Any], _: bytes) -> Reply:
        if self.engine is None:
            return {"engine": False}, b""
        door = self.engine.listen(meta["host"], meta["port"])
        self.engine.open(meta["epoch"], meta["index"], meta["count"], bytes.fromhex(meta["key"]), meta["ack_timeout_ms"], meta["silence_timeout_ms"])
        return {"engine": True, **door}, b""

    def ring_connect(self, meta: dict[str, Any], _: bytes) -> Reply:
        assert self.engine is not None
        self.engine.connect(meta["links"])
        return {}, b""

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

    def __init__(self, process: subprocess.Popen[bytes], connection: Connection) -> None:
        self.process = process
        self.connection = connection
        self.lock = threading.Lock()

    def _exchange(self, operation: str, payload: bytes, meta: dict[str, Any]) -> Reply:
        try:
            self.connection.send_bytes(json.dumps({"op": operation, **meta}).encode())
            self.connection.send_bytes(payload)
            answer = json.loads(self.connection.recv_bytes())
            data = self.connection.recv_bytes()
        except (EOFError, OSError) as exc:
            raise StageLost(f"the stage process died (exit code {self.process.poll()})", self) from exc
        if not answer.pop("ok"):
            raise LlamaError(answer.get("error", "stage call failed"))
        return answer, data

    def call(self, operation: str, payload: bytes = b"", **meta: Any) -> Reply:
        with self.lock:
            return self._exchange(operation, payload, meta)

    def ask_if_idle(self, operation: str, **meta: Any) -> dict[str, Any] | None:
        """A quick question, or None if the stage is busy. For what must never wait on a pass."""
        if not self.lock.acquire(blocking=False):
            return None
        try:
            return self._exchange(operation, b"", meta)[0]
        finally:
            self.lock.release()

    def close(self) -> None:
        try:
            self.call("close")
        except LlamaError:
            pass
        self.connection.close()


class StageHost(Channel):
    """The worker's handle on a stage process."""

    def __init__(self) -> None:
        key = secrets.token_bytes(32)
        self.listener = Listener(authkey=key)
        environment = {**os.environ, KEY_VARIABLE: key.hex()}
        package_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, environment.get("PYTHONPATH")]))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "isharded.llama_host", json.dumps(self.listener.address)],
            env=environment,
            stdin=subprocess.DEVNULL,
        )
        try:
            super().__init__(self.process, self._accept())
        except LlamaError:
            self.process.kill()
            raise

    def channel(self) -> Channel:
        """Another connection to this process, with a thread and a handler of its own.

        A pass runs on the engine's thread and ctypes lets go of the GIL, so a model on a
        channel works beside the stage. The stage's gate does not cover it and need not:
        they are separate models and contexts. A crash in either still takes both.
        """
        with self.lock:
            self._exchange("channel", b"", {})
            return Channel(self.process, self._accept())

    def _accept(self) -> Connection:
        accepted: list[Connection] = []
        thread = threading.Thread(target=lambda: accepted.append(self.listener.accept()), daemon=True)
        thread.start()
        thread.join(CONNECT_SECONDS)
        if not accepted:
            raise LlamaError(f"the stage process did not start (exit code {self.process.poll()})")
        return accepted[0]

    def close(self) -> None:
        try:
            self.call("close")
            self.process.wait(EXIT_SECONDS)
        except (LlamaError, subprocess.TimeoutExpired):
            pass
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(EXIT_SECONDS)
        self.connection.close()
        self.listener.close()


if __name__ == "__main__":
    target = json.loads(sys.argv[1])
    run(tuple(target) if isinstance(target, list) else target, bytes.fromhex(os.environ[KEY_VARIABLE]))
