"""A separate process that runs one llama.cpp stage.

PyTorch and llama.cpp each load their own OpenMP runtime, and the two abort when
they meet in a process. Keeping them apart also means a native crash cannot take
the worker with it, and that unloading is a process exit that frees everything.

This module must not import torch, directly or through other iSharded modules.
"""

from __future__ import annotations

from collections import OrderedDict
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

KEY_VARIABLE = "ISHARDED_STAGE_KEY"
CONNECT_SECONDS = 60.0
EXIT_SECONDS = 15.0
# Encoded images kept for the next turn. A conversation resends its images every time.
ENCODED_IMAGE_BYTES = 128 * 1024 * 1024


class _Host:
    def __init__(self) -> None:
        self.library: LlamaLibrary | None = None
        self.stage: LlamaStage | None = None
        self.embedding: TokenEmbedding | None = None
        self.table: mmap.mmap | None = None
        self.table_view: memoryview | None = None
        # One saved conversation per sequence.
        self.checkpoints: dict[int, tuple[str, np.ndarray]] = {}
        self.scratch = 1
        self.projector: Any = None
        self.encoded: OrderedDict[str, tuple[dict[str, Any], bytes]] = OrderedDict()

    def load(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
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
        )
        table = meta.get("table")
        if table:
            with open(table["path"], "rb") as handle:
                self.table = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self.table_view = memoryview(self.table)[table["offset"] : table["offset"] + table["length"]]
            self.embedding = TokenEmbedding(self.library, self.table_view, table["ggml_type"], self.stage.hidden_size)
        return {"hidden_size": self.stage.hidden_size, "vocab_size": self.stage.vocab_size}, b""

    def embed(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        if self.embedding is None:
            raise LlamaError("this stage has no embedding table")
        return {}, self.embedding.rows(meta["tokens"]).tobytes()

    def pool(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None and self.embedding is not None
        inputs = meta["inputs"]
        rows = self.embedding.rows([token for item in inputs for token in item])
        pooled = self.stage.pool(rows, [len(item) for item in inputs])
        return {"width": self.stage.output_width}, pooled.tobytes()

    def forward(self, meta: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        hidden = np.frombuffer(payload, dtype=np.float32).reshape(-1, self.stage.hidden_size)
        places = meta.get("places")
        places = None if places is None else np.array(places, dtype=np.int32)
        return {}, self.stage.forward(hidden, meta["position"], meta.get("slot", 0), places).tobytes()

    def forward_many(self, meta: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        hidden = np.frombuffer(payload, dtype=np.float32).reshape(-1, self.stage.hidden_size)
        return {}, self.stage.forward_many(hidden, meta["positions"], meta["slots"]).tobytes()

    def clear(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        self.stage.clear(meta.get("slot"))
        return {}, b""

    def save(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        slot = meta.get("slot", 0)
        self.checkpoints[slot] = (meta["key"], self.stage.save_state(slot))
        return {"bytes": int(self.checkpoints[slot][1].size)}, b""

    def restore(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        slot = meta.get("slot", 0)
        held = self.checkpoints.get(slot)
        if held is None or held[0] != meta["key"]:
            raise LlamaError(f"no checkpoint {meta['key']} here")
        self.stage.restore_state(held[1], slot)
        return {}, b""

    def commit(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        self.stage.decode_tokens(meta["tokens"], meta["position"], sequence_id=meta.get("slot", 0))
        return {}, b""

    def truncate(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        self.stage.truncate(meta["keep"], meta.get("slot", 0))
        return {}, b""

    def verify(self, meta: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        hidden = np.frombuffer(payload, dtype=np.float32).reshape(-1, self.stage.hidden_size)
        sampling = {key: meta[key] for key in ("guesses", "temperature", "top_p", "top_k", "seed")}
        sampling["slot"] = meta.get("slot", 0)
        return {"tokens": self.stage.verify(hidden, meta["position"], **sampling)}, b""

    def load_draft(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        self.library = LlamaLibrary.load(meta["folder"], tuple(meta["extra_paths"]))
        self.stage = LlamaStage(
            self.library,
            meta["path"],
            last=True,
            context_tokens=meta["context_tokens"],
            gpu_layers=meta["gpu_layers"],
            threads=meta["threads"],
            sequences=meta.get("slots", 1) + 1,
            unified_cache=True,
            map_file=False,
        )
        self.scratch = meta.get("slots", 1)
        return {}, b""

    def load_projector(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        from .mtmd import Projector

        self.library = LlamaLibrary.load(meta["folder"], tuple(meta["extra_paths"]))
        self.projector = Projector(self.library, meta["projector"], meta["stage"], meta["width"], meta["threads"], meta["use_gpu"])
        return {}, b""

    def measure_image(self, _: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        shape = self.projector.measure(payload)
        return {"rows": shape.rows, "span": shape.span}, b""

    def encode_image(self, meta: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        digest = meta["digest"]
        if digest not in self.encoded:
            shape, rows = self.projector.encode(payload)
            self.encoded[digest] = {"rows": shape.rows, "span": shape.span}, shape.places.tobytes() + rows.tobytes()
        self.encoded.move_to_end(digest)
        while len(self.encoded) > 1 and sum(len(data) for _, data in self.encoded.values()) > ENCODED_IMAGE_BYTES:
            self.encoded.popitem(last=False)
        return self.encoded[digest]

    def guess(self, meta: dict[str, Any], _: bytes) -> tuple[dict[str, Any], bytes]:
        assert self.stage is not None
        tokens = self.stage.guess(meta["commit"], meta["position"], meta["count"], meta.get("slot", 0), self.scratch)
        return {"tokens": tokens}, b""


def run(address: Any, key: bytes) -> None:
    connection = Client(address, authkey=key)
    host = _Host()
    while True:
        try:
            meta = json.loads(connection.recv_bytes())
            payload = connection.recv_bytes()
        except EOFError:
            return  # the worker went away, so this stage has no purpose left
        operation = meta.pop("op")
        if operation == "close":
            connection.send_bytes(b'{"ok":true}')
            connection.send_bytes(b"")
            return
        try:
            answer, data = getattr(host, operation)(meta, payload)
            connection.send_bytes(json.dumps({"ok": True, **answer}).encode())
            connection.send_bytes(data)
        except (LlamaError, OSError, ValueError, AttributeError, AssertionError) as exc:
            connection.send_bytes(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}).encode())
            connection.send_bytes(b"")


class StageHost:
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
        self.connection = self._accept()
        self.lock = threading.Lock()

    def _accept(self) -> Connection:
        accepted: list[Connection] = []
        thread = threading.Thread(target=lambda: accepted.append(self.listener.accept()), daemon=True)
        thread.start()
        thread.join(CONNECT_SECONDS)
        if not accepted:
            self.process.kill()
            raise LlamaError(f"the stage process did not start (exit code {self.process.poll()})")
        return accepted[0]

    def call(self, operation: str, payload: bytes = b"", **meta: Any) -> tuple[dict[str, Any], bytes]:
        with self.lock:
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
