"""The native token path, where its library is present. Everything works without it.

The engine is a Rust library loaded into each stage process. It takes over the passes that
happen for every token, so a token crosses a stage with no interpreter and no HTTP. Anything
it cannot do stays on the Python path, and a failure on it raises the same errors the HTTP
path does, so a reply recovers the same way.

This module must not import torch: it loads inside the stage process.
"""

from __future__ import annotations

import asyncio
import ctypes as C
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import sys
from typing import Any

import numpy as np

ABI_VERSION = 1
MAGIC = 0x5369
WIRE_VERSION = 1
HEADER = struct.Struct("<HBBI")
KIND_HELLO, KIND_HELLO_OK, KIND_AUTH, KIND_PASS, KIND_ACK, KIND_RESULT, KIND_NACK, KIND_PING, KIND_PONG = range(1, 10)
ROLE_LOCAL = 3
FLAG_VERIFY, FLAG_TOKENS = 1, 2
NO_ROLLBACK = 0xFFFFFFFF
SERVER_LABEL = b"isharded link accept"
CLIENT_LABEL = b"isharded link dial"
REFUSE_RETRY, REFUSE_EPOCH, REFUSE_FAILED_STAGE, REFUSE_STAGE_ERROR = 1, 2, 3, 4
TEXT_BYTES = 8192
MAX_FRAME_BYTES = 1 << 20  # answers and refusals are small


class TensorApi(C.Structure):
    _fields_ = [("name", C.c_void_p), ("bytes", C.c_void_p), ("get", C.c_void_p)]


class StageApi(C.Structure):
    _fields_ = [
        ("decode", C.c_void_p),
        ("logits_ith", C.c_void_p),
        ("seq_rm", C.c_void_p),
        ("context", C.c_void_p),
        ("memory", C.c_void_p),
        ("width", C.c_int32),
        ("vocab", C.c_int32),
        ("last", C.c_bool),
        ("wire_bf16", C.c_bool),
        ("table", C.c_void_p),
        ("table_rows", C.c_int64),
        ("row_bytes", C.c_int64),
        ("to_float", C.c_void_p),
    ]


def library_name() -> str:
    if sys.platform == "win32":
        return "isharded_engine.dll"
    return "libisharded_engine.dylib" if sys.platform == "darwin" else "libisharded_engine.so"


def find_library() -> Path | None:
    """Where the engine is, or None. ``ISHARDED_ENGINE=0`` turns it off."""
    if os.getenv("ISHARDED_ENGINE", "1") == "0":
        return None
    configured = os.getenv("ISHARDED_ENGINE_LIB")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).parent / "native" / library_name())
    return next((path for path in candidates if path.is_file()), None)


def _address(function: Any) -> int:
    return C.cast(function, C.c_void_p).value


class Engine:
    """One stage's engine, in the stage process."""

    def __init__(self, path: Path, target: bytes, library: Any) -> None:
        self.lib = C.CDLL(str(path))
        lib = self.lib
        lib.isharded_engine_abi.restype = C.c_uint32
        if lib.isharded_engine_abi() != ABI_VERSION:
            raise OSError(f"{path} is a different version of the engine")
        lib.isharded_engine_new.argtypes = [C.c_char_p, C.POINTER(TensorApi)]
        lib.isharded_engine_new.restype = C.c_void_p
        for name in ("capture_begin", "lock", "unlock", "close_ring"):
            function = getattr(lib, f"isharded_engine_{name}")
            function.argtypes = [C.c_void_p]
            function.restype = None
        lib.isharded_engine_capture_read.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        lib.isharded_engine_capture_read.restype = C.c_size_t
        lib.isharded_engine_attach.argtypes = [C.c_void_p, C.POINTER(StageApi)]
        lib.isharded_engine_attach.restype = None
        lib.isharded_engine_listen.argtypes = [C.c_void_p, C.c_char_p, C.c_uint16, C.c_char_p, C.c_size_t]
        lib.isharded_engine_listen.restype = C.c_int32
        lib.isharded_engine_open.argtypes = [C.c_void_p, C.c_char_p]
        lib.isharded_engine_open.restype = C.c_int32
        lib.isharded_engine_connect.argtypes = [C.c_void_p, C.c_char_p]
        lib.isharded_engine_connect.restype = C.c_int32
        lib.isharded_engine_stats.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
        lib.isharded_engine_stats.restype = C.c_int32
        base = library.ggml_base
        tensors = TensorApi(_address(base.ggml_get_name), _address(base.ggml_nbytes), _address(base.ggml_backend_tensor_get))
        self.handle = lib.isharded_engine_new(target, C.byref(tensors))
        if not self.handle:
            raise OSError("the engine could not start")
        self.library = library
        self.eval_callback = _address(lib.isharded_engine_eval)
        self.api: StageApi | None = None
        self.table_anchor: Any = None

    @classmethod
    def create(cls, target: bytes, library: Any) -> "Engine | None":
        path = find_library()
        if path is None:
            return None
        try:
            return cls(path, target, library)
        except (OSError, AttributeError) as exc:
            print(f"the native engine is not available: {exc}", flush=True)
            return None

    def attach(self, stage: Any, wire_bf16: bool, table: Any = None, table_rows: int = 0, row_bytes: int = 0, to_float: int | None = None) -> None:
        llama = self.library.llama
        self.table_anchor = None if table is None else np.frombuffer(table, dtype=np.uint8)
        self.api = StageApi(
            decode=_address(llama.llama_decode),
            logits_ith=_address(llama.llama_get_logits_ith),
            seq_rm=_address(llama.llama_memory_seq_rm),
            context=stage.context,
            memory=stage.memory,
            width=stage.hidden_size,
            vocab=stage.vocab_size,
            last=stage.last,
            wire_bf16=wire_bf16,
            table=None if self.table_anchor is None else self.table_anchor.ctypes.data,
            table_rows=table_rows,
            row_bytes=row_bytes,
            to_float=to_float,
        )
        self.lib.isharded_engine_attach(self.handle, C.byref(self.api))

    def capture_begin(self) -> None:
        self.lib.isharded_engine_capture_begin(self.handle)

    def capture_read(self, expected: int) -> np.ndarray:
        rows = np.empty(expected, dtype=np.float32)
        size = self.lib.isharded_engine_capture_read(self.handle, rows.ctypes.data, expected)
        if size > expected:
            rows = np.empty(size, dtype=np.float32)
            self.lib.isharded_engine_capture_read(self.handle, rows.ctypes.data, size)
        return rows[:size]

    def lock(self) -> None:
        self.lib.isharded_engine_lock(self.handle)

    def unlock(self) -> None:
        self.lib.isharded_engine_unlock(self.handle)

    def listen(self, host: str, port: int) -> dict[str, Any]:
        out = C.create_string_buffer(TEXT_BYTES)
        if self.lib.isharded_engine_listen(self.handle, host.encode(), port, out, TEXT_BYTES) < 0:
            raise OSError(out.value.decode("utf-8", "replace") or "the engine could not listen")
        return json.loads(out.value)

    def open(self, epoch: int, index: int, count: int, key: bytes, ack_timeout_ms: int, silence_timeout_ms: int) -> None:
        plan = {"epoch": epoch, "index": index, "count": count, "key": list(key), "ack_timeout_ms": ack_timeout_ms, "silence_timeout_ms": silence_timeout_ms}
        if self.lib.isharded_engine_open(self.handle, json.dumps(plan).encode()) != 0:
            raise OSError("the engine refused its place in the route")

    def connect(self, links: dict[str, Any]) -> None:
        """Starts dialling, and returns at once. ``stats`` says how it went."""
        if self.lib.isharded_engine_connect(self.handle, json.dumps(links).encode()) != 0:
            raise OSError("the engine could not read its links")

    def close_ring(self) -> None:
        self.lib.isharded_engine_close_ring(self.handle)

    def stats(self) -> dict[str, Any]:
        out = C.create_string_buffer(TEXT_BYTES)
        if self.lib.isharded_engine_stats(self.handle, out, TEXT_BYTES) < 0:
            return {}
        return json.loads(out.value)


class EngineRefusal(Exception):
    """A pass did not finish, and why."""

    def __init__(self, refusal: int, stage: int, message: str) -> None:
        super().__init__(message)
        self.refusal = refusal
        self.stage = stage
        self.message = message


def _frame(kind: int, body: bytes) -> bytes:
    return HEADER.pack(MAGIC, WIRE_VERSION, kind, len(body)) + body


def _proof(key: bytes, label: bytes, role: int, epoch: int, client_nonce: bytes, server_nonce: bytes) -> bytes:
    message = label + bytes([role]) + struct.pack("<Q", epoch) + client_nonce + server_nonce
    return hmac.new(key, message, hashlib.sha256).digest()


def pass_frame(
    identity: int,
    epoch: int,
    slot: int,
    position: int,
    rollback_to: int | None,
    tokens: list[int],
    guesses: list[int],
    verify: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
) -> bytes:
    flags = FLAG_TOKENS | (FLAG_VERIFY if verify else 0)
    rollback = NO_ROLLBACK if rollback_to is None else rollback_to
    body = struct.pack("<QQIIIBffIBQ", identity, epoch, slot, position, rollback, flags, temperature, top_p, top_k, seed is not None, (seed or 0) & (2**64 - 1))
    body += struct.pack(f"<I{len(guesses)}I", len(guesses), *guesses)
    body += struct.pack(f"<I{len(tokens)}I", len(tokens), *tokens)
    body += struct.pack("<II", 0, 0)
    return _frame(KIND_PASS, body)


class EngineClient:
    """The worker's end of the entry stage's local door. Many replies share it."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.waiting: dict[int, asyncio.Future[list[int]]] = {}
        # What each slot has out. A pipeline can walk away from a pass, and a stage cannot.
        self.out: dict[int, set[asyncio.Future[list[int]]]] = {}
        self.next_id = 1
        self.alive = True
        self.task: asyncio.Task[None] | None = None

    @classmethod
    async def open(cls, port: int, key: bytes) -> "EngineClient":
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        raw = writer.get_extra_info("socket")
        if raw is not None:
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        nonce = secrets.token_bytes(32)
        writer.write(_frame(KIND_HELLO, bytes([ROLE_LOCAL]) + struct.pack("<Q", 0) + nonce))
        await writer.drain()
        kind, body = await cls._read_frame(reader)
        if kind != KIND_HELLO_OK or len(body) != 64:
            raise OSError("the engine did not greet this worker")
        theirs, mac = body[:32], body[32:]
        if not hmac.compare_digest(mac, _proof(key, SERVER_LABEL, ROLE_LOCAL, 0, nonce, theirs)):
            raise OSError("the engine does not hold this worker's key")
        writer.write(_frame(KIND_AUTH, _proof(key, CLIENT_LABEL, ROLE_LOCAL, 0, nonce, theirs)))
        await writer.drain()
        client = cls(reader, writer)
        client.task = asyncio.create_task(client._read_loop())
        return client

    @staticmethod
    async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
        magic, version, kind, length = HEADER.unpack(await reader.readexactly(HEADER.size))
        if magic != MAGIC or version != WIRE_VERSION or length > MAX_FRAME_BYTES:
            raise OSError("the engine sent something that is not a frame")
        return kind, await reader.readexactly(length)

    async def _read_loop(self) -> None:
        reason: Exception = EngineRefusal(REFUSE_RETRY, 0, "the link to the engine closed")
        try:
            while True:
                kind, body = await self._read_frame(self.reader)
                if kind == KIND_RESULT:
                    identity, _epoch, count = struct.unpack_from("<QQI", body)
                    tokens = list(struct.unpack_from(f"<{count}I", body, 20))
                    future = self.waiting.pop(identity, None)
                    if future is not None and not future.done():
                        future.set_result(tokens)
                elif kind == KIND_NACK:
                    identity, _epoch, refusal, stage, length = struct.unpack_from("<QQBHI", body)
                    message = body[23 : 23 + length].decode("utf-8", "replace")
                    refused = EngineRefusal(refusal, stage, message)
                    if identity:
                        future = self.waiting.pop(identity, None)
                        if future is not None and not future.done():
                            future.set_exception(refused)
                    else:
                        # A link failed, and nothing can say which passes it took with it.
                        reason = refused
                        return
        except (OSError, asyncio.IncompleteReadError, struct.error) as exc:
            reason = EngineRefusal(REFUSE_RETRY, 0, f"the link to the engine failed: {exc}")
        finally:
            self.alive = False
            for future in self.waiting.values():
                if not future.done():
                    future.set_exception(reason)
            self.waiting.clear()
            self.writer.close()

    def send(self, **fields: Any) -> asyncio.Future[list[int]]:
        """Sends a pass and does not wait. The answer lands in the future, wanted or not."""
        if not self.alive:
            raise EngineRefusal(REFUSE_RETRY, 0, "the link to the engine closed")
        identity = self.next_id
        self.next_id += 1
        future: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
        self.waiting[identity] = future
        out = self.out.setdefault(fields["slot"], set())
        out.add(future)
        future.add_done_callback(lambda done: self._landed(out, done))
        self.writer.write(pass_frame(identity, **fields))
        return future

    @staticmethod
    def _landed(out: set[asyncio.Future[list[int]]], future: asyncio.Future[list[int]]) -> None:
        out.discard(future)
        if not future.cancelled():
            future.exception()  # nobody may be waiting, and that is fine

    async def run(self, deadline: float, **fields: Any) -> list[int]:
        """Sends a pass and waits for its answer."""
        return await asyncio.wait_for(self.send(**fields), deadline)

    async def settled(self, slot: int, seconds: float) -> bool:
        """Waits until nothing of this slot's is out, and says whether that happened."""
        out = self.out.get(slot)
        if out:
            await asyncio.wait(set(out), timeout=seconds)
        return not out

    def close(self) -> None:
        self.alive = False
        if self.task is not None:
            self.task.cancel()
        self.writer.close()
