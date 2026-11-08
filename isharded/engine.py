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
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import sys
import time
from typing import Any

import numpy as np

ABI_VERSION = 5
MAGIC = 0x5369
WIRE_VERSION = 3
HEADER = struct.Struct("<HBBI")
RESULT_HEAD = struct.Struct("<QQQII")
NACK_HEAD = struct.Struct("<QQQBHI")
KIND_HELLO, KIND_HELLO_OK, KIND_AUTH, KIND_PASS, KIND_ACK, KIND_RESULT, KIND_NACK, KIND_PING, KIND_PONG = range(1, 10)
ROLE_LOCAL = 3
FLAG_VERIFY, FLAG_TOKENS, FLAG_KEEP = 1, 2, 8
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
        ("state_size", C.c_void_p),
        ("state_get", C.c_void_p),
        ("state_set", C.c_void_p),
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
        ("keep_limit", C.c_uint32),
    ]


# Each call into the library, after ``isharded_engine_``: its arguments, and what it returns.
SIGNATURES: dict[str, tuple[list[Any], Any]] = {
    "new": ([C.c_char_p, C.POINTER(TensorApi)], C.c_void_p),
    "capture_begin": ([C.c_void_p], None),
    "lock": ([C.c_void_p], None),
    "unlock": ([C.c_void_p], None),
    "close_ring": ([C.c_void_p], None),
    "forget": ([C.c_void_p, C.c_int32], None),
    "begin": ([C.c_void_p, C.c_uint32, C.c_uint64], None),
    "capture_read": ([C.c_void_p, C.c_void_p, C.c_size_t], C.c_size_t),
    "attach": ([C.c_void_p, C.POINTER(StageApi)], None),
    "listen": ([C.c_void_p, C.c_char_p, C.c_uint16, C.c_char_p, C.c_size_t], C.c_int32),
    "open": ([C.c_void_p, C.c_char_p], C.c_int32),
    "connect": ([C.c_void_p, C.c_char_p], C.c_int32),
    "stats": ([C.c_void_p, C.c_char_p, C.c_size_t], C.c_int32),
    "probe": ([C.c_void_p, C.c_uint32, C.c_char_p, C.c_size_t], C.c_int32),
}


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
        for name, (arguments, answer) in SIGNATURES.items():
            function = getattr(lib, f"isharded_engine_{name}")
            function.argtypes = arguments
            function.restype = answer
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
    def create(cls, target: bytes, library: Any) -> Engine | None:
        path = find_library()
        if path is None:
            return None
        try:
            return cls(path, target, library)
        except (OSError, AttributeError) as exc:
            print(f"the native engine is not available: {exc}", flush=True)
            return None

    def attach(
        self, stage: Any, wire_bf16: bool, table: Any = None, table_rows: int = 0, row_bytes: int = 0, to_float: int | None = None, keep_limit: int = 2
    ) -> None:
        llama = self.library.llama
        self.table_anchor = None if table is None else np.frombuffer(table, dtype=np.uint8)
        self.api = StageApi(
            decode=_address(llama.llama_decode),
            logits_ith=_address(llama.llama_get_logits_ith),
            seq_rm=_address(llama.llama_memory_seq_rm),
            state_size=_address(llama.llama_state_seq_get_size_ext),
            state_get=_address(llama.llama_state_seq_get_data_ext),
            state_set=_address(llama.llama_state_seq_set_data_ext),
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
            keep_limit=keep_limit,
        )
        self.lib.isharded_engine_attach(self.handle, C.byref(self.api))

    def probe(self, slots: int) -> dict[str, Any]:
        """How far merged passes round on this stage as rows run alone, over this many slots.

        It runs passes of its own on every slot and empties them all afterwards, so it must
        come before any reply. ``exact_merge_limit`` is the most companions a row had and
        still came out byte for byte the same; ``verify_exact`` says whether one reply's rows
        in one pass came out as they do one at a time.
        """
        out = C.create_string_buffer(TEXT_BYTES)
        if self.lib.isharded_engine_probe(self.handle, slots, out, TEXT_BYTES) < 0:
            raise OSError(out.value.decode("utf-8", "replace") or "the probe failed")
        return json.loads(out.value)

    def forget(self, slot: int | None) -> None:
        """A slot's history changed, so what its rewinds could go back to is wrong. None means every slot."""
        self.lib.isharded_engine_forget(self.handle, -1 if slot is None else slot)

    def begin(self, slot: int, generation: int) -> None:
        """A slot now holds this reply. Passes of any other still on their way to it are dropped.

        The caller holds the engine's lock, so no pass is running while this changes.
        """
        self.lib.isharded_engine_begin(self.handle, slot, generation & (2**64 - 1))

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

    def open(
        self, epoch: int, index: int, count: int, keys: dict[str, bytes], ack_timeout_ms: int, silence_timeout_ms: int, pass_ms: int = 0, row_ms: int = 0, merge_passes: bool = True,
        merge_most: int | None = None,
    ) -> None:
        """Takes a place in a route, with the keys of this stage's own links: back, next, closing.

        ``merge_most`` is the most passes one trip may hold. None leaves it to the engine.
        """
        plan = {
            "merge_passes": merge_passes,
            "merge_most": merge_most,
            "epoch": epoch,
            "index": index,
            "count": count,
            "keys": {name: list(key) for name, key in keys.items()},
            "ack_timeout_ms": ack_timeout_ms,
            "silence_timeout_ms": silence_timeout_ms,
            "pass_ms": pass_ms,
            "row_ms": row_ms,
        }
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
    generation: int,
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
    keep: bool = False,
) -> bytes:
    flags = FLAG_TOKENS | (FLAG_VERIFY if verify else 0) | (FLAG_KEEP if keep else 0)
    rollback = NO_ROLLBACK if rollback_to is None else rollback_to
    body = struct.pack(
        f"<QQQIIIBffIBQI{len(guesses)}II{len(tokens)}III",
        identity, generation, epoch, slot, position, rollback, flags, temperature, top_p, top_k, seed is not None, (seed or 0) & (2**64 - 1),
        len(guesses), *guesses,
        len(tokens), *tokens,
        0, 0,
    )  # fmt: skip
    return _frame(KIND_PASS, body)


@dataclass(slots=True)
class Flight:
    """A pass that is out on the ring."""

    future: asyncio.Future[list[int]]
    slot: int
    generation: int
    sent: float


class EngineClient:
    """The worker's end of the entry stage's local door. Many replies share it."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.waiting: dict[int, Flight] = {}
        # The newest pass of each slot. Passes run in order, so when it is back all of them are.
        self.newest: dict[int, asyncio.Future[list[int]]] = {}
        # The reply on each slot. A slot that has settled has none, so its next pass starts one.
        self.generations: dict[int, int] = {}
        # What a pass has been taking, there and back, or None until one has come back.
        self.round_trip: float | None = None
        self.next_id = 1
        self.alive = True
        self.task: asyncio.Task[None] | None = None

    @classmethod
    async def open(cls, port: int, key: bytes) -> EngineClient:
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
                    # The seal after the tokens was for the entry, which has checked it.
                    identity, generation, _epoch, _slot, count = RESULT_HEAD.unpack_from(body)
                    tokens = list(struct.unpack_from(f"<{count}I", body, RESULT_HEAD.size))
                    flight = self._landing(identity, generation)
                    if flight is not None:
                        flight.future.set_result(tokens)
                elif kind == KIND_NACK:
                    identity, generation, _epoch, refusal, stage, length = NACK_HEAD.unpack_from(body)
                    message = body[NACK_HEAD.size : NACK_HEAD.size + length].decode("utf-8", "replace")
                    refused = EngineRefusal(refusal, stage, message)
                    if identity:
                        flight = self._landing(identity, generation)
                        if flight is not None:
                            flight.future.set_exception(refused)
                    else:
                        # A link failed, and nothing can say which passes it took with it.
                        reason = refused
                        return
        except (OSError, asyncio.IncompleteReadError, struct.error) as exc:
            reason = EngineRefusal(REFUSE_RETRY, 0, f"the link to the engine failed: {exc}")
        finally:
            self.alive = False
            for flight in self.waiting.values():
                if not flight.future.done():
                    flight.future.set_exception(reason)
            self.waiting.clear()
            self.writer.close()

    def _landing(self, identity: int, generation: int) -> Flight | None:
        """The pass this frame answers. Anything else is left over, or a repeat, and is dropped."""
        flight = self.waiting.get(identity)
        if flight is None or flight.generation != generation or flight.future.done():
            return None
        del self.waiting[identity]
        took = time.perf_counter() - flight.sent
        self.round_trip = took if self.round_trip is None else (self.round_trip * 7 + took) / 8
        return flight

    def send(self, **fields: Any) -> asyncio.Future[list[int]]:
        """Sends a pass and does not wait. Cancelling the future says the answer is not wanted."""
        if not self.alive:
            raise EngineRefusal(REFUSE_RETRY, 0, "the link to the engine closed")
        identity = self.next_id
        self.next_id += 1
        future: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
        slot = fields["slot"]
        given = fields.pop("generation", 0)
        generation = self.generations.get(slot)
        if given and given != generation:
            # The entry's own number for this reply, the one its restores and first pass carried.
            generation = self.generations[slot] = given
        elif generation is None:
            generation = self.generations[slot] = secrets.randbits(64) or 1  # zero means every pass
        self.waiting[identity] = Flight(future, slot, generation, time.perf_counter())
        self.newest[slot] = future
        future.add_done_callback(lambda done: self._landed(identity, done))
        self.writer.write(pass_frame(identity, generation, **fields))
        return future

    def _landed(self, identity: int, future: asyncio.Future[list[int]]) -> None:
        self.waiting.pop(identity, None)
        if not future.cancelled():
            future.exception()  # nobody may be waiting, and that is fine

    async def run(self, deadline: float, **fields: Any) -> list[int]:
        """Sends a pass and waits for its answer."""
        return await asyncio.wait_for(self.send(**fields), deadline)

    async def settled(self, slot: int, seconds: float) -> bool:
        """Waits until nothing of this slot's is out, and says whether that happened.

        Every reply ends here, so this also ends its generation. Whatever is still out can
        now only be dropped, here and at every stage, so nothing waits for it.
        """
        newest = self.newest.get(slot)
        if newest is not None:
            await asyncio.wait({newest}, timeout=seconds)
        landed = newest is None or newest.done()
        self.generations.pop(slot, None)
        for flight in list(self.waiting.values()):
            if flight.slot == slot:
                flight.future.cancel()
        return landed

    def close(self) -> None:
        self.alive = False
        if self.task is not None:
            self.task.cancel()
        self.writer.close()
