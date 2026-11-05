"""Read GGUF headers and write stage-sized GGUF files.

A GGUF header lists every tensor with its byte offset, so a node can fetch only
the tensors of its own layers, the same way safetensors shards are fetched. A
stage file holds a contiguous slice of layers, renumbered from zero, and looks
to llama.cpp like a small complete model.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
import re
import struct
from typing import Any, BinaryIO, Callable, Iterable

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
READ_CHUNK = 4 * 1024 * 1024

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = range(8)
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(8, 13)

SCALARS = {
    T_UINT8: "<B",
    T_INT8: "<b",
    T_UINT16: "<H",
    T_INT16: "<h",
    T_UINT32: "<I",
    T_INT32: "<i",
    T_FLOAT32: "<f",
    T_BOOL: "<?",
    T_UINT64: "<Q",
    T_INT64: "<q",
    T_FLOAT64: "<d",
}
SCALAR_SIZES = {kind: struct.calcsize(code) for kind, code in SCALARS.items()}
GGML_F32 = 0
SKELETON_VOCAB = "shardmesh.vocab_size"
LAYER_NAME = re.compile(r"^blk\.(\d+)\.(.+)$")


class GgufError(ValueError):
    pass


def parse_source(source: str) -> tuple[str, str] | None:
    """Split a GGUF model id into where it lives and the file's name.

    ``owner/repo:file.gguf`` names a file in a hub repository, and a path ending
    in ``.gguf`` names a local file. Any other name, or a path that is not on this
    machine, means whichever node in the pool has that file, which is how a pool
    on a closed network refers to its models. An empty location says so.
    """
    if not source.lower().endswith(".gguf"):
        return None
    path = Path(source)
    if path.is_file():
        return str(path.parent), path.name
    repository, separator, name = source.rpartition(":")
    if separator and "/" in repository and "\\" not in repository:
        return repository, name
    return "", source.replace("\\", "/").rsplit("/", 1)[-1]


@dataclass
class KeyValue:
    key: str
    kind: int
    raw: bytes  # the value as stored, so unchanged entries copy across exactly
    value: Any = None  # scalars and strings, plus the length of arrays
    item_kind: int | None = None


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int  # from the start of the data section
    nbytes: int = 0


@dataclass
class GgufHeader:
    version: int
    keys: dict[str, KeyValue]
    tensors: dict[str, TensorInfo]
    data_start: int
    file_size: int
    alignment: int = DEFAULT_ALIGNMENT

    @property
    def architecture(self) -> str:
        return str(self.get("general.architecture"))

    def get(self, key: str, default: Any = None) -> Any:
        entry = self.keys.get(key)
        return default if entry is None else entry.value

    def arch_key(self, name: str, default: Any = None) -> Any:
        return self.get(f"{self.architecture}.{name}", default)

    @property
    def layer_count(self) -> int:
        """Layers in the main stack, leaving out multi-token-prediction blocks."""
        return int(self.arch_key("block_count", 0)) - int(self.arch_key("nextn_predict_layers", 0) or 0)

    @property
    def hidden_size(self) -> int:
        return int(self.arch_key("embedding_length", 0))

    @property
    def tied_embeddings(self) -> bool:
        return "output.weight" not in self.tensors

    @property
    def vocab_size(self) -> int:
        tokens = self.keys.get("tokenizer.ggml.tokens")
        return int(tokens.value) if tokens else int(self.get(SKELETON_VOCAB, 0))

    def layer_tensors(self, layer: int) -> list[TensorInfo]:
        prefix = f"blk.{layer}."
        return [tensor for name, tensor in self.tensors.items() if name.startswith(prefix)]

    def skeleton(self) -> dict[str, Any]:
        """The header without its tokenizer, small enough to send to every stage."""
        kept = [entry for key, entry in self.keys.items() if not key.startswith("tokenizer.")]
        kept.append(_scalar(SKELETON_VOCAB, T_UINT32, self.vocab_size))
        return {
            "alignment": self.alignment,
            "keys": [
                [entry.key, entry.kind, entry.item_kind, base64.b64encode(entry.raw).decode("ascii")] for entry in kept
            ],
            "tensors": [
                [tensor.name, tensor.dims, tensor.ggml_type, tensor.offset, tensor.nbytes]
                for tensor in self.tensors.values()
            ],
        }

    @classmethod
    def from_skeleton(cls, data: dict[str, Any]) -> "GgufHeader":
        keys: dict[str, KeyValue] = {}
        for key, kind, item_kind, raw in data["keys"]:
            content = base64.b64decode(raw)
            value, _ = _read_value(_Cursor(lambda offset, length: content[offset : offset + length], len(content)), kind)
            keys[key] = KeyValue(key, kind, content, value, item_kind)
        tensors = {item[0]: TensorInfo(*item) for item in data["tensors"]}
        return cls(3, keys, tensors, 0, 0, int(data["alignment"]))


class _Cursor:
    def __init__(self, read: Callable[[int, int], bytes], size: int) -> None:
        self.read = read
        self.size = size
        self.buffer = b""
        self.base = 0
        self.position = 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        if end > self.size:
            raise GgufError("GGUF header runs past the end of the file")
        if self.position < self.base or end > self.base + len(self.buffer):
            start = self.position
            self.buffer = self.read(start, min(max(count, READ_CHUNK), self.size - start))
            self.base = start
            if len(self.buffer) < count:
                raise GgufError("short read inside the GGUF header")
        offset = self.position - self.base
        self.position = end
        return self.buffer[offset : offset + count]

    def unpack(self, code: str) -> Any:
        return struct.unpack(code, self.take(struct.calcsize(code)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8")


def _read_value(cursor: _Cursor, kind: int) -> tuple[Any, int | None]:
    if kind in SCALARS:
        return cursor.unpack(SCALARS[kind]), None
    if kind == T_STRING:
        return cursor.string(), None
    if kind == T_ARRAY:
        item_kind = cursor.unpack("<I")
        count = cursor.unpack("<Q")
        if item_kind in SCALARS:
            cursor.take(count * SCALAR_SIZES[item_kind])
        elif item_kind == T_STRING:
            for _ in range(count):
                cursor.take(cursor.unpack("<Q"))
        else:
            raise GgufError(f"unsupported GGUF array of type {item_kind}")
        return count, item_kind
    raise GgufError(f"unknown GGUF value type {kind}")


def parse_header(read: Callable[[int, int], bytes], file_size: int) -> GgufHeader:
    """Parse a header through ``read(offset, length)``, a local file or a range request."""
    cursor = _Cursor(read, file_size)
    if cursor.take(4) != MAGIC:
        raise GgufError("not a GGUF file")
    version = cursor.unpack("<I")
    if version < 2:
        raise GgufError(f"GGUF version {version} is too old")
    tensor_count = cursor.unpack("<Q")
    key_count = cursor.unpack("<Q")

    keys: dict[str, KeyValue] = {}
    for _ in range(key_count):
        key = cursor.string()
        kind = cursor.unpack("<I")
        start = cursor.position
        value, item_kind = _read_value(cursor, kind)
        end = cursor.position
        cursor.position = start
        raw = _take_span(cursor, end - start)
        keys[key] = KeyValue(key, kind, raw, value, item_kind)

    alignment = int(keys["general.alignment"].value) if "general.alignment" in keys else DEFAULT_ALIGNMENT
    tensors: dict[str, TensorInfo] = {}
    for _ in range(tensor_count):
        name = cursor.string()
        rank = cursor.unpack("<I")
        dims = [cursor.unpack("<Q") for _ in range(rank)]
        ggml_type = cursor.unpack("<I")
        tensors[name] = TensorInfo(name, dims, ggml_type, cursor.unpack("<Q"))

    data_start = _align(cursor.position, alignment)
    ordered = sorted(tensors.values(), key=lambda tensor: tensor.offset)
    for current, following in zip(ordered, ordered[1:] + [None]):
        end = following.offset if following else file_size - data_start
        current.nbytes = end - current.offset
    return GgufHeader(version, keys, tensors, data_start, file_size, alignment)


def _take_span(cursor: _Cursor, count: int) -> bytes:
    pieces = []
    while count:
        piece = min(count, READ_CHUNK)
        pieces.append(cursor.take(piece))
        count -= piece
    return b"".join(pieces)


def read_file_header(path: str) -> GgufHeader:
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()

        def read(offset: int, length: int) -> bytes:
            handle.seek(offset)
            return handle.read(length)

        return parse_header(read, size)


def _align(value: int, alignment: int) -> int:
    return value + (-value % alignment)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _key_bytes(entry: KeyValue) -> bytes:
    return _string(entry.key) + struct.pack("<I", entry.kind) + entry.raw


def _scalar(key: str, kind: int, value: Any) -> KeyValue:
    return KeyValue(key, kind, struct.pack(SCALARS[kind], value), value)


def _text(key: str, value: str) -> KeyValue:
    return KeyValue(key, T_STRING, _string(value), value)


def _slice_array(entry: KeyValue, start: int, end: int) -> KeyValue:
    width = SCALAR_SIZES[entry.item_kind]
    items = entry.raw[12 + start * width : 12 + end * width]
    raw = struct.pack("<IQ", entry.item_kind, end - start) + items
    return KeyValue(entry.key, T_ARRAY, raw, end - start, entry.item_kind)


@dataclass
class StagePlan:
    """What a stage file contains: its keys, and tensors as (new name, source)."""

    keys: list[KeyValue]
    tensors: list[tuple[str, TensorInfo]]
    stubs: list[TensorInfo] = field(default_factory=list)  # zero-filled placeholders

    @property
    def fetch_bytes(self) -> int:
        return sum(tensor.nbytes for _, tensor in self.tensors)


STUB_VOCAB = 4


def plan_stage(header: GgufHeader, start: int, end: int, *, last: bool) -> StagePlan:
    """Describe a stage file for layers ``start`` to ``end - 1``.

    Stages take hidden states as input and ShardMesh does its own tokenizing, so
    llama.cpp's vocabulary and embedding table are only there to satisfy its
    loader. They become placeholders, and so does anything else the stage never
    evaluates. The last stage keeps the real vocabulary size for its logits.
    """
    if not 0 <= start < end <= header.layer_count:
        raise GgufError(f"layers {start}..{end - 1} are outside 0..{header.layer_count - 1}")
    arch = header.architecture
    interval = header.arch_key("full_attention_interval")
    if interval and start % int(interval):
        raise GgufError(f"{arch} stages must start on a multiple of {interval}, not {start}")

    block_key = f"{arch}.block_count"
    source_blocks = int(header.get(block_key))
    replaced: dict[str, KeyValue] = {block_key: _scalar(block_key, header.keys[block_key].kind, end - start)}
    nextn_key = f"{arch}.nextn_predict_layers"
    if nextn_key in header.keys:
        replaced[nextn_key] = _scalar(nextn_key, header.keys[nextn_key].kind, 0)
    for key, entry in header.keys.items():
        if entry.kind == T_ARRAY and entry.value == source_blocks and entry.item_kind in SCALARS and key.startswith(arch + "."):
            replaced[key] = _slice_array(entry, start, end)

    keys = [
        replaced.get(key, entry)
        for key, entry in header.keys.items()
        if not key.startswith("tokenizer.") and key != SKELETON_VOCAB
    ]
    tensors: list[tuple[str, TensorInfo]] = []
    for layer in range(start, end):
        for tensor in header.layer_tensors(layer):
            suffix = LAYER_NAME.match(tensor.name).group(2)  # type: ignore[union-attr]
            tensors.append((f"blk.{layer - start}.{suffix}", tensor))

    table = header.tensors["token_embd.weight"]
    norm = header.tensors["output_norm.weight"]
    stubs: list[TensorInfo] = []
    if last:
        tensors.append(("output_norm.weight", norm))
        if header.tied_embeddings:
            tensors.append(("token_embd.weight", table))
        else:
            tensors.append(("output.weight", header.tensors["output.weight"]))
            stubs.append(TensorInfo("token_embd.weight", list(table.dims), table.ggml_type, 0, table.nbytes))
        vocabulary = header.vocab_size
    else:
        width = header.hidden_size
        stubs.append(TensorInfo("output_norm.weight", list(norm.dims), norm.ggml_type, 0, norm.nbytes))
        stubs.append(TensorInfo("token_embd.weight", [width, STUB_VOCAB], GGML_F32, 0, width * STUB_VOCAB * 4))
        vocabulary = STUB_VOCAB
    return StagePlan(keys + _stub_vocabulary(vocabulary), tensors, stubs)


def _stub_vocabulary(size: int) -> list[KeyValue]:
    tokens = [f"<t{index}>".encode("ascii") for index in range(size)]
    raw = struct.pack("<IQ", T_STRING, size) + b"".join(struct.pack("<Q", len(item)) + item for item in tokens)
    merges = struct.pack("<IQ", T_STRING, 0)
    token_types = struct.pack("<IQ", T_INT32, size) + struct.pack(f"<{size}i", *([1] * size))
    return [
        _text("tokenizer.ggml.model", "gpt2"),
        _text("tokenizer.ggml.pre", "default"),
        KeyValue("tokenizer.ggml.tokens", T_ARRAY, raw, size, T_STRING),
        KeyValue("tokenizer.ggml.token_type", T_ARRAY, token_types, size, T_INT32),
        KeyValue("tokenizer.ggml.merges", T_ARRAY, merges, 0, T_STRING),
    ]


def read_strings(entry: KeyValue) -> list[str]:
    """Decode an array of strings from a full header."""
    if entry.kind != T_ARRAY or entry.item_kind != T_STRING:
        raise GgufError(f"{entry.key} is not an array of strings")
    items: list[str] = []
    position = 12
    raw = entry.raw
    for _ in range(entry.value):
        (length,) = struct.unpack_from("<Q", raw, position)
        position += 8
        items.append(raw[position : position + length].decode("utf-8", "surrogatepass"))
        position += length
    return items


def read_numbers(entry: KeyValue) -> list[Any]:
    """Decode an array of numbers from a full header."""
    if entry.kind != T_ARRAY or entry.item_kind not in SCALARS:
        raise GgufError(f"{entry.key} is not an array of numbers")
    code = SCALARS[entry.item_kind]
    return list(struct.unpack(f"<{entry.value}{code[1]}", entry.raw[12:]))


def write_stage(
    header: GgufHeader,
    plan: StagePlan,
    out: BinaryIO,
    read_tensor: Callable[[TensorInfo], Iterable[bytes]],
) -> None:
    """Write a stage file. ``read_tensor`` yields a source tensor's bytes in order."""
    alignment = header.alignment
    entries: list[tuple[str, TensorInfo, TensorInfo | None]] = [(name, tensor, tensor) for name, tensor in plan.tensors]
    entries += [(stub.name, stub, None) for stub in plan.stubs]

    infos = []
    offset = 0
    for name, tensor, _ in entries:
        infos.append(
            _string(name)
            + struct.pack("<I", len(tensor.dims))
            + b"".join(struct.pack("<Q", dim) for dim in tensor.dims)
            + struct.pack("<IQ", tensor.ggml_type, offset)
        )
        offset += _align(tensor.nbytes, alignment)

    out.write(MAGIC + struct.pack("<IQQ", 3, len(entries), len(plan.keys)))
    for entry in plan.keys:
        out.write(_key_bytes(entry))
    for info in infos:
        out.write(info)
    position = out.tell()
    out.write(bytes(_align(position, alignment) - position))

    for _, tensor, source in entries:
        written = 0
        if source is None:
            out.seek(tensor.nbytes, 1)  # zeros, without writing them
            written = tensor.nbytes
        else:
            for piece in read_tensor(source):
                out.write(piece)
                written += len(piece)
        if written != tensor.nbytes:
            raise GgufError(f"{tensor.name}: got {written} bytes, expected {tensor.nbytes}")
        padding = -tensor.nbytes % alignment
        if padding:
            out.seek(padding, 1)
    out.truncate()


def file_reader(path: str) -> Callable[[TensorInfo], Iterable[bytes]]:
    """Read tensors from a local GGUF. The header must have been parsed from the same file."""
    header = read_file_header(path)

    def read(tensor: TensorInfo) -> Iterable[bytes]:
        with open(path, "rb") as handle:
            handle.seek(header.data_start + tensor.offset)
            remaining = tensor.nbytes
            while remaining:
                piece = handle.read(min(remaining, 16 * 1024 * 1024))
                if not piece:
                    raise GgufError(f"{tensor.name}: file ended early")
                remaining -= len(piece)
                yield piece

    return read
