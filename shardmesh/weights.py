"""Fetch individual tensors from a checkpoint without downloading the rest.

A safetensors file is an 8-byte length, a JSON header of tensor offsets, then
raw data. Reading the header costs a few kilobytes, after which any tensor can
be taken with an HTTP range request. The orchestrator reads the headers once
and sends each worker an allowlist; workers pull only those byte ranges,
straight from the model host.

Each weight unit lands in its own small safetensors file, so a finished unit
survives restarts and a rebalance only fetches what a worker does not have.

Integrity: the Hub publishes a SHA-256 for whole files, which a partial reader
cannot check. Instead every worker reads from one pinned commit, and each unit
is checked for exact size against the header. That catches truncation and
mismatched revisions, but not corruption inside a range.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import shutil
import ssl
import struct
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import torch
from safetensors import safe_open

from .gguf import GgufError, GgufHeader, parse_header, parse_source
from .protocol import auth_headers, TensorRef, WeightUnit
from .tickets import client_auth
from .tls import client_verify

# Mapped weights are read-only by design, and nothing writes to them.
warnings.filterwarnings("ignore", message="The given buffer is not writable")

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# Ranges closer than this are fetched together: re-reading a small gap is
# cheaper than another round trip.
COALESCE_GAP_BYTES = 256 * 1024
CHUNK_BYTES = 1024 * 1024
FETCH_RETRIES = 6
# Disk space a fetch must leave behind, so a download cannot fill the volume.
DISK_RESERVE_BYTES = 2 * 1024**3

TORCH_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


# Enough of a GGUF to cover the start of its header.
FINGERPRINT_BYTES = 1024 * 1024
LOCAL_SCAN_SECONDS = 60.0
RANGE_ATTEMPTS = 4
RANGE_RETRY_SECONDS = 3.0
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
FP8 = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = "_scale_inv"
FP8_BLOCK = 128
# Tensor start alignment inside a slab, matching what GPU allocators give.
SLAB_ALIGNMENT = 256


class WeightSourceError(RuntimeError):
    """The checkpoint cannot be read or is in a format ShardMesh refuses."""


def tls_context() -> ssl.SSLContext:
    """TLS context trusting the OS store as well as certifi.

    httpx defaults to certifi alone, which fails on machines whose antivirus or
    corporate proxy re-signs traffic with a root held only in the OS store.
    Verification stays on either way.
    """
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        pass
    return context


class LocalReader:
    """Reads a checkpoint directory on this machine."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def resolve_revision(self, requested: str) -> str:
        # Directories have no commit id, so the weight files stand in for one.
        digest = hashlib.sha1()
        for path in sorted(self.root.glob("*.safetensors")):
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
        return "local-" + digest.hexdigest()[:12]

    def list_files(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_file())

    def read_file(self, name: str) -> bytes | None:
        path = self.root / name
        return path.read_bytes() if path.is_file() else None

    def file_size(self, name: str) -> int:
        return (self.root / name).stat().st_size

    def read_range(self, name: str, start: int, end: int) -> bytes:
        with (self.root / name).open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start)

    def iter_range(self, name: str, start: int, end: int) -> Iterator[bytes]:
        with (self.root / name).open("rb") as handle:
            handle.seek(start)
            remaining = end - start
            while remaining > 0:
                chunk = handle.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise WeightSourceError(f"{name} ended early at byte {end - remaining}")
                remaining -= len(chunk)
                yield chunk

    def close(self) -> None:
        pass


class HubReader:
    """Reads a Hugging Face repository over HTTP range requests."""

    def __init__(self, repo: str, revision: str = "main") -> None:
        self.repo = repo
        self.revision = revision
        self.endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=15.0),
            verify=tls_context(),
        )

    def _url(self, name: str) -> str:
        return f"{self.endpoint}/{self.repo}/resolve/{self.revision}/{name}"

    def resolve_revision(self, requested: str) -> str:
        response = self.client.get(f"{self.endpoint}/api/models/{self.repo}/revision/{requested}")
        if response.status_code in {401, 403}:
            raise WeightSourceError(f"{self.repo} needs access; set HF_TOKEN on this node")
        if response.status_code == 404:
            raise WeightSourceError(f"model {self.repo!r} at revision {requested!r} was not found")
        response.raise_for_status()
        self.revision = response.json()["sha"]
        return self.revision

    def list_files(self) -> list[str]:
        response = self.client.get(f"{self.endpoint}/api/models/{self.repo}/revision/{self.revision}")
        response.raise_for_status()
        return [item["rfilename"] for item in response.json().get("siblings", [])]

    def read_file(self, name: str) -> bytes | None:
        response = self.client.get(self._url(name))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

    def file_size(self, name: str) -> int:
        with self.client.stream("GET", self._url(name), headers={"Range": "bytes=0-0"}) as response:
            if response.status_code == 404:
                raise WeightSourceError(f"{self.repo} has no file {name}")
            total = response.headers.get("content-range", "").rpartition("/")[2]
        if not total.isdigit():
            raise WeightSourceError(f"{name}: host did not report a file size")
        return int(total)

    def read_range(self, name: str, start: int, end: int) -> bytes:
        return b"".join(self.iter_range(name, start, end))

    def iter_range(self, name: str, start: int, end: int) -> Iterator[bytes]:
        """Stream a byte range, resuming from the last byte after a failure."""
        cursor = start
        failures = 0
        while cursor < end:
            try:
                headers = {"Range": f"bytes={cursor}-{end - 1}"}
                with self.client.stream("GET", self._url(name), headers=headers) as response:
                    if response.status_code != 206:
                        raise WeightSourceError(
                            f"{name}: host answered HTTP {response.status_code} to a range request"
                        )
                    for chunk in response.iter_bytes(CHUNK_BYTES):
                        cursor += len(chunk)
                        failures = 0
                        yield chunk
            except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                failures += 1
                if failures > FETCH_RETRIES:
                    raise WeightSourceError(f"{name}: gave up after {failures} retries: {exc}") from exc
                time.sleep(min(2**failures, 30))

    def close(self) -> None:
        self.client.close()


class PeerReader:
    """Reads a GGUF from other nodes in the pool, with no model host involved."""

    def __init__(self, urls: list[str], token: str | None = None) -> None:
        self.urls = [url.rstrip("/") for url in urls]
        self.client = httpx.Client(
            headers=auth_headers(token),
            timeout=httpx.Timeout(60.0, connect=5.0),
            trust_env=False,
            verify=client_verify(),
            auth=client_auth(),
        )

    def file_size(self, name: str) -> int:
        for url in self.urls:
            try:
                response = self.client.get(url + "/internal/gguf/index")
                response.raise_for_status()
                sizes = {item["name"]: item["size"] for item in response.json()["files"]}
            except (httpx.HTTPError, ValueError, KeyError):
                continue
            if name in sizes:
                return int(sizes[name])
        raise WeightSourceError(f"no node in the pool answered for {name}")

    def read_range(self, name: str, start: int, end: int) -> bytes:
        return b"".join(self.iter_range(name, start, end))

    def iter_range(self, name: str, start: int, end: int) -> Iterator[bytes]:
        """Stream a byte range, moving to the next holder if one fails part way."""
        cursor = start
        reason = "no holders"
        # Gigabytes over Wi-Fi will meet a dropped connection, and one must not end
        # a load. Each try resumes where the last one stopped.
        for attempt in range(RANGE_ATTEMPTS):
            if attempt:
                time.sleep(RANGE_RETRY_SECONDS * attempt)
            for url in self.urls:
                try:
                    params = {"name": name, "start": cursor, "end": end}
                    with self.client.stream("GET", url + "/internal/gguf/range", params=params) as response:
                        if response.status_code != 200:
                            reason = f"{url} answered {response.status_code}"
                            continue
                        for chunk in response.iter_bytes(CHUNK_BYTES):
                            cursor += len(chunk)
                            yield chunk
                    if cursor >= end:
                        return
                except httpx.HTTPError as exc:
                    reason = f"{url}: {type(exc).__name__}"
        raise WeightSourceError(f"{name}: no node could send bytes {cursor} to {end} ({reason})")

    def close(self) -> None:
        self.client.close()


Reader = LocalReader | HubReader | PeerReader


def open_reader(source: str, revision: str = "main") -> Reader:
    located = parse_source(source)
    root = located[0] if located else source
    if located and not root:
        raise WeightSourceError(f"no node in the pool has {located[1]}")
    return LocalReader(root) if Path(root).is_dir() else HubReader(root, revision)


def file_fingerprint(reader: Reader, name: str, size: int) -> str:
    """Identify a GGUF by its content, so nodes that find it separately agree.

    Paths and timestamps differ between machines. The header lists every tensor,
    so the start of the file plus its size is as good as hashing all of it.
    """
    head = reader.read_range(name, 0, min(size, FINGERPRINT_BYTES))
    return "gguf-" + hashlib.sha1(head + str(size).encode()).hexdigest()[:16]


@dataclass(slots=True)
class Manifest:
    """Everything known about a checkpoint without downloading its weights."""

    source: str
    revision: str
    config: dict[str, Any]
    tensors: dict[str, TensorRef] = field(default_factory=dict)


def read_header(reader: Reader, name: str) -> dict[str, TensorRef]:
    prefix = reader.read_range(name, 0, 8)
    if len(prefix) != 8:
        raise WeightSourceError(f"{name} is too short to be a safetensors file")
    (length,) = struct.unpack("<Q", prefix)
    if length > 512 * 1024 * 1024:
        raise WeightSourceError(f"{name} declares an implausible {length}-byte header")
    header = json.loads(reader.read_range(name, 8, 8 + length))
    base = 8 + length
    return {
        tensor: TensorRef(
            name=tensor,
            file=name,
            dtype=info["dtype"],
            shape=info["shape"],
            start=base + info["data_offsets"][0],
            end=base + info["data_offsets"][1],
        )
        for tensor, info in header.items()
        if tensor != "__metadata__"
    }


def read_gguf_header(reader: Reader, name: str) -> GgufHeader:
    def read(offset: int, length: int) -> bytes:
        return reader.read_range(name, offset, offset + length)

    try:
        return parse_header(read, reader.file_size(name))
    except GgufError as exc:
        raise WeightSourceError(f"{name}: {exc}") from exc


def gguf_reader(source: str, revision: str, holders: list[str] | None, token: str | None) -> Reader:
    """Prefer a copy inside the pool: it is closer, and may be the only one."""
    return PeerReader(holders, token) if holders else open_reader(source, revision)


def build_gguf_manifest(
    source: str, name: str, revision: str, holders: list[str] | None = None, token: str | None = None
) -> Manifest:
    """Read a GGUF header. Costs megabytes, however large the model is."""
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", name, re.IGNORECASE):
        raise WeightSourceError(f"{name} is one part of a split GGUF, which is not supported yet")
    reader = gguf_reader(source, revision, holders, token)
    try:
        header = read_gguf_header(reader, name)
        pinned = reader.resolve_revision(revision) if isinstance(reader, HubReader) else file_fingerprint(reader, name, header.file_size)
    finally:
        reader.close()
    tensors = {
        tensor.name: TensorRef(
            name=tensor.name,
            file=name,
            dtype="U8",
            shape=[tensor.nbytes],
            start=header.data_start + tensor.offset,
            end=header.data_start + tensor.offset + tensor.nbytes,
        )
        for tensor in header.tensors.values()
    }
    tokens = header.keys.get("tokenizer.ggml.tokens")
    config = {
        "model_type": "gguf",
        "gguf": header.skeleton(),
        "gguf_vocabulary": hashlib.sha1(tokens.raw).hexdigest() if tokens else "",
        "gguf_file": name,
        "gguf_file_size": header.file_size,
    }
    return Manifest(source=source, revision=pinned, config=config, tensors=tensors)


def build_manifest(
    source: str, revision: str = "main", holders: list[str] | None = None, token: str | None = None
) -> Manifest:
    """Read config and tensor offsets. Costs kilobytes, not gigabytes."""
    located = parse_source(source)
    if located:
        return build_gguf_manifest(source, located[1], revision, holders, token)
    reader = open_reader(source, revision)
    try:
        pinned = reader.resolve_revision(revision)
        raw_config = reader.read_file("config.json")
        if raw_config is None:
            raise WeightSourceError(f"{source} has no config.json")
        config = json.loads(raw_config)
        raw_index = reader.read_file(INDEX_FILE)
        if raw_index is not None:
            files = sorted(set(json.loads(raw_index)["weight_map"].values()))
        else:
            files = [name for name in reader.list_files() if name.endswith(".safetensors")]
        if not files:
            # Pickle checkpoints can run code when loaded, and workers would be
            # running it with data from the internet.
            raise WeightSourceError(f"{source} has no safetensors weights; pickle checkpoints are refused")
        with ThreadPoolExecutor(max_workers=8) as pool:
            headers = list(pool.map(lambda name: read_header(reader, name), files))
    finally:
        reader.close()
    tensors: dict[str, TensorRef] = {}
    for header in headers:
        tensors.update(header)
    return Manifest(source=source, revision=pinned, config=config, tensors=tensors)


def unit_filename(unit: str) -> str:
    return unit + ".safetensors"


def _slug(model_id: str) -> str:
    tail = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
    return f"{tail[:48]}-{hashlib.sha1(model_id.encode()).hexdigest()[:10]}"


def _itemsize(dtype: str) -> int:
    try:
        return torch.empty((), dtype=TORCH_DTYPES[dtype]).element_size()
    except KeyError as exc:
        raise WeightSourceError(f"unsupported tensor dtype {dtype}") from exc


def _unit_layout(unit: WeightUnit) -> tuple[bytes, dict[str, int]]:
    """Build a safetensors header for a unit and each tensor's data offset.

    Wider dtypes go first so that every tensor stays naturally aligned, which
    the disk tier needs to map tensors straight out of the file.
    """
    ordered = sorted(unit.tensors, key=lambda item: (-_itemsize(item.dtype), item.file, item.start))
    offsets: dict[str, int] = {}
    entries: dict[str, Any] = {}
    cursor = 0
    for tensor in ordered:
        size = tensor.end - tensor.start
        offsets[tensor.name] = cursor
        entries[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header, offsets


def _coalesce(tensors: list[TensorRef]) -> list[list[TensorRef]]:
    """Group a unit's tensors into runs that one range request can cover."""
    runs: list[list[TensorRef]] = []
    for tensor in sorted(tensors, key=lambda item: (item.file, item.start)):
        last = runs[-1][-1] if runs else None
        if last and last.file == tensor.file and tensor.start - last.end <= COALESCE_GAP_BYTES:
            runs[-1].append(tensor)
        else:
            runs.append([tensor])
    return runs


def fetch_unit(reader: Reader, unit: WeightUnit, destination: Path, on_bytes: Callable[[int], None]) -> None:
    """Write one unit to ``destination`` atomically."""
    prefix, offsets = _unit_layout(unit)
    expected = len(prefix) + unit.nbytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        with partial.open("wb") as handle:
            handle.write(prefix)
            base = len(prefix)
            for run in _coalesce(unit.tensors):
                position = run[0].start
                index = 0
                for chunk in reader.iter_range(run[0].file, run[0].start, run[-1].end):
                    view = memoryview(chunk)
                    while view.nbytes and index < len(run):
                        tensor = run[index]
                        if position < tensor.start:  # bytes in a gap between tensors
                            skip = min(tensor.start - position, view.nbytes)
                            view = view[skip:]
                            position += skip
                            continue
                        take = min(tensor.end - position, view.nbytes)
                        handle.seek(base + offsets[tensor.name] + position - tensor.start)
                        handle.write(view[:take])
                        view = view[take:]
                        position += take
                        on_bytes(take)
                        if position == tensor.end:
                            index += 1
        actual = partial.stat().st_size
        if actual != expected:
            raise WeightSourceError(f"{unit.name}: wrote {actual} bytes, header says {expected}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def unit_tensor_span(path: Path, name: str) -> tuple[int, int]:
    """Where one tensor's bytes sit inside a unit file, as (offset, length)."""
    header = _read_unit_header(path)
    with path.open("rb") as handle:
        (prefix,) = struct.unpack("<Q", handle.read(8))
    start, end = header[name]["data_offsets"]
    return 8 + prefix + start, end - start


def iter_unit_tensor(path: Path, name: str) -> Iterator[bytes]:
    offset, remaining = unit_tensor_span(path, name)
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise WeightSourceError(f"{path.name} ended early")
            remaining -= len(chunk)
            yield chunk


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Expand a block-scaled FP8 matrix. Each 128x128 block has one scale."""
    rows, columns = weight.shape
    scale = scale.to(weight.device, dtype)
    if rows % FP8_BLOCK == 0 and columns % FP8_BLOCK == 0:
        blocks = weight.to(dtype).view(rows // FP8_BLOCK, FP8_BLOCK, columns // FP8_BLOCK, FP8_BLOCK)
        return (blocks * scale[:, None, :, None]).view(rows, columns)
    spread = scale.repeat_interleave(FP8_BLOCK, 0)[:rows].repeat_interleave(FP8_BLOCK, 1)[:, :columns]
    return weight.to(dtype) * spread


class LoadedUnit:
    """A unit's tensors in a pool, plus the mapping that backs them if any."""

    def __init__(self, tensors: dict[str, torch.Tensor], mapping: mmap.mmap | None = None) -> None:
        self.tensors = tensors
        self.mapping = mapping

    def release(self) -> None:
        self.tensors = {}
        if self.mapping is not None:
            try:
                self.mapping.close()
            except BufferError:
                pass  # a live view remains; the OS unmaps when it is collected
            self.mapping = None


def _read_unit_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def _resident_form(
    name: str,
    stored: torch.dtype,
    dtype: torch.dtype | None,
    keep_float32: Callable[[str], bool],
    compact: bool,
) -> torch.dtype | None:
    """Dtype a tensor takes in its pool, or None if it does not stay resident."""
    if name.endswith(FP8_SCALE_SUFFIX):
        return stored if compact else None
    if stored == FP8:
        return FP8 if compact else dtype or torch.bfloat16
    if dtype is not None and stored.is_floating_point and not keep_float32(name):
        return dtype
    return stored


def _carve_slab(layout: dict[str, tuple[torch.dtype, list[int]]], device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate a unit as one block and return a view for each tensor.

    The CUDA and ROCm allocators round mid-sized requests into 20 MiB blocks.
    For MoE experts of a few MiB each, that wastes over 20% of the memory, and
    an exact plan then runs out of a hard budget. One allocation wastes nothing.
    """
    offsets: dict[str, int] = {}
    cursor = 0
    for name, (dtype, shape) in layout.items():
        cursor = -(-cursor // SLAB_ALIGNMENT) * SLAB_ALIGNMENT
        offsets[name] = cursor
        cursor += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    slab = torch.empty(max(cursor, 1), dtype=torch.uint8, device=device)
    views: dict[str, torch.Tensor] = {}
    for name, (dtype, shape) in layout.items():
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        views[name] = slab[offsets[name] : offsets[name] + size].view(dtype).view(shape)
    return views


def load_unit(
    path: Path,
    device: str,
    dtype: torch.dtype | None,
    keep_float32: Callable[[str], bool],
    compact: bool = False,
) -> LoadedUnit:
    """Read a unit into RAM or an accelerator, one tensor at a time.

    Peak host memory is a single tensor, so a checkpoint never has to fit in
    RAM to reach a GPU. FP8 weights are expanded here, once, unless ``compact``
    keeps them as stored: expanding per use costs far more than reading BF16.
    """
    target = torch.device(device)
    layout: dict[str, tuple[torch.dtype, list[int]]] = {}
    for name, info in _read_unit_header(path).items():
        form = _resident_form(name, TORCH_DTYPES[info["dtype"]], dtype, keep_float32, compact)
        if form is not None:
            layout[name] = (form, info["shape"])
    slab = _carve_slab(layout, target) if target.type != "cpu" else None

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name, (form, _) in layout.items():
            tensor = handle.get_tensor(name)
            if tensor.dtype == FP8 and form != FP8:
                scale = handle.get_tensor(name + FP8_SCALE_SUFFIX).to(target)
                tensor = dequantize_fp8(tensor.to(target), scale, form)
            if slab is None:
                tensors[name] = tensor.to(form)
            else:
                tensors[name] = slab[name].copy_(tensor)
    return LoadedUnit(tensors)


def map_unit(path: Path) -> LoadedUnit:
    """Map a unit from disk without reading it.

    Tensors are views of the page cache: the OS reads pages on first touch and
    drops them under memory pressure, so the resident set follows what the
    model is actually using. The mapping must stay read-only: on Windows a
    copy-on-write mapping charges its full size against the commit limit, so
    a model bigger than RAM plus page file could not be mapped at all.
    """
    with path.open("rb") as handle:
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    (length,) = struct.unpack("<Q", mapping[:8])
    header = json.loads(mapping[8 : 8 + length])
    base = 8 + length
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = TORCH_DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        if end == begin:
            tensors[name] = torch.empty(info["shape"], dtype=dtype)
            continue
        count = (end - begin) // torch.empty((), dtype=dtype).element_size()
        flat = torch.frombuffer(mapping, dtype=dtype, count=count, offset=base + begin)
        tensors[name] = flat.view(info["shape"])
    return LoadedUnit(tensors, mapping)


@dataclass(slots=True)
class FetchProgress:
    """Live state of one prepare job, read by the orchestrator."""

    job_id: str
    bytes_total: int = 0
    bytes_done: int = 0
    bytes_reused: int = 0
    bytes_from_peers: int = 0
    units_total: int = 0
    units_done: int = 0
    phase: str = "fetching"  # fetching | ready | failed
    error: str | None = None
    cancelled: bool = False
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        return {
            "job_id": self.job_id,
            "phase": self.phase,
            "bytes_total": self.bytes_total,
            "bytes_done": self.bytes_done,
            "bytes_reused": self.bytes_reused,
            "bytes_from_peers": self.bytes_from_peers,
            "units_total": self.units_total,
            "units_done": self.units_done,
            "bytes_per_second": self.bytes_done / elapsed if elapsed > 0 else 0.0,
            "error": self.error,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(manifest: Manifest) -> str:
    """Covers what nodes are told about a model beyond its weights: config, shapes, offsets."""
    tensors = {name: tensor.model_dump() for name, tensor in sorted(manifest.tensors.items())}
    body = json.dumps({"config": manifest.config, "tensors": tensors}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


MANIFEST_KEY = "manifest"


def apply_model_hashes(manifest: Manifest, units: list[WeightUnit], hashes: dict[str, str]) -> None:
    """Check a model's description, and tell each unit what it must hash to."""
    if hashes.get(MANIFEST_KEY) != manifest_sha256(manifest):
        raise WeightSourceError("this model's description does not match the hashes given; it is a different file, or altered")
    missing = [unit.name for unit in units if unit.name not in hashes]
    if missing:
        raise WeightSourceError(f"the hashes given have no entry for {missing[0]} and {len(missing) - 1} others")
    for unit in units:
        unit.sha256 = hashes[unit.name]


def hash_model(manifest: Manifest, units: list[WeightUnit], reader: Reader, folder: Path) -> dict[str, str]:
    """Hash every unit exactly as a worker would build it. Reads the whole model."""
    hashes = {MANIFEST_KEY: manifest_sha256(manifest)}
    scratch = folder / "unit.hashing"
    try:
        for unit in units:
            fetch_unit(reader, unit, scratch, lambda count: None)
            hashes[unit.name] = file_sha256(scratch)
    finally:
        scratch.unlink(missing_ok=True)
    return hashes


def fetch_from_peer(
    peer_url: str,
    token: str | None,
    model_id: str,
    revision: str,
    unit: WeightUnit,
    destination: Path,
    on_bytes: Callable[[int], None],
    client: httpx.Client | None = None,
) -> bool:
    """Copy a finished unit from another worker. False means use the model host.

    Unit files are built deterministically from the manifest, so a good copy is
    byte-identical in length and header. A peer that fails either check, or
    fails at all, costs nothing but the attempt.
    """
    prefix, _ = _unit_layout(unit)
    expected = len(prefix) + unit.nbytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.{os.getpid()}.{threading.get_ident()}.peer.part")
    owned = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0), verify=client_verify(), auth=client_auth())
    headers = auth_headers(token)
    params = {"model_id": model_id, "revision": revision, "unit": unit.name}
    received = 0
    try:
        with owned.stream("GET", peer_url.rstrip("/") + "/internal/shards/file", params=params, headers=headers) as response:
            if response.status_code != 200:
                return False
            with partial.open("wb") as handle:
                checked = False
                head = b""
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    if not checked:
                        head += chunk[: len(prefix) - len(head)]
                        if len(head) == len(prefix):
                            if head != prefix:
                                return False
                            checked = True
                    if received + len(chunk) > expected:
                        return False  # a peer must not be able to fill the disk
                    handle.write(chunk)
                    received += len(chunk)
                    on_bytes(len(chunk))
        if received != expected:
            return False
        os.replace(partial, destination)
        return True
    except (httpx.HTTPError, OSError):
        return False
    finally:
        if not destination.is_file() and received:
            on_bytes(-received)  # the model host will deliver these bytes instead
        partial.unlink(missing_ok=True)
        if client is None:
            owned.close()


class UnitCache:
    """On-disk store of fetched units, shared by every model on a worker."""

    def __init__(
        self,
        root: Path,
        budget_bytes: int = 0,
        concurrency: int = 4,
        local_dirs: tuple[Path, ...] = (),
    ) -> None:
        self.root = root
        self.budget_bytes = budget_bytes
        self.concurrency = concurrency
        # Folders that may already hold whole GGUF files, such as LM Studio's.
        self.local_dirs = local_dirs
        self._local_files: dict[str, Path] = {}
        self._scanned_at = -LOCAL_SCAN_SECONDS
        self.root.mkdir(parents=True, exist_ok=True)

    def local_files(self) -> dict[str, Path]:
        """GGUF files in the local folders, by name. Scanning a big drive is slow."""
        if time.monotonic() - self._scanned_at > LOCAL_SCAN_SECONDS:
            found: dict[str, Path] = {}
            for folder in self.local_dirs:
                for path in folder.rglob("*.gguf"):
                    if path.is_file() and not path.name.startswith("stage-"):
                        found.setdefault(path.name, path)
            self._local_files = found
            self._scanned_at = time.monotonic()
        return self._local_files

    def local_copy(self, name: str, size: int) -> Path | None:
        """A file already on this machine with this name and exact size."""
        for folder in self.local_dirs:
            for path in folder.rglob(name):
                if path.is_file() and path.stat().st_size == size:
                    return path
        return None

    def directory(self, model_id: str, revision: str) -> Path:
        if not SAFE_NAME.fullmatch(revision):
            raise ValueError(f"not a revision: {revision[:80]!r}")
        return self.root / _slug(model_id) / revision

    def path(self, model_id: str, revision: str, unit: str) -> Path:
        if not SAFE_NAME.fullmatch(unit):
            raise ValueError(f"not a unit: {unit[:80]!r}")
        return self.directory(model_id, revision) / unit_filename(unit)

    def has(self, model_id: str, revision: str, unit: WeightUnit) -> bool:
        path = self.path(model_id, revision, unit.name)
        if not path.is_file():
            return False
        return path.stat().st_size == len(_unit_layout(unit)[0]) + unit.nbytes

    def _discard(self, model_id: str, revision: str, unit: WeightUnit) -> None:
        try:
            self.path(model_id, revision, unit.name).unlink()
        except PermissionError as exc:  # Windows will not delete a file a running stage has open
            raise WeightSourceError(f"{unit.name} does not match its hash, and a running stage is using it") from exc

    def _intact(self, model_id: str, revision: str, unit: WeightUnit) -> bool:
        return file_sha256(self.path(model_id, revision, unit.name)) == unit.sha256

    def held(self, model_id: str, revision: str) -> dict[str, int]:
        """Finished units of a model, by name, with their file sizes."""
        if not SAFE_NAME.fullmatch(revision):
            return {}
        directory = self.directory(model_id, revision)
        if not directory.is_dir():
            return {}
        suffix = ".safetensors"
        return {path.name[: -len(suffix)]: path.stat().st_size for path in directory.glob(f"*{suffix}")}

    def peer_file(self, model_id: str, revision: str, unit: str) -> Path | None:
        """A unit's file for serving, or None. Names arrive from the network."""
        if not SAFE_NAME.fullmatch(unit) or not SAFE_NAME.fullmatch(revision):
            return None
        path = self.path(model_id, revision, unit)
        if not path.is_file() or self.root.resolve() not in path.resolve().parents:
            return None
        return path

    def _files(self) -> list[Path]:
        """Everything the cache budget covers: units, and stage files built from them."""
        return [*self.root.rglob("*.safetensors"), *self.root.rglob("stage-*.gguf")]

    def used_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._files())

    def make_room(self, needed: int, protected: set[Path]) -> None:
        """Evict least recently used units until ``needed`` bytes fit."""
        files = sorted(
            (path for path in self._files() if path not in protected),
            key=lambda path: path.stat().st_atime,
        )
        used = self.used_bytes()

        def short() -> bool:
            free = shutil.disk_usage(self.root).free - DISK_RESERVE_BYTES
            over_budget = bool(self.budget_bytes) and used + needed > self.budget_bytes
            return needed > free or over_budget

        while short() and files:
            victim = files.pop(0)
            size = victim.stat().st_size
            try:
                victim.unlink()
                used -= size
            except PermissionError:
                continue  # still mapped by the disk tier
        if short():
            free = shutil.disk_usage(self.root).free
            raise WeightSourceError(
                f"need {needed / 1024**3:.1f} GiB in {self.root}, but only "
                f"{free / 1024**3:.1f} GiB is free and nothing else can be evicted"
            )

    def fetch(
        self,
        source: str,
        model_id: str,
        revision: str,
        units: list[WeightUnit],
        progress: FetchProgress,
        source_revision: str = "main",
        peers: dict[str, list[str]] | None = None,
        token: str | None = None,
        file_sizes: dict[str, int] | None = None,
        file_sources: list[str] | None = None,
    ) -> None:
        """Bring every unit into the cache, skipping any already present."""
        self.root.mkdir(parents=True, exist_ok=True)  # it may have been cleared by hand
        for unit in units:
            # What is already on disk gets no more trust than what arrives.
            if unit.sha256 and self.has(model_id, revision, unit) and not self._intact(model_id, revision, unit):
                self._discard(model_id, revision, unit)
        missing = [unit for unit in units if not self.has(model_id, revision, unit)]
        progress.units_total = len(units)
        progress.units_done = len(units) - len(missing)
        progress.bytes_total = sum(unit.nbytes for unit in missing)
        progress.bytes_reused = sum(unit.nbytes for unit in units) - progress.bytes_total
        protected = {self.path(model_id, revision, unit.name) for unit in units}
        self.make_room(progress.bytes_total, protected)
        lock = threading.Lock()

        def add_bytes(count: int) -> None:
            if progress.cancelled:
                raise WeightSourceError("load cancelled")
            with lock:
                progress.bytes_done += count

        local = {name: self.local_copy(name, size) for name, size in (file_sizes or {}).items()}

        def accepted(unit: WeightUnit) -> bool:
            if not unit.sha256 or self._intact(model_id, revision, unit):
                return True
            self._discard(model_id, revision, unit)
            return False

        def fetch_one(unit: WeightUnit) -> None:
            destination = self.path(model_id, revision, unit.name)
            copy = local.get(unit.tensors[0].file)
            if copy is not None:
                reader = LocalReader(str(copy.parent))
                fetch_unit(reader, unit, destination, add_bytes)
                if accepted(unit):
                    with lock:
                        progress.units_done += 1
                    return
            for peer in (peers or {}).get(unit.name, []):
                if fetch_from_peer(peer, token, model_id, revision, unit, destination, add_bytes) and accepted(unit):
                    with lock:
                        # A unit file is its tensors plus a small header.
                        progress.bytes_done += unit.nbytes - destination.stat().st_size
                        progress.bytes_from_peers += unit.nbytes
                        progress.units_done += 1
                    return
            reader = gguf_reader(source, source_revision, file_sources, token)
            try:
                fetch_unit(reader, unit, self.path(model_id, revision, unit.name), add_bytes)
            finally:
                reader.close()
            if not accepted(unit):
                raise WeightSourceError(f"{unit.name} did not match its hash from any source")
            with lock:
                progress.units_done += 1

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            list(pool.map(fetch_one, missing))
