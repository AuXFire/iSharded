import json
import ssl
import struct
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import save_file

from isharded import weights
from isharded.protocol import TensorRef, WeightUnit
from isharded.weights import (
    FetchProgress,
    HubReader,
    LocalReader,
    UnitCache,
    WeightSourceError,
    build_manifest,
    fetch_unit,
    load_unit,
    map_unit,
    read_header,
    tls_context,
)


def checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "a": torch.arange(10, dtype=torch.float32),
        "b": torch.arange(7, dtype=torch.bfloat16),  # odd length tests alignment
        "c": torch.arange(5000, dtype=torch.float32),
        "d": torch.arange(9, dtype=torch.float32),
    }
    save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text('{"model_type": "gpt2"}')
    return tensors


def unit_of(directory: Path, *names: str) -> WeightUnit:
    header = read_header(LocalReader(str(directory)), "model.safetensors")
    return WeightUnit(name="unit", kind="dense", layer=0, tensors=[header[name] for name in names])


class CountingReader(LocalReader):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.requests: list[tuple[int, int]] = []

    def iter_range(self, name: str, start: int, end: int):
        self.requests.append((start, end))
        return super().iter_range(name, start, end)


def test_only_allowlisted_tensors_reach_the_unit_file(tmp_path: Path) -> None:
    tensors = checkpoint(tmp_path / "model")
    unit = unit_of(tmp_path / "model", "a", "d")
    destination = tmp_path / "out.safetensors"
    moved: list[int] = []

    fetch_unit(CountingReader(str(tmp_path / "model")), unit, destination, moved.append)

    loaded = load_unit(destination, "cpu", None, lambda name: False).tensors
    assert set(loaded) == {"a", "d"}
    assert torch.equal(loaded["a"], tensors["a"])
    assert torch.equal(loaded["d"], tensors["d"])
    assert sum(moved) == unit.nbytes


def test_nearby_tensors_share_one_request_and_distant_ones_do_not(tmp_path: Path, monkeypatch) -> None:
    checkpoint(tmp_path / "model")
    reader = CountingReader(str(tmp_path / "model"))
    unit = unit_of(tmp_path / "model", "a", "d", "c", "b")
    fetch_unit(reader, unit, tmp_path / "joined.safetensors", lambda count: None)
    assert len(reader.requests) == 1

    monkeypatch.setattr(weights, "COALESCE_GAP_BYTES", 0)
    reader = CountingReader(str(tmp_path / "model"))
    spread = unit_of(tmp_path / "model", "a", "d")
    ordered = sorted(spread.tensors, key=lambda item: item.start)
    if ordered[1].start > ordered[0].end:
        fetch_unit(reader, spread, tmp_path / "apart.safetensors", lambda count: None)
        assert len(reader.requests) == 2


def test_gap_bytes_between_tensors_never_land_in_the_unit(tmp_path: Path) -> None:
    tensors = checkpoint(tmp_path / "model")
    header = read_header(LocalReader(str(tmp_path / "model")), "model.safetensors")
    ordered = sorted(header.values(), key=lambda item: item.start)
    unit = WeightUnit(name="unit", kind="dense", layer=0, tensors=[ordered[0], ordered[2]])
    destination = tmp_path / "out.safetensors"

    fetch_unit(LocalReader(str(tmp_path / "model")), unit, destination, lambda count: None)

    loaded = load_unit(destination, "cpu", None, lambda name: False).tensors
    for name, tensor in loaded.items():
        assert torch.equal(tensor, tensors[name])


def test_mapped_units_match_loaded_units_across_mixed_dtypes(tmp_path: Path) -> None:
    tensors = checkpoint(tmp_path / "model")
    unit = unit_of(tmp_path / "model", "b", "a", "c")
    destination = tmp_path / "out.safetensors"
    fetch_unit(LocalReader(str(tmp_path / "model")), unit, destination, lambda count: None)

    mapped = map_unit(destination)
    try:
        for name in ("a", "b", "c"):
            assert mapped.tensors[name].dtype == tensors[name].dtype
            assert torch.equal(mapped.tensors[name], tensors[name])
    finally:
        mapped.release()


def test_loading_converts_to_the_pool_dtype_except_protected_tensors(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model")
    unit = unit_of(tmp_path / "model", "a", "b")
    destination = tmp_path / "out.safetensors"
    fetch_unit(LocalReader(str(tmp_path / "model")), unit, destination, lambda count: None)

    loaded = load_unit(destination, "cpu", torch.bfloat16, lambda name: name == "a").tensors
    assert loaded["a"].dtype == torch.float32
    assert loaded["b"].dtype == torch.bfloat16
    upcast = load_unit(destination, "cpu", torch.float32, lambda name: False).tensors
    assert upcast["b"].dtype == torch.float32


def test_a_second_fetch_reuses_finished_units(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model")
    unit = unit_of(tmp_path / "model", "a")
    cache = UnitCache(tmp_path / "cache")
    cache.fetch(str(tmp_path / "model"), "m", "r", [unit], FetchProgress("first"))

    again = FetchProgress("second")
    cache.fetch(str(tmp_path / "model"), "m", "r", [unit], again)

    assert again.bytes_total == 0
    assert again.bytes_reused == unit.nbytes


def test_a_truncated_unit_is_fetched_again(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model")
    unit = unit_of(tmp_path / "model", "c")
    cache = UnitCache(tmp_path / "cache")
    cache.fetch(str(tmp_path / "model"), "m", "r", [unit], FetchProgress("first"))
    path = cache.path("m", "r", "unit")
    path.write_bytes(path.read_bytes()[:100])

    assert not cache.has("m", "r", unit)
    progress = FetchProgress("second")
    cache.fetch(str(tmp_path / "model"), "m", "r", [unit], progress)
    assert progress.bytes_done == unit.nbytes


def test_cancelling_keeps_finished_units_and_leaves_no_partial_file(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model")
    finished = unit_of(tmp_path / "model", "a")
    finished.name = "finished"
    pending = unit_of(tmp_path / "model", "c")
    pending.name = "pending"
    cache = UnitCache(tmp_path / "cache", concurrency=1)
    cache.fetch(str(tmp_path / "model"), "m", "r", [finished], FetchProgress("first"))

    progress = FetchProgress("second", cancelled=True)
    with pytest.raises(WeightSourceError, match="cancelled"):
        cache.fetch(str(tmp_path / "model"), "m", "r", [finished, pending], progress)

    assert cache.has("m", "r", finished)
    assert not cache.has("m", "r", pending)
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_eviction_removes_old_units_but_never_the_ones_in_use(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = UnitCache(root, budget_bytes=2500)
    old, keep = root / "old.safetensors", root / "keep.safetensors"
    old.write_bytes(b"x" * 1000)
    keep.write_bytes(b"x" * 1000)

    cache.make_room(1000, {keep})

    assert not old.exists()
    assert keep.exists()


def test_running_out_of_space_says_how_much_is_needed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = UnitCache(root, budget_bytes=1500)
    keep = root / "keep.safetensors"
    keep.write_bytes(b"x" * 1000)
    with pytest.raises(WeightSourceError, match="GiB"):
        cache.make_room(1000, {keep})


def test_manifest_reads_config_and_offsets_without_the_data(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model")
    manifest = build_manifest(str(tmp_path / "model"))
    assert manifest.config["model_type"] == "gpt2"
    assert set(manifest.tensors) == {"a", "b", "c", "d"}
    assert manifest.revision.startswith("local-")


def test_pickle_only_checkpoints_are_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(WeightSourceError, match="pickle"):
        build_manifest(str(tmp_path))


def test_an_absurd_header_length_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(WeightSourceError, match="implausible"):
        read_header(LocalReader(str(tmp_path)), "bad.safetensors")


def hub_reader(handler) -> HubReader:
    reader = HubReader("org/model", "abc")
    reader.client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return reader


class Flaky(httpx.SyncByteStream):
    """Delivers some bytes, then drops the connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data
        raise httpx.ReadError("connection reset")


def test_a_dropped_download_resumes_from_the_last_byte(monkeypatch) -> None:
    monkeypatch.setattr(weights.time, "sleep", lambda seconds: None)
    # httpx hands over whole chunks, so bytes count once a chunk is delivered.
    monkeypatch.setattr(weights, "CHUNK_BYTES", 10)
    payload = bytes(range(100))
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ranges.append(request.headers["range"])
        start, end = (int(part) for part in request.headers["range"].removeprefix("bytes=").split("-"))
        if len(ranges) == 1:
            return httpx.Response(206, stream=Flaky(payload[start : start + 30]))
        return httpx.Response(206, content=payload[start : end + 1])

    assert hub_reader(handler).read_range("model.safetensors", 0, 100) == payload
    assert ranges == ["bytes=0-99", "bytes=30-99"]


def test_a_host_without_range_support_fails_instead_of_downloading_everything() -> None:
    reader = hub_reader(lambda request: httpx.Response(200, content=b"whole file"))
    with pytest.raises(WeightSourceError, match="range request"):
        reader.read_range("model.safetensors", 0, 5)


def test_repeated_failures_give_up(monkeypatch) -> None:
    monkeypatch.setattr(weights.time, "sleep", lambda seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(WeightSourceError, match="gave up"):
        hub_reader(handler).read_range("model.safetensors", 0, 5)


def test_unknown_or_gated_models_are_explained() -> None:
    assert_message = {404: "was not found", 401: "HF_TOKEN"}
    for status, message in assert_message.items():
        reader = hub_reader(lambda request, status=status: httpx.Response(status, content=json.dumps({})))
        with pytest.raises(WeightSourceError, match=message):
            reader.resolve_revision("main")


def test_tls_verification_stays_on() -> None:
    context = tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_ref_size_counts_every_tensor() -> None:
    first = TensorRef(name="a", file="f", dtype="BF16", shape=[2], start=0, end=4)
    second = TensorRef(name="b", file="f", dtype="BF16", shape=[3], start=4, end=10)
    assert WeightUnit(name="u", kind="dense", tensors=[first, second]).nbytes == 10


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")


def many_expert_tensors(directory: Path) -> WeightUnit:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {f"expert.{index}": torch.full((704, 2048), float(index), dtype=torch.bfloat16) for index in range(24)}
    save_file(tensors, directory / "model.safetensors")
    return unit_of(directory, *tensors)


@needs_gpu
def test_gpu_units_do_not_lose_memory_to_allocator_rounding(tmp_path: Path) -> None:
    unit = many_expert_tensors(tmp_path / "model")
    destination = tmp_path / "unit.safetensors"
    fetch_unit(LocalReader(str(tmp_path / "model")), unit, destination, lambda count: None)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved(0)

    loaded = load_unit(destination, "cuda:0", torch.bfloat16, lambda name: False)

    reserved = torch.cuda.memory_reserved(0) - before
    assert reserved <= unit.nbytes * 1.02 + 2 * 1024**2
    assert torch.equal(loaded.tensors["expert.7"].cpu(), torch.full((704, 2048), 7.0, dtype=torch.bfloat16))
    loaded.release()
    torch.cuda.empty_cache()


@needs_gpu
def test_gpu_slab_keeps_mixed_dtypes_and_conversions_correct(tmp_path: Path) -> None:
    tensors = checkpoint(tmp_path / "model")
    unit = unit_of(tmp_path / "model", "a", "b", "c", "d")
    destination = tmp_path / "unit.safetensors"
    fetch_unit(LocalReader(str(tmp_path / "model")), unit, destination, lambda count: None)

    loaded = load_unit(destination, "cuda:0", torch.bfloat16, lambda name: name == "d").tensors

    assert loaded["a"].dtype == torch.bfloat16
    assert loaded["d"].dtype == torch.float32
    assert torch.equal(loaded["d"].cpu(), tensors["d"])
    assert torch.equal(loaded["c"].cpu(), tensors["c"].to(torch.bfloat16))
    assert all(tensor.data_ptr() % 256 == 0 for tensor in loaded.values())


@needs_gpu
def test_gpu_slab_holds_fp8_in_either_form(tmp_path: Path) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    weight = torch.randn(256, 256).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.bfloat16)
    save_file({"w.weight": weight, "w.weight_scale_inv": scale}, directory / "model.safetensors")
    unit = unit_of(directory, "w.weight", "w.weight_scale_inv")
    destination = tmp_path / "unit.safetensors"
    fetch_unit(LocalReader(str(directory)), unit, destination, lambda count: None)

    expanded = load_unit(destination, "cuda:0", torch.bfloat16, lambda name: False).tensors
    compact = load_unit(destination, "cuda:0", torch.bfloat16, lambda name: False, compact=True).tensors

    assert set(expanded) == {"w.weight"} and expanded["w.weight"].dtype == torch.bfloat16
    assert compact["w.weight"].dtype == torch.float8_e4m3fn and "w.weight_scale_inv" in compact
    rebuilt = weights.dequantize_fp8(compact["w.weight"], compact["w.weight_scale_inv"], torch.bfloat16)
    assert torch.equal(rebuilt, expanded["w.weight"])
