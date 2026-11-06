import io
import os
from pathlib import Path
import re
import struct

import httpx
import numpy as np
import pytest

from isharded.gguf import (
    GgufError,
    GgufHeader,
    STUB_VOCAB,
    T_ARRAY,
    T_INT32,
    T_STRING,
    T_UINT32,
    parse_header,
    parse_source,
    plan_stage,
    read_numbers,
    read_strings,
    write_stage,
)
from isharded.gguf import GgufHeader, split_parts
from isharded.gguf_stage import cpu_tensor_patterns
from isharded.models import create_adapter
from isharded.planner import Candidate, ModelProfile, plan_placement, stored_bytes
from isharded.protocol import MemoryPool, TensorRef, UnitPlacement, WeightUnit
from isharded.weights import build_manifest, iter_unit_tensor, unit_tensor_span, UnitCache, FetchProgress

WIDTH = 8
LAYERS = 8


def text(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def key(name, kind, payload):
    return text(name) + struct.pack("<I", kind) + payload


def build_file(*, tied=False, interval=4, arch="qwen35"):
    """A tiny but valid GGUF: named tensors holding recognizable bytes."""
    tensors = [("token_embd.weight", 4 * WIDTH * 4), ("output_norm.weight", WIDTH * 4)]
    if not tied:
        tensors.append(("output.weight", 4 * WIDTH * 4))
    for layer in range(LAYERS):
        tensors.append((f"blk.{layer}.attn_norm.weight", WIDTH * 4))
        tensors.append((f"blk.{layer}.ffn_up.weight", WIDTH * WIDTH * 4))
    tensors.append((f"blk.{LAYERS}.nextn.eh_proj.weight", WIDTH * 4))

    tokens = ["a", "b", "c", "d"]
    keys = [
        key("general.architecture", T_STRING, text(arch)),
        key(f"{arch}.block_count", T_UINT32, struct.pack("<I", LAYERS + 1)),
        key(f"{arch}.nextn_predict_layers", T_UINT32, struct.pack("<I", 1)),
        key(f"{arch}.embedding_length", T_UINT32, struct.pack("<I", WIDTH)),
        key(f"{arch}.attention.head_count", T_UINT32, struct.pack("<I", 2)),
        key(f"{arch}.attention.head_count_kv", T_UINT32, struct.pack("<I", 1)),
        key(f"{arch}.feed_forward_length", T_ARRAY, struct.pack("<IQ", T_INT32, LAYERS + 1) + struct.pack(f"<{LAYERS + 1}i", *range(LAYERS + 1))),
        key("tokenizer.ggml.tokens", T_ARRAY, struct.pack("<IQ", T_STRING, 4) + b"".join(text(t) for t in tokens)),
    ]
    if interval:
        keys.append(key(f"{arch}.full_attention_interval", T_UINT32, struct.pack("<I", interval)))

    infos = b""
    data = b""
    for index, (name, size) in enumerate(tensors):
        infos += text(name) + struct.pack("<I", 1) + struct.pack("<Q", size // 4) + struct.pack("<IQ", 0, len(data))
        data += bytes([index % 251]) * size
        data += bytes(-len(data) % 32)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keys)) + b"".join(keys) + infos
    head += bytes(-len(head) % 32)
    return head + data


def header_of(blob):
    return parse_header(lambda offset, length: blob[offset : offset + length], len(blob))


def write_to_bytes(header, plan, blob):
    def read_tensor(tensor):
        yield blob[header.data_start + tensor.offset : header.data_start + tensor.offset + tensor.nbytes]

    out = io.BytesIO()
    write_stage(header, plan, out, read_tensor)
    return out.getvalue()


def test_parse_reads_keys_and_tensors():
    header = header_of(build_file())
    assert header.architecture == "qwen35"
    assert header.layer_count == LAYERS
    assert header.hidden_size == WIDTH
    assert header.vocab_size == 4
    assert not header.tied_embeddings
    assert header.tensors["blk.3.ffn_up.weight"].nbytes == WIDTH * WIDTH * 4
    assert read_strings(header.keys["tokenizer.ggml.tokens"]) == ["a", "b", "c", "d"]
    assert read_numbers(header.keys["qwen35.feed_forward_length"])[:3] == [0, 1, 2]


def test_a_small_read_window_still_parses(monkeypatch):
    monkeypatch.setattr("isharded.gguf.READ_CHUNK", 7)
    assert header_of(build_file()).layer_count == LAYERS


def test_not_a_gguf_is_refused():
    with pytest.raises(GgufError):
        header_of(b"nope" + bytes(64))


def test_stage_renumbers_its_layers_and_keeps_their_bytes():
    blob = build_file()
    header = header_of(blob)
    written = write_to_bytes(header, plan_stage(header, 4, 8, last=True), blob)
    stage = header_of(written)
    assert stage.get("qwen35.block_count") == 4
    assert stage.get("qwen35.nextn_predict_layers") == 0
    assert "blk.0.ffn_up.weight" in stage.tensors and "blk.4.ffn_up.weight" not in stage.tensors
    source = header.tensors["blk.5.ffn_up.weight"]
    moved = stage.tensors["blk.1.ffn_up.weight"]
    expected = blob[header.data_start + source.offset : header.data_start + source.offset + source.nbytes]
    assert written[stage.data_start + moved.offset : stage.data_start + moved.offset + moved.nbytes] == expected


def test_per_layer_arrays_are_sliced():
    blob = build_file()
    header = header_of(blob)
    stage = header_of(write_to_bytes(header, plan_stage(header, 4, 8, last=False), blob))
    assert read_numbers(stage.keys["qwen35.feed_forward_length"]) == [4, 5, 6, 7]


def test_a_middle_stage_fetches_only_its_layers():
    header = header_of(build_file())
    plan = plan_stage(header, 0, 4, last=False)
    names = {source.name for _, source in plan.tensors}
    assert names == {tensor.name for layer in range(4) for tensor in header.layer_tensors(layer)}
    assert {stub.name for stub in plan.stubs} == {"output_norm.weight", "token_embd.weight"}
    blob = build_file()
    stage = header_of(write_to_bytes(header, plan, blob))
    assert stage.vocab_size == STUB_VOCAB


def test_last_stage_has_the_real_vocabulary_size_and_output():
    blob = build_file()
    header = GgufHeader.from_skeleton(header_of(blob).skeleton())
    plan = plan_stage(header, 4, 8, last=True)
    assert "output.weight" in {name for name, _ in plan.tensors}
    assert [stub.name for stub in plan.stubs] == ["token_embd.weight"]
    assert next(entry for entry in plan.keys if entry.key == "tokenizer.ggml.tokens").value == 4


def test_tied_model_keeps_the_table_on_the_last_stage():
    header = header_of(build_file(tied=True))
    plan = plan_stage(header, 4, 8, last=True)
    assert "token_embd.weight" in {name for name, _ in plan.tensors}
    assert not plan.stubs


def test_skeleton_round_trip_drops_only_the_tokenizer():
    header = header_of(build_file())
    copy = GgufHeader.from_skeleton(header.skeleton())
    assert copy.architecture == header.architecture
    assert copy.layer_count == header.layer_count
    assert copy.vocab_size == header.vocab_size
    assert "tokenizer.ggml.tokens" not in copy.keys
    assert copy.tensors["blk.2.ffn_up.weight"].nbytes == header.tensors["blk.2.ffn_up.weight"].nbytes


def test_hybrid_stage_must_start_on_the_attention_pattern():
    header = header_of(build_file(interval=4))
    with pytest.raises(GgufError, match="multiple of 4"):
        plan_stage(header, 2, 8, last=True)


def test_layers_outside_the_stack_are_refused():
    header = header_of(build_file())
    with pytest.raises(GgufError):
        plan_stage(header, 4, LAYERS + 1, last=True)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("owner/repo:model-Q4.gguf", ("owner/repo", "model-Q4.gguf")),
        ("Qwen/Qwen3-8B", None),
    ],
)
def test_parse_source(source, expected):
    assert parse_source(source) == expected


def test_parse_source_finds_a_local_file(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    assert parse_source(str(path)) == (str(tmp_path), "m.gguf")


@pytest.mark.parametrize("source", ["m.gguf", r"C:\nowhere\m.gguf", "/models/m.gguf"])
def test_a_name_not_on_this_machine_means_whoever_has_it(source):
    assert parse_source(source) == ("", "m.gguf")


def test_a_file_nobody_has_is_a_clear_error():
    from isharded.weights import WeightSourceError, open_reader

    with pytest.raises(WeightSourceError, match="no node in the pool has m.gguf"):
        open_reader("m.gguf")


@pytest.fixture
def local_model(tmp_path):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(build_file())
    return path


def test_manifest_becomes_placeable_units(local_model):
    manifest = build_manifest(str(local_model))
    adapter = create_adapter(manifest.config)
    assert adapter.num_layers == LAYERS
    assert not adapter.supports_kv_snapshots
    units = {unit.name: unit for unit in adapter.partition(manifest.tensors)}
    assert {"embed", "head", "layer-00000.dense", "layer-00007.dense"} <= set(units)
    assert not any("00008" in name for name in units), "prediction blocks stay behind"
    assert units["layer-00003.dense"].raw
    assert {tensor.name for tensor in units["head"].tensors} == {"output.weight", "output_norm.weight"}


def test_unsupported_architecture_is_refused(tmp_path):
    path = tmp_path / "other.gguf"
    path.write_bytes(build_file(arch="gemma4"))
    manifest = build_manifest(str(path))
    with pytest.raises(ValueError, match="cannot be split"):
        create_adapter(manifest.config)


def test_raw_units_stay_their_size_in_every_pool():
    unit = WeightUnit(name="u", kind="dense", layer=0, tensors=[TensorRef(name="t", file="f", dtype="U8", shape=[100], start=0, end=100)])
    pool = MemoryPool(pool_id="p", kind="vram", device="cuda:0", capacity_group="g", dtype="float32")
    assert stored_bytes(unit, pool) == 100


def pool(kind, gib, group):
    device = "cuda:0" if kind == "vram" else "cpu"
    size = int(gib * 1024**3)
    return MemoryPool(pool_id=f"{group}-{kind}", kind=kind, device=device, capacity_group=f"{group}-{kind}", budget_bytes=size, free_bytes=size, bandwidth_gbps=500 if kind == "vram" else 30)


def profile_of(layers=16, size=256 * 1024**2, step=4, workspace=0, experts=0):
    def unit(name, kind, layer=None, nbytes=size):
        return WeightUnit(name=name, kind=kind, layer=layer, tensors=[TensorRef(name=name, file="f", dtype="U8", shape=[nbytes], start=0, end=nbytes)])

    units = [unit("embed", "embed", nbytes=1024), unit("head", "head", nbytes=1024)]
    units += [unit(f"layer-{i:05d}.dense", "dense", i) for i in range(layers)]
    if experts:
        units += [unit(f"layer-{i:05d}.experts", "experts", i, experts) for i in range(layers)]
    return ModelProfile(
        units=units,
        num_layers=layers,
        expert_active_fraction=1.0,
        kv_bytes_per_token=[0] * layers,
        state_bytes=[0] * layers,
        layer_step=step,
        stage_workspace_bytes=workspace,
    )


def test_stage_boundaries_follow_the_attention_pattern():
    big = Candidate("big", "http://a", [pool("vram", 2.4, "big")])
    small = Candidate("small", "http://b", [pool("vram", 2.4, "small")])
    route = plan_placement([big, small], [], profile_of(), spread=True)
    assert len(route) == 2
    assert all(stage.start_layer % 4 == 0 for stage in route)


def test_plan_stage_accepts_every_stage_the_planner_makes():
    header = header_of(build_file())
    for gib in (2.4, 1.3, 0.6):
        nodes = [Candidate(name, f"http://{name}", [pool("vram", gib, name), pool("ram", 8, name)]) for name in ("a", "b", "c")]
        route = plan_placement(nodes[:2], nodes[2:], profile_of(layers=LAYERS), spread=True)
        for stage in route:
            plan_stage(header, stage.start_layer, stage.end_layer, last=stage.end_layer == LAYERS)


def test_a_spilled_block_reaches_llama_cpp_as_expert_patterns():
    # The first stage's card is short of its second block by less than one layer's
    # experts, so one layer's experts goes to RAM and the block stays whole.
    gib = 1024**3
    profile = profile_of(layers=12, size=gib // 4, experts=gib)
    small = Candidate("small", "http://a", [pool("vram", 9.8, "small"), pool("ram", 8, "small")])
    big = Candidate("big", "http://b", [pool("vram", 5.4, "big")])
    stage = next(item for item in plan_placement([small, big], [], profile, spread=True) if item.worker_id == "small")
    spilled = [item for item in stage.placements if item.pool_id == "small-ram"]
    assert len(spilled) == 1 and spilled[0].unit.endswith(".experts")

    pools = {item.pool_id: item for item in small.pools}
    patterns = cpu_tensor_patterns(stage.placements, pools, stage.start_layer)
    local = int(spilled[0].unit[6:11]) - stage.start_layer
    assert patterns == [rf"^blk\.({local})\.ffn_[a-z_]*_exps\."]
    assert re.search(patterns[0], f"blk.{local}.ffn_down_exps.weight")
    assert re.search(patterns[0], f"blk.{local}.ffn_gate_up_exps.weight")
    assert not re.search(patterns[0], f"blk.{local}.ffn_gate_shexp.weight")
    assert not re.search(patterns[0], f"blk.{local}.attn_norm.weight")
    assert not re.search(patterns[0], f"blk.{local + 1}.ffn_down_exps.weight")


def test_stage_workspace_is_kept_out_of_gpu_memory():
    node = Candidate("one", "http://a", [pool("vram", 4.5, "one")])
    profile = profile_of(workspace=1024**3)
    with pytest.raises(ValueError):
        plan_placement([node], [], profile, spread=True)


def test_cpu_patterns_cover_only_slow_placements():
    pools = {
        "fast": MemoryPool(pool_id="fast", kind="vram", device="cuda:0", capacity_group="v"),
        "ram": MemoryPool(pool_id="ram", kind="ram", device="cpu", capacity_group="r"),
        "disk": MemoryPool(pool_id="disk", kind="disk", device="cpu", capacity_group="d"),
    }
    placements = [
        UnitPlacement(unit="layer-00012.dense", pool_id="fast", bytes=1),
        UnitPlacement(unit="layer-00013.dense", pool_id="ram", bytes=1),
        UnitPlacement(unit="layer-00014.experts", pool_id="disk", bytes=1),
        UnitPlacement(unit="head", pool_id="ram", bytes=1),
    ]
    patterns = cpu_tensor_patterns(placements, pools, 12)
    assert r"^blk\.(1)\." in patterns
    assert r"^blk\.(2)\.ffn_[a-z_]*_exps\." in patterns
    assert r"^output(_norm)?\.weight$" in patterns
    assert not any("(0)" in pattern for pattern in patterns)


def test_all_fast_placements_need_no_patterns():
    pools = {"fast": MemoryPool(pool_id="fast", kind="vram", device="cuda:0", capacity_group="v")}
    assert cpu_tensor_patterns([UnitPlacement(unit="layer-00000.dense", pool_id="fast", bytes=1)], pools, 0) == []


def test_cached_units_round_trip_their_bytes(local_model, tmp_path):
    manifest = build_manifest(str(local_model))
    adapter = create_adapter(manifest.config)
    units = adapter.partition(manifest.tensors)
    cache = UnitCache(tmp_path / "cache")
    cache.fetch(str(local_model), "m", manifest.revision, units, FetchProgress("job"))
    blob = local_model.read_bytes()
    unit = next(item for item in units if item.name == "layer-00002.dense")
    tensor = next(item for item in unit.tensors if item.name == "blk.2.ffn_up.weight")
    path = cache.path("m", manifest.revision, unit.name)
    assert b"".join(iter_unit_tensor(path, tensor.name)) == blob[tensor.start : tensor.end]
    offset, length = unit_tensor_span(path, tensor.name)
    assert path.read_bytes()[offset : offset + length] == blob[tensor.start : tensor.end]


def test_a_local_copy_is_used_instead_of_the_source(local_model, tmp_path):
    manifest = build_manifest(str(local_model))
    adapter = create_adapter(manifest.config)
    units = adapter.partition(manifest.tensors)
    other = tmp_path / "other"
    other.mkdir()
    copy = other / local_model.name
    copy.write_bytes(local_model.read_bytes())
    cache = UnitCache(tmp_path / "cache2", local_dirs=(other,))
    assert cache.local_copy(local_model.name, local_model.stat().st_size) == copy
    assert cache.local_copy(local_model.name, 1) is None
    cache.fetch(
        "owner/does-not-exist:tiny.gguf",
        "m",
        manifest.revision,
        units,
        FetchProgress("job"),
        file_sizes={local_model.name: local_model.stat().st_size},
    )
    assert all(cache.has("m", manifest.revision, unit) for unit in units)


BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
needs_real_model = pytest.mark.skipif(
    not (BUILD and MODEL and Path(BUILD).is_dir() and Path(MODEL).is_file()), reason="needs a llama.cpp build and a real GGUF"
)


@needs_real_model
def test_tokenizer_matches_llama_cpp():
    import ctypes as C

    from isharded.gguf import read_file_header
    from isharded.llamacpp import LlamaLibrary
    from isharded.tokenizer import tokenizer_from_gguf

    library = LlamaLibrary.load(BUILD)
    lib = library.llama
    params = lib.llama_model_default_params()
    params.vocab_only = True
    model = lib.llama_model_load_from_file(MODEL.encode(), params)
    vocab = lib.llama_model_get_vocab(model)
    lib.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32), C.c_int32, C.c_bool, C.c_bool]
    lib.llama_tokenize.restype = C.c_int32

    def reference(value):
        raw = value.encode()
        output = (C.c_int32 * (len(raw) + 16))()
        count = lib.llama_tokenize(vocab, raw, len(raw), output, len(output), False, True)
        return list(output[:count])

    tokenizer = tokenizer_from_gguf(read_file_header(MODEL))
    samples = [
        "Hello, world!",
        "The capital of France is Paris.\n\nIt's 2026; don't panic.",
        "<|im_start|>user\nWhat is 12 x 12?<|im_end|>\n<|im_start|>assistant\n",
        "def add(a, b):\n    return a + b  # sum\n",
        "café naïve 日本語 🙂 emoji",
        "   leading and   multiple   spaces\t\ttabs\r\n",
    ]
    for sample in samples:
        assert tokenizer.encode(sample) == reference(sample), sample
        assert tokenizer.decode(tokenizer.encode(sample)).replace("<|im_start|>", "").replace("<|im_end|>", "") == sample.replace("<|im_start|>", "").replace("<|im_end|>", "")
    assert tokenizer.stop_token_ids
    assert tokenizer.chat_template
    lib.llama_model_free(model)


@needs_real_model
def test_two_stages_match_the_whole_model(tmp_path):
    """Runs in stage processes, as workers do: torch is loaded here and they must stay apart."""
    from isharded.gguf import file_reader, read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    reader = file_reader(MODEL)
    cut = 12
    paths = []
    for index, (start, end, last) in enumerate(((0, cut, False), (cut, header.layer_count, True))):
        path = tmp_path / f"stage{index}.gguf"
        with path.open("wb") as out:
            write_stage(header, plan_stage(header, start, end, last=last), out, reader)
        paths.append(str(path))
    table = header.tensors["token_embd.weight"]
    embed = {
        "path": MODEL,
        "offset": header.data_start + table.offset,
        "length": table.nbytes,
        "ggml_type": table.ggml_type,
    }
    common = {"folder": BUILD, "extra_paths": [], "context_tokens": 256, "gpu_layers": -1, "threads": 4, "cpu_tensors": []}
    tokens = [9707, 11, 1246, 525, 498]

    whole = StageHost()
    first = StageHost()
    second = StageHost()
    try:
        whole.call("load", path=MODEL, last=True, table=embed, **common)
        rows = whole.call("embed", tokens=tokens)[1]
        expected = np.frombuffer(whole.call("forward", rows, position=0)[1], dtype=np.float32)
        whole.close()
        first.call("load", path=paths[0], last=False, table=None, **common)
        second.call("load", path=paths[1], last=True, table=None, **common)
        middle = first.call("forward", rows, position=0)[1]
        assert len(middle) == len(rows)
        got = np.frombuffer(second.call("forward", middle, position=0)[1], dtype=np.float32)
    finally:
        first.close()
        second.close()
    assert int(np.argmax(got)) == int(np.argmax(expected))
    assert np.allclose(got, expected, atol=1e-2)


@needs_real_model
def test_a_failed_stage_reports_instead_of_crashing_the_worker(tmp_path):
    from isharded.llama_host import StageHost
    from isharded.llamacpp import LlamaError

    broken = tmp_path / "broken.gguf"
    broken.write_bytes(b"GGUF" + bytes(64))
    host = StageHost()
    try:
        with pytest.raises(LlamaError):
            host.call(
                "load", folder=BUILD, extra_paths=[], path=str(broken), last=True, context_tokens=128,
                gpu_layers=0, threads=2, cpu_tensors=[], table=None,
            )
    finally:
        host.close()


@needs_real_model
def test_layers_kept_in_ram_give_the_same_answer_as_layers_on_the_gpu():
    """The GPU plus RAM split: the plan's CPU placements must change speed, not results."""
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    common = {"folder": BUILD, "extra_paths": [], "path": MODEL, "last": True, "context_tokens": 256, "gpu_layers": -1, "threads": 4, "table": embed}
    tokens = [9707, 11, 1246, 525, 498]
    answers = []
    for patterns in ([], [r"^blk\.(1|2|3|4|5|6|7)\.", r"^output(_norm)?\.weight$"]):
        host = StageHost()
        try:
            host.call("load", cpu_tensors=patterns, **common)
            rows = host.call("embed", tokens=tokens)[1]
            answers.append(np.frombuffer(host.call("forward", rows, position=0)[1], dtype=np.float32).copy())
        finally:
            host.close()
    assert int(np.argmax(answers[0])) == int(np.argmax(answers[1]))
    cosine = float(np.dot(answers[0], answers[1]) / (np.linalg.norm(answers[0]) * np.linalg.norm(answers[1])))
    assert cosine > 0.999


def one_hot_rows(*choices, vocabulary=12):
    """Rows of logits whose greedy choice is each given token."""
    rows = np.zeros((len(choices), vocabulary), dtype=np.float32)
    for row, choice in enumerate(choices):
        rows[row, choice] = 10.0
    return rows


@pytest.mark.parametrize(
    "guesses, choices, expected",
    [
        ([5, 6, 7], [5, 6, 7, 8], [5, 6, 7, 8]),  # every guess right, plus one free
        ([5, 6, 7], [5, 9, 7, 8], [5, 9]),  # stops at the first disagreement
        ([5, 6, 7], [9, 6, 7, 8], [9]),  # no guess right still yields one token
        ([], [4], [4]),
    ],
)
def test_greedy_keeps_guesses_while_they_match(guesses, choices, expected):
    from isharded.llamacpp import speculative_choice

    rows = one_hot_rows(*choices)
    assert speculative_choice(lambda row: rows[row], guesses, 0.0, 1.0, 0, np.random.default_rng(0)) == expected


@pytest.mark.parametrize(
    "temperature, top_p, top_k, guess",
    [(1.0, 1.0, 0, 1), (0.7, 1.0, 0, 0), (1.0, 0.9, 0, 2), (0.7, 0.9, 3, 1), (1.0, 1.0, 2, 4)],
)
def test_sampling_with_a_draft_keeps_the_model_distribution(temperature, top_p, top_k, guess):
    """The first token out of a pass must follow the same law with or without a guess."""
    from isharded.llamacpp import filtered_distribution, speculative_choice

    logits = np.array([2.0, 1.4, 1.0, 0.2, -0.5, -1.0], dtype=np.float32)
    ids, expected = filtered_distribution(logits, temperature, top_p, top_k)
    wanted = dict(zip(ids.tolist(), expected.tolist()))
    generator = np.random.default_rng(7)
    draws = 60000
    counts = np.zeros(logits.size)
    for _ in range(draws):
        first = speculative_choice(lambda row: logits, [guess], temperature, top_p, top_k, generator)[0]
        counts[first] += 1
    for token in range(logits.size):
        assert counts[token] / draws == pytest.approx(wanted.get(token, 0.0), abs=0.008)


def test_a_filtered_out_guess_is_never_accepted():
    from isharded.llamacpp import speculative_choice

    logits = np.array([5.0, 4.0, -9.0, -9.0], dtype=np.float32)
    generator = np.random.default_rng(1)
    for _ in range(300):
        assert speculative_choice(lambda row: logits, [3], 1.0, 1.0, 2, generator)[0] in {0, 1}


def test_a_pass_never_returns_more_than_guesses_plus_one():
    from isharded.llamacpp import speculative_choice

    logits = np.zeros(4, dtype=np.float32)
    generator = np.random.default_rng(2)
    for _ in range(200):
        assert 1 <= len(speculative_choice(lambda row: logits, [1, 2, 3], 1.0, 1.0, 0, generator)) <= 4


def test_filtered_distribution_matches_the_regular_sampler():
    import torch

    from isharded.llamacpp import filtered_distribution

    logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    temperature, top_p, top_k = 0.7, 0.9, 5
    cutoff = torch.topk(logits, top_k).values[-1]
    masked = logits.masked_fill(logits < cutoff, float("-inf"))
    reference = torch.softmax(masked / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(reference, descending=True)
    sorted_probs[torch.cumsum(sorted_probs, dim=-1) - sorted_probs > top_p] = 0
    reference = torch.zeros_like(reference).scatter(0, sorted_indices, sorted_probs)
    reference /= reference.sum()
    ids, probabilities = filtered_distribution(logits.numpy(), temperature, top_p, top_k)
    ours = np.zeros(8)
    ours[ids] = probabilities
    assert np.allclose(ours, reference.numpy(), atol=1e-5)


@needs_real_model
def test_a_rewound_stage_continues_exactly(tmp_path):
    """Run wrong guesses through a stage, rewind, and carry on: nothing may leak."""
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    common = {
        "folder": BUILD, "extra_paths": [], "path": MODEL, "last": True, "context_tokens": 256,
        "gpu_layers": -1, "threads": 4, "table": embed, "cpu_tensors": [], "rollback_tokens": 4,
    }
    prompt = [9707, 11, 1246, 525, 498]
    host = StageHost()
    try:
        host.call("load", **common)

        def rows(tokens):
            return host.call("embed", tokens=tokens)[1]

        def choice(payload, position):
            return int(np.argmax(np.frombuffer(host.call("forward", payload, position=position)[1], dtype=np.float32)))

        expected = [choice(rows(prompt), 0)]
        for step in range(5):
            expected.append(choice(rows(expected[-1:]), len(prompt) + step))

        host.call("clear")
        got = [choice(rows(prompt), 0)]
        position = len(prompt)
        for _ in range(5):
            answer = host.call(
                "verify", rows([got[-1], 7, 8, 9]), position=position, guesses=[7, 8, 9],
                temperature=0.0, top_p=1.0, top_k=0, seed=None,
            )[0]["tokens"]
            got.append(answer[0])
            position += 1
            host.call("truncate", keep=position)
    finally:
        host.close()
    assert got == expected


@needs_real_model
def test_draft_guesses_extend_from_committed_text():
    from isharded.llama_host import StageHost

    host = StageHost()
    common = {"folder": BUILD, "extra_paths": [], "path": MODEL, "context_tokens": 256, "gpu_layers": -1, "threads": 4}
    prompt = [9707, 11, 1246, 525, 498]
    try:
        host.call("load_draft", **common)
        first = host.call("guess", commit=prompt, position=0, count=4)[0]["tokens"]
        assert len(first) == 4
        # Committing the first guess must predict the same continuation, shifted by one.
        second = host.call("guess", commit=first[:1], position=len(prompt), count=3)[0]["tokens"]
        assert second == first[1:]
    finally:
        host.close()


def test_a_draft_needs_a_matching_vocabulary(local_model, tmp_path, monkeypatch):
    from isharded import orchestrator
    from isharded.protocol import LoadModelRequest

    manifest = build_manifest(str(local_model))
    adapter = create_adapter(manifest.config)
    other = tmp_path / "other.gguf"
    blob = build_file().replace(b"\x01\x00\x00\x00\x00\x00\x00\x00a", b"\x01\x00\x00\x00\x00\x00\x00\x00z", 1)
    other.write_bytes(blob)
    request = LoadModelRequest(model_id=str(local_model), draft_model=str(other))
    with pytest.raises(ValueError, match="different vocabulary"):
        orchestrator.resolve_draft(request, manifest, adapter)
    assert orchestrator.resolve_draft(LoadModelRequest(model_id=str(local_model), draft_model="none"), manifest, adapter) is None


def test_auto_draft_never_fails_a_load(local_model, monkeypatch):
    from isharded import orchestrator
    from isharded.protocol import LoadModelRequest

    def unavailable(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(orchestrator, "build_manifest", unavailable)
    manifest = build_manifest(str(local_model))
    adapter = create_adapter(manifest.config)
    assert orchestrator.resolve_draft(LoadModelRequest(model_id=str(local_model)), manifest, adapter) is None


def test_forward_can_return_a_speculative_answer(monkeypatch):
    """Verify passes return a list of choices, which must survive the HTTP layer."""
    import torch
    from fastapi.testclient import TestClient

    from isharded import worker
    from isharded.protocol import InferRequest, encode_activation, encode_meta

    class Runtime:
        lock = __import__("asyncio").Lock()
        last_error = None

        def _record(self, *args):
            pass

        async def execute(self, hidden, request, places=None):
            assert request.verify and request.rollback_to == 5
            return {"token_ids": [1, 2, 3]}

    monkeypatch.setattr(worker, "runtime", Runtime())
    monkeypatch.setattr(worker, "verify_token", lambda value: None)
    meta = InferRequest(session_id="s", epoch=1, input_ids=[1], verify=True, rollback_to=5)
    client = TestClient(worker.app, raise_server_exceptions=False)
    response = client.post(
        "/internal/forward",
        content=encode_activation(torch.zeros(1, 1, 4)),
        headers={"x-isharded-meta": encode_meta(meta.model_dump())},
    )
    assert response.status_code == 200
    assert response.json() == {"token_ids": [1, 2, 3]}


@pytest.mark.parametrize(
    "current, right, limit, expected",
    [
        (3, 3, 6, 4),  # all landed: reach further
        (6, 6, 6, 6),  # never past the limit, which is what stages can rewind
        (3, 1, 6, 3),  # some landed: hold
        (3, 0, 6, 2),  # none landed: pull back
        (2, 0, 6, 2),  # but keep guessing
        (1, 0, 1, 1),  # a limit of one stays one
    ],
)
def test_guess_count_follows_what_lands(current, right, limit, expected):
    from isharded.gguf_stage import next_guess_count

    assert next_guess_count(current, right, limit) == expected


@needs_real_model
def test_a_restored_checkpoint_continues_exactly():
    """Save mid-conversation, wander off, restore, and get the same answer as before."""
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost
    from isharded.llamacpp import LlamaError

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    host = StageHost()
    try:
        host.call(
            "load", folder=BUILD, extra_paths=[], path=MODEL, last=True, context_tokens=256,
            gpu_layers=-1, threads=4, table=embed, cpu_tensors=[], rollback_tokens=4, state_ram_bytes=1024**3,
        )

        def rows(tokens):
            return host.call("embed", tokens=tokens)[1]

        def logits(tokens, position):
            return np.frombuffer(host.call("forward", rows(tokens), position=position)[1], dtype=np.float32).copy()

        history = [9707, 11, 1246, 525, 498]
        logits(history, 0)
        assert host.call("save", key="history", length=len(history))[0]["states"]["ram_bytes"] > 0
        expected = logits([3351, 30], len(history))

        logits([100, 200, 300], len(history) + 2)  # the conversation carries on
        assert host.call("restore", key="history")[0]["tier"] == "ram"
        assert np.array_equal(logits([3351, 30], len(history)), expected)

        with pytest.raises(LlamaError, match="no saved state"):
            host.call("restore", key="another conversation of the same length")
    finally:
        host.close()


class Holder:
    """A fake worker serving a file through httpx."""

    def __init__(self, blob, fail_after=None):
        self.blob = blob
        self.fail_after = fail_after
        self.ranges = []

    def __call__(self, request):
        if request.url.path.endswith("/index"):
            return httpx.Response(200, json={"files": [{"name": "m.gguf", "size": len(self.blob)}]})
        start, end = int(request.url.params["start"]), int(request.url.params["end"])
        self.ranges.append((start, end))
        if self.fail_after is not None and start >= self.fail_after:
            return httpx.Response(404)
        return httpx.Response(200, content=self.blob[start:end])


def peer_reader(handler, urls=("http://a",)):
    from isharded.weights import PeerReader

    reader = PeerReader(list(urls))
    reader.client = httpx.Client(transport=httpx.MockTransport(handler))
    return reader


def test_a_model_can_be_read_entirely_from_a_peer():
    from isharded.weights import read_gguf_header

    blob = build_file()
    reader = peer_reader(Holder(blob))
    assert reader.file_size("m.gguf") == len(blob)
    assert read_gguf_header(reader, "m.gguf").layer_count == LAYERS
    assert reader.read_range("m.gguf", 10, 50) == blob[10:50]


def test_a_failed_holder_hands_over_to_the_next():
    blob = bytes(range(256))
    good = Holder(blob)
    bad = Holder(blob, fail_after=0)

    def handler(request):
        return (bad if request.url.host == "bad" else good)(request)

    reader = peer_reader(handler, ("http://bad", "http://good"))
    assert reader.read_range("m.gguf", 5, 25) == blob[5:25]


def test_nobody_answering_is_an_error_not_short_data():
    from isharded.weights import WeightSourceError

    reader = peer_reader(Holder(bytes(64), fail_after=0))
    with pytest.raises(WeightSourceError, match="no node could send"):
        reader.read_range("m.gguf", 0, 10)


def test_every_node_fingerprints_a_file_the_same_way(tmp_path):
    from isharded.weights import LocalReader, file_fingerprint

    blob = build_file()
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "m.gguf").write_bytes(blob)
    (two / "m.gguf").write_bytes(blob)
    first = file_fingerprint(LocalReader(str(one)), "m.gguf", len(blob))
    assert first == file_fingerprint(LocalReader(str(two)), "m.gguf", len(blob))
    assert first == file_fingerprint(peer_reader(Holder(blob)), "m.gguf", len(blob))
    (two / "m.gguf").write_bytes(b"x" + blob[1:])
    assert first != file_fingerprint(LocalReader(str(two)), "m.gguf", len(blob))


def test_a_manifest_can_come_from_the_pool_alone(monkeypatch):
    from isharded import weights

    blob = build_file()
    reader = peer_reader(Holder(blob))
    monkeypatch.setattr(weights, "PeerReader", lambda holders, token=None: reader)
    manifest = build_manifest("m.gguf", "main", ["http://a"])
    assert manifest.revision.startswith("gguf-")
    assert create_adapter(manifest.config).num_layers == LAYERS


def test_workers_only_serve_files_from_their_index(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from isharded import worker

    (tmp_path / "m.gguf").write_bytes(b"0123456789")
    (tmp_path / "secret.txt").write_bytes(b"private")

    class Runtime:
        peer_serve = True
        unit_cache = UnitCache(tmp_path / "cache", local_dirs=(tmp_path,))

    monkeypatch.setattr(worker, "runtime", Runtime())
    monkeypatch.setattr(worker, "verify_token", lambda value: None)
    client = TestClient(worker.app, raise_server_exceptions=False)
    assert client.get("/internal/gguf/index").json() == {"files": [{"name": "m.gguf", "size": 10}]}
    assert client.get("/internal/gguf/range", params={"name": "m.gguf", "start": 2, "end": 5}).content == b"234"
    assert client.get("/internal/gguf/range", params={"name": "secret.txt", "start": 0, "end": 3}).status_code == 404
    assert client.get("/internal/gguf/range", params={"name": "../secret.txt", "start": 0, "end": 3}).status_code == 404
    assert client.get("/internal/gguf/range", params={"name": "m.gguf", "start": 0, "end": 99}).status_code == 404


@needs_real_model
def test_two_conversations_do_not_touch_each_other():
    """The same stage, two sequences, interleaved, each matching a run on its own."""
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    host = StageHost()
    try:
        host.call(
            "load", folder=BUILD, extra_paths=[], path=MODEL, last=True, context_tokens=128, gpu_layers=-1,
            threads=4, table=embed, cpu_tensors=[], rollback_tokens=0, slots=2,
        )

        def step(tokens, position, slot):
            rows = host.call("embed", tokens=tokens)[1]
            data = host.call("forward", rows, position=position, slot=slot)[1]
            return int(np.argmax(np.frombuffer(data, dtype=np.float32)))

        def alone(prompt, slot):
            host.call("clear", slot=slot)
            out = [step(prompt, 0, slot)]
            for index in range(4):
                out.append(step(out[-1:], len(prompt) + index, slot))
            return out

        first, second = [9707, 11, 1246, 525, 498], [785, 6722, 315, 9625, 374]
        expected = (alone(first, 0), alone(second, 1))
        host.call("clear", slot=0)
        host.call("clear", slot=1)
        got = ([step(first, 0, 0)], [step(second, 0, 1)])
        for index in range(4):
            got[0].append(step(got[0][-1:], len(first) + index, 0))
            got[1].append(step(got[1][-1:], len(second) + index, 1))
        assert got == expected
    finally:
        host.close()


def test_only_disk_tier_stages_map_their_file():
    from isharded.gguf_stage import needs_mapping
    from isharded.protocol import MemoryPool, UnitPlacement

    def pool(kind):
        return MemoryPool(pool_id=kind, kind=kind, device="cpu", label=kind, capacity_group=kind, total_bytes=1, budget_bytes=1, free_bytes=1, bandwidth_gbps=1.0, dtype="bfloat16")

    pools = {kind: pool(kind) for kind in ("vram", "ram", "disk")}
    fast = [UnitPlacement(unit="a", pool_id="vram", bytes=5), UnitPlacement(unit="b", pool_id="ram", bytes=5)]
    assert not needs_mapping(fast, pools)
    assert needs_mapping([*fast, UnitPlacement(unit="c", pool_id="disk", bytes=5)], pools)
    assert not needs_mapping([UnitPlacement(unit="c", pool_id="disk", bytes=0)], pools)



def test_every_platform_has_a_cpu_build_with_a_full_hash() -> None:
    from isharded.llamacpp import BUILDS

    for platform_name, kinds in BUILDS.items():
        assert "cpu" in kinds, platform_name
        for files in kinds.values():
            assert all(len(digest) == 64 for _, digest in files)


def test_release_archives_nest_their_libraries(tmp_path, monkeypatch) -> None:
    import io
    import tarfile

    from isharded import llamacpp

    monkeypatch.setattr(llamacpp.os, "name", "posix")
    monkeypatch.setattr(llamacpp.sys, "platform", "linux")
    archive = tmp_path / "build.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        entry = tarfile.TarInfo("llama-b1/libllama.so")
        entry.size = 4
        bundle.addfile(entry, io.BytesIO(b"none"))
    folder = tmp_path / "cpu"
    assert llamacpp.build_folder(folder) is None
    llamacpp.unpack(archive, folder)
    assert llamacpp.build_folder(folder) == folder / "llama-b1"


def test_a_missing_system_library_names_its_package(tmp_path, monkeypatch) -> None:
    import pytest

    from isharded import llamacpp

    def refuse(_):
        raise OSError("libgomp.so.1: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(llamacpp.C, "CDLL", refuse)
    with pytest.raises(llamacpp.LlamaError, match="sudo apt-get install libgomp1"):
        llamacpp.load_native(tmp_path / "libggml-base.so")


def test_a_system_with_an_old_glibc_is_told_what_to_do(tmp_path, monkeypatch) -> None:
    import pytest

    from isharded import llamacpp

    def refuse(_):
        raise OSError("/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by libllama.so)")

    monkeypatch.setattr(llamacpp.C, "CDLL", refuse)
    with pytest.raises(llamacpp.LlamaError, match="needs glibc 2.38 or newer.*--llama-dir"):
        llamacpp.load_native(tmp_path / "libllama.so")


def test_a_platform_without_that_build_says_what_it_offers(monkeypatch) -> None:
    import pytest

    from isharded import llamacpp

    monkeypatch.setattr(llamacpp, "this_platform", lambda: "macos-arm64")
    with pytest.raises(llamacpp.LlamaError, match="it offers cpu"):
        llamacpp.ensure_build("cuda")


def test_a_gpu_reading_a_card_cannot_give_is_left_out(monkeypatch) -> None:
    from types import SimpleNamespace

    from isharded import inventory

    monkeypatch.setattr(inventory.shutil, "which", lambda _: "nvidia-smi")
    answer = SimpleNamespace(returncode=0, stdout="RTX 3060, 57, [N/A], 3, 3971, 6144\nbroken line\n")
    (gpu,) = inventory.nvidia_gpus(run=lambda *_, **__: answer)
    assert gpu == {"name": "RTX 3060", "temperature_c": 57.0, "load_percent": 3.0, "memory_used_mib": 3971.0, "memory_total_mib": 6144.0}


def split_blob(blob, name_of, folder):
    """Two valid parts of a GGUF, the way llama-gguf-split writes them."""
    header = header_of(blob)
    names = list(header.tensors)
    halves = [names[: len(names) // 2], names[len(names) // 2 :]]
    files = []
    for index, chosen in enumerate(halves):
        entries = [
            key("split.no", 4, struct.pack("<H", index)),  # uint16 in real files; the value is not read
            key("split.count", 4, struct.pack("<H", 2)),
            key("split.tensors.count", T_UINT32, struct.pack("<I", len(names))),
        ][1 if False else 2 :]
        if index == 0:
            entries = entries + [key(entry.key, entry.kind, entry.raw) for entry in header.keys.values()]
        infos, data = b"", b""
        for tensor_name in chosen:
            tensor = header.tensors[tensor_name]
            infos += text(tensor_name) + struct.pack("<I", 1) + struct.pack("<Q", tensor.nbytes // 4) + struct.pack("<IQ", 0, len(data))
            data += blob[header.data_start + tensor.offset : header.data_start + tensor.offset + tensor.nbytes]
            data += bytes(-len(data) % 32)
        head = b"GGUF" + struct.pack("<IQQ", 3, len(chosen), len(entries)) + b"".join(entries) + infos
        head += bytes(-len(head) % 32)
        path = folder / name_of(index + 1)
        path.write_bytes(head + data)
        files.append(path)
    return files


def test_every_part_is_named_from_any_part():
    assert split_parts("a/Model-Q3-00002-of-00003.gguf") == [f"a/Model-Q3-0000{index}-of-00003.gguf" for index in (1, 2, 3)]
    assert split_parts("whole.gguf") == ["whole.gguf"]


def test_a_split_gguf_reads_as_one_model_with_each_tensor_in_its_own_part(tmp_path):
    blob = build_file()
    whole = header_of(blob)
    first, second = split_blob(blob, lambda index: f"m-{index:05d}-of-00002.gguf", tmp_path)

    manifest = build_manifest(str(second))  # any part names the model

    assert set(manifest.tensors) == set(whole.tensors)
    assert {ref.file for ref in manifest.tensors.values()} == {first.name, second.name}
    assert manifest.config["gguf_parts"] == {first.name: first.stat().st_size, second.name: second.stat().st_size}
    assert manifest.config["gguf_file"] == first.name
    for name, ref in manifest.tensors.items():
        tensor = whole.tensors[name]
        expected = blob[whole.data_start + tensor.offset : whole.data_start + tensor.offset + tensor.nbytes]
        got = (tmp_path / ref.file).read_bytes()[ref.start : ref.end]
        assert got == expected, name
    skeleton = GgufHeader.from_skeleton(manifest.config["gguf"])
    assert not [name for name in skeleton.keys if name.startswith("split.")]  # a stage is one whole file
    assert skeleton.vocab_size == whole.vocab_size


def test_a_missing_part_or_tensor_is_refused_by_name(tmp_path):
    blob = build_file()
    first, second = split_blob(blob, lambda index: f"m-{index:05d}-of-00002.gguf", tmp_path)
    second.unlink()
    with pytest.raises(Exception, match="m-00002-of-00002"):
        build_manifest(str(first))
