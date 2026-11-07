"""The light install: a node with no PyTorch runs GGUF models, and refuses the rest.

Tests that compare with PyTorch skip themselves where it is missing. The others run
either way, and hide PyTorch with ``no_torch`` where it is installed.
"""
import asyncio
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest

from isharded import llamacpp, memory, node, optional, up, worker
from isharded.llamacpp import Draws, draw, filtered_distribution
from isharded.memory import Accelerator, read_nvidia_devices
from isharded.models import create_adapter
from isharded.orchestrator import Orchestrator
from isharded.planner import PlacementError
from isharded.protocol import HardwareInfo, InferRequest, MemoryPool, decode_rows, encode_rows, from_bfloat16, needs_pytorch, to_bfloat16

GGUF = "Q/M-GGUF:m.gguf"
SPECIAL_BITS = [
    0x00000000, 0x80000000, 0x7F800000, 0xFF800000,  # zeroes and infinities
    0x00000001, 0x80000001, 0x007FFFFF,  # subnormals
    0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000,  # the largest values, which round up to infinity
    0x3F808000, 0x3F818000, 0x3F808001, 0x3F817FFF,  # exact ties go to even
    0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF,  # NaNs
]


@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)


def special_rows() -> np.ndarray:
    return np.array(SPECIAL_BITS, dtype=np.uint32).view(np.float32).reshape(1, 3, 6)


def torch_bits(rows: np.ndarray) -> np.ndarray:
    import torch

    return torch.from_numpy(rows.copy()).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def test_rows_round_to_nearest_even_with_nothing_but_numpy():
    rows = np.array([0x3F808000, 0x3F818000, 0x3F808001, 0x7F7FFFFF], dtype=np.uint32).view(np.float32)
    assert to_bfloat16(rows).tolist() == [0x3F80, 0x3F82, 0x3F81, 0x7F80]


def test_a_nan_stays_a_nan_with_nothing_but_numpy():
    rows = np.array([0x7F800001, 0xFFFFFFFF], dtype=np.uint32).view(np.float32)
    assert np.isnan(from_bfloat16(to_bfloat16(rows))).all()


def same_bfloat16(ours: np.ndarray, theirs: np.ndarray) -> bool:
    """Bit for bit, except that a NaN may be any NaN: PyTorch writes a different one on Apple chips."""
    ours, theirs = ours.view(np.uint16).ravel(), theirs.view(np.uint16).ravel()

    def nan(bits: np.ndarray) -> np.ndarray:
        return ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)

    return bool(np.array_equal(nan(ours), nan(theirs)) and np.array_equal(ours[~nan(ours)], theirs[~nan(theirs)]))


def test_bfloat16_conversion_is_the_same_bytes_as_pytorch():
    assert same_bfloat16(to_bfloat16(special_rows()), torch_bits(special_rows()))
    every_pattern = np.random.default_rng(1).integers(0, 2**32, 1 << 20, dtype=np.uint32).view(np.float32)
    assert same_bfloat16(to_bfloat16(every_pattern), torch_bits(every_pattern))


@pytest.mark.parametrize("narrow", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("dtype", [None, np.int64, np.int32], ids=["no-places", "int64", "int32"])
@pytest.mark.parametrize("shape", [(1, 1, 8), (1, 5, 12)], ids=["one-row", "rows"])
def test_rows_cross_in_the_same_bytes_as_pytorch_sends(narrow, dtype, shape):
    import torch

    from isharded.protocol import encode_activation

    rows = np.random.default_rng(2).standard_normal(shape, dtype=np.float32)
    places = None if dtype is None else np.arange(8, dtype=dtype).reshape(4, 2)
    hidden = torch.from_numpy(rows)
    expected = encode_activation(hidden.to(torch.bfloat16) if narrow else hidden, None if places is None else torch.from_numpy(places))
    assert encode_rows(rows, places, narrow) == expected


@pytest.mark.parametrize("narrow", [False, True], ids=["fp32", "bf16"])
def test_each_side_reads_what_the_other_sent(narrow):
    import torch

    from isharded.protocol import decode_pass, encode_activation

    rows = np.random.default_rng(3).standard_normal((1, 3, 8), dtype=np.float32)
    places = np.arange(6, dtype=np.int32).reshape(3, 2)
    hidden = torch.from_numpy(rows)
    read, read_places = decode_rows(encode_activation(hidden.to(torch.bfloat16) if narrow else hidden, torch.from_numpy(places)))
    assert np.array_equal(read, hidden.to(torch.bfloat16).float().numpy() if narrow else rows)
    assert np.array_equal(read_places, places)
    tensor, tensor_places = decode_pass(encode_rows(rows, places, narrow))
    assert tensor.dtype == (torch.bfloat16 if narrow else torch.float32)
    assert np.array_equal(tensor_places.numpy(), places)


def test_rows_survive_the_wire_with_nothing_but_numpy():
    rows = np.array([[[1.0, -2.5, 0.0, 3.0]]], dtype=np.float32)
    read, places = decode_rows(encode_rows(rows, None, True))
    assert read.dtype == np.float32 and np.array_equal(read, rows) and places is None
    read, places = decode_rows(encode_rows(rows, np.array([[7]], dtype=np.int64)))
    assert np.array_equal(read, rows) and places.tolist() == [[7]]


def test_a_stage_with_no_pytorch_refuses_what_is_not_a_pass():
    wrong = encode_rows(np.zeros((1, 1, 2), dtype=np.float32)).replace(b"hidden", b"weight")
    with pytest.raises(ValueError, match="'hidden'"):
        decode_rows(wrong)
    with pytest.raises(ValueError, match="cannot read"):
        decode_rows(encode_rows(np.zeros((1, 1, 4), dtype=np.float32)).replace(b'"F32"', b'"U32"'))


def test_nothing_on_a_gguf_node_loads_pytorch_by_being_imported():
    modules = "isharded.worker, isharded.orchestrator, isharded.up, isharded.node, isharded.cli, isharded.assistant"
    check = f"import sys, {modules}; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", check], capture_output=True, text=True).returncode == 0


def test_a_model_can_be_planned_where_there_is_no_pytorch(no_torch):
    adapter = create_adapter({"model_type": "gpt2", "n_layer": 2, "n_embd": 8, "n_head": 2, "vocab_size": 16, "n_positions": 8})
    assert adapter.num_layers == 2
    assert memory_width_is_known()


def memory_width_is_known() -> bool:
    from isharded.weights import _itemsize

    return (_itemsize("BF16"), _itemsize("F8_E4M3"), _itemsize("I64")) == (2, 1, 8)


def test_pytorch_code_with_no_pytorch_says_how_to_add_it():
    missing = optional.LazyModule("isharded_has_no_such_module")
    with pytest.raises(optional.TorchMissing, match=r'pip install "isharded\[torch\]"') as refused:
        missing.tensor
    assert "--full" in str(refused.value)


def test_a_lazy_module_is_the_module_once_it_has_loaded():
    lazy = optional.LazyModule("json")
    assert lazy.dumps([1]) == "[1]" and "dumps" in vars(lazy)
    with pytest.raises(AttributeError):
        lazy.no_such_name


def test_has_torch_follows_what_can_be_imported(no_torch):
    assert not optional.has_torch() and not optional.torch_loaded()


def test_only_a_gguf_file_runs_without_pytorch():
    assert not needs_pytorch("Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf")
    assert not needs_pytorch(r"C:\models\QWEN.GGUF")
    assert needs_pytorch("Qwen/Qwen3-0.6B")


def test_a_node_says_whether_it_has_pytorch(no_torch):
    info = worker.hardware_info([], [], "h", {"inference"}, 0, 0, 0, False)
    assert info.pytorch is False


def test_a_node_from_before_this_counts_as_having_pytorch():
    assert HardwareInfo(device="cpu", backend="cpu", cpu_cores=1, ram_total_bytes=1).pytorch is True


def configure_request(architecture):
    return SimpleNamespace(
        model_id="Qwen/Qwen3-0.6B",
        architecture=architecture,
        assignment=SimpleNamespace(worker_id="w", start_layer=0, end_layer=1),
        term=1,
    )


def test_a_node_with_no_pytorch_refuses_a_pytorch_model_before_it_loads_anything(no_torch):
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.worker_id, runtime.term, runtime.host_models, runtime.roles = "w", 0, [], {"inference"}
    runtime.ring_stop = lambda: None
    runtime._materialize = lambda *_: pytest.fail("loaded a model it cannot run")
    architecture = {"model_type": "gpt2", "n_layer": 2, "n_embd": 8, "n_head": 2, "vocab_size": 16, "n_positions": 8}
    with pytest.raises(ValueError, match=r'pip install "isharded\[torch\]"'):
        asyncio.run(runtime.configure(configure_request(architecture)))


def test_a_node_with_no_pytorch_will_not_download_a_pytorch_model(no_torch, monkeypatch):
    monkeypatch.setattr(worker, "verify_token", lambda value: None)
    monkeypatch.setattr(worker, "verify_source", lambda value: None)
    monkeypatch.setattr(worker.runtime, "start_prepare", lambda body: pytest.fail("started a download"))
    body = {"job_id": "j", "model_id": "Qwen/Qwen3-0.6B", "source": "Qwen/Qwen3-0.6B", "revision": "r", "source_revision": "r", "units": []}
    with TestClient(worker.app, raise_server_exceptions=False) as client:
        response = client.post("/internal/prepare", json=body)
    assert response.status_code == 400 and "isharded[torch]" in response.json()["detail"]


def test_the_prepare_refusal_still_comes_after_the_token_check(no_torch, monkeypatch):
    def refuse(value):
        raise HTTPException(status_code=401, detail="no")

    monkeypatch.setattr(worker, "verify_token", refuse)
    body = {"job_id": "j", "model_id": "Qwen/Qwen3-0.6B", "source": "Qwen/Qwen3-0.6B", "revision": "r", "source_revision": "r", "units": []}
    with TestClient(worker.app, raise_server_exceptions=False) as client:
        assert client.post("/internal/prepare", json=body).status_code == 401


def planning(model, **hardware):
    fields = {"backend": "cpu", "pytorch": True, **hardware}
    return Orchestrator._runs_on(SimpleNamespace(planning_model=model, model_id=None), SimpleNamespace(**fields))


def test_the_planner_keeps_a_pytorch_model_off_a_node_with_no_pytorch():
    assert not planning("Qwen/Qwen3-0.6B", pytorch=False)
    assert planning("Qwen/Qwen3-0.6B")


def test_the_planner_still_gives_such_a_node_gguf_models():
    assert planning(GGUF, pytorch=False)
    assert planning(None, pytorch=False)


def test_the_planner_still_keeps_vulkan_to_gguf_models():
    assert not planning("Qwen/Qwen3-0.6B", backend="vulkan")
    assert planning(GGUF, backend="vulkan")


def pool_with(*workers):
    records = {
        name: SimpleNamespace(live=live, registration=SimpleNamespace(hardware=SimpleNamespace(pytorch=pytorch)))
        for name, live, pytorch in workers
    }
    return SimpleNamespace(planning_model="Qwen/Qwen3-0.6B", model_id=None, workers=records)


def test_a_refused_pytorch_model_names_who_was_left_out_and_how_to_fix_it():
    pool = pool_with(("light", True, False), ("full", True, True), ("gone", False, False))
    note = Orchestrator._left_out(pool)
    assert note.startswith(" light left out for having no PyTorch.") and "isharded[torch]" in note


def test_a_refusal_says_nothing_of_pytorch_when_it_is_not_the_reason():
    assert Orchestrator._left_out(pool_with(("full", True, True))) == ""
    pool = pool_with(("light", True, False))
    pool.planning_model = GGUF
    assert Orchestrator._left_out(pool) == ""


def test_kv_snapshots_only_go_where_they_can_be_read():
    def offer(pytorch, budget):
        contribution = SimpleNamespace(roles=["kv-cache"], ram_budget_bytes=budget)
        hardware = SimpleNamespace(pytorch=pytorch, contribution=contribution)
        return SimpleNamespace(live=True, quarantined=False, registration=SimpleNamespace(hardware=hardware, advertised_url=f"http://{budget}"))

    pool = SimpleNamespace(workers={"light": offer(False, 9), "full": offer(True, 1)})
    assert Orchestrator.choose_cache_url(pool, []).startswith("http://1")
    pool.workers.pop("full")
    assert Orchestrator.choose_cache_url(pool, []) is None


def test_the_refusal_reaches_whoever_asked():
    pool = pool_with(("light", True, False))
    pool._left_out = lambda: Orchestrator._left_out(pool)

    def plan(*_, **__):
        raise PlacementError("no worker offers any memory for weights")

    pool.plan = plan
    with pytest.raises(HTTPException) as refused:
        Orchestrator.plan_or_refuse(pool, SimpleNamespace(draft=None))
    assert refused.value.status_code == 507
    assert refused.value.detail.startswith("no worker offers any memory for weights. light left out") and "isharded[torch]" in refused.value.detail


def test_status_shows_which_nodes_have_no_pytorch():
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=1, ram_total_bytes=1, pytorch=False)
    registration = worker.WorkerRegistration(worker_id="light", advertised_url="http://light:1", hardware=hardware)
    record = SimpleNamespace(registration=registration, live=True, quarantined=False, quarantine_reason=None, control_latency_ms=None, health=None)
    assert Orchestrator._worker_view(record, False)["hardware"]["pytorch"] is False


def smi(stdout, returncode=0):
    return lambda *_, **__: SimpleNamespace(returncode=returncode, stdout=stdout)


def test_an_nvidia_gpu_is_read_from_its_driver_tool(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    found = read_nvidia_devices(smi("0, NVIDIA GeForce RTX 3060, 6144, 5000, 581.08\n1, Tesla T4, 15360, 15000, 581.08\n"))
    assert [(item.device, item.backend, item.kind) for item in found] == [("cuda:0", "cuda", "vram"), ("cuda:1", "cuda", "vram")]
    assert found[0].name == "NVIDIA GeForce RTX 3060" and found[0].total_bytes == 6144 * 1024**2 and found[0].free_bytes == 5000 * 1024**2


def test_a_missing_or_broken_driver_tool_means_no_nvidia_gpu_and_no_crash():
    def missing(*_, **__):
        raise FileNotFoundError

    assert read_nvidia_devices(missing) == []
    assert read_nvidia_devices(smi("", 9)) == []
    assert read_nvidia_devices(smi("0, Card, [N/A], [N/A], 1\n")) == []


TWO_CARDS = "0, Small, 6144, 5000, 581.08\n1, Large, 15360, 15000, 581.08\n"


@pytest.mark.parametrize(
    "visible, expected",
    [("", []), ("-1", []), ("1", [("cuda:0", "Large")]), ("1,0", [("cuda:0", "Large"), ("cuda:1", "Small")]), ("0,-1,1", [("cuda:0", "Small")]), ("7", [])],
)
def test_a_card_hidden_from_cuda_is_hidden_from_the_pool(monkeypatch, visible, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert [(item.device, item.name) for item in read_nvidia_devices(smi(TWO_CARDS))] == expected


def test_cards_named_by_uuid_are_left_as_the_tool_said(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    assert len(read_nvidia_devices(smi(TWO_CARDS))) == 2


def card(device):
    return Accelerator(device, "vram", device.split(":")[0], "card", 8 * 1024**3, 7 * 1024**3, None)


def test_a_node_with_no_pytorch_prefers_cuda_then_vulkan(no_torch, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(memory, "nvidia_accelerators", lambda: (card("cuda:0"),))
    monkeypatch.setattr(memory, "vulkan_accelerators", lambda: pytest.fail("asked Vulkan with CUDA there"))
    assert [item.device for item in memory._all_accelerators()] == ["cuda:0"]
    monkeypatch.setattr(memory, "nvidia_accelerators", lambda: ())
    monkeypatch.setattr(memory, "vulkan_accelerators", lambda: (card("vulkan:0"),))
    assert [item.device for item in memory._all_accelerators()] == ["vulkan:0"]


def test_an_arm_mac_with_no_pytorch_still_gets_its_gpu(no_torch, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(memory.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(memory.platform, "mac_ver", lambda: ("15.0", ("", "", ""), ""))
    assert [(item.device, item.kind) for item in memory._all_accelerators()] == [("mps", "unified")]


def test_a_node_with_no_pytorch_can_still_ask_if_it_has_a_gpu(no_torch, monkeypatch):
    monkeypatch.setattr(memory, "_accelerators_without_torch", lambda: [card("cuda:0")])
    assert node._has_accelerator()
    monkeypatch.setattr(memory, "_accelerators_without_torch", lambda: [])
    assert not node._has_accelerator()


@pytest.mark.parametrize("cards, expected", [((card("cuda:0"),), "cuda"), ((), "cpu")])
def test_a_node_with_no_pytorch_picks_its_llama_build(no_torch, monkeypatch, cards, expected):
    monkeypatch.setattr(memory, "nvidia_accelerators", lambda: cards)
    assert llamacpp.pick_build("auto") == expected
    assert llamacpp.pick_build("cpu") == "cpu" and llamacpp.pick_build("vulkan") == "vulkan"


def test_ram_is_timed_with_numpy_and_a_gpu_is_assumed(no_torch):
    assert memory.measure_bandwidth("cpu", size_mib=4, rounds=2) > 0
    assert memory.measure_bandwidth("cuda:0") == memory.ASSUMED_VULKAN_GBPS


def nvidia_pool():
    return MemoryPool(pool_id="cuda:0", kind="vram", device="cuda:0", capacity_group="g", total_bytes=8, budget_bytes=8, free_bytes=5, bandwidth_gbps=1.0, dtype="bfloat16")


def forget_vram_readings(monkeypatch):
    memory._read_nvidia_free.cache_clear()
    monkeypatch.setattr(memory, "NVIDIA_FREE", {})


def test_free_vram_is_read_from_the_driver_and_not_every_time(no_torch, monkeypatch, tmp_path):
    asked = threading.Event()
    times = []

    def read():
        times.append(1)
        memory.NVIDIA_FREE["cuda:0"] = 7 * 1024**3
        asked.set()
        return []

    forget_vram_readings(monkeypatch)
    monkeypatch.setattr(memory, "read_nvidia_devices", read)
    memory.live_free_bytes(nvidia_pool(), tmp_path)
    assert asked.wait(5)
    assert memory.live_free_bytes(nvidia_pool(), tmp_path) == 7 * 1024**3
    assert len(times) == 1


def test_a_slow_driver_tool_never_holds_up_a_health_check(no_torch, monkeypatch, tmp_path):
    release = threading.Event()
    forget_vram_readings(monkeypatch)
    monkeypatch.setattr(memory, "read_nvidia_devices", lambda: release.wait(5) and [])
    started = time.perf_counter()
    assert memory.live_free_bytes(nvidia_pool(), tmp_path) == 5  # what discovery saw
    assert time.perf_counter() - started < 1
    release.set()


def test_free_vram_falls_back_to_what_was_last_known(no_torch, monkeypatch, tmp_path):
    forget_vram_readings(monkeypatch)
    monkeypatch.setattr(memory, "read_nvidia_devices", lambda: [])
    assert memory.live_free_bytes(nvidia_pool(), tmp_path) == 5


def test_an_explicit_vram_budget_does_nothing_with_no_pytorch_to_hold_it(no_torch):
    memory.apply_accelerator_limits([nvidia_pool()], explicit_budget=True)


def test_freeing_memory_leaves_pytorch_alone_when_nothing_loaded_it(no_torch):
    worker.free_memory()


class Stage:
    """A GGUF stage that echoes its rows, doubled."""

    last = False
    width = 4

    def embed(self, ids):
        return np.ones((len(ids), self.width), dtype=np.float32)

    def forward(self, rows, position, slot, places, cuts, saves):
        return rows * 2

    def forward_many(self, rows, positions, slots):
        return rows * 2

    def clear(self, slot):
        pass


def gguf_runtime():
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.adapter = runtime.assignment = object()
    runtime.gguf = Stage()
    runtime.compute_ms_ema = runtime.row_ms_ema = None
    runtime.inference_calls = runtime.tokens_processed = runtime.merged_passes = runtime.merged_rows = 0
    runtime._pass_entry = lambda request: SimpleNamespace(length=0)
    return runtime


def test_a_gguf_stage_holds_its_rows_in_numpy(no_torch):
    runtime = gguf_runtime()
    request = InferRequest(session_id="s", epoch=1, input_ids=[1, 2])
    result = runtime._run_local(np.ones((1, 2, 4), dtype=np.float32), request)
    assert isinstance(result, np.ndarray) and result.shape == (1, 2, 4) and (result == 2).all()


def test_merged_rows_stay_in_numpy(no_torch):
    runtime = gguf_runtime()
    request = InferRequest(session_id="s", epoch=1, input_ids=[1])
    steps = [SimpleNamespace(hidden=np.ones((1, 1, 4), dtype=np.float32), request=request) for _ in range(2)]
    results = runtime._run_merged(steps)
    assert all(isinstance(item, np.ndarray) and item.shape == (1, 1, 4) and (item == 2).all() for item in results)


def last_stage():
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf = object()
    runtime.gguf_logits = np.array([0.0, 3.0, 1.0, 2.5], dtype=np.float32)
    runtime.caches = {"s": SimpleNamespace(length=9)}
    return runtime


def test_a_last_stage_samples_with_no_pytorch(no_torch):
    runtime = last_stage()
    greedy = InferRequest(session_id="s", epoch=1, input_ids=[1], temperature=0)
    assert runtime._sample(None, greedy) == 1
    seeded = InferRequest(session_id="s", epoch=1, input_ids=[1], temperature=0.8, seed=5)
    ids, chances = filtered_distribution(runtime.gguf_logits, 0.8, 1.0, 0)
    assert runtime._sample(None, seeded) == draw(ids, chances, Draws(5, 8))
    unseeded = InferRequest(session_id="s", epoch=1, input_ids=[1], temperature=0.8, top_k=2)
    assert {runtime._sample(None, unseeded) for _ in range(40)} <= {1, 3}


def test_a_stage_with_no_pytorch_reads_a_pass_before_it_has_a_model(no_torch, monkeypatch):
    seen = {}

    class Runtime:
        lock = asyncio.Lock()
        last_error = None
        gguf = None

        def _record(self, *args):
            pass

        async def execute(self, hidden, request, places=None):
            seen["rows"] = hidden
            raise HTTPException(status_code=409, detail="worker is not configured")

    monkeypatch.setattr(worker, "runtime", Runtime())
    monkeypatch.setattr(worker, "verify_token", lambda value: None)
    meta = worker.encode_meta(InferRequest(session_id="s", epoch=1, input_ids=[1]).model_dump())
    client = TestClient(worker.app, raise_server_exceptions=False)
    response = client.post("/internal/forward", content=encode_rows(np.ones((1, 1, 4), dtype=np.float32)), headers={"x-isharded-meta": meta})
    assert response.status_code == 409 and isinstance(seen["rows"], np.ndarray)


def test_the_doctor_says_what_a_light_node_can_run(no_torch):
    line = next(item for item in up.checks() if "PyTorch" in item[1])
    assert line[0] and "GGUF models only" in line[1] and "isharded[torch]" in line[2]
