import asyncio

import httpx
import pytest
import torch

from shardmesh.orchestrator import ModelTarget, Orchestrator
from shardmesh.worker import MAX_PENDING_CHECKPOINTS, CacheEntry, WorkerRuntime, choose_dtype
from shardmesh.planner import ModelProfile, PlacementError
from shardmesh.protocol import (
    META_HEADER,
    CacheRestoreRequest,
    ConfigureRequest,
    HardwareInfo,
    InferRequest,
    MemoryPool,
    ResourceContribution,
    StageAssignment,
    TensorRef,
    WeightUnit,
    WorkerRegistration,
    decode_meta,
    decode_tensors,
    encode_tensors,
)


def registration(
    worker_id: str,
    backend: str,
    roles: list[str],
    standby: bool = False,
    ram_budget: int = 0,
) -> WorkerRegistration:
    return WorkerRegistration(
        worker_id=worker_id,
        advertised_url=f"http://{worker_id}:8000",
        hardware=HardwareInfo(
            host_id="host",
            device="cuda:0" if backend == "cuda" else "cpu",
            backend=backend,
            cpu_cores=8,
            ram_total_bytes=16 * 1024**3,
            accelerator_memory_bytes=6 * 1024**3 if backend == "cuda" else 0,
            contribution=ResourceContribution(
                roles=roles,
                cpu_threads=4 if backend == "cpu" else 0,
                ram_budget_bytes=ram_budget,
                accelerator_memory_budget_bytes=4 * 1024**3 if backend == "cuda" else 0,
                standby=standby,
            ),
        ),
    )


def test_gpu_is_preferred_cpu_is_standby_and_memory_is_service() -> None:
    orchestrator = Orchestrator()
    try:
        orchestrator.register(registration("gpu", "cuda", ["inference"]))
        orchestrator.register(registration("cpu", "cpu", ["inference"], standby=True))
        orchestrator.register(registration("memory", "cpu", ["kv-cache"], ram_budget=1024))

        assert [candidate.worker_id for candidate in orchestrator.candidates()] == ["gpu"]
        assert orchestrator.choose_cache_url([]) == "http://memory:8000"

        orchestrator.workers["gpu"].quarantined = True
        assert [candidate.worker_id for candidate in orchestrator.candidates()] == ["cpu"]
    finally:
        asyncio.run(orchestrator.close())


def test_worker_model_sources_can_differ_from_logical_model_id() -> None:
    target = ModelTarget.__new__(ModelTarget)
    target.model_id = "distilbert/distilgpt2"
    target.worker_model_ids = {"local-gpu": r"D:\models\distilgpt2"}

    assert target.source_for("local-gpu").endswith("models\\distilgpt2")
    assert target.source_for("remote-rocm") == "distilbert/distilgpt2"


def test_a_lost_pinned_worker_is_dropped_so_recovery_can_continue() -> None:
    orchestrator = Orchestrator()
    try:
        for name in ("a", "b"):
            pool = MemoryPool(
                pool_id="ram",
                kind="ram",
                device="cpu",
                capacity_group=f"{name}:ram",
                budget_bytes=1024**3,
                free_bytes=1024**3,
                bandwidth_gbps=20,
            )
            record = registration(name, "cpu", ["inference"])
            record.hardware.memory_pools = [pool]
            orchestrator.register(record)
        target = ModelTarget.__new__(ModelTarget)
        unit = WeightUnit(
            name="x",
            kind="embed",
            tensors=[TensorRef(name="x", file="f", dtype="BF16", shape=[1], start=0, end=2)],
        )
        head = unit.model_copy(update={"name": "h", "kind": "head"})
        dense = unit.model_copy(update={"name": "d", "kind": "dense", "layer": 0})
        target.profile = ModelProfile(
            units=[unit, head, dense],
            num_layers=1,
            expert_active_fraction=1.0,
            kv_bytes_per_token=[0],
            state_bytes=[0],
        )
        target.spread = True
        target.entry_worker = "a"
        target.exit_worker = None
        target.reuse_cached = False
        target.peer_fetch = False
        target.held = {}
        target.serving = set()
        assert orchestrator.plan(target)[0].worker_id == "a"

        orchestrator.workers["a"].quarantined = True
        assert orchestrator.plan(target)[0].worker_id == "b"
        assert any("entry_worker pin" in event["detail"] for event in orchestrator.events)
        with pytest.raises(PlacementError, match="cannot be pinned"):
            orchestrator.plan(target, honor_pins_strictly=True)
    finally:
        asyncio.run(orchestrator.close())


def test_every_node_defaults_to_bf16_and_can_opt_into_full_precision() -> None:
    assert choose_dtype("auto") == torch.bfloat16
    assert choose_dtype("float32") == torch.float32
    assert choose_dtype("checkpoint") is None


def test_a_failed_load_gives_its_memory_back() -> None:
    worker = WorkerRuntime()

    class Unit:
        released = False

        def release(self) -> None:
            self.released = True

    unit = Unit()
    worker.loaded = {"model/unit/cuda:0/False": unit}
    worker.unit_weights = {"unit": {}}
    worker.pool_used = {"cuda:0": 1024}
    worker.worker_id = "w"
    stage = StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)
    request = ConfigureRequest(
        model_id="m",
        revision="r",
        architecture={"model_type": "gpt2", "n_layer": 1, "n_head": 1, "n_embd": 8},
        epoch=1,
        assignment=stage,
        route=[stage],
    )

    def out_of_memory(*args) -> None:
        raise RuntimeError("CUDA out of memory")

    worker._materialize = out_of_memory
    with pytest.raises(RuntimeError):
        asyncio.run(worker.configure(request))

    assert unit.released
    assert worker.loaded == {}
    assert worker.pool_used == {}
    assert worker.assignment is None


def checkpointing_worker() -> WorkerRuntime:
    worker = WorkerRuntime()
    worker.cache_url = "http://store"
    worker.assignment = StageAssignment(worker_id="w", url="http://w", start_layer=0, end_layer=1)
    return worker


def kv_entry(tokens: int) -> CacheEntry:
    key = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1)
    return CacheEntry(epoch=1, states={0: {"key": key, "value": key.clone()}}, length=tokens)


def infer(tokens: int = 1) -> InferRequest:
    return InferRequest(session_id="s", epoch=1, input_ids=[1] * tokens)


def test_snapshots_do_not_hold_up_the_token_and_arrive_in_order() -> None:
    async def scenario() -> list[int]:
        worker = checkpointing_worker()
        sent: list[int] = []
        release = asyncio.Event()

        async def slow_put(url, content, headers, timeout):
            await release.wait()
            meta = decode_meta(headers[META_HEADER])
            sent.append(meta["start_position"])
            return httpx.Response(200, request=httpx.Request("PUT", url))

        worker.client.put = slow_put
        for length in (1, 2, 3):
            worker.caches["s"] = kv_entry(length)
            worker.schedule_checkpoint(infer())  # returns while the send is stuck
        assert worker.checkpoints_pending == 3
        release.set()
        await worker.checkpoint_tail
        await worker.client.aclose()
        return sent

    assert asyncio.run(scenario()) == [0, 1, 2]


def test_snapshot_keeps_the_token_it_was_taken_for_even_if_the_cache_moves_on() -> None:
    async def scenario() -> list[list[float]]:
        worker = checkpointing_worker()
        stored: list[list[float]] = []

        async def put(url, content, headers, timeout):
            stored.append(decode_tensors(content)["layer_0_key"].flatten().tolist())
            return httpx.Response(200, request=httpx.Request("PUT", url))

        worker.client.put = put
        worker.caches["s"] = kv_entry(1)
        worker.schedule_checkpoint(infer())
        worker.caches["s"] = kv_entry(2)  # the next token lands before the send
        await worker.checkpoint_tail
        await worker.client.aclose()
        return stored

    assert asyncio.run(scenario()) == [[0.0]]


def test_a_slow_store_is_dropped_rather_than_queued_forever() -> None:
    async def scenario() -> tuple[int, int]:
        worker = checkpointing_worker()
        hold = asyncio.Event()

        async def stuck(url, content, headers, timeout):
            await hold.wait()
            return httpx.Response(200, request=httpx.Request("PUT", url))

        worker.client.put = stuck
        worker.caches["s"] = kv_entry(1)
        for _ in range(MAX_PENDING_CHECKPOINTS + 5):
            worker.schedule_checkpoint(infer())
        result = worker.checkpoints_pending, worker.checkpoints_dropped
        hold.set()
        await worker.checkpoint_tail
        await worker.client.aclose()
        return result

    assert asyncio.run(scenario()) == (MAX_PENDING_CHECKPOINTS, 5)


def test_a_store_that_is_behind_is_refused_so_recovery_replays() -> None:
    async def scenario(expected: int) -> None:
        worker = checkpointing_worker()
        worker.unit_weights = {"layer-00000.dense": {"w": torch.zeros(1)}}
        payload = encode_tensors({"layer_0_key": torch.zeros(1, 1, 3, 1), "layer_0_value": torch.zeros(1, 1, 3, 1)})

        async def get(url, params, headers, timeout):
            return httpx.Response(200, content=payload, request=httpx.Request("GET", url))

        worker.client.get = get
        try:
            await worker.restore_cache(
                CacheRestoreRequest(
                    session_id="s", source_epoch=1, target_epoch=2, cache_url="http://store", expected_length=expected
                )
            )
        finally:
            await worker.client.aclose()

    asyncio.run(scenario(3))
    with pytest.raises(ValueError, match="store has 3 tokens"):
        asyncio.run(scenario(5))
