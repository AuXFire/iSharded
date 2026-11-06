import pytest
import torch

from isharded.protocol import decode_tensors, encode_tensors
from isharded.worker import MemoryStore


def snapshot(layer: int, value: float) -> bytes:
    return encode_tensors(
        {
            f"layer_{layer}_key": torch.full((1, 1, 2, 2), value),
            f"layer_{layer}_value": torch.full((1, 1, 2, 2), value + 1),
        }
    )


def test_memory_store_returns_requested_layer_range() -> None:
    first = snapshot(0, 1)
    second = snapshot(1, 3)
    store = MemoryStore(len(first) + len(second) + 64)
    store.put("session:1:a", first, {"session_id": "session", "epoch": 1})
    store.put("session:1:b", second, {"session_id": "session", "epoch": 1})

    restored = decode_tensors(store.get_layers("session", 1, 0, 2))
    assert set(restored) == {
        "layer_0_key",
        "layer_0_value",
        "layer_1_key",
        "layer_1_value",
    }


def test_memory_store_evicts_least_recent_snapshot() -> None:
    payload = snapshot(0, 1)
    store = MemoryStore(len(payload) + 8)
    store.put("old", payload, {"session_id": "old", "epoch": 1})
    store.put("new", payload, {"session_id": "new", "epoch": 1})
    assert store.stats()["evictions"] == 1
    with pytest.raises(KeyError):
        store.get_layers("old", 1, 0, 1)


def test_memory_store_reassembles_incremental_kv_chunks() -> None:
    first = snapshot(0, 1)
    second = encode_tensors(
        {
            "layer_0_key": torch.full((1, 1, 1, 2), 5.0),
            "layer_0_value": torch.full((1, 1, 1, 2), 6.0),
        }
    )
    store = MemoryStore(len(first) + len(second) + 64)
    store.put(
        "session:1:a:0",
        first,
        {"session_id": "session", "epoch": 1, "start_position": 0},
    )
    store.put(
        "session:1:a:2",
        second,
        {"session_id": "session", "epoch": 1, "start_position": 2},
    )
    restored = decode_tensors(store.get_layers("session", 1, 0, 1))
    assert restored["layer_0_key"].shape[-2] == 3
    assert restored["layer_0_key"][0, 0, -1, 0] == 5
