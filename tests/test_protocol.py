import torch

from isharded.protocol import decode_activation, decode_meta, encode_activation, encode_meta


def test_activation_round_trip_is_lossless() -> None:
    value = torch.randn(1, 3, 8, dtype=torch.float16)
    restored = decode_activation(encode_activation(value))
    assert restored.dtype == value.dtype
    assert torch.equal(restored, value)


def test_metadata_round_trip() -> None:
    value = {"session_id": "a/b", "epoch": 3, "reset": True}
    assert decode_meta(encode_meta(value)) == value
