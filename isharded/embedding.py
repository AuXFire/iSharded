"""Packing text into embedding passes, and turning pooled output into vectors.

The stage process imports this, so it must not import torch.
"""

from __future__ import annotations

import base64
import re
from typing import Iterable

import numpy as np

# Inputs one pass can hold. Each is a sequence, and sequences are cheap here because
# they share one cache.
PASS_SEQUENCES = 64
# Attention in a shared cache grows with the square of a pass, whichever input each
# token belongs to, so many short inputs do best in a modest pass.
PASS_TOKENS = 1024
MOST_INPUTS = 2048
# Models trained so their leading dimensions stand alone, by name, and how far they
# go. A GGUF does not say, and cutting any other model's vector just damages it.
SHORTENED_FAMILIES = {"qwen3-embedding": 32}
# GGUF pooling types that give one vector per input.
POOLED_TYPES = {1: "mean", 2: "cls", 3: "last"}


def passes(lengths: list[int], budget: int = PASS_TOKENS, sequences: int = PASS_SEQUENCES) -> list[list[int]]:
    """Group input indexes into passes, keeping their order.

    An input longer than the budget gets a pass of its own, and whoever calls this
    has already checked that it fits the context.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, length in enumerate(lengths):
        if current and (used + length > budget or len(current) >= sequences):
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        groups.append(current)
    return groups


def shaped(vector: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    """Shorten a vector if asked, then scale it to unit length.

    Cosine search and dot products both expect this, and a shortened vector is no
    longer unit length until it happens after the cut.
    """
    kept = vector[:dimensions].astype(np.float64)
    length = float(np.linalg.norm(kept))
    return (kept / length if length else kept).astype(np.float32)


def encoded(vector: np.ndarray, encoding_format: str) -> list[float] | str:
    """A vector as JSON numbers, or base64 of little-endian float32 as OpenAI sends it."""
    if encoding_format == "base64":
        return base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
    return vector.tolist()


def smallest_dimensions(names: Iterable[str]) -> int | None:
    """The shortest a model's vectors may be cut, or None for a model not trained for it."""
    for name in names:
        plain = re.sub(r"[\s_]+", "-", str(name).lower())
        for family, smallest in SHORTENED_FAMILIES.items():
            if family in plain:
                return smallest
    return None


def dimensions_problem(asked: int | None, width: int, smallest: int | None) -> str | None:
    """What is wrong with a request's ``dimensions``, in words, or None if it can be met."""
    if asked is None or asked == width:
        return None
    if smallest is None:
        return f"this model gives {width} dimensions and was not trained to give fewer; leave dimensions out"
    if not smallest <= asked <= width:
        return f"dimensions must be between {smallest} and {width} for this model"
    return None


def inputs_problem(texts: list[str]) -> str | None:
    """What is wrong with a request's ``input``, in words, or None."""
    if not texts:
        return "input must hold at least one text"
    if len(texts) > MOST_INPUTS:
        return f"input holds {len(texts)} texts, and one request can hold {MOST_INPUTS}"
    empty = next((index for index, text in enumerate(texts) if not text), None)
    return None if empty is None else f"input {empty} is empty"


def length_problem(lengths: list[int], limit: int) -> str | None:
    """Text past the context is refused. Cutting it would change what the vector means."""
    over = next(((index, length) for index, length in enumerate(lengths) if length > limit), None)
    if over is None:
        return None
    return (
        f"input {over[0]} is {over[1]} tokens, and this model was loaded with room for {limit}; "
        "shorten it or load the model with a larger --context-tokens"
    )
