"""Checking whether a route gives honest answers.

A node in a route can return anything, and nothing in a reply proves otherwise. What
can be measured is agreement: the same prompt, with no randomness, on routes built
from different machines. Honest routes run together, because they hold the same
weights. Prompts here are made up and look like ordinary chat, so a person's real
words are never sent anywhere extra, and a node cannot easily tell it is being tested.

This catches a stage that is broken or tampers with what it handles. It does not
catch one that behaves whenever it might be watched, or only touches certain topics.
It is a smoke alarm, not a proof.
"""
from __future__ import annotations

import random
import statistics
from typing import Any

SAMPLES = 8
MOST_SAMPLES = 64
REPLY_TOKENS = 48
# Hardware rounds differently, so honest routes can part at a close call. They still
# run together for most of a reply, while a tampered route parts almost at once.
AGREEMENT_FLOOR = 0.5
SUBJECTS = (
    "rivers", "the moon", "bread", "bridges", "honey", "chess", "volcanoes", "clocks", "maps", "bicycles", "coffee", "lighthouses",
    "glass", "whales", "paper", "trains", "salt", "tides", "bees", "compasses", "rain", "wool", "stairs", "kites",
)  # fmt: skip
ASKS = (
    "Explain {} to a child.",
    "List three facts about {}.",
    "Write two sentences about {}.",
    "What should a beginner know about {}?",
    "Why do people care about {}?",
    "Give me a short history of {}.",
)


def canaries(count: int, rng: Any = None) -> list[str]:
    chooser = rng or random.SystemRandom()
    return [chooser.choice(ASKS).format(chooser.choice(SUBJECTS)) for _ in range(count)]


def agreement(first: str, second: str) -> float:
    """How far two replies run together, word by word, as a share of the shorter one."""
    left, right = first.split(), second.split()
    if not left or not right:
        return 1.0 if left == right else 0.0
    shared = 0
    for one, other in zip(left, right):
        if one != other:
            break
        shared += 1
    return shared / min(len(left), len(right))


def judge(replies: dict[str, list[str]], floor: float = AGREEMENT_FLOOR) -> dict[str, Any]:
    """Compare every pair of routes. Blame takes a majority, so two routes can only disagree."""
    names = sorted(replies)
    pairs = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            scores = [agreement(one, other) for one, other in zip(replies[left], replies[right])]
            pairs.append({"routes": [left, right], "agreement": round(statistics.median(scores), 3) if scores else 0.0})
    out_of_line = [name for name in names if all(pair["agreement"] < floor for pair in pairs if name in pair["routes"])]
    if not any(pair["agreement"] < floor for pair in pairs):
        return {"verdict": "agree", "pairs": pairs, "suspects": []}
    if len(names) >= 3 and 0 < len(out_of_line) < len(names):
        return {"verdict": "suspect", "pairs": pairs, "suspects": out_of_line}
    return {"verdict": "disagree", "pairs": pairs, "suspects": []}
