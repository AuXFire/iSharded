"""A draft that guesses the exit's own draw, and the seed the entry gives a reply so it can."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from isharded import worker as worker_module
from isharded.gguf_stage import DraftModel, DraftText
from isharded.llamacpp import Draws, LlamaStage, coupled_choice, draw, filtered_distribution
from isharded.protocol import GenerateRequest

from test_pipeline import Ring, entry, lookup_only, on_ring, stream_of  # noqa: F401

FILTERS = [(0.8, 1.0, 0), (1.0, 1.0, 0), (0.7, 0.9, 0), (1.0, 1.0, 20), (0.8, 0.95, 40)]


def logits_like_a_model(seed: int) -> np.ndarray:
    """A few strong tokens, a middle, and a long tail far below, with ties among them."""
    rng = np.random.default_rng(seed)
    logits = np.where(rng.random(5000) < 0.8, rng.normal(-40, 5, 5000), rng.normal(0, 3, 5000)).astype(np.float32)
    logits[rng.integers(0, 5000, 4)] = 12.0
    logits[rng.integers(0, 5000, 30)] = 5.0
    return logits


@pytest.mark.parametrize("temperature, top_p, top_k", FILTERS)
def test_a_coupled_guess_is_the_token_the_exit_draws_from_the_same_logits(temperature, top_p, top_k):
    for sample in range(40):
        logits = logits_like_a_model(sample)
        ids, chances = filtered_distribution(logits, temperature, top_p, top_k)
        for position in range(10):
            expected = draw(ids, chances, Draws(sample, position))
            assert coupled_choice(logits, temperature, top_p, top_k, Draws(sample, position)) == expected


def test_a_coupled_guess_follows_the_draw_and_not_the_best_token():
    logits = np.zeros(8, dtype=np.float32)  # every token as likely, so the draw alone decides
    made = {coupled_choice(logits, 1.0, 1.0, 0, Draws(3, position)) for position in range(64)}
    assert len(made) > 4


def stage_with(logits_after: dict[int, np.ndarray]) -> tuple[LlamaStage, list[tuple[list[int], int, int]]]:
    """A stage whose logits depend on the last position it decoded, with no library under it."""
    stage = LlamaStage.__new__(LlamaStage)
    decoded: list[tuple[list[int], int, int]] = []
    last = {"position": -1}
    stage.library = SimpleNamespace(llama=SimpleNamespace(llama_memory_seq_rm=lambda *_: True, llama_memory_seq_cp=lambda *_: None))
    stage.memory = None

    def decode_tokens(tokens, position, *, sequence_id=0):
        decoded.append((list(tokens), position, sequence_id))
        if tokens:
            last["position"] = position + len(tokens) - 1

    stage.decode_tokens = decode_tokens
    stage.logits_at = lambda _: logits_after[last["position"]]
    return stage, decoded


@pytest.mark.parametrize("restart", [True, False])
def test_each_guess_draws_with_the_position_of_the_row_before_it(restart):
    logits = {position: logits_like_a_model(100 + position) for position in range(20)}
    stage, _ = stage_with(logits)
    sampling = {"temperature": 0.9, "top_p": 1.0, "top_k": 0, "seed": 11}
    commit, fresh = ([7, 8], [9]) if restart else ([], [7, 8, 9])
    guesses = stage.guess(commit, 5, 3, fresh=fresh, restart=restart, sampling=sampling)
    # The text runs to position 7, so the guesses are what rows 7, 8 and 9 draw.
    expected = []
    for row in (7, 8, 9):
        expected.append(coupled_choice(logits[row], 0.9, 1.0, 0, Draws(11, row)))
    assert guesses == expected


@pytest.mark.parametrize("sampling", [None, {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 4}, {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "seed": None}])
def test_a_guess_with_no_draw_to_follow_is_the_best_token(sampling):
    logits = {position: logits_like_a_model(position) for position in range(10)}
    stage, _ = stage_with(logits)
    assert stage.guess([], 3, 2, fresh=[5], restart=False, sampling=sampling) == [int(np.argmax(logits[3])), int(np.argmax(logits[4]))]


def test_a_draft_passes_the_reply_sampling_to_its_process():
    calls = []

    class Host:
        def call(self, operation, payload=b"", **meta):
            calls.append(meta)
            return {"tokens": [1]}, b""

    draft = DraftModel.__new__(DraftModel)
    draft.host, draft.slots, draft.text, draft.last_rows = Host(), 1, {0: DraftText(committed=3)}, {}
    sampling = {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "seed": 9}
    draft.guess([4], 1, 0, sampling)
    draft.guess([1], 1, 0)
    assert calls[0]["sampling"] == sampling and calls[1]["sampling"] is None


class SeedRing(Ring):
    def __init__(self) -> None:
        super().__init__()
        self.seeds: list[int | None] = []

    def send(self, request, token, position, rollback_to):
        self.seeds.append(request.seed)
        return super().send(request, token, position, rollback_to)


class Draft:
    """Guesses one token past what it has seen, and remembers the sampling it was given."""

    def __init__(self) -> None:
        self.read = 0
        self.sampling: list[dict | None] = []

    def prepare(self, prompt, plan, slot) -> None:
        self.read = len(prompt)

    def seen(self, slot) -> int:
        return self.read

    def rewind(self, keep, slot) -> None:
        self.read = min(self.read, keep)

    def guess(self, fresh, count, slot, sampling=None) -> list[int]:
        self.read += len(fresh)
        self.sampling.append(sampling)
        return [5] * count


def run_reply(monkeypatch, temperature: float, seed: int | None) -> tuple[SeedRing, Draft]:
    runtime = worker_module.runtime
    draft = Draft()
    monkeypatch.setattr(runtime, "draft", draft)
    monkeypatch.setattr(runtime, "prompt_lookup", False)
    ring = SeedRing()
    on_ring(monkeypatch, ring)
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[5, 6, 7], max_tokens=6, reset=True, temperature=temperature, seed=seed)
    tokens = asyncio.run(stream_of(body))
    assert len(tokens) == 6
    return ring, draft


def test_a_sampled_reply_with_no_seed_gets_one_from_the_entry_for_every_pass(lookup_only, monkeypatch):
    monkeypatch.delenv("ISHARDED_ENTRY_SEED", raising=False)
    ring, draft = run_reply(monkeypatch, 0.8, None)
    [seed] = set(ring.seeds)
    assert isinstance(seed, int) and 0 <= seed < 2**63
    assert draft.sampling and all(item == {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "seed": seed} for item in draft.sampling)
    again, _ = run_reply(monkeypatch, 0.8, None)
    assert set(again.seeds) != {seed}, "each reply gets a seed of its own"


@pytest.mark.parametrize("temperature, seed, setting", [(0.8, 21, None), (0.0, None, None), (0.8, None, "0")])
def test_the_entry_leaves_a_seed_alone_when_it_has_one_has_no_use_for_one_or_is_told_not_to(lookup_only, monkeypatch, temperature, seed, setting):
    if setting is None:
        monkeypatch.delenv("ISHARDED_ENTRY_SEED", raising=False)
    else:
        monkeypatch.setenv("ISHARDED_ENTRY_SEED", setting)
    ring, draft = run_reply(monkeypatch, temperature, seed)
    assert set(ring.seeds) == {seed}
    assert all(item["seed"] == seed for item in draft.sampling)
