"""A draft with no recurrent state goes back by cutting its guesses off, not by reading text again."""

import os
from pathlib import Path

import pytest

from isharded.gguf_stage import DraftModel, PromptPlan

BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
needs_real_model = pytest.mark.skipif(
    not (BUILD and MODEL and Path(BUILD).is_dir() and Path(MODEL).is_file()), reason="needs a llama.cpp build and a real GGUF"
)
# Plain English in this model family's tokens, so the next token is rarely a near tie.
PROMPT = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374]


class Host:
    """Stands in for a draft's stage process. ``out_of_step`` answers a cut the way a scratch sequence that moved would."""

    def __init__(self, out_of_step=False):
        self.calls = []
        self.next = 100
        self.out_of_step = out_of_step
        self.rows = 1

    def call(self, operation, payload=b"", **meta):
        self.calls.append((operation, meta))
        if self.out_of_step and meta.get("catch_up") is not None:
            return {"tokens": None, "decodes": 0}, b""
        tokens = list(range(self.next, self.next + meta.get("count", 0)))
        self.next += len(tokens)
        return {"tokens": tokens, "decodes": 1, "decoded_rows": self.rows}, b""


def draft_on(host, rewinds=True):
    draft = DraftModel.__new__(DraftModel)
    draft.slots, draft.text, draft.host, draft.rewinds = 1, {}, host, rewinds
    draft.prepare([1, 2, 3], PromptPlan(), 0)
    host.calls.clear()
    return draft


def missed(draft):
    """Three guesses that landed, then the model's own token in place of the last one."""
    first = draft.guess([4], 1)
    second = draft.guess(first, 1)
    draft.guess(second, 1)
    draft.host.calls.clear()
    draft.rewind(5)
    return first


def test_a_miss_cuts_scratch_and_moves_what_landed_to_the_slot_without_reading_it_again():
    draft = draft_on(Host())
    first = missed(draft)
    assert draft.seen() == 5
    draft.guess([50], 1)
    [(operation, meta)] = draft.host.calls
    assert (operation, meta["restart"], meta["commit"], meta["position"], meta["fresh"]) == ("guess", False, [], 5, [50])
    assert meta["catch_up"] == 4, "the slot ends where the reply's first guess left it, and takes what landed from there"
    assert (draft.text[0].committed, draft.text[0].chain) == (5, [50]), "the model's own token stays a guess's text, on scratch"
    assert draft.seen() == 6
    assert first == [100]


def test_after_a_cut_guesses_go_on_as_usual():
    draft = draft_on(Host())
    missed(draft)
    guess = draft.guess([50], 1)
    draft.guess(guess, 1)
    meta = draft.host.calls[-1][1]
    assert (meta["catch_up"], meta["position"], meta["fresh"]) == (None, 6, guess)


def test_two_misses_before_a_guess_cut_once_at_the_earlier():
    draft = draft_on(Host())
    missed(draft)
    draft.rewind(4)
    draft.guess([60], 1)
    meta = draft.host.calls[-1][1]
    assert (meta["position"], meta["catch_up"], meta["fresh"]) == (4, 4, [60])
    assert draft.text[0].committed == 4


def test_a_scratch_sequence_out_of_step_starts_again_from_the_slot():
    draft = draft_on(Host(out_of_step=True))
    first = missed(draft)
    assert draft.guess([50], 1) == [100 + 3]
    cut, again = draft.host.calls
    assert cut[1]["catch_up"] == 4
    assert again[1].get("restart", True) and again[1]["commit"] == [first[0], 50] and again[1]["position"] == 4
    assert (draft.text[0].committed, draft.text[0].chain, draft.text[0].stale) == (6, [], False)


def test_a_recurrent_draft_still_starts_again_after_a_miss():
    draft = draft_on(Host(), rewinds=False)
    first = missed(draft)
    draft.guess([50], 1)
    [(_, meta)] = draft.host.calls
    assert meta.get("restart", True) and meta["commit"] == [first[0], 50] and meta.get("catch_up") is None


def test_the_first_guess_of_a_reply_copies_the_slot_even_for_a_draft_that_cuts():
    # Scratch was cleared for the reply, so there is nothing on it to cut.
    draft = draft_on(Host())
    draft.guess([4], 1)
    [(_, meta)] = draft.host.calls
    assert meta.get("restart", True) and meta["commit"] == [4]


def test_a_draft_counts_the_decodes_its_guesses_cost():
    draft = draft_on(Host())
    missed(draft)
    before = draft.decodes, draft.decoded_rows
    draft.host.rows = 3
    draft.guess([50], 1)
    assert (draft.decodes, draft.decoded_rows) == (before[0] + 1, before[1] + 3)


def real_draft(slots=1):
    return DraftModel(BUILD, (), Path(MODEL), 256, 2, slots)


@needs_real_model
def test_a_miss_on_a_real_draft_reads_one_row_if_it_can_cut_and_guesses_stay_right():
    """Against a fresh draft that read the same text, on both slots, with a scratch knocked out of step on the way."""
    reference = real_draft()
    draft = real_draft(2)

    def fresh(text, count):
        reference.prepare(text[:-1], PromptPlan())
        return reference.guess(text[-1:], count)

    try:
        other = [3838, 374, 279, 7192, 315, 220, 18, 323, 220, 19, 30]
        draft.prepare(PROMPT[:-1], PromptPlan(), 0)
        draft.prepare(other[:-1], PromptPlan(), 1)
        text, fed, other_fed = list(PROMPT), PROMPT[-1:], other[-1:]
        costs = []
        for round_number in range(4):
            for _ in range(3):
                [guess] = draft.guess(fed, 1)
                other_fed = draft.guess(other_fed, 1, 1)
                text.append(guess)
                fed = [guess]
            keep = len(text) - 2
            draft.rewind(keep)
            text = [*text[:keep], (text[keep] + 1 + round_number) % 5000 + 200]
            fed = text[-1:]
            if round_number == 2:
                draft.host.call("clear", slot=draft.slots)  # scratch for slot 0, behind the draft's back
            before = draft.decoded_rows
            [guess] = draft.guess(fed, 1)
            costs.append(draft.decoded_rows - before)
            assert [guess] == fresh(text, 1)
            text.append(guess)
            fed = [guess]
        if draft.rewinds:
            assert [cost == 1 for cost in costs] == [True, True, False, True], "a cut reads only the model's token; scratch out of step reads the slot again"
        else:
            assert all(cost > 1 for cost in costs), "recurrent state goes back by reading the confirmed guesses again"
    finally:
        draft.close()
        reference.close()


@needs_real_model
def test_a_draft_knows_whether_its_model_can_go_back():
    from isharded.gguf import read_file_header

    architecture = read_file_header(MODEL).keys["general.architecture"].value
    draft = real_draft()
    try:
        assert draft.rewinds == (architecture in ("qwen3", "qwen3moe", "llama")), architecture
    finally:
        draft.close()
