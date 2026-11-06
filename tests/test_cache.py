"""Reusing prompts: where they are cut, what is saved, and what happens when a state is gone."""

import asyncio
import json
import os
from types import SimpleNamespace

from fastapi import HTTPException
import numpy as np
import pytest

from isharded import worker
from isharded.gguf_stage import plan_prompt
from isharded.llamacpp import LlamaError
from isharded.protocol import GenerateRequest, InferRequest, SavePoint
from isharded.statestore import CUT_LEAST, HISTORY, PASS_ROWS, PREFIX, PrefixIndex, cut_points, keys_at, pass_ends, prefix_key

TURN = 7  # stands for the token that opens a turn


def chat(*lengths):
    """A prompt whose turns have these lengths, each opened by the turn token."""
    tokens = []
    for index, length in enumerate(lengths):
        tokens += [TURN] + [1000 + index] * (length - 1)
    return tokens


def test_cuts_fall_before_a_turn_once_a_piece_is_long_enough():
    assert cut_points(chat(100, 100, 100), TURN) == [100, 200]
    assert cut_points(chat(10, 10, 100, 100), TURN) == [120]


def test_without_a_turn_token_nothing_is_cut():
    assert cut_points(chat(100, 100), None) == []


@pytest.mark.parametrize("seed", range(20))
def test_cuts_do_not_move_as_a_conversation_grows(seed):
    """The rule everything else rests on: a saved state must sit on a cut in every later run."""
    rng = np.random.default_rng(seed)
    prompt = chat(*rng.integers(1, 3 * CUT_LEAST, size=30).tolist())
    whole = cut_points(prompt, TURN)
    for stop in rng.integers(1, len(prompt), size=25).tolist():
        assert cut_points(prompt[:stop], TURN) == [cut for cut in whole if cut < stop]


def test_a_pass_stops_at_every_cut_and_every_full_pass_between():
    assert pass_ends([100], 1400) == [100, 100 + PASS_ROWS, 100 + 2 * PASS_ROWS]
    assert pass_ends([], PASS_ROWS) == [], "nothing would be left to run"


@pytest.mark.parametrize("seed", range(10))
def test_where_passes_stop_does_not_depend_on_where_a_run_started(seed):
    """A run resuming from a state must meet the same stops as one that read everything."""
    rng = np.random.default_rng(seed)
    prompt = chat(*rng.integers(1, 900, size=12).tolist())
    cuts = cut_points(prompt, TURN)
    ends = pass_ends(cuts, len(prompt))
    for start in ends:
        edges = [start, *[cut for cut in cuts if cut > start], len(prompt)]
        stops = []
        for first, last in zip(edges, edges[1:]):
            stops += list(range(first + PASS_ROWS, last, PASS_ROWS)) + [last]
        assert [stop for stop in stops if stop < len(prompt)] == [end for end in ends if end > start]


def test_the_first_cut_of_a_chat_is_saved_for_other_conversations():
    prompt = chat(200, 30, 10)
    plan = plan_prompt(PrefixIndex(), prompt, TURN, 230)
    assert [(save.at, save.kind) for save in plan.saves] == [(200, PREFIX)]


def test_the_end_of_settled_history_is_saved_for_the_next_turn():
    prompt = chat(200, 100, 10)
    plan = plan_prompt(PrefixIndex(), prompt, TURN, 300)
    assert [(save.at, save.kind) for save in plan.saves] == [(200, PREFIX), (300, HISTORY)]


def test_a_second_conversation_starts_from_the_shared_prefix():
    index = PrefixIndex()
    system = chat(200)
    index.add(prefix_key(system), 200, PREFIX)
    plan = plan_prompt(index, [*system, *chat(100, 10)], TURN, 300)
    assert (plan.start, plan.kind) == (200, PREFIX)
    assert plan.cuts == [100], "cuts count from where this pass starts"
    assert [(save.at, save.length, save.kind) for save in plan.saves] == [(100, 300, HISTORY)]


def test_a_longer_state_wins_over_a_shorter_one():
    index = PrefixIndex()
    prompt = chat(200, 100, 100, 10)
    for length in (200, 300, 400):
        index.add(prefix_key(prompt[:length]), length, HISTORY)
    assert plan_prompt(index, prompt, TURN, 400).start == 400


def test_a_state_of_the_same_length_with_other_words_is_not_reused():
    index = PrefixIndex()
    index.add(prefix_key(chat(200)), 200, HISTORY)
    other = chat(200)
    other[50] = 99
    assert plan_prompt(index, [*other, *chat(100)], TURN, None).start == 0


def test_a_state_needs_something_left_to_run():
    index = PrefixIndex()
    prompt = chat(200)
    index.add(prefix_key(prompt), 200, HISTORY)
    assert plan_prompt(index, prompt, TURN, None).start == 0


def test_the_next_turn_replaces_the_last_turn_but_never_a_shared_prefix():
    index = PrefixIndex()
    prompt = chat(200, 100, 100, 10)
    index.add(prefix_key(prompt[:300]), 300, HISTORY)
    assert plan_prompt(index, prompt, TURN, 400).saves[0].replaces == prefix_key(prompt[:300])
    index = PrefixIndex()
    index.add(prefix_key(prompt[:300]), 300, PREFIX)
    assert plan_prompt(index, prompt, TURN, 400).saves[0].replaces is None


def test_a_prompt_repeating_an_earlier_one_marks_the_place_as_shared():
    """Text from the plain completions API has no turns to cut at, so repeats show what is shared."""
    index = PrefixIndex()
    shared = list(range(3000, 3000 + 2 * PASS_ROWS))
    first = [*shared, *range(100, 400)]
    second = [*shared, *range(500, 800)]
    assert plan_prompt(index, first, None, None).saves == []
    plan = plan_prompt(index, second, None, None)
    assert [(save.length, save.kind) for save in plan.saves] == [(2 * PASS_ROWS, PREFIX)]


def test_nothing_is_reused_or_saved_with_the_cache_off():
    index = PrefixIndex()
    prompt = chat(200, 100, 10)
    index.add(prefix_key(prompt[:200]), 200, PREFIX)
    plan = plan_prompt(index, prompt, TURN, 300, caching=False)
    assert (plan.start, plan.key, plan.saves) == (0, None, [])


def test_the_cache_never_changes_where_a_prompt_is_cut():
    """Whatever is reused, the rows run in the same pieces as a run with the cache off."""
    prompt = chat(200, 100, 100, 10)
    off = plan_prompt(PrefixIndex(), prompt, TURN, 400, caching=False)
    for held in (200, 300, 400):
        index = PrefixIndex()
        index.add(prefix_key(prompt[:held]), held, HISTORY)
        on = plan_prompt(index, prompt, TURN, 400)
        assert on.start == held
        assert [cut + held for cut in on.cuts] == [cut for cut in off.cuts if cut > held]


def test_the_index_forgets_what_a_stage_says_it_lost():
    index = PrefixIndex()
    index.add("a", 5, PREFIX)
    index.drop("a")
    index.drop("never there")
    assert "a" not in index


def test_finding_a_prefix_checks_tokens_not_lengths():
    index = PrefixIndex()
    tokens = list(range(50))
    index.add(keys_at(tokens, [10])[10], 10, PREFIX)
    assert index.longest(tokens)[0] == 10
    assert index.longest([99, *tokens[1:]]) is None


# The entry worker: a reply must survive any state going missing.


class Gate:
    def __init__(self):
        self.released = []

    async def acquire(self, choose):
        return 0

    async def release(self, slot):
        self.released.append(slot)


def fake_entry(monkeypatch, results):
    """An entry worker whose stages answer with ``results``, one per pass."""
    runtime = SimpleNamespace(
        assignment=SimpleNamespace(start_layer=0), waiting=0, gate=Gate(), epoch=1, gguf=object(), draft=None, speculate_tokens=0,
        prompt_cache=True, prefix=PrefixIndex(), slot_used_at={}, state_misses=0, last_error=None, lock=asyncio.Lock(),
        choose_slot=min, _record=lambda *args: None, _prepare_entry=lambda request, part: request, passes=[],
    )

    async def execute(hidden, request, places):
        runtime.passes.append(request)
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    runtime.execute = execute
    monkeypatch.setattr(worker, "runtime", runtime)
    monkeypatch.setattr(worker, "verify_token", lambda authorization: None)
    return runtime


def generate(prompt, **fields):
    body = GenerateRequest(session_id="s", epoch=1, input_ids=prompt, reset=True, max_tokens=1, cut_token=TURN, **fields)

    async def run():
        response = await worker.generate(body, None)
        return [json.loads(chunk) async for chunk in response.body_iterator]

    return asyncio.run(run())


def missing(key):
    return HTTPException(status_code=503, detail={"retry": True, "missing": key, "message": "gone"})


def test_a_prompt_passes_once_along_the_route_however_much_it_saves(monkeypatch):
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    prompt = chat(200, 100, 10)
    assert generate(prompt, checkpoint_at=300) == [{"token_id": 5}]
    (only,) = runtime.passes
    assert only.prompt_rows and only.reset and only.cuts == [200, 300]
    assert [save.length for save in only.saves] == [200, 300]


def test_states_become_reusable_only_once_the_pass_went_through(monkeypatch):
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    prompt = chat(200, 100, 10)
    generate(prompt, checkpoint_at=300)
    assert prefix_key(prompt[:300]) in runtime.prefix

    runtime = fake_entry(monkeypatch, [HTTPException(status_code=503, detail={"retry": True, "message": "down"})])
    generate(prompt, checkpoint_at=300)
    assert prefix_key(prompt[:300]) not in runtime.prefix


def test_a_state_a_stage_lost_costs_a_replay_and_not_the_reply(monkeypatch):
    prompt = chat(200, 100, 10)
    key = prefix_key(prompt[:200])
    runtime = fake_entry(monkeypatch, [missing(key), {"token_id": 5}])
    runtime.prefix.add(key, 200, PREFIX)
    assert generate(prompt) == [{"token_id": 5}]
    first, second = runtime.passes
    assert first.checkpoint_key == key and first.input_ids == prompt[200:]
    assert second.checkpoint_key is None and second.reset and second.input_ids == prompt
    assert runtime.state_misses == 1
    assert key in [save.key for save in second.saves], "the replay puts back what was lost"


def test_a_missing_state_falls_back_to_the_next_longest(monkeypatch):
    prompt = chat(200, 100, 10)
    short, long = prefix_key(prompt[:200]), prefix_key(prompt[:300])
    runtime = fake_entry(monkeypatch, [missing(long), {"token_id": 5}])
    runtime.prefix.add(short, 200, PREFIX)
    runtime.prefix.add(long, 300, HISTORY)
    assert generate(prompt) == [{"token_id": 5}]
    assert runtime.passes[1].checkpoint_key == short


def test_other_failures_still_reach_the_control_plane(monkeypatch):
    prompt = chat(200, 10)
    detail = {"failed_worker": "w2", "message": "gone"}
    runtime = fake_entry(monkeypatch, [HTTPException(status_code=502, detail=detail)])
    assert generate(prompt) == [{"error": detail}]
    assert len(runtime.passes) == 1


def test_a_stage_that_lost_a_state_says_which_one(monkeypatch):
    class Stage:
        def restore(self, key, slot):
            raise LlamaError("no saved state")

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf, runtime.caches, runtime.tokens_reused = Stage(), {}, {}
    request = InferRequest(session_id="s", epoch=1, input_ids=[1], restore_length=10, checkpoint_key="k")
    with pytest.raises(HTTPException) as caught:
        runtime._session(request)
    assert caught.value.status_code == 503
    assert caught.value.detail["missing"] == "k" and caught.value.detail["retry"]


def test_restored_tokens_are_counted_by_where_they_came_from():
    class Stage:
        def restore(self, key, slot):
            return "disk"

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf, runtime.caches, runtime.tokens_reused = Stage(), {}, {}
    request = InferRequest(session_id="s", epoch=1, input_ids=[1], restore_length=10, reused_rows=10, checkpoint_key="k", restore_kind=PREFIX)
    assert runtime._session(request).length == 10
    assert runtime.tokens_reused == {"prefix_disk": 10}


def test_prompt_rows_never_merge_with_other_conversations():
    import torch

    step = SimpleNamespace(verify=False, reset=False, prompt_rows=True, restore_length=None, next_position=None)
    assert not worker.WorkerRuntime._can_merge(SimpleNamespace(merge_passes=True, gguf=object()), torch.zeros(1, 1, 4), step)


def test_a_save_point_crosses_stages_intact():
    save = SavePoint(at=5, length=205, key="k", kind=PREFIX, replaces="old")
    sent = InferRequest(session_id="s", epoch=1, input_ids=[1], saves=[save], cuts=[5])
    assert InferRequest.model_validate(json.loads(sent.model_dump_json())).saves == [save]


# Real models. See tests/test_gguf.py for what these need.

BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
needs_real_model = pytest.mark.skipif(not (BUILD and MODEL), reason="set ISHARDED_TEST_LLAMA_DIR and ISHARDED_TEST_GGUF")


@pytest.fixture
def stage():
    # A stage process of its own, because this test process has imported torch.
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    host = StageHost()
    host.call(
        "load", folder=BUILD, extra_paths=[], path=MODEL, last=True, context_tokens=2048, gpu_layers=0,
        threads=2, table=embed, cpu_tensors=[], slots=3, state_ram_bytes=1024**3,
    )
    yield host
    host.close()


def tokens_for_test(count=740):
    return np.random.default_rng(7).integers(1000, 20000, size=count).tolist()


def run(stage, tokens, start=0, slot=0, cuts=(), saves=()):
    """Tokens from ``start`` on, as one pass. Returns the last row's logits."""
    rows = stage.call("embed", tokens=tokens[start:])[1]
    data = stage.call("forward", rows, position=start, slot=slot, cuts=list(cuts), saves=list(saves))[1]
    return np.frombuffer(data, dtype=np.float32).copy()


def save_at(row, key, length=None):
    return {"at": row, "length": length or row, "key": key, "kind": PREFIX}


@needs_real_model
def test_a_restored_state_continues_bit_for_bit(stage):
    """With the same cuts, reading everything and restoring must give the very same logits."""
    tokens = tokens_for_test()
    expected = run(stage, tokens, cuts=[300, 700], saves=[save_at(700, "history")])
    stage.call("clear", slot=0)
    assert stage.call("restore", key="history", slot=1)[0]["tier"] == "ram"
    assert np.array_equal(run(stage, tokens, start=700, slot=1), expected)


@needs_real_model
def test_a_state_saved_at_a_cut_inside_a_pass_is_the_state_at_that_cut(stage):
    tokens = tokens_for_test()
    expected = run(stage, tokens, cuts=[300, 700], saves=[save_at(300, "early")])
    stage.call("restore", key="early", slot=1)
    assert np.array_equal(run(stage, tokens, start=300, slot=1, cuts=[400]), expected)


@needs_real_model
def test_a_state_saved_where_a_full_pass_ends_is_exact_too(stage):
    tokens = tokens_for_test()
    expected = run(stage, tokens, saves=[save_at(PASS_ROWS, "grid")])
    stage.call("restore", key="grid", slot=1)
    assert np.array_equal(run(stage, tokens, start=PASS_ROWS, slot=1), expected)


@needs_real_model
def test_one_state_serves_several_sequences(stage):
    tokens = tokens_for_test()
    run(stage, tokens[:300], cuts=[200], saves=[save_at(200, "shared")])
    answers = []
    for slot in (1, 2):
        stage.call("restore", key="shared", slot=slot)
        answers.append(run(stage, [*tokens[:200], *tokens[500:540]], start=200, slot=slot))
    assert np.array_equal(*answers)
