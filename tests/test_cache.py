"""Reusing prompts: where they are cut, what is saved, and what happens when a state is gone."""

import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import numpy as np
import pytest

from isharded import worker
from isharded.gguf_stage import plan_prompt
from isharded.llamacpp import KV_TYPES, PINNED_BUILD, LlamaError, StageLost
from isharded.orchestrator import Orchestrator
from isharded.protocol import GenerateRequest, InferRequest, LoadModelRequest, SavePoint
from isharded.statestore import (
    CUT_LEAST,
    HISTORY,
    PASS_ROWS,
    PREFIX,
    PrefixIndex,
    clear_other_stores,
    cut_points,
    keys_at,
    pass_ends,
    prefix_key,
    stage_namespace,
)

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
        choose_slot=min, _record=lambda *args: None, ring_ready=lambda request: False, caches={}, _prepare_entry=lambda request, part: request, passes=[], prepared=[], can_fetch=True, pipes={}, route=[],
    )

    async def sync_index():
        pass

    async def prepare_state(key):
        runtime.prepared.append(key)
        return runtime.can_fetch

    async def execute(hidden, request, places):
        runtime.passes.append(request)
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def ring_settle(slot):
        pass

    runtime.execute, runtime.sync_index, runtime.prepare_state, runtime.ring_settle = execute, sync_index, prepare_state, ring_settle
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
    runtime.gguf, runtime.caches, runtime.tokens_reused, runtime.fetched = Stage(), {}, {}, set()
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
    runtime.gguf, runtime.caches, runtime.tokens_reused, runtime.fetched = Stage(), {}, {}, set()
    request = InferRequest(session_id="s", epoch=1, input_ids=[1], restore_length=10, reused_rows=10, checkpoint_key="k", restore_kind=PREFIX)
    assert runtime._session(request).length == 10
    assert runtime.tokens_reused == {"prefix_disk": 10}

    runtime.fetched.add("k")
    runtime._session(request)
    assert runtime.tokens_reused == {"prefix_disk": 10, "prefix_copy": 10}, "a copy from another node counts as one, once"
    assert not runtime.fetched


def test_a_state_on_another_node_is_fetched_before_the_pass(monkeypatch):
    prompt = chat(200, 100, 10)
    key = prefix_key(prompt[:300])
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    runtime.prefix.add(key, 300, HISTORY, elsewhere=True)
    assert generate(prompt) == [{"token_id": 5}]
    assert runtime.prepared == [key]
    assert runtime.passes[0].checkpoint_key == key
    assert key not in runtime.prefix.elsewhere, "only the first reply has to ask"


def test_a_copy_nobody_can_fetch_costs_a_replay_and_not_the_reply(monkeypatch):
    prompt = chat(200, 100, 10)
    key = prefix_key(prompt[:300])
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    runtime.can_fetch = False
    runtime.prefix.add(key, 300, HISTORY, elsewhere=True)
    assert generate(prompt) == [{"token_id": 5}]
    assert runtime.passes[0].checkpoint_key is None and runtime.passes[0].input_ids == prompt
    assert runtime.state_misses == 1


@pytest.mark.parametrize("failing", ["sync_index", "prepare_state"])
def test_nothing_about_the_cache_can_cost_a_reply(monkeypatch, failing):
    """Reuse only ever saves work. Whatever breaks in it, the prompt can still be read."""
    prompt = chat(200, 100, 10)
    key = prefix_key(prompt[:300])
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    runtime.prefix.add(key, 300, HISTORY, elsewhere=True)

    async def broken(*args):
        raise RuntimeError("something nobody thought of")

    setattr(runtime, failing, broken)
    assert generate(prompt) == [{"token_id": 5}]
    assert "something nobody thought of" in runtime.last_error


def test_a_peer_answering_nonsense_is_treated_as_holding_nothing(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"states": [{"no": "fields"}, 7]})

    runtime = entry_runtime(tmp_path, handler, ["w1", "w2"], held=["both"])
    asyncio.run(runtime.sync_index())
    assert not runtime.prefix.entries


def test_a_state_held_here_needs_no_fetching(monkeypatch):
    prompt = chat(200, 100, 10)
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    runtime.prefix.add(prefix_key(prompt[:300]), 300, HISTORY)
    generate(prompt)
    assert runtime.prepared == []


def test_however_many_states_go_missing_the_last_attempt_needs_none(monkeypatch):
    prompt = chat(200, 100, 100, 100, 10)
    keys = [prefix_key(prompt[:length]) for length in (200, 300, 400, 500)]
    runtime = fake_entry(monkeypatch, [{"token_id": 5}])
    runtime.can_fetch = False
    for key, length in zip(keys, (200, 300, 400, 500)):
        runtime.prefix.add(key, length, HISTORY, elsewhere=True)
    assert generate(prompt) == [{"token_id": 5}]
    assert runtime.passes[-1].input_ids == prompt and runtime.passes[-1].reset


# What a stage does when a reply needs a state it does not have.


class Host:
    """A stage process's side of holding states."""

    def __init__(self, held=()):
        self.held = set(held)
        self.adopted = []

    def call(self, operation, **meta):
        if operation == "state_has":
            return {"held": meta["key"] in self.held}, b""
        if operation == "state_adopt":
            self.adopted.append(meta["key"])
            return {"adopted": True}, b""
        if operation == "state_list":
            return {"states": [{"key": key, "length": 300, "kind": HISTORY} for key in self.held]}, b""
        raise AssertionError(operation)


def stage_runtime(tmp_path, handler, held=(), peers=("http://peer:1",)):
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf = SimpleNamespace(host=Host(held))
    runtime.namespace, runtime.state_peers, runtime.fetched, runtime.copies_fetched = "space", list(peers), set(), 0
    runtime.pool_token, runtime.cache_dir, runtime.worker_id = None, tmp_path, "w1"
    (runtime.state_root / "space").mkdir(parents=True)
    runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return runtime


def test_workers_sharing_a_cache_never_clear_each_others_states(tmp_path):
    """A node's workers share --cache-dir, and each clears whatever is not its own."""
    first, second = (worker.WorkerRuntime.__new__(worker.WorkerRuntime) for _ in range(2))
    first.cache_dir = second.cache_dir = tmp_path
    first.worker_id, second.worker_id = "node-gpu", "node-cpu"
    (first.state_root / "layers-0-14").mkdir(parents=True)
    (second.state_root / "layers-14-28").mkdir(parents=True)
    clear_other_stores(second.state_root, {"another-model"})
    assert (first.state_root / "layers-0-14").is_dir()
    assert first.state_root != second.state_root


def test_a_worker_name_cannot_lead_out_of_the_cache(tmp_path):
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.cache_dir, runtime.worker_id = tmp_path, "../../somewhere/else"
    assert runtime.state_root.parent == tmp_path / "states"


def test_a_stage_holding_a_state_fetches_nothing(tmp_path):
    def handler(request):
        raise AssertionError("no need to ask anyone")

    assert asyncio.run(stage_runtime(tmp_path, handler, held=["abc"]).ensure_state("abc"))


def test_a_stage_taking_over_fetches_its_copy(tmp_path):
    runtime = stage_runtime(tmp_path, lambda request: httpx.Response(200, content=b"a state"))
    assert asyncio.run(runtime.ensure_state("abc"))
    assert runtime.gguf.host.adopted == ["abc"]
    assert "abc" in runtime.fetched and runtime.copies_fetched == 1
    assert (runtime.state_root / "space" / "abc.part").read_bytes() == b"a state"


def test_a_stage_tries_each_node_for_a_copy(tmp_path):
    def handler(request):
        return httpx.Response(200 if request.url.host == "second" else 404, content=b"a state")

    runtime = stage_runtime(tmp_path, handler, peers=("http://first:1", "http://second:1"))
    assert asyncio.run(runtime.ensure_state("abc"))


def test_with_no_copy_anywhere_a_stage_says_so(tmp_path):
    assert not asyncio.run(stage_runtime(tmp_path, lambda request: httpx.Response(404)).ensure_state("abc"))


def test_a_silent_holder_only_costs_a_listing_its_timeout(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("silent")

    runtime = stage_runtime(tmp_path, handler, held=["own"])
    assert list(asyncio.run(runtime.restorable())) == ["own"]


def test_what_others_hold_for_these_layers_is_marked_as_elsewhere(tmp_path):
    def handler(request):
        assert request.url.path == "/internal/state/held/space"
        states = [{"key": "own", "length": 300, "kind": HISTORY}, {"key": "theirs", "length": 900, "kind": HISTORY}]
        return httpx.Response(200, json={"states": states})

    found = asyncio.run(stage_runtime(tmp_path, handler, held=["own"]).restorable())
    assert not found["own"]["elsewhere"] and found["theirs"]["elsewhere"]


def entry_runtime(tmp_path, handler, stages, held=()):
    runtime = stage_runtime(tmp_path, handler, held=held, peers=())
    runtime.worker_id, runtime.epoch, runtime.index_epoch = "w1", 3, 1
    runtime.prompt_cache, runtime.prefix, runtime.index_lock = True, PrefixIndex(), asyncio.Lock()
    runtime.route = [SimpleNamespace(worker_id=name, url=f"http://{name}:1") for name in stages]
    return runtime


def test_a_new_entry_worker_learns_only_what_every_stage_can_restore(tmp_path):
    def handler(request):
        states = [{"key": "both", "length": 300, "kind": HISTORY, "elsewhere": True}, {"key": "theirs", "length": 900, "kind": HISTORY, "elsewhere": False}]
        return httpx.Response(200, json={"states": states})

    runtime = entry_runtime(tmp_path, handler, ["w1", "w2"], held=["both", "mine"])
    asyncio.run(runtime.sync_index())
    assert list(runtime.prefix.entries) == ["both"]
    assert runtime.prefix.elsewhere == {"both"}, "any stage needing a fetch means asking them all first"


def test_a_stage_that_will_not_answer_leaves_nothing_to_reuse(tmp_path):
    def handler(request):
        raise httpx.ConnectError("gone")

    runtime = entry_runtime(tmp_path, handler, ["w1", "w2"], held=["both"])
    asyncio.run(runtime.sync_index())
    assert not runtime.prefix.entries


def test_the_stages_are_only_asked_once_per_route(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"states": []})

    runtime = entry_runtime(tmp_path, handler, ["w1", "w2"])
    asyncio.run(runtime.sync_index())
    asyncio.run(runtime.sync_index())
    assert len(calls) == 1
    runtime.epoch = 4
    asyncio.run(runtime.sync_index())
    assert len(calls) == 2


def test_every_stage_must_have_a_state_before_a_reply_counts_on_it(tmp_path):
    answers = {"w2": True, "w3": False}

    def handler(request):
        return httpx.Response(200, json={"ready": answers[request.url.host]})

    runtime = entry_runtime(tmp_path, handler, ["w1", "w2", "w3"], held=["abc"])
    assert not asyncio.run(runtime.prepare_state("abc"))
    answers["w3"] = True
    assert asyncio.run(runtime.prepare_state("abc"))


def test_a_stage_that_stops_answering_is_not_waited_on_for_a_state(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "FETCH_SECONDS", 0.05)

    async def handler(request):
        await asyncio.sleep(30)

    runtime = entry_runtime(tmp_path, handler, ["w1", "w2"], held=["abc"])
    began = time.monotonic()
    assert not asyncio.run(runtime.prepare_state("abc"))
    assert time.monotonic() - began < 5


# Who keeps the copy.


def registered(name, machine, roles=("inference",)):
    hardware = SimpleNamespace(host_id=machine, contribution=SimpleNamespace(roles=list(roles)))
    registration = SimpleNamespace(hardware=hardware, advertised_url=f"http://{name}:1/")
    return SimpleNamespace(registration=registration, live=True, quarantined=False)


def holders(workers, order):
    plane = Orchestrator.__new__(Orchestrator)
    plane.workers = workers
    return plane.copy_holders([SimpleNamespace(worker_id=name) for name in order])


def test_each_stage_copies_to_exactly_one_other_machine():
    found = holders({"a": registered("a", "m1"), "b": registered("b", "m2"), "c": registered("c", "m3")}, ["a", "b", "c"])
    assert found == {"a": "http://b:1", "b": "http://c:1", "c": "http://a:1"}


def test_a_copy_on_the_same_machine_is_no_copy():
    found = holders({"a": registered("a", "m1"), "b": registered("b", "m1"), "c": registered("c", "m2")}, ["a", "b", "c"])
    assert found == {"a": "http://c:1", "b": "http://c:1", "c": "http://a:1"}


def test_a_node_offering_memory_for_this_goes_first():
    memory = registered("m", "m9", ("kv-cache", "checkpoint"))
    found = holders({"a": registered("a", "m1"), "b": registered("b", "m2"), "m": memory}, ["a", "b"])
    assert set(found.values()) == {"http://m:1"}


def test_a_lone_stage_uses_any_other_node():
    assert holders({"a": registered("a", "m1"), "b": registered("b", "m2")}, ["a"]) == {"a": "http://b:1"}


def test_with_nowhere_else_to_go_a_state_stays_local():
    assert holders({"a": registered("a", "m1")}, ["a"]) == {}


def test_nodes_that_are_down_keep_no_copies():
    gone = registered("b", "m2")
    gone.live = False
    held = registered("c", "m3")
    held.quarantined = True
    assert holders({"a": registered("a", "m1"), "b": gone, "c": held}, ["a"]) == {}


def plane_with_caches(**caches):
    plane = Orchestrator.__new__(Orchestrator)
    plane.workers = {name: SimpleNamespace(health={"telemetry": {"cache": cache}}) for name, cache in caches.items()}
    plane.route = [SimpleNamespace(worker_id=name) for name in caches]
    return plane


def test_the_saving_can_be_read_off_one_number():
    cache = {"prompt_tokens_computed": 250, "tokens_reused": {"prefix_ram": 600, "history_disk": 150}, "ram_bytes": 10, "disk_bytes": 20, "held_bytes": 5}
    summary = plane_with_caches(a=cache, b=cache).cache_summary()
    assert summary["saved_percent"] == 75.0


def test_a_token_is_counted_once_however_many_stages_ran_it():
    cache = {"prompt_tokens_computed": 250, "tokens_reused": {"prefix_ram": 750}}
    summary = plane_with_caches(a=cache, b=cache, c=cache).cache_summary()
    assert (summary["prompt_tokens_computed"], summary["prompt_tokens_reused"]) == (250, 750)


def test_bytes_held_add_up_across_nodes():
    summary = plane_with_caches(a={"ram_bytes": 10, "held_bytes": 1}, b={"ram_bytes": 20, "held_bytes": 2}).cache_summary()
    assert (summary["ram_bytes"], summary["held_bytes"]) == (30, 3)


def test_with_nothing_run_yet_there_is_no_percentage():
    assert plane_with_caches(a={}).cache_summary()["saved_percent"] is None


def test_with_no_route_the_summary_is_still_a_summary():
    plane = plane_with_caches(a={"ram_bytes": 10})
    plane.route = []
    assert plane.cache_summary()["prompt_tokens_computed"] == 0


def test_both_ends_agree_on_what_a_store_is_called():
    """A copy is found by this name, worked out separately by a control plane and a worker."""
    stage = SimpleNamespace(start_layer=14, end_layer=28)
    request = SimpleNamespace(model_id="m", revision="r", speculate_tokens=2, assignment=stage, kv_type="q8_0")
    assert worker.WorkerRuntime.state_namespace(request) == stage_namespace(PINNED_BUILD, "m", "r", 14, 28, 2, "q8_0")


def what_crosses_to_the_next_stage(rows, **fields):
    """The dtype that rows of a pass leave this stage in."""
    import torch

    from isharded.protocol import decode_pass

    sent = {}

    class Client:
        async def post(self, url, content, headers, timeout):
            sent["body"] = content
            return httpx.Response(200, json={"token_id": 3}, request=httpx.Request("POST", url))

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.assignment, runtime.model_id, runtime.epoch = object(), "m", 1
    runtime.gguf, runtime.dtype_name, runtime.pool_token = object(), "bfloat16", None
    runtime.merge_passes, runtime.activation_bytes_sent, runtime.downstream_ms_ema, runtime.last_error = False, 0, None, None
    runtime.client = Client()
    runtime._next_stage = lambda: SimpleNamespace(url="http://next:1")
    runtime._stages_after = lambda: 1
    runtime._record = lambda *args: None
    runtime.schedule_checkpoint = lambda request: None

    async def alone(hidden, request, places=None):
        return hidden

    runtime._step_alone = alone

    async def go():
        runtime.lock = asyncio.Lock()
        await runtime.execute(torch.zeros(1, rows, 4), InferRequest(session_id="s", epoch=1, input_ids=[1] * rows, **fields))

    asyncio.run(go())
    return decode_pass(sent["body"])[0].dtype


def test_one_prompt_row_crosses_stages_as_a_whole_prompt_does():
    """Otherwise a state that left one row to run would not match reading the whole prompt."""
    import torch

    assert what_crosses_to_the_next_stage(1, prompt_rows=True) == what_crosses_to_the_next_stage(9, prompt_rows=True) == torch.bfloat16


def test_a_generated_token_still_crosses_exactly():
    import torch

    assert what_crosses_to_the_next_stage(1) == torch.float32


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


def start_stage(folder=None, ram=1024**3, disk=0, kv_type="f16"):
    # A stage process of its own, because this test process has imported torch.
    from isharded.gguf import read_file_header
    from isharded.llama_host import StageHost

    header = read_file_header(MODEL)
    table = header.tensors["token_embd.weight"]
    embed = {"path": MODEL, "offset": header.data_start + table.offset, "length": table.nbytes, "ggml_type": table.ggml_type}
    host = StageHost()
    host.call(
        "load", folder=BUILD, extra_paths=[], path=MODEL, last=True, context_tokens=2048, gpu_layers=0, threads=2, table=embed,
        cpu_tensors=[], slots=3, state_dir=None if folder is None else str(folder), state_ram_bytes=ram, state_disk_bytes=disk, kv_type=kv_type,
    )
    return host


@pytest.fixture
def stage():
    host = start_stage()
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


def wait_for_file(path, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not path.is_file():
        assert time.monotonic() < deadline, f"{path.name} was never written"
        time.sleep(0.05)


@needs_real_model
def test_a_state_pushed_out_of_ram_comes_back_from_disk_exact(tmp_path):
    stage = start_stage(tmp_path, ram=1, disk=1024**3)
    try:
        tokens = tokens_for_test()
        expected = run(stage, tokens, cuts=[700], saves=[save_at(700, "history")])
        wait_for_file(tmp_path / "history.state")
        stage.call("clear", slot=0)
        assert stage.call("restore", key="history", slot=1)[0]["tier"] == "disk"
        assert np.array_equal(run(stage, tokens, start=700, slot=1), expected)
    finally:
        stage.close()


@needs_real_model
def test_a_state_outlives_the_stage_process_that_saved_it(tmp_path):
    """Killed, not closed: nothing gets the chance to flush or tidy up."""
    first = start_stage(tmp_path, disk=1024**3)
    tokens = tokens_for_test()
    expected = run(first, tokens, cuts=[700], saves=[save_at(700, "history")])
    wait_for_file(tmp_path / "history.state")
    first.process.kill()
    first.close()
    second = start_stage(tmp_path, disk=1024**3)
    try:
        assert second.call("restore", key="history", slot=1)[0]["tier"] == "disk"
        assert np.array_equal(run(second, tokens, start=700, slot=1), expected)
    finally:
        second.close()


@needs_real_model
@pytest.mark.parametrize("seed", range(4))
def test_a_false_state_fails_cleanly_and_harms_nothing_else(tmp_path, seed):
    """A checksum shows damage, not honesty: a pool member could send anything that passes."""
    from isharded.statestore import StateStore

    real = start_stage(tmp_path, disk=1024**3)
    tokens = tokens_for_test()
    expected = run(real, tokens, cuts=[700], saves=[save_at(700, "good")])
    wait_for_file(tmp_path / "good.state")
    genuine = np.fromfile(tmp_path / "good.state", dtype=np.uint8)
    real.close()

    rng = np.random.default_rng(seed)
    forged = {
        0: rng.integers(0, 256, size=100_000, dtype=np.uint8),  # nothing like a state
        1: genuine[-200_000:].copy(),  # the tail of a real one
        2: genuine[1000:].copy(),  # a real one, with the start missing
        3: genuine[300:].copy(),
    }[seed]
    store = StateStore(tmp_path, 1024**3, 1024**3)
    store.put("forged", forged, 700)
    store.close()

    stage = start_stage(tmp_path, disk=1024**3)
    try:
        with pytest.raises(LlamaError) as refused:
            stage.call("restore", key="forged", slot=1)
        assert not isinstance(refused.value, StageLost), "llama.cpp refused it, and the process lives"
        stage.call("restore", key="good", slot=1)
        assert np.array_equal(run(stage, tokens, start=700, slot=1), expected), "the stage still answers correctly"
    finally:
        stage.close()


def request_for(start=0, end=14, model="m", revision="r", speculate=0, kv_type="f16"):
    return SimpleNamespace(
        model_id=model, revision=revision, speculate_tokens=speculate, kv_type=kv_type, assignment=SimpleNamespace(start_layer=start, end_layer=end)
    )


@pytest.mark.parametrize(
    "other",
    [
        request_for(start=14, end=28),
        request_for(end=13),
        request_for(model="another"),
        request_for(revision="another"),
        request_for(speculate=4),
        request_for(kv_type="q8_0"),
    ],
)
def test_a_state_is_tied_to_its_model_and_layers(other):
    assert worker.WorkerRuntime.state_namespace(request_for()) != worker.WorkerRuntime.state_namespace(other)


def test_8_bit_kv_is_planned_at_what_it_really_takes():
    """Bytes for 32 values, so the planner does not reserve memory it will never use."""
    assert KV_TYPES["q8_0"][1] / KV_TYPES["f16"][1] == pytest.approx(0.53, abs=0.01)


def test_a_kv_type_nobody_supports_is_refused_at_the_door():
    with pytest.raises(ValueError):
        LoadModelRequest(kv_type="q4_0")


@needs_real_model
def test_a_state_saved_in_8_bits_restores_in_8_bits(tmp_path):
    stage = start_stage(kv_type="q8_0")
    try:
        tokens = tokens_for_test()
        expected = run(stage, tokens, cuts=[700], saves=[save_at(700, "history")])
        stage.call("clear", slot=0)
        stage.call("restore", key="history", slot=1)
        assert np.array_equal(run(stage, tokens, start=700, slot=1), expected)
    finally:
        stage.close()


def test_the_same_stage_finds_its_states_again():
    assert worker.WorkerRuntime.state_namespace(request_for()) == worker.WorkerRuntime.state_namespace(request_for())


def test_a_busy_stage_is_not_waited_for():
    """Health checks read the store's numbers, and a long prompt must not hold them up."""
    from isharded.gguf_stage import GgufStage
    from isharded.llama_host import StageHost

    host = StageHost.__new__(StageHost)
    host.lock = threading.Lock()
    stage = GgufStage.__new__(GgufStage)
    stage.host, stage.states = host, {"states": 1}
    with host.lock:
        stage.refresh_states()
    assert stage.states == {"states": 1}


class Slots:
    """A stage's llama.cpp sequences, as far as restoring goes."""

    def __init__(self, held=("k",)):
        self.held, self.calls = set(held), []

    def restore(self, key, slot):
        self.calls.append(("restore", key, slot))
        if key not in self.held:
            raise LlamaError(f"no saved state {key} here")
        return "ram"

    def clear(self, slot):
        self.calls.append(("clear", slot))


def later_stage(slots=None):
    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.gguf, runtime.caches, runtime.tokens_reused, runtime.fetched = slots or Slots(), {}, {}, set()
    runtime.epoch, runtime.lock = 1, asyncio.Lock()
    return runtime


def first_pass(session="s", key="k", length=10, slot=0):
    return InferRequest(session_id=session, epoch=1, input_ids=[1], restore_length=length, reused_rows=length, checkpoint_key=key, slot=slot)


def test_a_state_restored_ahead_is_what_the_first_pass_runs_on():
    runtime = later_stage()
    assert asyncio.run(runtime.take_restore("s", 1, 0, "k", 10))
    assert runtime._session(first_pass()).length == 10
    assert runtime.gguf.calls == [("restore", "k", 0)], "restored once, before the pass"
    assert runtime.tokens_reused == {"history_ram": 10}
    assert runtime.caches["s"].ahead is None, "only that one pass runs on it"


def test_a_restore_that_comes_after_its_pass_does_nothing():
    runtime = later_stage()
    entry = runtime._session(first_pass())
    entry.length = 40  # the pass ran on from the restored state
    assert not asyncio.run(runtime.take_restore("s", 1, 0, "k", 10))
    assert runtime.gguf.calls == [("restore", "k", 0)]
    assert runtime._session(first_pass().model_copy(update={"restore_length": None, "checkpoint_key": None})) is entry


def test_a_restore_for_another_reply_leaves_nothing_to_go_on_from():
    runtime = later_stage(Slots(held=("k", "other")))
    runtime._session(first_pass())
    # Something old, held up somewhere, lands on the slot after this reply's first pass.
    assert asyncio.run(runtime.take_restore("t", 1, 0, "other", 20))
    with pytest.raises(HTTPException) as caught:
        runtime._session(first_pass().model_copy(update={"restore_length": None, "checkpoint_key": None}))
    assert caught.value.detail["retry"], "replayed, never run on a state that is not its own"
    # And a first pass that is not the one it was for restores its own state.
    assert runtime._session(first_pass()).length == 10
    assert runtime.gguf.calls[-1] == ("restore", "k", 0)


def test_an_older_restore_of_the_same_conversation_is_not_gone_on_from():
    runtime = later_stage(Slots(held=("k", "older")))
    runtime._session(first_pass())
    assert asyncio.run(runtime.take_restore("s", 1, 0, "older", 5))
    with pytest.raises(HTTPException) as caught:
        runtime._session(first_pass().model_copy(update={"restore_length": None, "checkpoint_key": None}))
    assert caught.value.detail["retry"]


def test_a_state_a_stage_cannot_restore_ahead_still_comes_back_as_missing():
    runtime = later_stage(Slots(held=()))
    assert not asyncio.run(runtime.take_restore("s", 1, 0, "k", 10))
    with pytest.raises(HTTPException) as caught:
        runtime._session(first_pass())
    assert caught.value.detail["missing"] == "k"


def test_a_restore_for_another_route_or_a_lost_stage_is_refused():
    runtime = later_stage()
    assert not asyncio.run(runtime.take_restore("s", 2, 0, "k", 10))
    assert runtime.gguf.calls == []

    class Lost(Slots):
        def restore(self, key, slot):
            raise StageLost("the stage process is gone")

    runtime = later_stage(Lost())
    with pytest.raises(StageLost):
        asyncio.run(runtime.take_restore("s", 1, 0, "k", 10))


def test_a_reply_has_every_later_stage_restore_at_once_and_waits_for_them(monkeypatch):
    prompt = chat(200, 100, 10)
    key = prefix_key(prompt[:300])
    runtime = fake_entry(monkeypatch, [missing(key), {"token_id": 5}])
    runtime.route = [SimpleNamespace(worker_id=name) for name in ("w1", "w2", "w3")]
    runtime.prefix.add(key, 300, HISTORY)
    done = []

    async def restore_ahead(request):
        await asyncio.sleep(0.01)
        done.append((request.checkpoint_key, request.restore_length, len(runtime.passes)))

    runtime.restore_ahead = restore_ahead
    monkeypatch.setattr(worker, "PROMPT_PARTS", False)  # the replay reads it whole, here in one pass
    assert generate(prompt) == [{"token_id": 5}]
    # It went out with the first pass, and the reply waited for it even though that failed.
    assert done == [(key, 300, 1)]
    assert runtime.passes[1].reset and runtime.passes[1].checkpoint_key is None


def test_restore_ahead_goes_to_every_stage_but_this_one():
    seen = []

    def handler(request):
        seen.append((request.url.host, json.loads(request.content)))
        return httpx.Response(200, json={"restored": True})

    runtime = worker.WorkerRuntime.__new__(worker.WorkerRuntime)
    runtime.worker_id, runtime.pool_token = "w1", "secret"
    runtime.route = [SimpleNamespace(worker_id=name, url=f"http://{name}:1/") for name in ("w1", "w2", "w3")]

    async def run():
        runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await runtime.restore_ahead(first_pass(length=300))
        await runtime.client.aclose()

    asyncio.run(run())
    assert sorted(host for host, _ in seen) == ["w2", "w3"]
    assert seen[0][1] == {"session_id": "s", "epoch": 1, "slot": 0, "key": "k", "length": 300}
