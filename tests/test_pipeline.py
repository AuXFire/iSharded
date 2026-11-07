"""A pipeline against a fake route that runs passes the way stages do: in order, and blindly."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import random
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from isharded import worker as worker_module
from isharded.pipeline import Pipeline
from isharded.protocol import GenerateRequest, InferRequest, SavePoint
from isharded.statestore import pass_ends, prompt_parts

STOP = 999


def model(text: list[int]) -> int:
    """The next token, which depends on the whole text, so a wrong context shows up."""
    return (sum((index + 1) * token for index, token in enumerate(text[-4:])) * 31 + len(text)) % 500


class Route:
    """Holds what stages hold. A pass takes ``length`` turns of the loop to come back."""

    def __init__(self, text: list[int], length: int = 4) -> None:
        self.held = list(text)
        self.length = length
        self.waiting: deque[tuple[asyncio.Future[list[int]], int, int | None, int, int]] = deque()
        self.clock = 0
        self.most_out = 0
        self.sent = 0
        self.furthest = 0
        self.work: asyncio.Task[None] | None = None

    def send(self, token: int, position: int, rollback_to: int | None) -> asyncio.Future[list[int]]:
        answer = asyncio.get_running_loop().create_future()
        self.waiting.append((answer, token, rollback_to, position, self.clock + self.length))
        self.sent += 1
        self.furthest = max(self.furthest, position)
        self.most_out = max(self.most_out, len(self.waiting))
        if self.work is None:
            self.work = asyncio.create_task(self.run())
        return answer

    async def run(self) -> None:
        while self.waiting:
            while self.clock < self.waiting[0][4]:
                self.clock += 1
                await asyncio.sleep(0)
            answer, token, rollback_to, position, _ = self.waiting.popleft()
            if rollback_to is not None:
                del self.held[rollback_to:]
            assert len(self.held) == position, "a pass ran at the wrong place"
            self.held.append(token)
            if not answer.cancelled():
                answer.set_result([model(self.held)])
            await asyncio.sleep(0)
        self.work = None


class Draft:
    """Guesses right as often as it is told to, and holds text like a real draft.

    A guess takes a few turns of the loop, as a real one takes time, so answers can land
    while it runs.
    """

    def __init__(self, right: float, seed: int = 1) -> None:
        self.right = right
        self.random = random.Random(seed)
        self.calls = 0
        self.committed = 0
        self.rewinds: list[int] = []

    async def guess(self, text: list[int]) -> int:
        self.calls += 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.committed = len(text)
        truth = model(text)
        return truth if self.random.random() < self.right else (truth + 1) % 500

    async def rewind(self, keep: int) -> None:
        self.rewinds.append(keep)
        self.committed = min(self.committed, keep)


def plain(prompt: list[int], first: int, count: int, stops=()) -> list[int]:
    text = [*prompt, first]
    made = [first]
    while len(made) < count and made[-1] not in stops:
        made.append(model(text))
        text.append(made[-1])
    return made


def pipeline(route: Route, draft: Draft, history: list[int], room: int, limit: int = 6, stops=(), sharing=lambda: 1) -> Pipeline:
    async def wait(answer: asyncio.Future[list[int]]) -> int:
        return (await answer)[0]

    return Pipeline(history, room, stops, 4, limit, route.send, wait, draft.guess, draft.rewind, sharing)


async def reply(right: float, count: int = 200, seed: int = 1, limit: int = 6, length: int = 4) -> tuple[list[int], Route, Draft, Pipeline]:
    prompt = [7, 8, 9]
    first = model(prompt)
    route, draft = Route(prompt, length), Draft(right, seed)
    history = [*prompt, first]
    pipe = pipeline(route, draft, history, count - 1, limit)
    made = [first]
    while pipe.room > 0:
        async for token in pipe.tokens(24):
            made.append(token)
    await pipe.close()
    return made, route, draft, pipe


@pytest.mark.parametrize("right", [0.0, 0.3, 0.6, 0.9, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_reply_is_the_plain_reply_however_the_guesses_go(right: float, seed: int) -> None:
    made, _, _, _ = asyncio.run(reply(right, seed=seed))
    prompt = [7, 8, 9]
    assert made == plain(prompt, model(prompt), 200)


def test_good_guesses_put_several_passes_on_the_route() -> None:
    _, route, _, pipe = asyncio.run(reply(1.0, length=40))
    assert route.most_out >= 4
    assert pipe.misses == 0


def test_a_wrong_guess_is_counted_and_the_draft_goes_back() -> None:
    _, _, draft, pipe = asyncio.run(reply(0.5, length=40))
    assert pipe.misses > 10
    assert draft.rewinds


def test_passes_never_run_past_the_room_the_reply_has() -> None:
    made, route, _, _ = asyncio.run(reply(1.0, count=20))
    assert len(made) == 20
    assert route.furthest <= 3 + 20


def test_depth_stops_at_what_guessing_may_cost_a_stage() -> None:
    _, route, _, _ = asyncio.run(reply(1.0, limit=3, length=40))
    assert route.most_out <= 3


def test_nothing_is_sent_past_a_guess_that_ends_the_reply() -> None:
    async def run() -> tuple[list[int], Route]:
        prompt = [1, 2]
        text = [*prompt, model(prompt)]
        route = Route(text[:-1])
        end = model(text)

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await answer)[0]

        async def guess(_: list[int]) -> int:
            return end

        async def rewind(_: int) -> None:
            return None

        pipe = Pipeline(text, 50, {end}, 4, 6, route.send, wait, guess, rewind)
        made = [token async for token in pipe.tokens(24)]
        await pipe.close()
        return made, route

    made, route = asyncio.run(run())
    assert len(made) == 1
    assert route.sent == 1


def test_a_route_full_of_other_replies_gets_no_guesses() -> None:
    async def run() -> tuple[Route, Draft]:
        prompt = [3, 4]
        history = [*prompt, model(prompt)]
        route, draft = Route(history[:-1]), Draft(1.0)
        pipe = pipeline(route, draft, history, 30, sharing=lambda: 6)
        async for _ in pipe.tokens(30):
            pass
        await pipe.close()
        return route, draft

    route, draft = asyncio.run(run())
    assert route.most_out == 1
    assert draft.calls == 0


def test_depth_fills_the_route_without_running_away() -> None:
    pipe = pipeline(Route([1]), Draft(1.0), [1, 2], 500, limit=16)
    pipe.right = 0.9
    pipe.trip, pipe.step = 0.036, 0.009
    assert pipe.depth() == 5
    pipe.trip, pipe.step = 0.100, 0.009  # a far route needs more out to stay full
    assert pipe.depth() == 13
    pipe.right = 0.3  # and guesses that rarely land are not worth that many
    assert pipe.depth() == 3


def test_a_wrong_guess_is_answered_without_waiting_for_the_draft() -> None:
    async def run() -> tuple[list[int], list[str]]:
        prompt = [5, 6]
        history = [*prompt, model(prompt)]
        route = Route(history[:-1], length=1)
        order: list[str] = []
        released = asyncio.Event()

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await answer)[0]

        async def guess(text: list[int]) -> int:
            if len(order) >= 2:
                await released.wait()  # the draft is slow, and the route is not
            order.append("guess")
            return (model(text) + 1) % 500  # always wrong

        async def rewind(_: int) -> None:
            return None

        pipe = Pipeline(history, 6, (), 4, 6, route.send, wait, guess, rewind)
        made = []
        async for token in pipe.tokens(3):
            made.append(token)
            order.append("token")
            if len(made) == 3:
                released.set()
        await pipe.close()
        return made, order

    made, order = asyncio.run(run())
    assert len(made) == 3
    assert order.count("token") == 3


def test_a_guess_that_lands_after_a_miss_is_not_sent() -> None:
    async def run() -> tuple[list[int], Route]:
        prompt = [2, 3]
        history = [*prompt, model(prompt)]
        route = Route(history[:-1], length=2)
        draft = Draft(0.0, seed=4)
        pipe = pipeline(route, draft, history, 40)
        made = [token async for token in pipe.tokens(40)]
        await pipe.close()
        return made, route

    made, _ = asyncio.run(run())
    assert made == plain([2, 3], model([2, 3]), 41)[1:]


def test_a_guess_for_a_place_already_answered_is_thrown_away() -> None:
    async def run() -> tuple[list[int], int, Route]:
        prompt = [4, 4]
        history = [*prompt, model(prompt)]
        route = Route(history[:-1], length=1)
        draft = Draft(1.0)  # always right, and slower than the route
        slow = draft.guess

        async def guess(text: list[int]) -> int:
            for _ in range(12):
                await asyncio.sleep(0)
            return await slow(text)

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await answer)[0]

        pipe = Pipeline(history, 30, (), 4, 6, route.send, wait, guess, draft.rewind)
        made = [token async for token in pipe.tokens(30)]
        await pipe.close()
        return made, pipe.misses, route

    made, misses, _ = asyncio.run(run())
    assert made == plain([4, 4], model([4, 4]), 31)[1:]
    assert misses == 0  # a right guess that came late is not a wrong one


def test_a_pass_that_never_comes_back_ends_the_wait() -> None:
    async def run() -> None:
        history = [1, 2, 3]
        draft = Draft(1.0)

        def send(*_: object) -> asyncio.Future[list[int]]:
            return asyncio.get_running_loop().create_future()

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await asyncio.wait_for(answer, 0.05))[0]

        pipe = Pipeline(history, 30, (), 4, 2, send, wait, draft.guess, draft.rewind)
        try:
            async for _ in pipe.tokens(30):
                pass
        finally:
            await pipe.close()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(run(), 5))


def test_landing_leaves_the_reply_ready_to_go_on_another_way() -> None:
    async def run() -> tuple[list[int], Draft, Route]:
        prompt = [5, 6]
        history = [*prompt, model(prompt)]
        route, draft = Route(history[:-1]), Draft(1.0)
        pipe = pipeline(route, draft, history, 40)
        async for _ in pipe.tokens(10):
            pass
        await pipe.land()
        # A plain pass from the newest token, as the worker sends one.
        newest = len(history) - 1
        answer = route.send(history[newest], newest, newest)
        history.append((await answer)[0])
        return history, draft, route

    history, draft, _ = asyncio.run(run())
    assert draft.committed <= len(history) - 1
    assert history[3:] == plain([5, 6], model([5, 6]), 12)[1:]


def test_a_prompt_divides_where_a_whole_prompt_would_stop() -> None:
    assert prompt_parts([], 100) == [(0, 100)]
    assert prompt_parts([], 1500) == [(0, 512), (512, 1024), (1024, 1500)]
    # A cut starts a pass of its own, so the passes after it count from the cut.
    assert prompt_parts([700], 2000) == [(0, 512), (512, 700 + 512), (1212, 1724), (1724, 2000)]
    every = [end for _, end in prompt_parts([700], 2000)]
    assert set(every[:-1]) <= set(pass_ends([700], 2000))


class Parts:
    """A stage's ordering, without a stage."""

    def __init__(self) -> None:
        self.runtime = worker_module.WorkerRuntime.__new__(worker_module.WorkerRuntime)
        self.runtime.parts_run = {}
        self.runtime.part_ran = asyncio.Condition()
        self.seen: list[InferRequest] = []
        self.delays: dict[int, int] = {}
        self.fail_at: int | None = None

    async def step(self, part: InferRequest) -> dict[str, int]:
        if part.part_index == self.fail_at:
            raise HTTPException(status_code=502, detail="gone")
        self.seen.append(part)
        await self.runtime.note_part(part)
        for _ in range(self.delays.get(part.part_index, 0)):
            await asyncio.sleep(0)
        return {"token_id": part.part_index}


def whole_prompt(**fields) -> InferRequest:
    saves = [SavePoint(at=512, length=512, key="a", kind="prefix"), SavePoint(at=1100, length=1100, key="b", kind="history")]
    return InferRequest(session_id="s", epoch=1, input_ids=list(range(1500)), reset=True, prompt_rows=True, cuts=[1100], saves=saves, **fields)


PIECES = [(0, 512), (512, 1100), (1100, 1500)]


def test_parts_carry_their_own_rows_cuts_and_saves() -> None:
    parts = Parts()
    result = asyncio.run(parts.runtime.run_in_parts(whole_prompt(), PIECES, parts.step))
    assert result == {"token_id": 2}
    first, second, third = parts.seen
    assert [len(part.input_ids) for part in parts.seen] == [512, 588, 400]
    assert second.input_ids[0] == 512 and third.input_ids[0] == 1100
    assert [save.at for save in first.saves] == [512]
    assert [save.at for save in second.saves] == [588]
    assert third.saves == []
    assert first.cuts == second.cuts == third.cuts == []
    assert (first.reset, second.reset, third.reset) == (True, False, False)
    assert {part.part_count for part in parts.seen} == {3}
    assert len({part.part_of for part in parts.seen}) == 1


def test_only_the_first_part_restores() -> None:
    parts = Parts()
    whole = whole_prompt(restore_length=100, checkpoint_key="k", reused_rows=100)
    asyncio.run(parts.runtime.run_in_parts(whole, PIECES, parts.step))
    assert parts.seen[0].restore_length == 100
    assert parts.seen[1].restore_length is None and parts.seen[1].checkpoint_key is None and parts.seen[1].reused_rows == 0


def test_a_part_leaves_before_the_one_ahead_has_crossed_the_route() -> None:
    parts = Parts()
    parts.delays = {0: 1000}  # the first part is still away along the route
    asyncio.run(parts.runtime.run_in_parts(whole_prompt(), PIECES, parts.step))
    assert [part.part_index for part in parts.seen] == [0, 1, 2]


def test_a_part_that_is_through_before_anyone_looked_is_no_failure() -> None:
    async def run() -> dict[str, int]:
        parts = Parts()

        async def step(part: InferRequest) -> dict[str, int]:
            parts.seen.append(part)
            return {"token_id": part.part_index}  # says nothing on this stage, as a test can

        async def noted(part: InferRequest) -> dict[str, int]:
            answer = await step(part)
            parts.runtime.parts_run[part.slot] = (part.part_of, part.part_index)
            return answer

        return await parts.runtime.run_in_parts(whole_prompt(), PIECES, noted)

    assert asyncio.run(run()) == {"token_id": 2}


def test_a_failed_part_fails_the_prompt_and_stops_the_rest() -> None:
    parts = Parts()
    parts.fail_at = 1
    with pytest.raises(HTTPException) as caught:
        asyncio.run(parts.runtime.run_in_parts(whole_prompt(), PIECES, parts.step))
    assert caught.value.status_code == 502
    assert [part.part_index for part in parts.seen] == [0]


def test_a_stage_runs_parts_in_order_even_if_they_arrive_out_of_order() -> None:
    async def run() -> list[int]:
        parts = Parts()
        order: list[int] = []

        async def arrive(index: int) -> None:
            part = InferRequest(session_id="s", epoch=1, input_ids=[1], slot=2, part_of="g", part_index=index, part_count=3)
            if index:
                await parts.runtime.await_part(part, index - 1, 5)
            order.append(index)
            await parts.runtime.note_part(part)

        await asyncio.gather(arrive(2), arrive(1), arrive(0))
        return order

    assert asyncio.run(run()) == [0, 1, 2]


def test_a_part_whose_prompt_was_replaced_never_runs() -> None:
    async def run() -> None:
        parts = Parts()
        await parts.runtime.note_part(InferRequest(session_id="s", epoch=1, input_ids=[1], part_of="new", part_index=0))
        stale = InferRequest(session_id="s", epoch=1, input_ids=[1], part_of="old", part_index=1)
        await parts.runtime.await_part(stale, 0, 0.01)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(run())
    assert caught.value.status_code == 503


# A reply that failed part way carries on from what it wrote.


def following(**fields) -> InferRequest:
    return InferRequest(session_id="s", epoch=1, input_ids=[], **fields)


def test_written_tokens_each_get_their_own_pass_over_http(monkeypatch) -> None:
    passes = []

    async def reply_step(request):
        passes.append(request.input_ids)
        return {"token_id": 400 + len(passes)}

    monkeypatch.setattr(worker_module, "reply_step", reply_step)
    monkeypatch.setattr(worker_module.runtime, "ring_ready", lambda request: False)
    answer = asyncio.run(worker_module.read_written(following(), [7, 8, 9], 5))
    assert passes == [[7], [8], [9]]
    assert answer == 403, "only the last answer is the next token"


class Ring:
    """Answers after a few turns of the loop, whatever was sent, and in order."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.sent: list[tuple[int, int, int | None]] = []
        self.out: deque[asyncio.Future[list[int]]] = deque()
        self.most_out = 0
        self.fail_at = fail_at
        self.work: asyncio.Task[None] | None = None

    def send(self, request: InferRequest, token: int, position: int, rollback_to: int | None) -> asyncio.Future[list[int]]:
        answer = asyncio.get_running_loop().create_future()
        self.sent.append((token, position, rollback_to))
        self.out.append(answer)
        self.most_out = max(self.most_out, len(self.out))
        if self.work is None:
            self.work = asyncio.create_task(self.run())
        return answer

    async def run(self) -> None:
        answered = 0
        while True:
            await asyncio.sleep(0)
            if self.out:
                answered += 1
                answer = self.out.popleft()
                if answered == self.fail_at:
                    answer.set_exception(HTTPException(status_code=503, detail={"retry": True}))
                elif not answer.done():
                    answer.set_result([1000 + answered])

    async def answer(self, flight: asyncio.Future[list[int]]) -> int:
        return (await flight)[0]


def on_ring(monkeypatch, ring: Ring) -> None:
    monkeypatch.setattr(worker_module.runtime, "ring_ready", lambda request: True)
    monkeypatch.setattr(worker_module.runtime, "ring_send", ring.send)
    monkeypatch.setattr(worker_module.runtime, "ring_answer", ring.answer)


def test_written_tokens_go_back_to_back_on_the_ring_and_within_a_window(monkeypatch) -> None:
    ring = Ring()
    on_ring(monkeypatch, ring)
    monkeypatch.setattr(worker_module, "RESUME_WINDOW", 4)
    written = list(range(50, 70))

    async def run() -> int:
        answer = await worker_module.read_written(following(), written, 100)
        ring.work.cancel()
        return answer

    assert asyncio.run(run()) == 1000 + len(written), "the last pass answers for the reply"
    assert ring.sent == [(token, 100 + index, None) for index, token in enumerate(written)]
    assert 1 < ring.most_out <= 4


def test_a_pass_that_fails_on_the_ring_ends_the_read_and_drops_the_rest(monkeypatch) -> None:
    ring = Ring(fail_at=2)
    on_ring(monkeypatch, ring)
    monkeypatch.setattr(worker_module, "RESUME_WINDOW", 3)
    seen: list[asyncio.Future[list[int]]] = []
    sending = ring.send

    def send(*arguments) -> asyncio.Future[list[int]]:
        seen.append(sending(*arguments))
        return seen[-1]

    monkeypatch.setattr(worker_module.runtime, "ring_send", send)

    async def run() -> None:
        try:
            await worker_module.read_written(following(), list(range(10)), 0)
        finally:
            ring.work.cancel()

    with pytest.raises(HTTPException):
        asyncio.run(run())
    assert len(ring.sent) < 10
    assert not any(flight.cancelled() for flight in seen), "finish_reply waits on what is still out"


def test_a_reply_that_goes_away_leaves_its_passes_out_for_the_slot_to_wait_on(monkeypatch) -> None:
    """ring_settle waits on the newest pass. Cancelled, it would free the slot with passes still on the ring."""
    held: list[asyncio.Future[list[int]]] = []

    def send(*arguments) -> asyncio.Future[list[int]]:
        held.append(asyncio.get_running_loop().create_future())
        return held[-1]

    async def answer(flight: asyncio.Future[list[int]]) -> int:
        return (await flight)[0]

    monkeypatch.setattr(worker_module.runtime, "ring_ready", lambda request: True)
    monkeypatch.setattr(worker_module.runtime, "ring_send", send)
    monkeypatch.setattr(worker_module.runtime, "ring_answer", answer)
    monkeypatch.setattr(worker_module, "RESUME_WINDOW", 3)

    async def run() -> None:
        reading = asyncio.ensure_future(worker_module.read_written(following(), list(range(10)), 0))
        await asyncio.sleep(0)
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)

    asyncio.run(run())
    assert len(held) == 3
    assert not held[-1].cancelled()


async def stream_of(body) -> list[int]:
    response = await worker_module.generate(body, None)
    lines = [line async for line in response.body_iterator]
    return [json.loads(line)["token_id"] for line in lines]


@pytest.fixture
def entry(monkeypatch):
    """The entry worker of a one-stage route, with passes answered from ``model``."""
    runtime = worker_module.runtime
    monkeypatch.setattr(worker_module, "verify_token", lambda authorization: None)
    monkeypatch.setattr(runtime, "assignment", SimpleNamespace(start_layer=0))
    monkeypatch.setattr(runtime, "epoch", 1)
    monkeypatch.setattr(runtime, "gguf", None)
    monkeypatch.setattr(runtime, "draft", None)
    monkeypatch.setattr(runtime, "route", [])
    monkeypatch.setattr(runtime, "ring_ready", lambda request: False)
    passes: list[InferRequest] = []

    async def reply_step(request, part=None):
        passes.append(request)
        if request.verify:
            return {"token_ids": [300 + len(passes)]}
        return {"token_id": 300 + len(passes)}

    monkeypatch.setattr(worker_module, "reply_step", reply_step)
    return passes


def test_a_resumed_reply_reads_the_prompt_as_usual_and_streams_only_new_tokens(entry) -> None:
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[1, 2, 3], written=[7, 8], max_tokens=2, reset=True)
    assert asyncio.run(stream_of(body)) == [303, 304]
    prompt, *rest = entry
    assert prompt.input_ids == [1, 2, 3] and prompt.prompt_rows and prompt.reset
    assert [item.input_ids for item in rest] == [[7], [8], [303]]
    assert not any(item.prompt_rows or item.reset for item in rest), "plain decoding ran them, so these are plain passes"


def test_a_reply_with_nothing_written_runs_as_it_always_did(entry) -> None:
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[1, 2], max_tokens=2, reset=True)
    assert asyncio.run(stream_of(body)) == [301, 302]
    assert [item.input_ids for item in entry] == [[1, 2], [301]]


def test_a_draft_goes_on_from_the_whole_reply_after_a_resume(entry, monkeypatch) -> None:
    class Draft:
        texts: list[list[int]] = []

        def prepare(self, prompt, plan, slot) -> None:
            self.prompt = list(prompt)

        def guess(self, text, count, slot) -> list[int]:
            self.texts.append(list(text))
            return [5] * count

    draft = Draft()
    runtime = worker_module.runtime
    monkeypatch.setattr(runtime, "draft", draft)
    monkeypatch.setattr(runtime, "speculate_tokens", 2)
    monkeypatch.setattr(runtime, "draft_choice", worker_module.DraftChoice())
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[1, 2], written=[7, 8], max_tokens=2, reset=True)
    tokens = asyncio.run(stream_of(body))
    assert tokens[0] == 303
    assert draft.prompt == [1, 2], "it reads the prompt, and the rest as text it has not seen"
    assert draft.texts[0] == [1, 2, 7, 8, 303]
    assert entry[3].rollback_to == 4, "the next pass goes after everything already written"
