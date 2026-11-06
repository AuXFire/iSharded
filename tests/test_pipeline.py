"""A pipeline against a fake route that runs passes the way stages do: in order, and blindly."""

from __future__ import annotations

import asyncio
from collections import deque
import random

from fastapi import HTTPException
import pytest

from isharded import worker as worker_module
from isharded.pipeline import Pipeline
from isharded.protocol import InferRequest, SavePoint
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
        self.waiting: deque[tuple[asyncio.Future[list[int]], int, int | None, int]] = deque()
        self.most_out = 0
        self.sent = 0
        self.furthest = 0
        self.work: asyncio.Task[None] | None = None

    def send(self, token: int, position: int, rollback_to: int | None) -> asyncio.Future[list[int]]:
        answer = asyncio.get_running_loop().create_future()
        self.waiting.append((answer, token, rollback_to, position))
        self.sent += 1
        self.furthest = max(self.furthest, position)
        self.most_out = max(self.most_out, len(self.waiting))
        if self.work is None:
            self.work = asyncio.create_task(self.run())
        return answer

    async def run(self) -> None:
        for _ in range(self.length):
            await asyncio.sleep(0)
        while self.waiting:
            answer, token, rollback_to, position = self.waiting.popleft()
            if rollback_to is not None:
                del self.held[rollback_to:]
            assert len(self.held) == position, "a pass ran at the wrong place"
            self.held.append(token)
            answer.set_result([model(self.held)])
            await asyncio.sleep(0)
        self.work = None


class Draft:
    """Guesses right as often as it is told to, and holds text like a real draft."""

    def __init__(self, right: float, seed: int = 1) -> None:
        self.right = right
        self.random = random.Random(seed)
        self.calls = 0
        self.committed = 0
        self.rewinds: list[int] = []

    async def guess(self, text: list[int]) -> int:
        self.calls += 1
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


async def reply(right: float, count: int = 200, seed: int = 1, limit: int = 6) -> tuple[list[int], Route, Draft, Pipeline]:
    prompt = [7, 8, 9]
    first = model(prompt)
    route, draft = Route(prompt), Draft(right, seed)
    history = [*prompt, first]
    pipe = pipeline(route, draft, history, count - 1, limit)
    made = [first]
    while pipe.room > 0:
        async for token in pipe.tokens(24):
            made.append(token)
    return made, route, draft, pipe


@pytest.mark.parametrize("right", [0.0, 0.3, 0.6, 0.9, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_reply_is_the_plain_reply_however_the_guesses_go(right: float, seed: int) -> None:
    made, _, _, _ = asyncio.run(reply(right, seed=seed))
    prompt = [7, 8, 9]
    assert made == plain(prompt, model(prompt), 200)


def test_good_guesses_put_several_passes_on_the_route() -> None:
    _, route, _, pipe = asyncio.run(reply(1.0))
    assert route.most_out >= 5
    assert pipe.misses == 0


def test_a_wrong_guess_is_counted_and_the_draft_goes_back() -> None:
    _, _, draft, pipe = asyncio.run(reply(0.5))
    assert pipe.misses > 10
    assert len(draft.rewinds) >= pipe.misses - 2  # a rewind waits for the next guess


def test_passes_never_run_past_the_room_the_reply_has() -> None:
    made, route, _, _ = asyncio.run(reply(1.0, count=20))
    assert len(made) == 20
    assert route.furthest <= 3 + 20


def test_depth_stops_at_what_guessing_may_cost_a_stage() -> None:
    _, route, _, _ = asyncio.run(reply(1.0, limit=3))
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
