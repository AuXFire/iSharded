"""A pipeline against a fake route that runs passes the way stages do: in order, and blindly."""

from __future__ import annotations

import asyncio
from collections import deque
import random

import pytest

from isharded.pipeline import Pipeline

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
