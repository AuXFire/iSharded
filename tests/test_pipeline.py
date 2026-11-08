"""A pipeline against a fake route that runs passes the way stages do: in order, and blindly."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import psutil
import pytest

from isharded import worker as worker_module
from isharded import gguf_stage as gguf_stage_module
from isharded.engine import find_library
from isharded import pipeline as pipeline_module
from isharded.pipeline import LOOKUP_BEATS_DRAFT, LOOKUP_SHORTEST, LOOKUP_WINDOW, Pipeline, PromptLookup, RowClock
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
        self.text: list[int] = []
        self.rewinds: list[int] = []

    async def guess(self, fresh: list[int]) -> int:
        self.calls += 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.text += fresh
        truth = model(self.text)
        return truth if self.random.random() < self.right else (truth + 1) % 500

    async def rewind(self, keep: int) -> None:
        self.rewinds.append(keep)
        del self.text[keep:]


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


def priced(replies: int, trip: float, row: float, limit: int = 16) -> Pipeline:
    pipe = pipeline(Route([1]), Draft(1.0), [1, 2], 500, limit=limit, sharing=lambda: replies)
    pipe.right = 0.99
    pipe.trip, pipe.step = trip, row * replies
    pipe.clock.row = row
    return pipe


def test_a_route_full_of_other_replies_gets_no_guesses() -> None:
    # Six replies at one pass each already ask for all a route of these times can pass.
    async def run() -> tuple[Route, Draft]:
        prompt = [3, 4]
        history = [*prompt, model(prompt)]
        route, draft = Route(history[:-1]), Draft(1.0)
        pipe = pipeline(route, draft, history, 30, sharing=lambda: 6)
        pipe.trip, pipe.clock.row = 0.006, 0.001
        async for _ in pipe.tokens(30):
            pipe.trip, pipe.clock.row = 0.006, 0.001
        await pipe.close()
        return route, draft

    route, draft = asyncio.run(run())
    assert route.most_out == 1
    assert draft.calls == 0


def test_eight_replies_on_a_far_route_each_keep_more_than_one_pass_out() -> None:
    assert priced(8, trip=0.250, row=0.010).depth() == 3


@pytest.mark.parametrize("replies", [2, 3, 5, 8, 16, 32])
@pytest.mark.parametrize("trip", [0.02, 0.1, 0.25, 0.7])
@pytest.mark.parametrize("row", [0.001, 0.008, 0.03])
def test_the_rows_replies_ask_for_stay_within_what_a_route_passes(replies: int, trip: float, row: float) -> None:
    depth = priced(replies, trip, row).depth()
    assert depth == 1 or replies * depth / trip <= 1 / row


def test_the_price_never_lifts_the_fill_and_worth_limits() -> None:
    pipe = priced(2, trip=0.5, row=0.001)
    pipe.step = 0.1
    assert pipe.depth() == 6, "filling the route still limits it"
    pipe.step = 0.002
    pipe.right = 0.3
    assert pipe.depth() == 3, "and so does what guesses are worth"


def test_a_reply_on_its_own_pays_no_row_price() -> None:
    pipe = priced(1, trip=0.036, row=0.009)
    pipe.step = 0.009
    assert pipe.depth() == 5, "enough to fill the route, as it always was"


def test_the_price_follows_a_reply_that_joins_at_once() -> None:
    replies = [1]
    pipe = pipeline(Route([1]), Draft(1.0), [1, 2], 500, limit=16, sharing=lambda: replies[0])
    pipe.right = 0.99
    pipe.trip, pipe.step, pipe.clock.row = 0.1, 0.01, 0.01
    assert pipe.depth() == 11
    replies[0] = 5
    assert pipe.depth() == 2, "without waiting for the step to notice"


def test_a_reply_with_nothing_measured_still_gets_to_measure() -> None:
    pipe = pipeline(Route([1]), Draft(1.0), [1, 2], 500, sharing=lambda: 8)
    assert pipe.depth() == 5, "one pass out could never show a gap between answers"


def test_the_row_clock_keeps_the_shortest_gap_and_forgets_it_slowly() -> None:
    clock = RowClock()
    clock.tick(1.0)
    assert clock.row is None
    clock.tick(1.5)
    clock.tick(1.51)
    assert clock.row == pytest.approx(0.01)
    clock.tick(2.5)
    assert clock.row == pytest.approx(0.0102), "a route with room to spare says little"
    for at in range(200):
        clock.tick(3.0 + at)
    assert clock.row > 0.1, "but a route can get slower"


def test_answers_from_one_merged_pass_do_not_look_like_a_fast_route() -> None:
    clock = RowClock()
    for batch in range(5):
        for row in range(8):
            clock.tick(batch * 0.05 + row * 0.0002)
    assert clock.row == pytest.approx(0.05 - 7 * 0.0002)


def test_replies_on_one_route_share_a_row_clock_so_many_replies_do_not_look_like_a_slow_route(monkeypatch) -> None:
    # Eight replies take turns on a route passing one row every 10 ms, so each hears its own
    # answers 80 ms apart, while the route is as fast as ever.
    clock = RowClock()
    pipes = [pipeline(Route([1]), Draft(1.0), [1, 2], 500) for _ in range(8)]
    now = [0.0]
    monkeypatch.setattr(pipeline_module.time, "perf_counter", lambda: now[0])
    for pipe in pipes:
        pipe.clock = clock
    for answer in range(80):
        now[0] += 0.01
        pipes[answer % 8]._timed(pipeline_module.Flight(1, 1, None, 0.0, False))
    assert clock.row == pytest.approx(0.01)
    assert pipes[0].step == pytest.approx(0.08)


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
        draft = Draft(0.0)  # always wrong

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await answer)[0]

        async def guess(fresh: list[int]) -> int:
            if len(order) >= 2:
                await released.wait()  # the draft is slow, and the route is not
            order.append("guess")
            return await draft.guess(fresh)

        pipe = Pipeline(history, 6, (), 4, 6, route.send, wait, guess, draft.rewind)
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

        async def guess(fresh: list[int]) -> int:
            for _ in range(12):
                await asyncio.sleep(0)
            return await slow(fresh)

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
    assert len(draft.text) <= len(history) - 1
    assert history[3:] == plain([5, 6], model([5, 6]), 12)[1:]


@pytest.mark.parametrize("right", [1.0, 0.7, 0.0])
def test_the_draft_hears_only_what_it_has_not_seen(right: float) -> None:
    async def run() -> tuple[list[int], list[int], Pipeline]:
        prompt = list(range(3, 5000))
        history = [*prompt, model(prompt)]
        route, draft = Route(history[:-1]), Draft(right)
        draft.text = history[:-1]
        heard: list[int] = []

        async def guess(fresh: list[int]) -> int:
            heard.append(len(fresh))
            answer = await draft.guess(fresh)
            # Its text must be the reply so far, then the guesses out there. A miss can
            # land while a guess runs, and then that guess is thrown away anyway.
            if pipe.guessing_for[0] == pipe.misses:
                assert draft.text[: len(history)] == history[: len(draft.text)]
            return answer

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await answer)[0]

        pipe = Pipeline(history, 60, (), 4, 6, route.send, wait, guess, draft.rewind, told=len(prompt))
        made = [token async for token in pipe.tokens(60)]
        await pipe.close()
        return made, heard, pipe

    made, heard, pipe = asyncio.run(run())
    prompt = list(range(3, 5000))
    assert made == plain(prompt, model(prompt), 61)[1:]
    assert heard and max(heard) <= 2  # never the history, however long it is
    assert (pipe.misses == 0) == (right == 1.0)


def looked_up(text: list[int], shortest: int = LOOKUP_SHORTEST) -> int | None:
    return PromptLookup(text).guess(shortest)


def test_lookup_reads_on_from_where_the_newest_tokens_came_before() -> None:
    assert looked_up([1, 2, 3, 4, 9, 1, 2, 3]) == 4


def test_lookup_needs_a_match_long_enough() -> None:
    assert looked_up([1, 2, 3, 4, 9, 2, 3]) is None
    assert looked_up([5, 6, 7]) is None
    assert looked_up([7]) is None


def test_lookup_takes_the_longest_match_then_the_latest() -> None:
    assert looked_up([1, 2, 3, 4, 7, 2, 3, 4, 8, 2, 3, 4, 5, 9, 1, 2, 3, 4]) == 7, "the longest"
    assert looked_up([2, 3, 4, 7, 2, 3, 4, 8, 9, 2, 3, 4]) == 8, "and the latest of equals"


def test_lookup_follows_a_match_without_searching_again(monkeypatch) -> None:
    lookup = PromptLookup([1, 2, 3, 4, 5, 6, 9, 1, 2, 3])
    assert lookup.guess(LOOKUP_SHORTEST) == 4
    monkeypatch.setattr(lookup, "_search", lambda: pytest.fail("searched while a match held"))
    lookup.put(4, 10)
    assert lookup.guess(LOOKUP_SHORTEST) == 5
    lookup.put(5, 11)
    assert lookup.matched == 5


def test_lookup_drops_a_match_the_text_left_and_one_a_rewind_took_away() -> None:
    lookup = PromptLookup([1, 2, 3, 4, 5, 6, 9, 1, 2, 3])
    assert lookup.guess(LOOKUP_SHORTEST) == 4
    lookup.put(8, 10)
    assert lookup.source < 0
    lookup.put(4, 10)  # a rewind: the model said 4 goes where 8 was
    assert lookup.end == 11
    assert lookup.guess(LOOKUP_SHORTEST) == 5


def test_lookup_holds_a_bounded_window_however_long_the_text_gets() -> None:
    size = LOOKUP_WINDOW + 100
    lookup = PromptLookup(list(range(size)))
    assert lookup.text.size == 2 * LOOKUP_WINDOW
    assert lookup.base == 100
    for position in range(size, size + 3 * LOOKUP_WINDOW):
        lookup.put(position, position)
    assert lookup.text.size == 2 * LOOKUP_WINDOW
    assert lookup.end <= lookup.text.size
    newest = size + 3 * LOOKUP_WINDOW - 1
    assert lookup.text[lookup.end - 1] == newest and lookup.base + lookup.end == newest + 1


def test_lookup_forgets_what_slides_out_of_the_window() -> None:
    text = [1, 2, 3, 4, *range(10, 10 + 2 * LOOKUP_WINDOW - 6), 1, 2, 3]
    lookup = PromptLookup([])
    for position, token in enumerate(text):
        lookup.put(token, position)
    assert lookup.guess(LOOKUP_SHORTEST) is None


def test_lookup_follows_a_match_across_a_slide() -> None:
    quote = list(range(50, 58))
    filler = range(1000, 1000 + 2 * LOOKUP_WINDOW - 12)
    text = [*filler[: LOOKUP_WINDOW + 10], *quote, *filler[LOOKUP_WINDOW + 10 :], *quote[:4]]
    lookup = PromptLookup([])
    for position, token in enumerate(text):
        lookup.put(token, position)
    assert lookup.guess(LOOKUP_SHORTEST) == quote[4]
    assert lookup.end == lookup.text.size
    lookup.put(quote[4], len(text))
    assert lookup.end == LOOKUP_WINDOW + 1, "the window slid"
    assert lookup.guess(LOOKUP_SHORTEST) == quote[5]


class Quoting(Route):
    """A route whose model repeats the prompt word for word, then writes what ``model`` says."""

    def __init__(self, text: list[int], quote: list[int], at: int, length: int = 4) -> None:
        super().__init__(text, length)
        self.quote, self.at = quote, at

    def truth(self, text: list[int]) -> int:
        offset = len(text) - self.at
        return self.quote[offset] if 0 <= offset < len(self.quote) else model(text)

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
                answer.set_result([self.truth(self.held)])
            await asyncio.sleep(0)
        self.work = None


QUOTE = [(index * 37 + 11) % 400 for index in range(60)]


async def quoted_reply(draft: Draft | None, count: int = 120, stops=(), lookup: bool = True) -> tuple[list[int], list[int], Quoting, Pipeline]:
    """A reply that quotes its prompt. Returns it and what the same route says with no guesses."""
    prompt = [*QUOTE, 450, 451, *QUOTE[:4]]
    route = Quoting(prompt, QUOTE, len(prompt))
    first = route.truth(prompt)
    history = [*prompt, first]

    async def wait(answer: asyncio.Future[list[int]]) -> int:
        return (await answer)[0]

    pipe = Pipeline(history, count - 1, stops, 4, 6, route.send, wait, draft.guess if draft else None, draft.rewind if draft else None, lookup=lookup)
    made = [first]
    while pipe.room > 0:
        before = len(made)
        async for token in pipe.tokens(24):
            made.append(token)
        if made[-1] in stops or len(made) == before:
            break
    await pipe.close()
    text = list(prompt)
    truth = []
    while len(truth) < count:
        truth.append(route.truth(text))
        text.append(truth[-1])
        if truth[-1] in stops:
            break
    return made, truth, route, pipe


def test_lookup_alone_runs_a_pipeline_and_changes_nothing() -> None:
    made, truth, route, pipe = asyncio.run(quoted_reply(None))
    assert made == truth
    assert route.most_out > 1, "with no draft, the text alone filled the route"
    assert pipe.lookups > 40
    assert pipe.misses <= 3


def test_a_reply_that_repeats_nothing_goes_one_pass_at_a_time() -> None:
    async def run() -> tuple[list[int], list[int], Route, Pipeline]:
        prompt = [7, 8, 9]
        first = model(prompt)
        route = Route(prompt)
        history = [*prompt, first]

        async def wait(answer: asyncio.Future[list[int]]) -> int:
            return (await answer)[0]

        pipe = Pipeline(history, 80, (), 4, 6, route.send, wait, None, None, lookup=True)
        made = [first]
        async for token in pipe.tokens(80):
            made.append(token)
        await pipe.close()
        await pipe.land()
        return made, plain(prompt, first, 81), route, pipe

    made, truth, route, pipe = asyncio.run(run())
    assert made == truth
    assert pipe.misses == pipe.lookups
    assert route.sent == 80 + pipe.lookups, "only what the text suggested went out beside the model's own"


def test_a_short_match_leaves_the_guessing_to_the_draft() -> None:
    made, truth, route, pipe = asyncio.run(quoted_reply(Draft(0.5), count=4))
    assert made == truth
    assert pipe.lookups <= 4
    lookup = PromptLookup([*QUOTE, 450, *QUOTE[: LOOKUP_BEATS_DRAFT - 1]])
    assert lookup.guess(LOOKUP_BEATS_DRAFT) is None
    assert lookup.guess(LOOKUP_SHORTEST) == QUOTE[LOOKUP_BEATS_DRAFT - 1]


@pytest.mark.parametrize("right", [0.0, 0.5, 1.0])
def test_a_long_match_beats_the_draft_and_the_draft_still_ends_up_with_the_right_text(right: float) -> None:
    async def run() -> tuple[list[int], list[int], Draft, Pipeline]:
        draft = Draft(right)
        made, truth, _, pipe = await quoted_reply(draft, count=90)
        await pipe.land()
        return made, truth, draft, pipe

    made, truth, draft, pipe = asyncio.run(run())
    assert made == truth
    assert pipe.lookups > 20
    assert draft.calls < len(made), "the draft was left alone while the text did the guessing"
    assert draft.text == pipe.history[: len(draft.text)], "and what it missed reached it in order"


def test_a_long_match_does_not_wait_for_a_slow_draft() -> None:
    class Slow(Draft):
        async def guess(self, fresh: list[int]) -> int:
            for _ in range(200):
                await asyncio.sleep(0)
            return await super().guess(fresh)

    async def run() -> tuple[list[int], list[int], Slow, Quoting, Pipeline]:
        draft = Slow(0.5)
        made, truth, route, pipe = await quoted_reply(draft, count=50)
        await pipe.land()
        return made, truth, draft, route, pipe

    made, truth, draft, route, pipe = asyncio.run(run())
    assert made == truth
    assert route.most_out > 1
    assert draft.calls <= 3, "the text kept the route full while the draft thought"
    assert draft.text == pipe.history[: len(draft.text)]


def test_a_stop_in_the_text_ends_the_guessing_like_a_draft_expecting_the_end() -> None:
    stop = QUOTE[20]
    made, truth, route, pipe = asyncio.run(quoted_reply(None, stops={stop}))
    assert made == truth
    assert made[-1] == stop
    assert route.furthest <= len(QUOTE) + 6 + 20, "nothing went past where the text says the reply ends"


def test_landing_a_reply_with_no_draft_needs_nothing_to_rewind() -> None:
    async def run() -> Pipeline:
        made, truth, route, pipe = await quoted_reply(None, count=10)
        await pipe.land()
        return pipe

    assert asyncio.run(run()).flights == deque()


def test_a_prompt_divides_where_a_whole_prompt_would_stop() -> None:
    assert prompt_parts([], 100) == [(0, 100)]
    assert prompt_parts([], 1500) == [(0, 512), (512, 1024), (1024, 1500)]
    # A cut starts a pass of its own, so the passes after it count from the cut.
    assert prompt_parts([700], 2000) == [(0, 512), (512, 700), (700, 1212), (1212, 1724), (1724, 2000)]
    every = [end for _, end in prompt_parts([700], 2000)]
    assert every[:-1] == pass_ends([700], 2000)


def test_every_pass_of_a_prompt_is_a_part_of_its_own() -> None:
    # A system prompt of 300 rows, then a long turn: the short pass before the cut goes
    # alone, so the longest part, which every later stage waits for once, stays one pass.
    parts = prompt_parts([300, 3700], 3704)
    assert parts[0] == (0, 300) and parts[-1] == (3700, 3704)
    assert max(end - first for first, end in parts) == 512
    assert [first for first, _ in parts[1:]] == [end for _, end in parts[:-1]]
    assert prompt_parts([300], 520) == [(0, 300), (300, 520)]
    assert prompt_parts([], 0) == [(0, 0)]


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


def trying(way: str) -> worker_module.DraftChoice:
    """A choice that has just started a trial of ``way``. A fresh one measures plain decoding first."""
    choice = worker_module.DraftChoice()
    choice._start_trial(way, [way, "plain"])
    return choice


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
            self.read = len(prompt)

        def seen(self, slot) -> int:
            return self.read

        def rewind(self, keep, slot) -> None:
            self.read = keep

        def guess(self, fresh, count, slot, sampling=None) -> list[int]:
            self.texts.append(list(fresh))
            self.read += len(fresh)
            return [5] * count

    draft = Draft()
    runtime = worker_module.runtime
    monkeypatch.setattr(runtime, "draft", draft)
    monkeypatch.setattr(runtime, "speculate_tokens", 2)
    monkeypatch.setattr(runtime, "draft_choice", trying("guessed"))
    # One stage whose probe saw a checked pass round as one row, so a greedy reply may guess.
    monkeypatch.setattr(runtime, "route", [1])
    monkeypatch.setattr(runtime, "exactness", {"exact_merge_limit": 0, "verify_exact": True})
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[1, 2], written=[7, 8], max_tokens=2, reset=True)
    tokens = asyncio.run(stream_of(body))
    assert tokens[0] == 303
    assert draft.prompt == [1, 2], "it reads the prompt, and the rest as text it has not seen"
    assert draft.texts[0] == [7, 8, 303]
    assert entry[3].rollback_to == 4, "the next pass goes after everything already written"


class CountingDraft:
    """A draft that guesses 5 every time and remembers whether it was used at all."""

    def __init__(self) -> None:
        self.read = 0
        self.prepared = False

    def prepare(self, prompt, plan, slot) -> None:
        self.prepared, self.read = True, len(prompt)

    def seen(self, slot) -> int:
        return self.read

    def rewind(self, keep, slot) -> None:
        self.read = keep

    def guess(self, fresh, count, slot, sampling=None) -> list[int]:
        self.read += len(fresh)
        return [5] * count


@pytest.mark.parametrize(
    "temperature, seed, probed, guessing",
    [
        (0.0, None, False, False),  # greedy, and nothing proved a checked pass rounds as one row
        (0.0, 7, False, False),
        (0.8, 7, True, False),  # a checked pass draws another way than plain decoding
        (0.8, None, False, True),  # a sampled reply with no seed comes out different anyway
        (0.0, None, True, True),  # the probe proved it on this one stage
        (0.0, 7, True, True),
    ],
)
def test_a_reply_that_must_repeat_checks_guesses_in_one_pass_only_where_the_probe_proved_it(entry, monkeypatch, temperature, seed, probed, guessing) -> None:
    """Five greedy runs with the draft on gave two texts at one stage on an A40: which way ran
    was chosen by timing, and a checked pass of several rows rounds differently."""
    runtime = worker_module.runtime
    draft = CountingDraft()
    monkeypatch.setattr(runtime, "draft", draft)
    monkeypatch.setattr(runtime, "speculate_tokens", 3)
    monkeypatch.setattr(runtime, "draft_choice", trying("guessed"))
    monkeypatch.setattr(runtime, "route", [1])
    monkeypatch.setattr(runtime, "exact", False)
    monkeypatch.setattr(runtime, "exactness", {"exact_merge_limit": 0, "verify_exact": True} if probed else None)
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[1, 2], max_tokens=6, reset=True, temperature=temperature, seed=seed)
    assert len(asyncio.run(stream_of(body))) == 6
    assert any(item.verify for item in entry) == guessing
    assert draft.prepared == guessing, "a reply that cannot use the draft leaves it alone, as with no draft"
    if guessing and not temperature:
        assert all(len(item.input_ids) <= 1 + worker_module.EXACT_GUESSES for item in entry[1:]), "no more rows than the probe checked"


def test_the_first_pass_after_another_way_is_not_counted_against_either(entry, monkeypatch) -> None:
    """A pass of one row after passes of four, or the other way round, waits for the device to
    set up the other shape. Counted, it read plain decoding at 21.7 tokens a second, not 30."""
    runtime = worker_module.runtime
    notes: list[tuple[str, int]] = []

    class Noted(worker_module.DraftChoice):
        def note(self, way: str, tokens: int, seconds: float) -> None:
            notes.append((way, tokens))
            super().note(way, tokens, seconds)

    monkeypatch.setattr(runtime, "draft", CountingDraft())
    monkeypatch.setattr(runtime, "speculate_tokens", 3)
    choice = Noted()
    choice._start_trial("guessed", ["guessed", "plain"])
    monkeypatch.setattr(runtime, "draft_choice", choice)
    monkeypatch.setattr(runtime, "route", [1])
    monkeypatch.setattr(runtime, "exact", False)
    monkeypatch.setattr(gguf_stage_module, "PIPELINE_STRETCH", 3)
    monkeypatch.setattr(gguf_stage_module, "TRIAL_GLANCE", 3)
    body = GenerateRequest(session_id="s", epoch=1, input_ids=[1, 2], max_tokens=12, reset=True, temperature=0.8)
    assert len(asyncio.run(stream_of(body))) == 12
    ways = ["guessed" if item.verify else "plain" for item in entry[1:]]
    assert ways[:4] == ["guessed"] * 4 and "plain" in ways, "each way had its trial"
    # The first token was a plain pass, so the reply starts on the plain way.
    expected = [(way, 1) for before, way in zip(["plain", *ways], ways) if before == way]
    assert notes == expected


@pytest.fixture
def lookup_only(entry, monkeypatch):
    """The entry of a two-stage route with no draft, where lookup is on."""
    runtime = worker_module.runtime
    monkeypatch.setattr(runtime, "route", [1, 2])
    monkeypatch.setattr(runtime, "speculate_tokens", 4)
    monkeypatch.setattr(runtime, "prompt_lookup", True)
    monkeypatch.setattr(runtime, "draft_choice", trying("pipelined"))
    monkeypatch.setattr(worker_module, "PIPELINE_STRETCH", 4)
    # A trial as long as a stretch here, so one stretch measures the pipeline.
    monkeypatch.setattr(gguf_stage_module, "PIPELINE_STRETCH", 4)
    monkeypatch.setattr(gguf_stage_module, "TRIAL_GLANCE", 4)
    return entry


# The first token is 301, which makes the end of the text match its start.
QUOTING_PROMPT = [5, 6, 301, 7, 8, 9, 4, 5, 6]


def test_lookup_runs_a_pipeline_on_a_route_with_no_draft(lookup_only, monkeypatch) -> None:
    ring = Ring()
    on_ring(monkeypatch, ring)
    body = GenerateRequest(session_id="s", epoch=1, input_ids=QUOTING_PROMPT, max_tokens=4, reset=True)
    tokens = asyncio.run(stream_of(body))
    assert tokens[0] == 301 and len(tokens) == 4
    assert ring.sent[:2] == [(301, 9, 9), (7, 10, None)], "a guess from the text goes out behind the model's own token"
    assert worker_module.runtime.pipeline_lookups > 0


def test_a_guess_from_the_text_never_rides_in_a_pass_of_several_rows(lookup_only) -> None:
    # Rows checked together round differently from a row alone, and that would change a reply.
    body = GenerateRequest(session_id="s", epoch=1, input_ids=QUOTING_PROMPT, max_tokens=4, reset=True)
    assert asyncio.run(stream_of(body)) == [301, 302, 303, 304]
    assert [item.input_ids for item in lookup_only[1:]] == [[301], [302], [303]]
    assert not any(item.verify for item in lookup_only)


@pytest.mark.parametrize("route, lookup", [([1], True), ([1, 2], False)])
def test_a_reply_with_nothing_to_guess_with_runs_as_it_always_did(lookup_only, monkeypatch, route, lookup) -> None:
    monkeypatch.setattr(worker_module.runtime, "route", route)
    monkeypatch.setattr(worker_module.runtime, "prompt_lookup", lookup)
    body = GenerateRequest(session_id="s", epoch=1, input_ids=QUOTING_PROMPT, max_tokens=2, reset=True)
    assert asyncio.run(stream_of(body)) == [301, 302]
    assert lookup_only[1].rollback_to is None, "this is the plain path"


def test_the_first_pass_off_a_pipeline_does_not_count_against_the_way_it_went_to(lookup_only, monkeypatch) -> None:
    ring = Ring()
    on_ring(monkeypatch, ring)
    notes: list[tuple[str, int]] = []

    class Noted(worker_module.DraftChoice):
        def note(self, way: str, tokens: int, seconds: float) -> None:
            notes.append((way, tokens))
            super().note(way, tokens, seconds)

    choice = Noted()
    choice._start_trial("pipelined", ["pipelined", "plain"])
    monkeypatch.setattr(worker_module.runtime, "draft_choice", choice)
    body = GenerateRequest(session_id="s", epoch=1, input_ids=QUOTING_PROMPT, max_tokens=7, reset=True)
    assert len(asyncio.run(stream_of(body))) == 7
    assert notes == [("pipelined", 4), ("plain", 1)], "the pass that landed the pipeline went unmeasured"


def test_the_choice_can_leave_a_lookup_pipeline_and_come_back(lookup_only, monkeypatch) -> None:
    ring = Ring()
    on_ring(monkeypatch, ring)
    choice = worker_module.runtime.draft_choice
    choice.note("plain", 100, 1.0)
    choice.note("pipelined", 10, 1.0)
    choice.trial, choice.since_trial = None, 0  # a trial of the pipeline has just been lost
    body = GenerateRequest(session_id="s", epoch=1, input_ids=QUOTING_PROMPT, max_tokens=4, reset=True)
    assert len(asyncio.run(stream_of(body))) == 4
    assert ring.sent == [], "plain was faster, so nothing guessed"


BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
PIPELINE_TRIALS = 8
needs_real_route = pytest.mark.skipif(not (BUILD and MODEL and find_library()), reason="needs a real model and the built engine")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RealRoute:
    """A real control plane and CPU workers, two unless asked, each with its own stage process."""

    def __init__(self, folder: Path, workers: int = 2) -> None:
        models = folder / "models"
        models.mkdir()
        self.model, self.draft = models / "model.gguf", models / "draft.gguf"
        for link in (self.model, self.draft):
            shutil.copyfile(MODEL, link)  # a draft must go by another name, or it would only guess at itself
        self.folder = folder
        self.processes: list[subprocess.Popen] = []
        self.client = httpx.Client(timeout=300, trust_env=False)
        self.control, self.ports = free_port(), [free_port() for _ in range(workers)]
        self.env = {
            **{key: value for key, value in os.environ.items() if not key.startswith("ISHARDED_") or key == "ISHARDED_ENGINE_LIB"},
            "ISHARDED_LLAMA_DIR": BUILD, "ISHARDED_POOL_TOKEN_FILE": str(folder / "no-token"), "ISHARDED_HOST": "127.0.0.1",
            "CUDA_VISIBLE_DEVICES": "-1", "NO_PROXY": "*",
        }  # fmt: skip

    def launch(self, *arguments: str, url: str) -> None:
        self.processes.append(subprocess.Popen([sys.executable, "-m", *arguments], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.wait_for(lambda: self.client.get(url + "/health", timeout=2).status_code == 200, 120)

    def wait_for(self, check, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                if check():
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise AssertionError("that never happened")

    def start(self, exact: bool = False) -> None:
        url = f"http://127.0.0.1:{self.control}"
        files = [f"--{name}-file={self.folder / name}" for name in ("keys", "state", "audit")]
        self.launch("isharded.orchestrator", "--host", "127.0.0.1", "--port", str(self.control), *files, url=url)
        for index, port in enumerate(self.ports):
            own = f"http://127.0.0.1:{port}"
            self.launch(
                "isharded.worker", "--host", "127.0.0.1", "--port", str(port), "--worker-id", f"w{index}", "--orchestrator-url", url,
                "--advertise-url", own, "--device", "cpu", "--gguf-dir", str(self.model.parent), "--cache-dir", str(self.folder / f"cache{index}"),
                "--llama-dir", BUILD, "--state-disk-gb", "0", "--state-ram-gb", "0", "--weights-ram-gb", "4", url=own,
            )  # fmt: skip
        self.wait_for(lambda: len(self.status()["workers"]) == len(self.ports), 30)
        body = {"model_id": str(self.model), "wait": False, "placement": "spread", "context_tokens": 512, "parallel": 1, "draft_model": str(self.draft), "prompt_cache": False, "exact": exact}
        assert self.client.post(url + "/admin/load", json=body).status_code < 300
        self.wait_for(self.loaded, 240)

    def loaded(self) -> bool:
        status = self.status()
        refused = (status.get("load_job") or {}).get("error")
        if refused and "the pool offers" in refused:
            pytest.skip(f"this machine has too little memory free: {refused}")
        assert not refused, refused
        return status["status"] == "ready" and (len(self.ports) == 1 or bool(self.entry()["ring_live"]))

    def status(self) -> dict:
        return self.client.get(f"http://127.0.0.1:{self.control}/admin/status").json()

    def entry_port(self) -> int:
        return self.ports[int(self.status()["route"][0]["worker_id"][1:])]

    def entry(self) -> dict:
        return self.client.get(f"http://127.0.0.1:{self.entry_port()}/health").json()["telemetry"]

    def ask(self, prompt: str, tokens: int = 32, **sampling: float) -> str:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "temperature": 0, "enable_thinking": False, **sampling}
        answer = self.client.post(f"http://127.0.0.1:{self.control}/v1/chat/completions", json=body)
        answer.raise_for_status()
        return answer.json()["choices"][0]["message"]["content"]

    def kill_entry_stage(self) -> None:
        worker = self.processes[1 + self.ports.index(self.entry_port())]
        psutil.Process(worker.pid).children()[0].kill()

    def stop(self) -> None:
        for process in self.processes:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
            process.kill()
            process.wait()
        self.client.close()


@needs_real_route
def test_a_reply_goes_back_to_the_pipeline_after_the_entry_stage_restarts(tmp_path) -> None:
    """Before this, the ring stayed down for good, and replies kept to the guessed way."""
    if psutil.virtual_memory().available < 4 * 1024**3:
        pytest.skip("this machine has too little memory free for two stages and a draft")
    route = RealRoute(tmp_path)
    try:
        route.start()
        prompt = "List five facts about the moon."
        before = route.ask(prompt)
        made = route.entry()["pipeline"]["tokens"]
        assert made, "the reply never used the pipeline, so this would prove nothing"
        route.kill_entry_stage()
        assert route.ask(prompt)  # the reply survives, on whatever way is open
        assert route.entry()["stage_restarts"] == 1
        route.wait_for(lambda: route.entry()["ring_live"], 90)
        # The entry keeps to whichever way has been faster, and plain decoding can win on a
        # busy machine. It gives the pipeline a trial now and then, as long as the ring is up.
        for _ in range(PIPELINE_TRIALS):
            assert route.ask(prompt) == before
            if route.entry()["pipeline"]["tokens"] > made:
                break
        assert route.entry()["pipeline"]["tokens"] > made, route.entry()["draft_ways"]
    finally:
        route.stop()


IDENTICAL_RUNS = 5


@needs_real_route
@pytest.mark.parametrize("workers, exact", [(1, False), (1, True), (2, False)], ids=["one stage", "one stage, probed", "two stages"])
def test_one_request_gives_one_text_with_the_draft_on(tmp_path, workers, exact) -> None:
    """Five greedy runs at one stage on an A40 gave two texts: the way was chosen by timing, and
    a checked pass of several rows rounds differently. Probed, a stage whose checked passes
    round as one row may use them, and the text still may not change."""
    if psutil.virtual_memory().available < (2 + 2 * workers) * 1024**3:
        pytest.skip("this machine has too little memory free for the stages and a draft")
    route = RealRoute(tmp_path, workers)
    try:
        route.start(exact=exact)
        prompts = ["List five facts about the moon.", "Write a short paragraph about how rivers shape the land."]
        for prompt in prompts:
            for sampling in ({"temperature": 0}, {"temperature": 0.8, "seed": 11}):
                texts = {route.ask(prompt, 48, **sampling) for _ in range(IDENTICAL_RUNS)}
                assert len(texts) == 1, (prompt, sampling, texts, route.entry()["draft_trials"])
        trials = route.entry()["draft_trials"]
        if workers > 1:
            assert route.entry()["pipeline"]["tokens"], "the draft never ran, so this would prove little"
            assert trials["won"] + trials["lost"] >= 2, trials
        elif not exact:
            assert route.entry()["draft_passes"] == 0, "nothing proved a checked pass exact, so none ran"
        status = route.status()
        assert status["fingerprint"]["split"] == "/".join(f"{stage['start_layer']}-{stage['end_layer']}" for stage in status["route"])
    finally:
        route.stop()
