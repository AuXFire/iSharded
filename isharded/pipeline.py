"""Keeps every stage of a route working on one reply.

A token has to cross every stage in turn, so a reply on its own leaves all but one stage idle.
This treats a route the way a processor treats its pipeline. A pass goes out before the one
ahead of it has come back, carrying the draft's guess at what that pass will answer. The last
stage samples exactly as it does for a plain pass. If its answer is the guess, the pass behind
was already right and already on its way. If not, whatever followed is thrown away and the reply
goes on from the model's own token.

So a guess can only save time. Each pass is still one row at its own position, which is what
plain decoding runs, so every token is the one plain decoding gives, seed and all. One row also
matters on a mixture of experts, where a pass of 4 rows costs about 3.4 times a pass of one.

Stages keep nothing for this but the snapshots recurrent state needs to rewind, and only when a
pass asks. A stage runs what reaches it in order, and a rewind clears
whatever is queued past it, so a thrown-away pass only ever uses a stage that had nothing better
to do, and the pass that replaces it does not wait behind it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from dataclasses import dataclass
import math
import time
from typing import Any

# A guess this unlikely to count is not worth sending, however idle the stages are.
WORTH_SENDING = 0.03
WORTH_LOG = math.log(WORTH_SENDING)
# How far the share of guesses that land moves with each guess, and where it starts.
RIGHT_WEIGHT = 0.05
FIRST_RIGHT = 0.6
# How far the time between answers moves with each one.
STEP_WEIGHT = 0.2
# The fastest trip is forgotten slowly, because a route can get slower.
TRIP_DRIFT = 1.02
class RowClock:
    """How long one row takes on a route, measured like a pipeline's step, between answers.

    The clock is shared, because a reply on a busy route hears only its own answers, and the gap
    between those says how many replies there are as much as how fast the route is. The shortest
    gap lately is the one to keep: a route with room to spare answers less often than it could.
    """

    def __init__(self) -> None:
        self.row: float | None = None
        self.answered: float | None = None

    def tick(self, now: float) -> None:
        if self.answered is not None:
            gap = now - self.answered
            self.row = gap if self.row is None else min(gap, self.row * TRIP_DRIFT)
        self.answered = now


@dataclass(slots=True)
class Flight:
    token: int
    position: int
    answer: asyncio.Future[Any]
    sent: float
    # Nothing was ahead of it, so its trip says how long the route itself takes.
    alone: bool


class Pipeline:
    """One reply's passes on a route. ``history`` is the reply so far, and grows here.

    ``guess`` hears only the tokens the draft has not seen, and ``told`` is how much it had
    seen at the start, so a guess costs the same however long the reply is.
    """

    def __init__(
        self,
        history: list[int],
        room: int,
        stops: Collection[int],
        stages: int,
        limit: int,
        send: Callable[[int, int, int | None], asyncio.Future[Any]],
        wait: Callable[[asyncio.Future[Any]], Awaitable[int]],
        guess: Callable[[list[int]], Awaitable[int]],
        rewind: Callable[[int], Awaitable[None]],
        sharing: Callable[[], int] = lambda: 1,
        told: int = 0,
        clock: RowClock | None = None,
    ) -> None:
        self.history = history
        self.room = room
        self.stops = stops
        self.stages = stages
        self.limit = limit
        self.send = send
        self.wait = wait
        self.guess = guess
        self.rewind = rewind
        self.sharing = sharing
        self.told = told
        self.clock = clock or RowClock()
        self.flights: deque[Flight] = deque()
        self.right = FIRST_RIGHT
        self.trip: float | None = None
        self.step: float | None = None
        self.answered: float | None = None
        # Where the draft's text went wrong, until it has been told.
        self.wrong_from: int | None = None
        # The guess being made. It runs beside everything else, because a wrong guess must
        # be answered the moment it shows, not when the draft next looks up.
        self.guessing: asyncio.Task[int] | None = None
        # What that guess was built on, which may be gone by the time it lands.
        self.guessing_for = (-1, -1)
        # The draft expects the reply to end here, so there is nothing to send past it.
        self.ending = False
        self.tokens_made = 0
        self.misses = 0

    def _priced(self) -> int:
        """The most passes this reply may keep out, now that other replies want the route too.

        N replies with D passes out, each out for a round trip R, ask the route for N x D / R
        rows a second, and it passes 1 / T. So D stops at R / (N x T). Alone, a reply only
        competes with itself, and filling the route is already the limit.
        """
        replies, row = self.sharing(), self.clock.row
        if replies <= 1 or not self.trip or not row:
            return self.limit
        return max(1, int(self.trip / (replies * row)))

    def depth(self) -> int:
        """How many passes to keep out: enough to fill the route, and no more.

        A pass beyond that waits behind the others, and a wrong guess then has to wait behind
        passes that no longer count.
        """
        filled = self.stages + 1
        if self.trip is not None and self.step:
            filled = math.ceil(self.trip / self.step) + 1
        worth = math.ceil(WORTH_LOG / math.log(min(max(self.right, 0.05), 0.99)))
        return max(1, min(filled, worth, self.limit, self._priced()))

    def _launch(self, token: int, position: int, rollback_to: int | None = None) -> None:
        answer = self.send(token, position, rollback_to)
        self.flights.append(Flight(token, position, answer, time.perf_counter(), not self.flights))

    def _fresh(self) -> list[int]:
        """What the draft has not seen of the reply as it stands if every guess out there lands."""
        if self.wrong_from is not None:
            self.told = min(self.told, self.wrong_from)
        told, self.told = self.told, self.flights[-1].position + 1
        known = len(self.history)
        return self.history[told:] + [flight.token for flight in self.flights if flight.position >= max(known, told)]

    async def _next_guess(self, fresh: list[int]) -> int:
        if self.wrong_from is not None:
            keep, self.wrong_from = self.wrong_from, None
            await self.rewind(keep)
        return await self.guess(fresh)

    def _guess_on(self) -> None:
        """Starts the next guess, if the route has room for one."""
        if self.guessing is None and not self.ending and len(self.flights) < min(self.depth(), self.room):
            self.guessing_for = (self.misses, self.flights[-1].position)
            self.guessing = asyncio.ensure_future(self._next_guess(self._fresh()))

    def _take_guess(self) -> None:
        assert self.guessing is not None
        guess = self.guessing.result()
        self.guessing = None
        if self.guessing_for != (self.misses, self.flights[-1].position):
            return  # the reply moved on while it was being made
        if guess in self.stops:
            self.ending = True
        else:
            self._launch(guess, self.flights[-1].position + 1)

    async def _answer(self, flight: Flight) -> int:
        """Keeps guessing until the oldest pass is back, or ``wait`` gives up on it."""
        answer = asyncio.ensure_future(self.wait(flight.answer))
        try:
            while not answer.done():
                self._guess_on()
                await asyncio.wait({answer} if self.guessing is None else {answer, self.guessing}, return_when=asyncio.FIRST_COMPLETED)
                if self.guessing is not None and self.guessing.done():
                    self._take_guess()
            return answer.result()
        finally:
            answer.cancel()

    def _timed(self, flight: Flight) -> None:
        now = time.perf_counter()
        self.clock.tick(now)
        if flight.alone:
            took = now - flight.sent
            self.trip = took if self.trip is None else min(took, self.trip * TRIP_DRIFT)
        elif self.answered is not None:
            gap = now - self.answered
            self.step = gap if self.step is None else self.step + (gap - self.step) * STEP_WEIGHT
        self.answered = now

    def _abandon(self) -> None:
        for flight in self.flights:
            flight.answer.cancel()
        self.flights.clear()

    def _follow(self, token: int, position: int) -> None:
        """The model said ``token`` goes at ``position``. Keeps what is out, or starts again."""
        rollback_to = None
        if self.flights:
            landed = self.flights[0].token == token
            self.right += (float(landed) - self.right) * RIGHT_WEIGHT
            if landed:
                return
            self.misses += 1
            self._abandon()
            self.wrong_from = position if self.wrong_from is None else min(self.wrong_from, position)
            rollback_to = position
        self.ending = False
        self._launch(token, position, rollback_to)

    async def tokens(self, count: int) -> AsyncIterator[int]:
        """The next ``count`` tokens, or fewer if the reply ends."""
        if not self.flights:
            newest = len(self.history) - 1
            self._launch(self.history[newest], newest, newest)
        for _ in range(count):
            flight = self.flights[0]
            token = await self._answer(flight)
            self.flights.popleft()
            self._timed(flight)
            self.history.append(token)
            self.room -= 1
            self.tokens_made += 1
            finished = token in self.stops or self.room <= 0
            if not finished:
                # Before the token goes to the caller, so the route never waits for a reader.
                self._follow(token, flight.position + 1)
            yield token
            if finished:
                return

    async def land(self) -> None:
        """Lets go of what is out, so a reply can go on from its newest token some other way."""
        self._abandon()
        await self.close()
        known = len(self.history)
        await self.rewind(known if self.wrong_from is None else min(self.wrong_from, known))
        self.wrong_from = None

    async def close(self) -> None:
        """Waits for the guess being made, since the draft's text is not safe to touch until then."""
        guessing, self.guessing = self.guessing, None
        if guessing is not None:
            await asyncio.gather(guessing, return_exceptions=True)
