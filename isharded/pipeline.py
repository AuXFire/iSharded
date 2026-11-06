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

Stages keep no state for this. A stage runs what reaches it in order, and an answer nobody is
waiting for is dropped at the entry, so a thrown-away pass only ever uses a stage that had
nothing better to do.
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
# How far the share of guesses that land moves with each guess, and where it starts.
RIGHT_WEIGHT = 0.05
FIRST_RIGHT = 0.6
# How far the time between answers moves with each one.
STEP_WEIGHT = 0.2
# The fastest trip is forgotten slowly, because a route can get slower.
TRIP_DRIFT = 1.02


@dataclass(slots=True)
class Flight:
    token: int
    position: int
    answer: asyncio.Future[Any]
    sent: float
    # Nothing was ahead of it, so its trip says how long the route itself takes.
    alone: bool


class Pipeline:
    """One reply's passes on a route. ``history`` is the reply so far, and grows here."""

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
        self.flights: deque[Flight] = deque()
        self.right = FIRST_RIGHT
        self.trip: float | None = None
        self.step: float | None = None
        self.answered: float | None = None
        # Where the draft's text went wrong, until it has been told.
        self.wrong_from: int | None = None
        # The draft expects the reply to end here, so there is nothing to send past it.
        self.ending = False
        self.tokens_made = 0
        self.misses = 0

    def depth(self) -> int:
        """How many passes to keep out: enough to fill the route, and no more.

        A pass beyond that waits behind the others, and a wrong guess then has to wait behind
        passes that no longer count. Other replies on the route fill it too, so a busy route
        gets no guesses at all.
        """
        filled = self.stages + 1
        if self.trip is not None and self.step:
            filled = math.ceil(self.trip / self.step) + 1
        worth = math.ceil(math.log(WORTH_SENDING) / math.log(min(max(self.right, 0.05), 0.99)))
        return max(1, -(-min(filled, worth, self.limit) // max(1, self.sharing())))

    def _launch(self, token: int, position: int, rollback_to: int | None = None) -> None:
        answer = self.send(token, position, rollback_to)
        self.flights.append(Flight(token, position, answer, time.perf_counter(), not self.flights))

    def _text(self) -> list[int]:
        """The reply as it stands if every guess out there lands."""
        known = len(self.history)
        return self.history + [flight.token for flight in self.flights if flight.position >= known]

    async def _fill(self) -> None:
        while not self.ending and len(self.flights) < min(self.depth(), self.room) and not self.flights[0].answer.done():
            if self.wrong_from is not None:
                await self.rewind(self.wrong_from)
                self.wrong_from = None
            guess = await self.guess(self._text())
            if guess in self.stops:
                self.ending = True
                return
            self._launch(guess, self.flights[-1].position + 1)

    def _timed(self, flight: Flight) -> None:
        now = time.perf_counter()
        if flight.alone:
            took = now - flight.sent
            self.trip = took if self.trip is None else min(took, self.trip * TRIP_DRIFT)
        elif self.answered is not None:
            gap = now - self.answered
            self.step = gap if self.step is None else self.step + (gap - self.step) * STEP_WEIGHT
        self.answered = now

    def _follow(self, token: int, position: int) -> None:
        """The model said ``token`` goes at ``position``. Keeps what is out, or starts again."""
        if self.flights:
            landed = self.flights[0].token == token
            self.right += (float(landed) - self.right) * RIGHT_WEIGHT
            if landed:
                return
            self.misses += 1
            self.flights.clear()
            self.wrong_from = position if self.wrong_from is None else min(self.wrong_from, position)
            self.ending = False
            self._launch(token, position, position)
            return
        self.ending = False
        self._launch(token, position)

    async def tokens(self, count: int) -> AsyncIterator[int]:
        """The next ``count`` tokens, or fewer if the reply ends."""
        if not self.flights:
            newest = len(self.history) - 1
            self._launch(self.history[newest], newest, newest)
        for _ in range(count):
            await self._fill()
            flight = self.flights.popleft()
            token = await self.wait(flight.answer)
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
        self.flights.clear()
        known = len(self.history)
        await self.rewind(known if self.wrong_from is None else min(self.wrong_from, known))
        self.wrong_from = None
