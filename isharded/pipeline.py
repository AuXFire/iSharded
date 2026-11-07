"""Keeps every stage of a route working on one reply.

A token has to cross every stage in turn, so a reply on its own leaves all but one stage idle.
This treats a route the way a processor treats its pipeline. A pass goes out before the one
ahead of it has come back, carrying the draft's guess at what that pass will answer. The last
stage samples exactly as it does for a plain pass. If its answer is the guess, the pass behind
was already right and already on its way. If not, whatever followed is thrown away and the reply
goes on from the model's own token.

Guesses come from a draft, from the text itself, or both. Text often repeats what came before,
as code and quoted documents do, and finding that costs no model at all.

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

import numpy as np

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
# Answers closer than this left the route in one merged pass, so the gap says nothing about a row.
SAME_PASS_SECONDS = 0.002
# How much text a lookup searches. It holds up to twice this, at 4 bytes a token, and one search
# reads it once, so this sets both its memory and the most a token can cost.
LOOKUP_WINDOW = 1 << 16
# The fewest matching tokens worth a guess, and the most a search compares.
LOOKUP_SHORTEST = 3
LOOKUP_LONGEST = 8
# A match this long has been right more often than a draft, and takes no time to make.
LOOKUP_BEATS_DRAFT = 5


class PromptLookup:
    """Guesses by finding the last few tokens earlier in the text, and reading on from there.

    A search runs only when no match is being followed, so a run of good guesses costs almost
    nothing, and it searches a fixed window, so a long prompt costs no more than a short one.
    """

    def __init__(self, text: list[int]) -> None:
        recent = text[-LOOKUP_WINDOW:]
        self.text = np.zeros(2 * LOOKUP_WINDOW, dtype=np.int32)
        self.text[: len(recent)] = recent
        self.end = len(recent)
        # Where text[0] is in the reply.
        self.base = len(text) - len(recent)
        # Where the match being followed reads next, and how long it is so far.
        self.source = -1
        self.matched = 0

    def put(self, token: int, position: int) -> None:
        """The text now has ``token`` at ``position``, and nothing after it."""
        at = position - self.base
        if at < self.end:
            self.end, self.source = at, -1
        if self.end == self.text.size:
            self._slide()
        self.text[self.end] = token
        self.end += 1
        if self.source >= 0 and self.text[self.source] == token:
            self.source += 1
            self.matched += 1
        else:
            self.source = -1

    def _slide(self) -> None:
        """Forgets the oldest half. Each token moves once per window, so this costs next to nothing."""
        self.text[:LOOKUP_WINDOW] = self.text[-LOOKUP_WINDOW:]
        self.end = LOOKUP_WINDOW
        self.base += self.text.size - LOOKUP_WINDOW
        self.source = self.source - LOOKUP_WINDOW if self.source >= LOOKUP_WINDOW else -1

    def _search(self) -> None:
        """Finds the longest match for the newest tokens, and the latest among those."""
        newest = self.end - 1
        hits = np.flatnonzero(self.text[:newest] == self.text[newest])
        length = 1
        while hits.size and length < min(LOOKUP_LONGEST, newest + 1):
            hits_before = hits[hits >= length]
            longer = hits_before[self.text[hits_before - length] == self.text[newest - length]]
            if not longer.size:
                break
            hits, length = longer, length + 1
        if hits.size and length >= LOOKUP_SHORTEST:
            self.source, self.matched = int(hits[-1]) + 1, length

    def guess(self, shortest: int) -> int | None:
        """What comes next, if a match of at least ``shortest`` tokens says so."""
        if self.source < 0 and self.end > 1:
            self._search()
        return int(self.text[self.source]) if self.source >= 0 and self.matched >= shortest else None


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
        gap = None if self.answered is None else now - self.answered
        self.answered = now
        if gap is not None and gap >= SAME_PASS_SECONDS:
            self.row = gap if self.row is None else min(gap, self.row * TRIP_DRIFT)


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
    seen at the start, so a guess costs the same however long the reply is. With no draft,
    ``guess`` and ``rewind`` are None and every guess comes from ``lookup``.
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
        guess: Callable[[list[int]], Awaitable[int]] | None,
        rewind: Callable[[int], Awaitable[None]] | None,
        sharing: Callable[[], int] = lambda: 1,
        told: int = 0,
        lookup: bool = False,
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
        self.lookup = PromptLookup(history) if lookup else None
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
        self.lookups = 0

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
        if self.lookup is not None:
            self.lookup.put(token, position)
        self.flights.append(Flight(token, position, answer, time.perf_counter(), not self.flights))

    def _fresh(self) -> list[int]:
        """What the draft has not seen of the reply as it stands if every guess out there lands."""
        if self.wrong_from is not None:
            self.told = min(self.told, self.wrong_from)
        told, self.told = self.told, self.flights[-1].position + 1
        known = len(self.history)
        return self.history[told:] + [flight.token for flight in self.flights if flight.position >= max(known, told)]

    async def _next_guess(self, fresh: list[int]) -> int:
        assert self.guess is not None and self.rewind is not None
        if self.wrong_from is not None:
            keep, self.wrong_from = self.wrong_from, None
            await self.rewind(keep)
        return await self.guess(fresh)

    def _send_guess(self, guess: int) -> None:
        if guess in self.stops:
            self.ending = True
        else:
            self._launch(guess, self.flights[-1].position + 1)

    def _looked_up(self) -> int | None:
        """A guess from the text, if there is one the draft would not do better than."""
        if self.lookup is None:
            return None
        return self.lookup.guess(LOOKUP_SHORTEST if self.guess is None else LOOKUP_BEATS_DRAFT)

    def _guess_on(self) -> None:
        """Starts the next guess, while the route has room for one.

        A guess from the text is there at once, so several can go out here, and none waits for a
        slow draft. The draft is not told about them. They reach it with the rest of what it has
        not seen, next time it is asked, and a guess it was making for that place is thrown away.
        """
        while not self.ending and len(self.flights) < min(self.depth(), self.room):
            guess = self._looked_up()
            if guess is None:
                if self.guess is not None and self.guessing is None:
                    self.guessing_for = (self.misses, self.flights[-1].position)
                    self.guessing = asyncio.ensure_future(self._next_guess(self._fresh()))
                return
            self.lookups += 1
            self._send_guess(guess)

    def _take_guess(self) -> None:
        assert self.guessing is not None
        guess = self.guessing.result()
        self.guessing = None
        if self.guessing_for != (self.misses, self.flights[-1].position):
            return  # the reply moved on while it was being made
        self._send_guess(guess)

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
        if self.rewind is not None:
            await self.rewind(known if self.wrong_from is None else min(self.wrong_from, known))
        self.wrong_from = None

    async def close(self) -> None:
        """Waits for the guess being made, since the draft's text is not safe to touch until then."""
        guessing, self.guessing = self.guessing, None
        if guessing is not None:
            await asyncio.gather(guessing, return_exceptions=True)
