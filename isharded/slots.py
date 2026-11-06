"""Admission to a pipeline: how many replies run at once, and which sequence each gets.

Every stage holds a fixed number of independent sequences. The entry worker hands
them out here, so requests from any node queue in one place, and a pool change can
wait here for the replies in progress.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class SlotGate:
    def __init__(self, slots: int = 1) -> None:
        self.size = slots
        self.free = set(range(slots))
        self.held = False
        self.changed = asyncio.Condition()

    async def acquire(self, choose: Callable[[set[int]], int] = min) -> int:
        async with self.changed:
            await self.changed.wait_for(lambda: bool(self.free) and not self.held)
            slot = choose(self.free)
            self.free.discard(slot)
            return slot

    async def release(self, slot: int) -> None:
        async with self.changed:
            if slot < self.size:
                self.free.add(slot)
            self.changed.notify_all()

    async def hold(self, seconds: float) -> bool:
        """Stop admitting, and wait for what is running. False if it did not finish in time."""
        async with self.changed:
            self.held = True
            try:
                await asyncio.wait_for(self.changed.wait_for(lambda: len(self.free) == self.size), seconds)
            except asyncio.TimeoutError:
                self.held = False
                self.changed.notify_all()
                return False
            return True

    async def resume(self) -> None:
        async with self.changed:
            self.held = False
            self.changed.notify_all()

    def resize(self, slots: int) -> None:
        """Change the number of slots. Slots still in use return if they still exist."""
        in_use = set(range(self.size)) - self.free
        self.size = slots
        self.free = set(range(slots)) - in_use
