"""Choosing where a node gets its weights from, by what each source actually delivers.

The model host and other nodes can supply the same bytes, and which is fastest depends
on links nothing can know in advance. A host usually beats a neighbour's home uplink, a
neighbour on a LAN beats any host, and on a closed network there is no host at all. So
nothing is assumed: every transfer reports its speed, each unit goes to the source that
should finish it soonest, and a transfer that falls behind is given up for a faster one.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from types import TracebackType

MODEL_HOST = "model host"
ASSUMED_BYTES_PER_SECOND = 40e6  # for a source nobody has tried yet
HOST_ADVANTAGE = 1.25  # a node's uplink may be carrying replies, and the host's is not
SWITCH_MARGIN = 1.5  # starting over throws away what arrived, so it has to clearly pay
SETTLE_SECONDS = 5.0  # a transfer's first moments are connecting and ramping up
CHECK_SECONDS = 1.0
SMOOTHING = 0.5
WORTH_MEASURING_BYTES = 8 * 1024**2  # a smaller transfer mostly measures the round trip


class SlowSource(Exception):
    """Raised inside a transfer to give it up, because another source will finish sooner."""


class SourcePicker:
    """What each source has delivered during one job. Shared by that job's download threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.speeds: dict[str, float] = {}
        self.busy: Counter[str] = Counter()
        self.failed: set[str] = set()
        self.lock = threading.Lock()

    def _share(self, source: str, untried: float) -> float:
        """What one more transfer from this source can expect, given what it already carries."""
        if source in self.failed:
            return 0.0
        speed = self.speeds.get(source)
        if speed is None:
            speed = ASSUMED_BYTES_PER_SECOND if self.busy[source] else untried
        share = speed / (1 + self.busy[source])
        return share * HOST_ADVANTAGE if source == MODEL_HOST else share

    def pick(self, sources: list[str]) -> str:
        """The source for the next transfer. An untried one goes first, to find out what it can do."""
        with self.lock:
            return max(sources, key=lambda source: self._share(source, math.inf))

    def note(self, source: str, rate: float) -> None:
        with self.lock:
            known = self.speeds.get(source)
            self.speeds[source] = rate if known is None else SMOOTHING * rate + (1 - SMOOTHING) * known
            self.failed.discard(source)

    def fail(self, source: str) -> None:
        """It stays a last resort, since a slow or flaky source beats having none."""
        with self.lock:
            self.failed.add(source)

    def beaten(self, others: list[str], left: int, nbytes: int, rate: float) -> bool:
        """Would starting this unit over elsewhere clearly beat finishing it here?"""
        with self.lock:
            best = max((self._share(source, ASSUMED_BYTES_PER_SECOND) for source in others), default=0.0)
        return best > 0 and left / rate > SWITCH_MARGIN * nbytes / best

    def transfer(
        self, source: str, others: list[str], nbytes: int, forward: Callable[[int], None], *, patient: bool = False
    ) -> Transfer:
        return Transfer(self, source, others, nbytes, forward, patient)


class Transfer:
    """One unit on its way from one source. It is passed as the byte counter, so it sees every chunk.

    A patient transfer is never given up. That is how a source gets a second visit
    after a faster-looking one disappointed, and why a unit cannot bounce for ever.
    """

    def __init__(
        self,
        picker: SourcePicker,
        source: str,
        others: list[str],
        nbytes: int,
        forward: Callable[[int], None],
        patient: bool,
    ) -> None:
        self.picker = picker
        self.source = source
        self.others = others
        self.nbytes = nbytes
        self.forward = forward
        self.patient = patient
        self.arrived = 0
        self.rate = 0.0
        self.counted = 0
        self.started = self.checked = picker.clock()

    def __enter__(self) -> Transfer:
        with self.picker.lock:
            self.picker.busy[self.source] += 1
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        with self.picker.lock:
            self.picker.busy[self.source] -= 1
        elapsed = self.picker.clock() - self.started
        if kind is None and self.arrived >= WORTH_MEASURING_BYTES and elapsed > 0:
            self.picker.note(self.source, self.arrived / elapsed)

    def __call__(self, count: int) -> None:
        self.forward(count)
        self.arrived += count
        now = self.picker.clock()
        if count <= 0 or now - self.checked < CHECK_SECONDS:
            return
        recent = (self.arrived - self.counted) / (now - self.checked)
        self.rate = SMOOTHING * recent + (1 - SMOOTHING) * self.rate if self.rate else recent
        self.counted, self.checked = self.arrived, now
        if now - self.started < SETTLE_SECONDS:
            return
        self.picker.note(self.source, self.rate)
        if not self.patient and self.picker.beaten(self.others, self.nbytes - self.arrived, self.nbytes, self.rate):
            raise SlowSource(f"{self.source} is delivering {self.rate / 1e6:.1f} MB/s")
