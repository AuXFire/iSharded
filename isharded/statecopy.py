"""One copy of each saved state on another node, so that losing a node loses no state.

Only one other node holds a copy: a stage's KV is for its own layers, and copying it
everywhere would cost more than the replay it saves. A copy leaves behind the reply
that made it, at a set rate, and nothing ever waits for one. A newer turn of the same
conversation replaces one still queued, so a slow link falls behind and never backs
up. When a copy is missing or late, recovery reads the prompt instead.

A state is a person's conversation, so only nodes in the pool may send or fetch one.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import contextlib
import logging
from pathlib import Path
import time
from typing import Any, AsyncIterator, Callable

import httpx

from .protocol import META_HEADER, SavePoint, encode_meta
from .statestore import SUFFIX

# States too short to be worth a copy: reading them again costs less than sending them.
COPY_LEAST_TOKENS = 256
QUEUED_COPIES = 8
CHUNK_BYTES = 1024 * 1024
# How long a state file may take to appear, once a stage has saved it.
FILE_WAIT_SECONDS = 20.0
# A copy goes quiet on a silent link, and this is how long that goes unnoticed.
SILENCE_SECONDS = 30.0
CONNECT_SECONDS = 5.0
# Fetching a copy delays a reply, so a holder gets little time to start or to stall.
FETCH_SILENCE_SECONDS = 5.0
FETCH_CONNECT_SECONDS = 3.0

log = logging.getLogger(__name__)


class CopySender:
    """Sends a stage's new states to the node holding its copies, one at a time."""

    def __init__(self, client: httpx.AsyncClient, headers: Callable[[], dict[str, str]], rate_bytes: int) -> None:
        self.client = client
        self.headers = headers
        self.rate_bytes = rate_bytes
        self.url: str | None = None
        self.folder: Path | None = None
        self.namespace = ""
        self.queue: OrderedDict[str, SavePoint] = OrderedDict()
        self.task: asyncio.Task[None] | None = None
        self.sent = 0
        self.sent_bytes = 0
        self.failed = 0
        self.dropped = 0

    def aim(self, url: str | None, folder: Path | None, namespace: str) -> None:
        """Send from now on to this node. What was queued for another stage is no use."""
        self.url = url.rstrip("/") if url else None
        self.folder = folder
        self.namespace = namespace
        self.dropped += len(self.queue)
        self.queue.clear()

    def add(self, saves: list[SavePoint]) -> None:
        if not self.url or not self.folder or not self.rate_bytes:
            return
        for save in saves:
            if save.length < COPY_LEAST_TOKENS:
                continue
            if self.queue.pop(save.replaces or "", None) is not None:
                self.dropped += 1
            self.queue[save.key] = save
            while len(self.queue) > QUEUED_COPIES:
                self.queue.popitem(last=False)
                self.dropped += 1
        if self.queue and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self.queue:
            _, save = self.queue.popitem(last=False)
            try:
                await self._send(save)
            except (httpx.HTTPError, OSError, TimeoutError) as exc:
                # The node may be gone, and the next repair names another. Nobody is waiting.
                self.failed += 1
                log.info("a state copy did not arrive: %s", exc)

    async def _send(self, save: SavePoint) -> None:
        assert self.folder is not None and self.url is not None
        path = self.folder / (save.key + SUFFIX)
        deadline = time.monotonic() + FILE_WAIT_SECONDS
        while not path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{path.name} was not written")
            await asyncio.sleep(0.1)
        size = path.stat().st_size
        meta = {"namespace": self.namespace, "key": save.key, "replaces": save.replaces}
        headers = {**self.headers(), "content-length": str(size), META_HEADER: encode_meta(meta)}
        response = await self.client.put(
            self.url + "/internal/state/copy",
            content=self._paced(path),
            headers=headers,
            timeout=httpx.Timeout(SILENCE_SECONDS, connect=CONNECT_SECONDS),
        )
        response.raise_for_status()
        self.sent += 1
        self.sent_bytes += size

    async def _paced(self, path: Path) -> AsyncIterator[bytes]:
        """A file, no faster than the rate, so that tokens keep the link they need."""
        started = time.monotonic()
        total = 0
        with path.open("rb") as handle:
            while chunk := await asyncio.to_thread(handle.read, CHUNK_BYTES):
                yield chunk
                total += len(chunk)
                ahead = total / self.rate_bytes - (time.monotonic() - started)
                if ahead > 0:
                    await asyncio.sleep(ahead)

    def report(self) -> dict[str, Any]:
        return {
            "copies_sent": self.sent,
            "copy_bytes_sent": self.sent_bytes,
            "copies_failed": self.failed,
            "copies_dropped": self.dropped,
            "copies_queued": len(self.queue),
        }


async def fetch_file(client: httpx.AsyncClient, url: str, target: Path, headers: dict[str, str]) -> bool:
    """Download a state to ``target``. False, with nothing left behind, if that fails."""
    try:
        timeout = httpx.Timeout(FETCH_SILENCE_SECONDS, connect=FETCH_CONNECT_SECONDS)
        async with client.stream("GET", url, headers=headers, timeout=timeout) as response:
            if response.status_code != 200:
                return False
            with target.open("wb") as handle:
                async for chunk in response.aiter_bytes(CHUNK_BYTES):
                    await asyncio.to_thread(handle.write, chunk)
        return True
    except (httpx.HTTPError, OSError):
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
        return False
