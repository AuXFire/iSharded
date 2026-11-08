from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import sys
import time
import traceback
from typing import Any

import uvicorn
from uvicorn.protocols.http.auto import AutoHTTPProtocol

from .tls import CONNECTION, NODE_KEEP_ALIVE_SECONDS, PEER, peer_name, probe_options, RETIRED_KEYS, key_of_certificate, server_options

# Over the five seconds clients hold an idle connection, so that a client always closes
# first, and never sends on a connection the server is dropping that same moment.
KEEP_ALIVE_SECONDS = 15
# The same for a connection that proved it belongs to a node. Anyone else gets the short
# time above, because every open connection counts against ``limit_concurrency``.
NODE_KEEP_ALIVE_MARGIN_SECONDS = 60
# Every peer and worker holds a connection or two. Past this a node answers 503.
MAX_CONNECTIONS = 2048
# A client still sending a request must get this much across in this long. Heads and
# small bodies are done well inside it, and a stage's largest body moves far faster.
STALL_SECONDS = 30.0
STALL_BYTES = 1024


def set_nodelay(transport: asyncio.BaseTransport) -> None:
    sock = transport.get_extra_info("socket")
    if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def set_probes(transport: asyncio.BaseTransport) -> None:
    """Without these, a peer that vanished holds its place here until keep-alive runs out."""
    sock = transport.get_extra_info("socket")
    if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
        for option in probe_options():
            sock.setsockopt(*option)


class NoDelayHTTPProtocol(AutoHTTPProtocol):  # type: ignore[misc, valid-type]
    """Send each reply at once, and let go of clients that stop sending.

    Windows' Proactor event loop leaves TCP_NODELAY off on accepted sockets, and
    uvicorn writes headers and body separately. Nagle then holds the body until
    the delayed ACK for the headers arrives: measured at 60 ms on every hop,
    over Tailscale and the LAN alike.

    Uvicorn's keep-alive timeout starts when a reply ends, so a connection that never
    finishes a request has no timeout at all. Each one still counts against
    ``limit_concurrency``, so a stranger could hold a node at its limit with connections
    that say nothing. Nothing here looks at a request once it has arrived whole, so a
    reply can take as long as it likes.

    A connection is held open for long only once a route says a node is on the other end,
    so a stranger's stays short and cannot sit on a place in that limit.
    """

    def connection_made(self, transport: Any) -> None:
        set_nodelay(transport)
        set_probes(transport)
        super().connection_made(transport)
        self.heard_bytes = 0
        self.proven = False
        self.stall_watch = self.loop.call_later(STALL_SECONDS, self._check_stalled)
        peer = peer_name(transport.get_extra_info("peercert"))
        held = transport.get_extra_info("ssl_object") if peer else None
        # Worked out once for the connection, and looked up on every request, as a connection
        # can outlast its key's retirement.
        key = key_of_certificate(held.getpeercert(binary_form=True)) if held is not None else None
        inner = self.app

        async def with_identity(scope: Any, receive: Any, send: Any) -> None:
            CONNECTION.set(self)
            if peer and key not in RETIRED_KEYS:
                PEER.set(peer)
            await inner(scope, receive, send)

        self.app = with_identity

    def data_received(self, data: bytes) -> None:
        self.heard_bytes += len(data)
        super().data_received(data)

    def connection_lost(self, exc: Exception | None) -> None:
        self.stall_watch.cancel()
        super().connection_lost(exc)

    def on_response_complete(self) -> None:
        if self.proven:
            self.timeout_keep_alive = NODE_KEEP_ALIVE_SECONDS + NODE_KEEP_ALIVE_MARGIN_SECONDS
        super().on_response_complete()
        if not self.transport.is_closing():
            self._watch_again()  # the next request gets its full time, and keep-alive comes first

    def _watch_again(self) -> None:
        self.stall_watch.cancel()
        self.heard_bytes = 0
        self.stall_watch = self.loop.call_later(STALL_SECONDS, self._check_stalled)

    def _waiting_on_client(self) -> bool:
        cycle = self.cycle
        if cycle is None:
            return True
        if cycle.response_complete:
            return self.heard_bytes > 0  # with nothing heard it is idle, and that is keep-alive's to end
        # Reading paused means this side is behind, and the client is not to blame.
        return cycle.more_body and not self.flow.read_paused

    def _between_requests(self) -> bool:
        """A finished reply and nothing since is keep-alive's to end, however long it lasts."""
        return self.cycle is not None and self.cycle.response_complete and not self.heard_bytes

    def _check_stalled(self) -> None:
        if self.transport.is_closing():
            return
        if self._waiting_on_client() and self.heard_bytes < STALL_BYTES and not self._between_requests():
            self.transport.close()
            return
        self._watch_again()


ACCEPT_RETRY_SECONDS = 0.05


def steady_accept(accept: Any) -> Any:
    """IocpProactor.accept, except that one failed accept stays one failed accept.

    If a client drops as it connects, Windows fails the accept with WinError 64, and
    asyncio's Proactor loop answers by closing the listening socket for good. The
    process lives on and its outgoing gossip still works, so nothing restarts it and
    nothing notices. A live node sat deaf like that, with its port closed, until it
    was found by hand. So a failure here is retried while the socket is still open.
    """

    def steady(self: Any, listener: Any) -> Any:
        answer = self._loop.create_future()

        def attempt() -> None:
            if answer.done():
                return
            try:
                inner = accept(self, listener)
            except OSError as exc:
                answer.set_exception(exc)
                return
            answer.add_done_callback(lambda _: inner.cancel() if answer.cancelled() else None)
            inner.add_done_callback(settle)

        def settle(inner: Any) -> None:
            if answer.done():
                return
            if inner.cancelled():
                answer.cancel()
            elif inner.exception() is None:
                answer.set_result(inner.result())
            elif isinstance(inner.exception(), OSError) and listener.fileno() != -1:
                self._loop.call_later(ACCEPT_RETRY_SECONDS, attempt)
            else:
                answer.set_exception(inner.exception())

        attempt()
        return answer

    return steady


def keep_listening() -> None:
    if sys.platform != "win32":
        return  # the selector loops elsewhere do not close a socket over one failed accept
    from asyncio import windows_events

    proactor = windows_events.IocpProactor
    if not getattr(proactor, "keeps_listening", False):
        proactor.accept = steady_accept(proactor.accept)
        proactor.keeps_listening = True


# Generous for any prompt, and far short of what could exhaust a node.
MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Hidden states for a long prompt on a wide model, in FP32.
MAX_STAGE_BYTES = 1024 * 1024 * 1024
# What anyone may send with no credentials. A certificate or a signing request takes a
# kilobyte or two, however many addresses it lists.
MAX_STRANGER_BYTES = 16 * 1024
# A saved state goes straight to disk as it arrives, so this bounds disk and not memory.
MAX_STATE_BYTES = 64 * 1024 * 1024 * 1024


class BodyLimit:
    """Refuse an oversized request from its headers, before reading the body.

    Plain ASGI, because this sits on the path of every token between stages.
    Bodies are parsed before a route can check who sent them, so this also
    bounds what a stranger can make a node hold in memory. ``paths`` gives single
    routes their own limit, higher or lower.
    """

    def __init__(self, app: Any, limit: int = MAX_REQUEST_BYTES, paths: dict[str, int] | None = None) -> None:
        self.app = app
        self.limit = limit
        self.paths = paths or {}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            length = headers.get(b"content-length")
            limit = self.paths.get(scope["path"], self.limit)
            if b"transfer-encoding" in headers:
                await self._refuse(send, 411, "send a Content-Length")
                return
            if length is not None and (not length.isdigit() or int(length) > limit):
                await self._refuse(send, 413, f"requests here are limited to {limit} bytes")
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(send: Any, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"connection", b"close")]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


MOST_ADDRESSES = 4096


def client_address(host: str | None) -> str:
    """One client, as nearly as a limit can tell. One IPv6 customer holds a whole /64."""
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return host or "unknown"
    plain = getattr(address, "ipv4_mapped", None) or address
    return str(plain) if plain.version == 4 else str(ipaddress.ip_network(f"{plain}/64", strict=False))


class AddressLimit:
    """How much one address may ask of a route anyone can reach."""

    def __init__(self, burst: int, per_minute: float, clock: Any = time.monotonic) -> None:
        self.burst = float(burst)
        self.rate = per_minute / 60.0
        self.clock = clock
        self.held: dict[str, tuple[float, float]] = {}

    def _allowance(self, address: str, now: float) -> float:
        held = self.held.get(address)
        return self.burst if held is None else min(self.burst, held[0] + (now - held[1]) * self.rate)

    def wait(self, address: str) -> float:
        """Seconds until this address may ask again. Nothing is spent by asking this."""
        allowance = self._allowance(address, self.clock())
        return 0.0 if allowance >= 1.0 else (1.0 - allowance) / self.rate

    def charge(self, address: str) -> None:
        now = self.clock()
        allowance = self._allowance(address, now)
        if address not in self.held and len(self.held) >= MOST_ADDRESSES:
            # An address that has refilled is the same as one never seen.
            self.held = {item: value for item, value in self.held.items() if self._allowance(item, now) < self.burst}
            if len(self.held) >= MOST_ADDRESSES:
                del self.held[next(iter(self.held))]
        self.held.pop(address, None)  # to the end, so the oldest stays first
        self.held[address] = (max(allowance - 1.0, 0.0), now)


class ChunkedResponses:
    """Send one path's responses chunked, by leaving their length out.

    AVG's network filter inspects requests that look like MCP, even on loopback.
    A response to one that has a Content-Length comes out with the length removed
    and ``transfer-encoding: chunked`` added, around a body that is not chunked,
    so no client can read it. Measured with a bare app on a raw socket. Responses
    that really are chunked pass unharmed, and AVG still sees all of them.
    """

    def __init__(self, app: Any, path: str) -> None:
        self.app = app
        self.path = path

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return

        async def without_length(message: Any) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [item for item in message["headers"] if item[0].lower() != b"content-length"]}
            await send(message)

        await self.app(scope, receive, without_length)


def where_everything_is() -> dict[str, Any]:
    """Each task's and thread's innermost frames. Stalls show up here and nowhere else."""
    tasks = []
    for task in asyncio.all_tasks():
        frames = task.get_stack(limit=4)
        tasks.append(" < ".join(f"{frame.f_code.co_name}:{frame.f_lineno}" for frame in reversed(frames)))
    threads = [
        " < ".join(f"{item.name}:{item.lineno}" for item in reversed(traceback.extract_stack(frame)[-4:]))
        for frame in sys._current_frames().values()
    ]
    return {"tasks": sorted(tasks), "threads": sorted(threads)}


def serve(app: Any, host: str, port: int) -> None:
    keep_listening()
    # Redirected output defaults to the system code page on Windows, which cannot
    # write every model name or event.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    # Nodes poll each other constantly, so request logs would fill a disk.
    access_log = os.getenv("ISHARDED_ACCESS_LOG") == "1"
    connections = int(os.getenv("ISHARDED_MAX_CONNECTIONS") or MAX_CONNECTIONS)
    uvicorn.run(
        app,
        host=host,
        port=port,
        http=NoDelayHTTPProtocol,
        access_log=access_log,
        timeout_keep_alive=KEEP_ALIVE_SECONDS,
        limit_concurrency=connections,
        **server_options(),
    )
