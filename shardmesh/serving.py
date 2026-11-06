from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import traceback
from typing import Any

import uvicorn
from uvicorn.protocols.http.auto import AutoHTTPProtocol

from .tls import PEER, peer_name, server_options


def set_nodelay(transport: asyncio.BaseTransport) -> None:
    sock = transport.get_extra_info("socket")
    if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class NoDelayHTTPProtocol(AutoHTTPProtocol):  # type: ignore[misc, valid-type]
    """Send each reply at once.

    Windows' Proactor event loop leaves TCP_NODELAY off on accepted sockets, and
    uvicorn writes headers and body separately. Nagle then holds the body until
    the delayed ACK for the headers arrives: measured at 60 ms on every hop,
    over Tailscale and the LAN alike.
    """

    def connection_made(self, transport: Any) -> None:
        set_nodelay(transport)
        super().connection_made(transport)
        peer = peer_name(transport.get_extra_info("peercert"))
        if peer:
            inner = self.app

            async def with_identity(scope: Any, receive: Any, send: Any) -> None:
                PEER.set(peer)
                await inner(scope, receive, send)

            self.app = with_identity


# Generous for any prompt, and far short of what could exhaust a node.
MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Hidden states for a long prompt on a wide model, in FP32.
MAX_STAGE_BYTES = 1024 * 1024 * 1024


class BodyLimit:
    """Refuse an oversized request from its headers, before reading the body.

    Plain ASGI, because this sits on the path of every token between stages.
    Bodies are parsed before a route can check who sent them, so this also
    bounds what a stranger can make a node hold in memory.
    """

    def __init__(self, app: Any, limit: int = MAX_REQUEST_BYTES, large: dict[str, int] | None = None) -> None:
        self.app = app
        self.limit = limit
        self.large = large or {}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            length = headers.get(b"content-length")
            limit = self.large.get(scope["path"], self.limit)
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
    # Redirected output defaults to the system code page on Windows, which cannot
    # write every model name or event.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    # Nodes poll each other constantly, so request logs would fill a disk.
    access_log = os.getenv("SHARDMESH_ACCESS_LOG") == "1"
    uvicorn.run(app, host=host, port=port, http=NoDelayHTTPProtocol, access_log=access_log, **server_options())
