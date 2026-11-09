"""A reply sent whole stops when its client goes, on every API shape.

Only a real socket can drop a connection, so the app runs under uvicorn here, with a
model that says nothing until it is told to stop.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from isharded import orchestrator
from isharded.replies import ClientGone, unless_client_leaves
from test_tools import QWEN3, mesh  # noqa: F401

SAY = [{"role": "user", "content": "hi"}]
WHOLE = [
    ("/v1/chat/completions", {"model": "scripted", "messages": SAY}),
    ("/v1/completions", {"model": "scripted", "prompt": "hi"}),
    ("/v1/messages", {"model": "scripted", "max_tokens": 16, "messages": SAY}),
    ("/v1/responses", {"model": "scripted", "input": "hi"}),
    ("/api/chat", {"model": "scripted", "messages": SAY, "stream": False}),
    ("/api/generate", {"model": "scripted", "prompt": "hi", "stream": False}),
]


@pytest.fixture
def live(mesh):  # noqa: F811
    """The real app on a real socket. Its model counts what it was asked, and waits."""
    mesh.use(QWEN3, "Hello there")
    entered, closed = threading.Event(), threading.Event()
    state = SimpleNamespace(silent=True)

    async def token_stream(session, max_tokens, *rest):
        entered.set()
        try:
            if state.silent:
                await asyncio.sleep(60)
            for token in mesh.tokens[:max_tokens]:
                yield token
        finally:
            closed.set()

    mesh.node.token_stream = token_stream
    server = uvicorn.Server(uvicorn.Config(orchestrator.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    yield SimpleNamespace(mesh=mesh, port=port, entered=entered, closed=closed, state=state)
    server.should_exit = True
    thread.join(15)


def send(live, path: str, body: dict) -> socket.socket:
    """A request written by hand, which is the only way to walk away before the reply."""
    data = json.dumps(body).encode()
    head = [f"POST {path} HTTP/1.1", "Host: 127.0.0.1", "Content-Type: application/json", f"Content-Length: {len(data)}"]
    connection = socket.create_connection(("127.0.0.1", live.port), timeout=15)
    connection.sendall("\r\n".join(head).encode() + b"\r\n\r\n" + data)
    return connection


@pytest.mark.parametrize("path, body", WHOLE)
def test_a_whole_reply_stops_when_its_client_goes(live, path, body):
    connection = send(live, path, body)
    assert live.entered.wait(15), "the model should have been asked"
    began = time.monotonic()
    connection.close()
    assert live.closed.wait(5), "the reply kept running after its client went"
    stopped = time.monotonic() - began
    deadline = time.monotonic() + 5
    while live.mesh.node.replies_in_flight and time.monotonic() < deadline:
        time.sleep(0.02)
    assert live.mesh.node.replies_in_flight == 0
    assert stopped < 2


@pytest.mark.parametrize("path, body", WHOLE[:2])
def test_a_whole_reply_from_another_lane_stops_when_its_client_goes(live, monkeypatch, path, body):
    """The call to the lane is closed, which is what makes that lane stop the reply."""
    entered, closed = threading.Event(), threading.Event()

    async def locate(model):
        return "other"

    async def forward(name, route, payload, caller):
        entered.set()
        try:
            await asyncio.sleep(60)
        finally:
            closed.set()

    monkeypatch.setattr(orchestrator.runtime.lanes, "locate", locate)
    monkeypatch.setattr(orchestrator.runtime.lanes, "forward", forward)
    connection = send(live, path, body)
    assert entered.wait(15), "the lane should have been asked"
    connection.close()
    assert closed.wait(5), "the call to the lane went on after its client went"


@pytest.mark.parametrize("path, body", WHOLE)
def test_a_client_that_stays_gets_its_whole_reply(live, path, body):
    live.state.silent = False
    connection = send(live, path, body)
    connection.settimeout(15)
    received = b""
    while b"Hello there" not in received:
        piece = connection.recv(65536)
        assert piece, f"the connection closed with {received!r}"
        received += piece
    connection.close()
    assert received.startswith(b"HTTP/1.1 200")


def test_a_reply_that_finishes_first_is_returned_and_the_watch_ends():
    async def run():
        left = asyncio.Event()

        async def receive():
            await left.wait()
            return {"type": "http.disconnect"}

        async def work():
            return 7

        assert await unless_client_leaves(receive, work()) == 7
        assert await unless_client_leaves(None, work()) == 7

    asyncio.run(run())


def test_a_client_that_goes_first_cancels_the_work_and_waits_for_it_to_close():
    async def run():
        closed = []

        async def receive():
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        async def work():
            try:
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(0.01)  # closing a stream takes a moment, and is waited for
                closed.append(True)

        with pytest.raises(ClientGone):
            await unless_client_leaves(receive, work())
        assert closed == [True]

    asyncio.run(run())


def test_a_failure_in_the_reply_still_reaches_the_caller():
    async def run():
        async def receive():
            await asyncio.sleep(60)
            return {"type": "http.disconnect"}

        async def work():
            raise ValueError("refused")

        with pytest.raises(ValueError, match="refused"):
            await unless_client_leaves(receive, work())

    asyncio.run(run())
