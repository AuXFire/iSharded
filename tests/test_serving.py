import asyncio
import contextlib
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn

from isharded import serving
from isharded.serving import NoDelayHTTPProtocol, set_nodelay


class Transport:
    def __init__(self, sock):
        self.sock = sock

    def get_extra_info(self, name):
        return self.sock if name == "socket" else None


def test_accepted_socket_gets_nodelay():
    with socket.create_server(("127.0.0.1", 0)) as server:
        with socket.create_connection(server.getsockname()):
            accepted, _ = server.accept()
            with accepted:
                set_nodelay(Transport(accepted))
                assert accepted.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)


def test_transport_without_a_socket_is_ignored():
    set_nodelay(Transport(None))


def test_server_answers_through_the_protocol():
    seen = []

    class Recording(NoDelayHTTPProtocol):
        def connection_made(self, transport):
            super().connection_made(transport)
            sock = transport.get_extra_info("socket")
            seen.append(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, http=Recording, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert httpx.get(f"http://127.0.0.1:{port}/", trust_env=False).text == "ok"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert seen and all(seen)


@contextlib.contextmanager
def serving_at(app):
    """A real server, because the timers under test live in the protocol."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, http=NoDelayHTTPProtocol, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def answers_at_once(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": b"ok"})


def closed_by_server(connection, wait=5.0):
    connection.settimeout(wait)
    try:
        return connection.recv(4096) == b""
    except ConnectionResetError:
        return True
    except TimeoutError:
        return False


@pytest.fixture
def short_stall(monkeypatch):
    monkeypatch.setattr(serving, "STALL_SECONDS", 0.3)


def test_a_connection_that_never_sends_a_request_is_dropped(short_stall):
    with serving_at(answers_at_once) as port, socket.create_connection(("127.0.0.1", port)) as silent:
        assert closed_by_server(silent)


def test_a_request_head_sent_a_byte_at_a_time_is_dropped(short_stall):
    with serving_at(answers_at_once) as port, socket.create_connection(("127.0.0.1", port)) as slow:
        head = b"GET / HTTP/1.1\r\nHost: a\r\nX-Padding: " + b"x" * 200
        try:
            for index in range(len(head)):
                slow.sendall(head[index : index + 1])
                if closed_by_server(slow, 0.05):
                    return
        except OSError:
            return  # dropped mid-send, which is as good
        pytest.fail("a head that never ends was kept for its whole length")


def test_a_body_that_stops_coming_is_dropped(short_stall):
    received = []

    async def app(scope, receive, send):
        received.append(await receive())
        while (await receive())["type"] != "http.disconnect":
            pass

    with serving_at(app) as port, socket.create_connection(("127.0.0.1", port)) as stalled:
        stalled.sendall(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 100000\r\n\r\npartial")
        assert closed_by_server(stalled)


def test_a_reply_may_take_far_longer_than_a_client_may_stall(short_stall):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for _ in range(6):
            await asyncio.sleep(0.2)
            await send({"type": "http.response.body", "body": b"token ", "more_body": True})
        await send({"type": "http.response.body", "body": b""})

    with serving_at(app) as port:
        assert httpx.get(f"http://127.0.0.1:{port}/", trust_env=False, timeout=10).text == "token " * 6


def test_a_reply_may_take_far_longer_to_start_than_a_client_may_stall(short_stall):
    async def app(scope, receive, send):
        await asyncio.sleep(1.0)
        await answers_at_once(scope, receive, send)

    with serving_at(app) as port:
        assert httpx.get(f"http://127.0.0.1:{port}/", trust_env=False, timeout=10).text == "ok"


def test_a_large_body_is_safe_for_as_long_as_it_keeps_moving(short_stall):
    async def app(scope, receive, send):
        total, more = 0, True
        while more:
            message = await receive()
            total += len(message.get("body", b""))
            more = message.get("more_body", False)
        body = str(total).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    def pieces():
        for _ in range(10):
            time.sleep(0.1)
            yield b"x" * serving.STALL_BYTES

    with serving_at(app) as port:
        answer = httpx.post(f"http://127.0.0.1:{port}/", content=pieces(), headers={"content-length": str(10 * serving.STALL_BYTES)}, trust_env=False, timeout=10)
    assert answer.text == str(10 * serving.STALL_BYTES)


def test_a_body_this_side_is_too_busy_to_read_is_not_the_clients_fault(short_stall):
    async def app(scope, receive, send):
        await asyncio.sleep(1.0)  # past the watch, with reading paused by the amount waiting
        total, more = 0, True
        while more:
            message = await receive()
            total += len(message.get("body", b""))
            more = message.get("more_body", False)
        body = str(total).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    size = 4 * 1024 * 1024
    with serving_at(app) as port:
        assert httpx.post(f"http://127.0.0.1:{port}/", content=b"x" * size, trust_env=False, timeout=20).text == str(size)


def test_a_connection_between_requests_is_left_to_keep_alive(monkeypatch):
    monkeypatch.setattr(serving, "STALL_SECONDS", 0.5)
    ask = b"GET / HTTP/1.1\r\nHost: a\r\n\r\n"
    with serving_at(answers_at_once) as port, socket.create_connection(("127.0.0.1", port)) as kept:
        kept.settimeout(5)
        for _ in range(3):
            kept.sendall(ask)
            reply = b""
            while not reply.endswith(b"ok"):
                piece = kept.recv(4096)
                assert piece, "the connection was closed between requests"
                reply += piece
            time.sleep(0.35)  # a reply just before the watch fires must not count against the next


def test_a_node_has_a_keep_alive_timeout_and_a_connection_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(serving.uvicorn, "run", lambda app, **options: seen.update(options))
    monkeypatch.setattr(serving, "keep_listening", lambda: None)
    monkeypatch.delenv("ISHARDED_MAX_CONNECTIONS", raising=False)
    serving.serve(object(), "127.0.0.1", 1)
    assert seen["timeout_keep_alive"] == serving.KEEP_ALIVE_SECONDS
    assert seen["limit_concurrency"] == serving.MAX_CONNECTIONS
    monkeypatch.setenv("ISHARDED_MAX_CONNECTIONS", "9000")
    serving.serve(object(), "127.0.0.1", 1)
    assert seen["limit_concurrency"] == 9000


def test_uvicorn_really_takes_these_options():
    import inspect

    accepted = inspect.signature(uvicorn.run).parameters
    assert {"timeout_keep_alive", "limit_concurrency", "ssl_ciphers"} <= set(accepted)


def test_a_client_closes_an_idle_connection_before_a_node_does():
    """Otherwise a request can go out on a connection the node is closing that same moment."""
    from httpx._config import DEFAULT_LIMITS

    assert serving.KEEP_ALIVE_SECONDS > DEFAULT_LIMITS.keepalive_expiry
    assert serving.KEEP_ALIVE_SECONDS < serving.STALL_SECONDS


def test_a_body_limit_refuses_from_the_headers_alone():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from isharded.serving import BodyLimit

    async def echo(request):
        return PlainTextResponse(str(len(await request.body())))

    site = Starlette(routes=[Route("/small", echo, methods=["POST"]), Route("/big", echo, methods=["POST"])])
    site.add_middleware(BodyLimit, limit=10, paths={"/big": 20})
    client = TestClient(site)
    assert client.post("/small", content=b"x" * 10).text == "10"
    assert client.post("/small", content=b"x" * 11).status_code == 413
    assert client.post("/big", content=b"x" * 20).text == "20"
    assert client.post("/big", content=b"x" * 21).status_code == 413

    def chunks():
        yield b"x"

    assert client.post("/small", content=chunks()).status_code == 411


@pytest.mark.skipif(sys.platform != "win32", reason="the Proactor loop is Windows only")
@pytest.mark.parametrize("steady", [False, True])
def test_a_failed_accept_does_not_close_the_listening_socket(steady) -> None:
    """Without the fix this fails the way a live node did: alive, and deaf for good."""
    import asyncio
    from asyncio import windows_events
    import socket as sockets

    real = windows_events.IocpProactor.accept
    failures = [OSError(64, "The specified network name is no longer available")]

    def failing(self, listener):
        if failures:
            future = self._loop.create_future()
            future.set_exception(failures.pop())
            return future
        return real(self, listener)

    async def talk(port: int) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        answer = await asyncio.wait_for(reader.read(2), 5)
        writer.close()
        return answer

    async def run() -> bytes | None:
        async def greet(reader, writer):
            writer.write(b"ok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(greet, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await talk(port)
        except (OSError, asyncio.TimeoutError):
            return None
        finally:
            server.close()

    windows_events.IocpProactor.accept = serving.steady_accept(failing) if steady else failing
    loop = windows_events.ProactorEventLoop()
    loop.set_exception_handler(lambda *_: None)
    try:
        answer = loop.run_until_complete(run())
    finally:
        windows_events.IocpProactor.accept = real
        loop.close()
    assert answer == (b"ok" if steady else None)
