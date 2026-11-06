import asyncio
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


def test_a_body_limit_refuses_from_the_headers_alone():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from isharded.serving import BodyLimit

    async def echo(request):
        return PlainTextResponse(str(len(await request.body())))

    site = Starlette(routes=[Route("/small", echo, methods=["POST"]), Route("/big", echo, methods=["POST"])])
    site.add_middleware(BodyLimit, limit=10, large={"/big": 20})
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
