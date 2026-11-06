import asyncio
import socket
import threading
import time

import httpx
import uvicorn

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
