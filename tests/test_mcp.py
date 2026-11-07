"""MCP in both directions, against the published specification.

Clients of both eras are real here: a real server process in tests/mcp_test_server.py,
and a real orchestrator on a real socket. With ISHARDED_TEST_MCP_PYTHON naming a
Python that has the official ``mcp`` package, that client is tried as well.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient
import uvicorn

from isharded import cli, mcp_client, orchestrator
from isharded.access import AccessControl, create_key
from isharded.mcp_client import (
    SAFE_ENVIRONMENT,
    CutOff,
    HttpServer,
    McpConnection,
    ServerError,
    StdioServer,
    Toolbox,
    agent_loop,
    ask_at_terminal,
    result_text,
    run_agent,
    server_environment,
    split_command,
)
from isharded.mcp_server import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    INVALID_REQUEST,
    LEGACY_VERSIONS,
    METHOD_NOT_FOUND,
    MODERN_VERSIONS,
    NAME_PARAMETERS,
    PARSE_ERROR,
    REQUEST_CANCELLED,
    RESOURCE_NOT_FOUND,
    SERVER_KEY,
    SUPPORTED_VERSIONS,
    TOOLS,
    UNSUPPORTED_VERSION,
    VERSION_KEY,
    InFlight,
    McpHandler,
    MeshTools,
    accepts_both,
    as_header,
    header_value,
    serve_stdio,
)
from isharded.protocol import StageAssignment
from isharded.tokenizer import ChatTokenizer

ROOT = Path(__file__).resolve().parents[1]
TEST_SERVER = Path(__file__).with_name("mcp_test_server.py")
MODERN = MODERN_VERSIONS[0]
META = {VERSION_KEY: MODERN, "io.modelcontextprotocol/clientInfo": {"name": "tests", "version": "1"}, "io.modelcontextprotocol/clientCapabilities": {}}
WANTS = "application/json, text/event-stream"


def fake_mesh(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "small"}, {"id": "large"}]})
    if request.url.path == "/admin/status":
        status = {"status": "ready", "model": "small", "epoch": 3, "mesh": {"leader": "a", "members": {"a": "http://a"}}, "route": [{"worker_id": "a-cpu"}], "workers": {"a-cpu": {"state": "active", "registration": "long"}}, "lanes": [], "metrics": {"tokens_per_second_ema": 12.5}}
        return httpx.Response(200, json=status)
    body = json.loads(request.content)
    if request.url.path == "/v1/embeddings":
        return httpx.Response(200, json={"model": "embedder", "data": [{"embedding": [0.6, 0.8]}], "usage": {"total_tokens": 2}, "seen": body})
    answer = "<think>hmm</think>\n\nPong"
    return httpx.Response(200, json={"model": body.get("model", "small"), "choices": [{"message": {"content": answer}, "finish_reason": "stop"}], "usage": {"total_tokens": 9}, "seen": body})


def handler(transport=fake_mesh, stateless: bool = False) -> McpHandler:
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport), base_url="http://mesh")
    return McpHandler(MeshTools(client), stateless=stateless)


def rpc(target: McpHandler, method: str, params=None, identifier=1):
    message = {"jsonrpc": "2.0", "id": identifier, "method": method}
    if params is not None:
        message["params"] = params
    return asyncio.run(target.handle(message))


def modern(target: McpHandler, method: str, params=None):
    return rpc(target, method, {**(params or {}), "_meta": META})


def ready(transport=fake_mesh) -> McpHandler:
    target = handler(transport)
    assert "result" in rpc(target, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    return target


# --- the handshake, for clients from before 2026-07-28


@pytest.mark.parametrize("version", LEGACY_VERSIONS)
def test_initialize_answers_with_the_version_asked_for(version):
    reply = rpc(handler(), "initialize", {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    result = reply["result"]
    assert result["protocolVersion"] == version
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert result["serverInfo"]["name"] == "isharded"
    assert "resultType" not in result


def test_initialize_offers_our_own_version_when_it_does_not_know_theirs():
    assert rpc(handler(), "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == LEGACY_VERSIONS[0]


def test_initialize_needs_a_version():
    assert rpc(handler(), "initialize", {})["error"]["code"] == INVALID_PARAMS


def test_nothing_but_ping_comes_before_initialize():
    target = handler()
    assert rpc(target, "ping")["result"] == {}
    assert rpc(target, "tools/list")["error"]["code"] == INVALID_REQUEST


def test_a_legacy_session_lists_and_calls_tools():
    target = ready()
    names = [tool["name"] for tool in rpc(target, "tools/list")["result"]["tools"]]
    assert names == ["chat", "embed", "list_models", "mesh_status"]
    result = rpc(target, "tools/call", {"name": "list_models", "arguments": {}})["result"]
    assert result["structuredContent"] == {"models": ["small", "large"]}
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    assert "resultType" not in result


# --- per-request metadata, from 2026-07-28


def test_discover_says_what_is_supported_without_any_handshake():
    result = modern(handler(), "server/discover")["result"]
    assert result["supportedVersions"] == list(SUPPORTED_VERSIONS)
    assert result["capabilities"] == {"tools": {}, "resources": {}, "prompts": {}}
    assert result["resultType"] == "complete"
    assert result["_meta"][SERVER_KEY]["name"] == "isharded"
    assert result["ttlMs"] > 0 and result["cacheScope"] in {"public", "private"}


def test_modern_requests_need_no_initialize_and_carry_the_new_fields():
    target = handler()
    result = modern(target, "tools/list")["result"]
    assert [tool["name"] for tool in result["tools"]] == ["chat", "embed", "list_models", "mesh_status"]
    assert result["resultType"] == "complete" and result["ttlMs"] > 0 and result["cacheScope"] == "private"
    assert result["_meta"][SERVER_KEY]["name"] == "isharded"
    called = modern(target, "tools/call", {"name": "list_models", "arguments": {}})["result"]
    assert called["resultType"] == "complete" and not called.get("isError")


def test_a_version_we_do_not_speak_is_named_with_those_we_do():
    error = rpc(handler(), "tools/list", {"_meta": {VERSION_KEY: "1900-01-01"}})["error"]
    assert error["code"] == UNSUPPORTED_VERSION == -32022
    assert error["data"] == {"supported": list(SUPPORTED_VERSIONS), "requested": "1900-01-01"}


def test_ping_is_gone_from_the_modern_protocol():
    assert modern(handler(), "ping")["error"]["code"] == METHOD_NOT_FOUND


# --- errors


@pytest.mark.parametrize(
    "message, code",
    [
        ({"id": 1, "method": "ping"}, INVALID_REQUEST),
        ({"jsonrpc": "1.0", "id": 1, "method": "ping"}, INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": None, "method": "ping"}, INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": True, "method": "ping"}, INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1, "method": 7}, INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1]}, INVALID_PARAMS),
        ([{"jsonrpc": "2.0", "id": 1, "method": "ping"}], INVALID_REQUEST),
    ],
)
def test_malformed_messages_are_named(message, code):
    assert asyncio.run(handler().handle(message))["error"]["code"] == code


def test_an_unknown_method_is_method_not_found():
    assert rpc(ready(), "resources/subscribe")["error"]["code"] == METHOD_NOT_FOUND == -32601


def test_notifications_and_stray_answers_get_no_reply():
    assert asyncio.run(handler().handle({"jsonrpc": "2.0", "method": "notifications/initialized"})) is None
    assert asyncio.run(handler().handle({"jsonrpc": "2.0", "id": 4, "result": {}})) is None


def test_an_unknown_tool_is_a_protocol_error():
    error = rpc(ready(), "tools/call", {"name": "launch", "arguments": {}})["error"]
    assert error == {"code": INVALID_PARAMS, "message": "Unknown tool: launch"}


@pytest.mark.parametrize("params", [{}, {"name": 3}, {"name": "chat", "arguments": [1]}])
def test_a_malformed_call_is_a_protocol_error(params):
    assert rpc(ready(), "tools/call", params)["error"]["code"] == INVALID_PARAMS


def test_a_cursor_for_a_page_that_does_not_exist_is_refused():
    assert rpc(ready(), "tools/list", {"cursor": "next"})["error"]["code"] == INVALID_PARAMS


@pytest.mark.parametrize(
    "arguments, words",
    [
        ({}, "prompt is required"),
        ({"prompt": 3}, "prompt must be a string"),
        ({"prompt": "x", "max_tokens": "many"}, "max_tokens must be a integer"),
        ({"prompt": "x", "max_tokens": True}, "max_tokens must be a integer"),
        ({"prompt": "x", "max_tokens": 0}, "from 1 to 32768"),
        ({"prompt": "x", "temperature": 9}, "from 0 to 5"),
        ({"prompt": "x", "tools": []}, "not an argument"),
    ],
)
def test_bad_arguments_are_a_tool_error_a_model_can_read(arguments, words):
    result = rpc(ready(), "tools/call", {"name": "chat", "arguments": arguments})["result"]
    assert result["isError"] is True and words in result["content"][0]["text"]


def test_a_refusal_from_the_mesh_is_a_tool_error():
    def refuses(request):
        return httpx.Response(409, json={"detail": "no model is ready; load one first"})

    result = rpc(ready(refuses), "tools/call", {"name": "chat", "arguments": {"prompt": "x"}})["result"]
    assert result["isError"] is True and "no model is ready" in result["content"][0]["text"]


def test_a_mesh_that_is_down_is_a_tool_error_that_says_what_to_check():
    def down(request):
        raise httpx.ConnectError("connection refused")

    result = rpc(ready(down), "tools/call", {"name": "list_models", "arguments": {}})["result"]
    assert result["isError"] is True and "Check that the node is running" in result["content"][0]["text"]


# --- the tools


def test_chat_returns_the_answer_without_the_reasoning():
    result = rpc(ready(), "tools/call", {"name": "chat", "arguments": {"prompt": "Ping?"}})["result"]
    assert result["content"][0]["text"] == "Pong"
    assert result["structuredContent"]["finish_reason"] == "stop"


def test_chat_passes_on_what_it_was_given():
    seen = {}

    def spy(request):
        seen.update(json.loads(request.content))
        return fake_mesh(request)

    arguments = {"prompt": "Ping?", "system": "Be brief", "model": "large", "max_tokens": 9, "temperature": 0.5, "enable_thinking": False}
    rpc(ready(spy), "tools/call", {"name": "chat", "arguments": arguments})
    assert seen == {
        "messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Ping?"}],
        "model": "large", "max_tokens": 9, "temperature": 0.5, "enable_thinking": False,
    }


def test_chat_says_when_a_reply_was_cut_off():
    def cut(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "Pon"}, "finish_reason": "length"}]})

    assert "cut off" in rpc(ready(cut), "tools/call", {"name": "chat", "arguments": {"prompt": "x"}})["result"]["content"][0]["text"]


def test_embed_returns_the_vector_the_mesh_made():
    seen = {}

    def recording(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return fake_mesh(request)

    result = rpc(ready(recording), "tools/call", {"name": "embed", "arguments": {"text": "hello", "model": "embedder"}})["result"]
    assert seen == {"path": "/v1/embeddings", "body": {"input": "hello", "model": "embedder"}}
    assert result["structuredContent"]["embedding"] == [0.6, 0.8]
    assert result["structuredContent"]["model"] == "embedder"


def test_chat_passes_an_image_on_as_a_part_of_the_question():
    seen = {}

    def recording(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return fake_mesh(request)

    picture = "data:image/png;base64,AAAA"
    rpc(ready(recording), "tools/call", {"name": "chat", "arguments": {"prompt": "What is this?", "image": picture}})
    assert seen["body"]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": picture}}]}]


def test_mesh_status_is_a_summary_that_can_be_read():
    status = rpc(ready(), "tools/call", {"name": "mesh_status", "arguments": {}})["result"]["structuredContent"]
    assert status["status"] == "ready" and status["coordinator"] == "a"
    assert status["route"] == ["a-cpu"] and status["workers"] == {"a-cpu": "active"}


def test_every_tool_offered_is_a_real_function():
    assert {tool["name"] for tool in TOOLS} == {"chat", "embed", "list_models", "mesh_status"}
    assert all(callable(getattr(MeshTools, tool["name"])) for tool in TOOLS)
    assert all(tool["inputSchema"]["type"] == "object" and tool["description"] for tool in TOOLS)


def test_embed_is_offered_exactly_when_the_mesh_can_embed():
    routes = {route.path for route in orchestrator.app.routes}
    assert ("/v1/embeddings" in routes) == any(tool["name"] == "embed" for tool in TOOLS)


# --- over stdio


class Pipes:
    """Both ends of a stdio connection, without any process."""

    def __init__(self) -> None:
        self.incoming: queue.Queue[bytes] = queue.Queue()
        self.out = io.BytesIO()

    def readline(self) -> bytes:
        return self.incoming.get()

    def write(self, data: bytes) -> None:
        self.out.write(data)

    def flush(self) -> None:
        pass

    def send(self, message) -> None:
        self.incoming.put(json.dumps(message).encode() + b"\n")

    def replies(self) -> list[dict]:
        return [json.loads(line) for line in self.out.getvalue().splitlines()]

    def wait_for(self, identifier, seconds: float = 5.0) -> dict:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            found = [reply for reply in self.replies() if reply.get("id") == identifier]
            if found:
                return found[0]
            time.sleep(0.01)
        raise AssertionError(f"no reply to {identifier}")


@pytest.fixture
def stdio():
    """Serve over pipes, and always close them, or a failed test would hang the run."""
    started = []

    def serve(transport) -> tuple[Pipes, threading.Thread]:
        pipes = Pipes()

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://mesh") as client:
                await serve_stdio(client, pipes, pipes)

        thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
        thread.start()
        started.append(pipes)
        return pipes, thread

    yield serve
    for pipes in started:
        pipes.incoming.put(b"")


def test_stdio_answers_line_by_line_and_stops_when_the_input_ends(stdio):
    pipes, thread = stdio(httpx.MockTransport(fake_mesh))
    pipes.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    pipes.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert pipes.wait_for(1)["result"]["protocolVersion"] == "2025-06-18"
    pipes.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert len(pipes.wait_for(2)["result"]["tools"]) == len(TOOLS)
    pipes.incoming.put(b"not json\n")
    deadline = time.monotonic() + 5
    while not any(reply.get("error", {}).get("code") == PARSE_ERROR for reply in pipes.replies()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    pipes.incoming.put(b"")
    thread.join(5)
    assert not thread.is_alive()


class SlowThenFine(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.cancelled = threading.Event()

    async def handle_async_request(self, request):
        if b"slow" in request.content:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return fake_mesh(request)


def test_a_long_reply_holds_nothing_up_and_can_be_cancelled(stdio):
    transport = SlowThenFine()
    pipes, thread = stdio(transport)
    pipes.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
    pipes.wait_for(1)
    pipes.send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "chat", "arguments": {"prompt": "slow"}}})
    pipes.send({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    assert pipes.wait_for(3)["result"] == {}
    pipes.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2, "reason": "changed my mind"}})
    assert transport.cancelled.wait(5), "the request to the mesh should stop"
    pipes.send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "chat", "arguments": {"prompt": "fine"}}})
    assert pipes.wait_for(4)["result"]["content"][0]["text"] == "Pong"
    assert not [reply for reply in pipes.replies() if reply.get("id") == 2], "a cancelled request gets no reply"
    pipes.incoming.put(b"")
    thread.join(5)


def project_env(tmp_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("ISHARDED_")}
    env.update(ISHARDED_POOL_TOKEN_FILE=str(tmp_path / "no-token"), PYTHONUTF8="1")
    return env


def test_the_real_command_keeps_stdout_for_the_protocol(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-m", "isharded", "mcp", "--url", "http://127.0.0.1:9"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, env=project_env(tmp_path),
    )
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": META}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": META}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_models", "arguments": {}, "_meta": META}},
    ]
    output, _ = process.communicate(("\n".join(json.dumps(line) for line in lines) + "\n").encode(), timeout=120)
    assert process.returncode == 0, "it should stop cleanly when its input closes"
    replies = {reply["id"]: reply for reply in map(json.loads, output.decode().splitlines())}
    assert MODERN in replies[1]["result"]["supportedVersions"]
    assert len(replies[2]["result"]["tools"]) == len(TOOLS)
    assert replies[3]["result"]["isError"] is True


# --- over HTTP


class ScriptedChat(ChatTokenizer):
    def __init__(self) -> None:
        super().__init__(None, "{% for m in messages %}{{ m.content }}{% endfor %}", {0}, {})  # type: ignore[arg-type]

    def encode(self, text):
        return [1]

    def decode(self, ids):
        return "".join("Pong" for token in ids if token == 7)


@pytest.fixture
def site(monkeypatch, tmp_path):
    node = orchestrator.Orchestrator()
    node.model_id = "scripted"
    node.tokenizer = ScriptedChat()
    node.route = [StageAssignment(worker_id="w", url="http://worker", start_layer=0, end_layer=1)]

    async def token_stream(session, *rest):
        yield 7
        yield 0

    node.token_stream = token_stream
    monkeypatch.setattr(orchestrator, "runtime", node)
    return SimpleNamespace(node=node, client=TestClient(orchestrator.app), tmp=tmp_path)


def legacy_post(site, method, params=None, identifier=1, **headers):
    message = {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params or {}}
    return site.client.post("/mcp", json=message, headers={"Accept": WANTS, **headers})


def modern_post(site, method, params=None, version=MODERN, **headers):
    message = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {**(params or {}), "_meta": {**META, VERSION_KEY: version}}}
    sent = {"Accept": WANTS, "MCP-Protocol-Version": version, "Mcp-Method": method}
    if method in NAME_PARAMETERS:
        sent["Mcp-Name"] = (params or {}).get(NAME_PARAMETERS[method], "")
    return site.client.post("/mcp", json=message, headers={**sent, **headers})


def test_http_serves_the_handshake_and_makes_no_session(site):
    answer = legacy_post(site, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    assert answer.status_code == 200 and answer.json()["result"]["protocolVersion"] == "2025-11-25"
    assert "mcp-session-id" not in answer.headers


def test_http_needs_no_session_for_a_legacy_client(site):
    answer = legacy_post(site, "tools/list", **{"MCP-Protocol-Version": "2025-11-25"})
    assert answer.status_code == 200 and len(answer.json()["result"]["tools"]) == len(TOOLS)


def test_http_serves_modern_requests(site):
    answer = modern_post(site, "tools/list")
    assert answer.status_code == 200 and answer.json()["result"]["resultType"] == "complete"


def test_a_tool_over_http_takes_the_normal_reply_path(site):
    answer = modern_post(site, "tools/call", {"name": "chat", "arguments": {"prompt": "Ping?"}})
    assert answer.json()["result"]["content"][0]["text"] == "Pong"
    assert site.node.requests_total == 1


def test_a_notification_is_accepted_with_no_body(site):
    answer = site.client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers={"Accept": WANTS})
    assert answer.status_code == 202 and answer.content == b""


def test_a_modern_notification_needs_no_mirrored_headers(site):
    # The specification leaves these undefined, so a notification is not refused for lacking them.
    message = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}
    answer = site.client.post("/mcp", json=message, headers={"Accept": WANTS, "MCP-Protocol-Version": MODERN})
    assert answer.status_code == 202 and answer.content == b""


@pytest.mark.parametrize("method", ["get", "delete"])
def test_there_is_no_stream_and_no_session_to_end(site, method):
    answer = getattr(site.client, method)("/mcp")
    assert answer.status_code == 405 and answer.headers["allow"] == "POST"


@pytest.mark.parametrize(
    "accept, expected",
    [
        ("application/json, text/event-stream", True),
        ("text/event-stream;q=0.9, application/json", True),
        ("*/*", True),
        ("application/json", False),
        ("", False),
    ],
)
def test_a_client_must_accept_both_kinds_of_reply(accept, expected):
    assert accepts_both(accept) is expected


def test_refusals_at_the_door(site):
    message = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert site.client.post("/mcp", json=message, headers={"Accept": "application/json"}).status_code == 406
    assert site.client.post("/mcp", content="{}", headers={"Accept": WANTS, "Content-Type": "text/plain"}).status_code == 415
    bad = site.client.post("/mcp", content="{broken", headers={"Accept": WANTS, "Content-Type": "application/json"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == PARSE_ERROR and bad.json()["id"] is None
    batch = site.client.post("/mcp", json=[message], headers={"Accept": WANTS})
    assert batch.status_code == 400 and batch.json()["error"]["code"] == INVALID_REQUEST


def test_an_unsupported_version_header_is_a_400_naming_what_works(site):
    answer = legacy_post(site, "tools/list", **{"MCP-Protocol-Version": "1999-01-01"})
    assert answer.status_code == 400
    assert answer.json()["error"]["code"] == UNSUPPORTED_VERSION and answer.json()["error"]["data"]["supported"] == list(SUPPORTED_VERSIONS)


def test_an_unknown_modern_method_is_a_404_with_a_json_rpc_body(site):
    answer = modern_post(site, "resources/subscribe")
    assert answer.status_code == 404 and answer.json()["error"]["code"] == METHOD_NOT_FOUND


def test_a_legacy_unknown_method_stays_a_json_rpc_error(site):
    answer = legacy_post(site, "resources/subscribe", **{"MCP-Protocol-Version": "2025-11-25"})
    assert answer.status_code == 200 and answer.json()["error"]["code"] == METHOD_NOT_FOUND


@pytest.mark.parametrize(
    "headers",
    [
        {"Mcp-Method": "tools/list"},
        {"Mcp-Name": "mesh_status"},
    ],
)
def test_headers_that_disagree_with_the_body_are_refused(site, headers):
    answer = modern_post(site, "tools/call", {"name": "chat", "arguments": {"prompt": "x"}}, **headers)
    assert answer.status_code == 400 and answer.json()["error"]["code"] == HEADER_MISMATCH == -32020


def test_a_modern_body_with_no_version_header_is_refused(site):
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": META}}
    answer = site.client.post("/mcp", json=message, headers={"Accept": WANTS, "Mcp-Method": "tools/list"})
    assert answer.status_code == 400 and answer.json()["error"]["code"] == HEADER_MISMATCH


def test_a_header_and_body_on_different_versions_are_refused(site):
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {VERSION_KEY: "2025-11-25"}}}
    headers = {"Accept": WANTS, "Mcp-Method": "tools/list", "MCP-Protocol-Version": MODERN}
    assert site.client.post("/mcp", json=message, headers=headers).json()["error"]["code"] == HEADER_MISMATCH


def test_a_name_too_awkward_for_a_header_travels_as_base64():
    assert header_value("=?base64?" + base64.b64encode("héllo".encode()).decode() + "?=") == "héllo"
    assert header_value("plain") == "plain"


def test_base64_names_are_compared_after_decoding(site):
    encoded = "=?base64?" + base64.b64encode(b"chat").decode() + "?="
    answer = modern_post(site, "tools/call", {"name": "chat", "arguments": {"prompt": "x"}}, **{"Mcp-Name": encoded})
    assert answer.status_code == 200


@pytest.mark.parametrize("origin", ["http://evil.example", "http://evil.example:8000", "null"])
def test_a_page_from_elsewhere_is_turned_away(site, origin):
    # The attacker's name resolving to this machine is the whole trick, so Host cannot help.
    answer = legacy_post(site, "tools/list", Origin=origin, Host="evil.example")
    assert answer.status_code == 403 and answer.json()["id"] is None


@pytest.mark.parametrize("origin", ["http://localhost:6274", "http://127.0.0.1:3000", "http://[::1]:3000"])
def test_a_page_on_this_machine_is_let_in(site, origin):
    assert legacy_post(site, "tools/list", Origin=origin).status_code == 200


def keyed(site, requests_per_minute: int = 0) -> str:
    path = site.tmp / "keys.json"
    key = create_key(str(path), "someone", "user", requests_per_minute)
    site.node.access = AccessControl.load(None, str(path))
    return key


def test_http_is_behind_the_same_keys_as_v1(site):
    key = keyed(site)
    missing = legacy_post(site, "tools/list")
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    assert legacy_post(site, "tools/list", Authorization="Bearer wrong").status_code == 401
    assert legacy_post(site, "tools/list", Authorization=f"Bearer {key}").status_code == 200


def test_a_tool_acts_as_the_caller_and_the_audit_log_says_so(site, monkeypatch):
    key = keyed(site)
    recorded = []
    monkeypatch.setattr(site.node.audit, "record", lambda actor, method, path, status, client: recorded.append((actor, path, status, client)))
    arguments = {"name": "chat", "arguments": {"prompt": "Ping?"}}
    assert legacy_post(site, "tools/call", arguments, Authorization=f"Bearer {key}").json()["result"]["content"][0]["text"] == "Pong"
    assert recorded == [("someone", "/v1/chat/completions", 200, "testclient")]


def test_a_tool_pays_once_and_the_protocol_pays_too(site):
    key = keyed(site, requests_per_minute=2)
    bearer = {"Authorization": f"Bearer {key}"}
    assert legacy_post(site, "tools/call", {"name": "list_models", "arguments": {}}, **bearer).json()["result"].get("isError") is None
    assert legacy_post(site, "tools/list", **bearer).status_code == 200
    limited = legacy_post(site, "tools/list", **bearer)
    assert limited.status_code == 429 and "retry-after" in limited.headers


def test_a_limit_reached_inside_a_tool_is_a_tool_error(site):
    key = keyed(site, requests_per_minute=1)
    bearer = {"Authorization": f"Bearer {key}"}
    legacy_post(site, "tools/list", **bearer)
    result = legacy_post(site, "tools/call", {"name": "list_models", "arguments": {}}, **bearer).json()["result"]
    assert result["isError"] is True and "429" in result["content"][0]["text"]


def test_responses_carry_no_length_for_avg_to_mangle(site):
    # A Content-Length here is what AVG turns into an unreadable reply. See ChunkedResponses.
    for answer in (legacy_post(site, "tools/list"), site.client.get("/mcp"), site.client.post("/mcp", json={}, headers={"Accept": "text/plain"})):
        assert "content-length" not in answer.headers


@pytest.fixture(scope="module")
def real_socket(tmp_path_factory):
    """A real orchestrator on a real socket, on a port kept for this project's tests."""
    home = tmp_path_factory.mktemp("mcp-node")
    port = 9531
    env = {key: value for key, value in os.environ.items() if not key.startswith("ISHARDED_")}
    env.update(ISHARDED_POOL_TOKEN_FILE=str(home / "no-token"), PYTHONUTF8="1")
    command = [
        sys.executable, "-m", "isharded.orchestrator", "--host", "127.0.0.1", "--port", str(port), "--open",
        "--keys-file", str(home / "keys.json"), "--state-file", str(home / "state.json"), "--audit-file", str(home / "audit.jsonl"),
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                httpx.get(url + "/health", timeout=1, trust_env=False)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("the orchestrator did not start")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=15)


@pytest.mark.parametrize(
    "message, status",
    [
        ({"jsonrpc": "2.0", "id": 1, "method": "ping"}, 200),
        ({"jsonrpc": "2.0", "method": "notifications/initialized"}, 202),
        ({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, 200),
    ],
)
def test_every_reply_survives_a_real_socket(real_socket, message, status):
    answer = httpx.post(real_socket + "/mcp", json=message, headers={"Accept": WANTS, "MCP-Protocol-Version": "2025-11-25"}, timeout=20, trust_env=False)
    assert answer.status_code == status


def test_refusals_survive_a_real_socket_too(real_socket):
    assert httpx.get(real_socket + "/mcp", timeout=20, trust_env=False).status_code == 405
    answer = httpx.post(real_socket + "/mcp", json={}, headers={"Accept": "text/plain"}, timeout=20, trust_env=False)
    assert answer.status_code == 406 and answer.json()["error"]["code"] == INVALID_REQUEST


def test_an_inner_call_works_on_a_real_socket(real_socket):
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}}
    answer = httpx.post(real_socket + "/mcp", json=message, headers={"Accept": WANTS, "MCP-Protocol-Version": "2025-11-25"}, timeout=20, trust_env=False)
    assert answer.json()["result"]["structuredContent"] == {"models": []}


# --- the client, with a real server process


def server_command(*options: str) -> str:
    python = f'"{sys.executable}"' if " " in sys.executable else sys.executable
    return " ".join([python, f'"{TEST_SERVER}"', *options])


@pytest.fixture
def started():
    servers = []

    def start(*options: str) -> StdioServer:
        server = StdioServer(server_command(*options), server_environment([]))
        servers.append(server)
        server.start()
        return server

    yield start
    for server in servers:
        server.close()


@pytest.mark.parametrize(
    "options, modern_server",
    [(["--era", "legacy"], False), (["--era", "modern"], True), (["--era", "legacy", "--banner"], False)],
)
def test_the_client_finds_out_which_era_a_server_is_from(started, options, modern_server):
    server = started(*options)
    assert (server.modern_version == MODERN) is modern_server
    toolbox = Toolbox([server])
    assert {"add", "write_note"} <= set(toolbox.tools)
    assert toolbox.call("add", {"a": 2, "b": 3}, 10) == "5"


def test_a_legacy_server_that_ignores_the_probe_is_still_reached(started, monkeypatch):
    monkeypatch.setattr(mcp_client, "PROBE_SECONDS", 0.3)
    server = started("--era", "legacy", "--silent-probe")
    assert server.modern_version is None
    assert Toolbox([server]).call("add", {"a": 1, "b": 1}, 10) == "2"


def test_the_client_reports_a_tools_error_to_the_model(started):
    toolbox = Toolbox([started()])
    assert toolbox.call("fails", {}, 10) == "The tool reported an error: the disk is full"


def test_a_tool_the_server_refuses_is_reported_too(started):
    server = started()
    toolbox = Toolbox([server])
    toolbox.tools["gone"] = (server, {"name": "gone"})
    assert "could not be called" in toolbox.call("gone", {}, 10)


def test_a_server_that_dies_is_reported_and_the_run_goes_on(started):
    server = started()
    toolbox = Toolbox([server])
    server.process.kill()
    server.process.wait(5)
    assert "failed" in toolbox.call("add", {"a": 1, "b": 2}, 5)


def test_closing_ends_the_server_process(started):
    server = started()
    server.close()
    assert server.process.poll() is not None


def test_a_server_that_will_not_start_says_so():
    with pytest.raises(ServerError, match="could not start"):
        StdioServer("this-program-does-not-exist-anywhere --flag", server_environment([]))
    with pytest.raises(ServerError, match="empty command"):
        split_command("   ")


def test_quoted_paths_with_spaces_stay_whole():
    parts = split_command('"C:\\Program Files\\x y\\server.exe" --a "b c"' if os.name == "nt" else "'/opt/x y/server' --a 'b c'")
    assert parts[-2:] == ["--a", "b c"] and " " in parts[0]


def test_servers_see_only_the_basics_unless_told_otherwise(started, monkeypatch):
    monkeypatch.setenv("ISHARDED_POOL_TOKEN", "secret-value")
    monkeypatch.setenv("SOME_API_KEY", "secret-value")
    monkeypatch.setenv("WANTED_BY_SERVER", "yes")
    assert "SOME_API_KEY" not in server_environment([])
    assert "ISHARDED_POOL_TOKEN" not in server_environment(["WANTED_BY_SERVER"])
    assert server_environment(["WANTED_BY_SERVER"])["WANTED_BY_SERVER"] == "yes"
    seen = Toolbox([started()]).call("environment", {}, 10)
    assert "SOME_API_KEY" not in seen and "ISHARDED_POOL_TOKEN" not in seen
    assert {name.upper() for name in json.loads(seen)} - {name.upper() for name in SAFE_ENVIRONMENT} <= {"LC_CTYPE", "SYSTEMROOT", "PYTHONIOENCODING", "COMSPEC", "WINDIR"}


def test_a_tool_list_is_offered_to_the_model_as_openai_functions(started):
    tools = Toolbox([started()]).for_model()
    add = next(tool for tool in tools if tool["function"]["name"] == "add")
    assert add["type"] == "function" and add["function"]["parameters"]["required"] == ["a", "b"]


def test_two_servers_with_the_same_tool_stay_apart(started):
    first, second = started(), started()
    second.name = "second"
    toolbox = Toolbox([first, second])
    assert "add" in toolbox.tools and "second_add" in toolbox.tools
    assert toolbox.tools["second_add"][0] is second


def test_text_of_a_result_says_what_was_left_out():
    assert result_text({"content": [{"type": "text", "text": "hi"}, {"type": "image", "data": "x"}]}) == "hi\n[image content omitted]"
    assert result_text({"structuredContent": {"a": 1}}) == '{"a": 1}'
    assert "cut off" in result_text({"content": [{"type": "text", "text": "x" * 40000}]})


# --- the agent loop


def scripted_model(*replies):
    """A mesh that answers from a list, and remembers what it was asked."""
    asked = []
    remaining = list(replies)

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content))
        reply = remaining.pop(0)
        if isinstance(reply, str):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]})
        calls = [{"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}} for index, (name, arguments) in enumerate(reply)]
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}, "finish_reason": "tool_calls"}]})

    client = httpx.Client(transport=httpx.MockTransport(answer), base_url="http://mesh")
    return client, asked


def tool_results(asked) -> list[str]:
    return [message["content"] for message in asked[-1]["messages"] if message["role"] == "tool"]


def run_loop(client, toolbox, allowed=(), ask=None, max_turns=8):
    def refuse(name, arguments):
        raise AssertionError(f"should not have asked about {name}")

    return agent_loop(client, toolbox, "small", "go", set(allowed), max_turns, None, 10, ask or refuse)


def test_read_only_tools_run_without_asking(started):
    client, asked = scripted_model([("add", {"a": 2, "b": 3})], "<think>easy</think>\n\nIt is 5.")
    assert run_loop(client, Toolbox([started()])) == "It is 5."
    assert tool_results(asked) == ["5"]
    assert asked[1]["messages"][1]["tool_calls"][0]["id"] == "call_0"
    assert asked[1]["messages"][2]["tool_call_id"] == "call_0"


def test_a_tool_with_no_read_only_claim_waits_for_a_person(started, tmp_path):
    notes = tmp_path / "notes.txt"
    client, asked = scripted_model([("write_note", {"text": "hello"})], "Done.")
    questions = []

    def answer(name, arguments):
        questions.append((name, arguments))
        return True

    run_loop(client, Toolbox([started("--notes", str(notes))]), ask=answer)
    assert questions == [("write_note", {"text": "hello"})]
    assert notes.read_text(encoding="utf-8") == "hello\n"


def test_saying_no_stops_the_tool_and_tells_the_model(started, tmp_path):
    notes = tmp_path / "notes.txt"
    client, asked = scripted_model([("write_note", {"text": "hello"})], "I could not save it.")
    run_loop(client, Toolbox([started("--notes", str(notes))]), ask=lambda name, arguments: False)
    assert not notes.exists()
    assert "did not allow" in tool_results(asked)[0]


def test_allow_lets_one_tool_through_without_asking(started, tmp_path):
    notes = tmp_path / "notes.txt"
    client, _ = scripted_model([("write_note", {"text": "hello"})], "Done.")
    run_loop(client, Toolbox([started("--notes", str(notes))]), allowed={"write_note"})
    assert notes.exists()


def test_a_read_only_claim_is_only_the_servers_word(started, tmp_path):
    """Documented in the guide: a lying server is not stopped by this check."""
    notes = tmp_path / "notes.txt"
    client, _ = scripted_model([("claims_read_only", {"text": "written anyway"})], "Done.")
    run_loop(client, Toolbox([started("--notes", str(notes))]))
    assert notes.read_text(encoding="utf-8") == "written anyway\n"


def test_annotations_other_than_true_do_not_count(started):
    toolbox = Toolbox([started()])
    server, tool = toolbox.tools["add"]
    for claim in ("yes", 1, None):
        toolbox.tools["add"] = (server, {**tool, "annotations": {"readOnlyHint": claim}})
        assert not toolbox.read_only("add")
    toolbox.tools["add"] = (server, {**tool, "annotations": "readOnlyHint"})
    assert not toolbox.read_only("add")


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_with_no_terminal_nobody_can_say_yes(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert ask_at_terminal("write_note", {"text": "x"}) is False
    assert "--allow" in capsys.readouterr().err


def test_a_device_that_only_claims_to_be_a_terminal_is_refused_with_the_reason(monkeypatch, capsys):
    # Windows' NUL says it is a terminal, and then reads as empty.
    monkeypatch.setattr("sys.stdin", Terminal(""))
    assert ask_at_terminal("write_note", {"text": "x"}) is False
    assert "--allow" in capsys.readouterr().err


def test_at_a_terminal_the_answer_decides(monkeypatch, capsys):
    for typed, expected in (("y\n", True), ("YES\n", True), ("n\n", False), ("\n", False)):
        monkeypatch.setattr("sys.stdin", Terminal(typed))
        assert ask_at_terminal("write_note", {"text": "x"}) is expected


def test_the_question_stays_off_stdout(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", Terminal("y\n"))
    ask_at_terminal("write_note", {"text": "x"})
    captured = capsys.readouterr()
    assert captured.out == "" and "Allow?" in captured.err


def test_a_tool_cannot_rewrite_the_question_with_control_characters(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", Terminal("n\n"))
    ask_at_terminal("evil\x1b[2K", {"text": "\x1b[1A\rAllow? harmless"})
    assert "\x1b" not in capsys.readouterr().err


def test_a_call_to_nothing_or_with_bad_arguments_is_explained_to_the_model(started):
    toolbox = Toolbox([started()])
    client, asked = scripted_model([("no_such", {})], "ok")
    run_loop(client, toolbox)
    assert "no tool named" in tool_results(asked)[0]


def test_a_model_that_never_stops_is_stopped(started):
    client, _ = scripted_model(*[[("add", {"a": 1, "b": 1})] for _ in range(5)])
    with pytest.raises(ServerError, match="--max-turns"):
        run_loop(client, Toolbox([started()]), max_turns=3)


def test_a_refusal_from_the_mesh_is_reported(started):
    def refuses(request):
        return httpx.Response(400, json={"detail": "this model's chat template has no tool format"})

    client = httpx.Client(transport=httpx.MockTransport(refuses), base_url="http://mesh")
    with pytest.raises(ServerError, match="no tool format"):
        run_loop(client, Toolbox([started()]))


def agent_arguments(*servers: str, **options) -> SimpleNamespace:
    defaults = {"url": "http://mesh", "token": None, "model": "small", "prompt": "go", "server": list(servers), "allow": [], "pass_env": [], "max_turns": 8, "max_tokens": None, "tool_timeout": 10.0}
    return SimpleNamespace(**{**defaults, **options})


def test_the_command_prints_the_answer_and_nothing_else_on_stdout(monkeypatch, capsys):
    client, _ = scripted_model([("add", {"a": 40, "b": 2})], "It is 42.")
    monkeypatch.setattr(mcp_client.httpx, "Client", lambda **options: client)
    assert run_agent(agent_arguments(server_command())) == 0
    captured = capsys.readouterr()
    assert captured.out == "It is 42.\n"
    assert "Calling add" in captured.err


def test_the_command_refuses_to_allow_a_tool_nobody_offers(capsys):
    assert run_agent(agent_arguments(server_command(), allow=["rm_rf"])) == 1
    assert "--allow names rm_rf" in capsys.readouterr().err


def test_the_command_says_when_a_server_will_not_start(capsys):
    assert run_agent(agent_arguments("this-program-does-not-exist-anywhere")) == 1
    assert "could not start" in capsys.readouterr().err


def test_the_key_goes_to_the_mesh_and_never_to_a_server(monkeypatch, capsys):
    seen = {}
    client, _ = scripted_model("Hello.")

    def build(**options):
        seen.update(options)
        return client

    monkeypatch.setattr(mcp_client.httpx, "Client", build)
    assert run_agent(agent_arguments(server_command(), token="a-secret-key")) == 0
    assert seen["headers"] == {"Authorization": "Bearer a-secret-key"}
    assert "a-secret-key" not in "".join(capsys.readouterr())


def parse(*words: str):
    return cli.build_parser().parse_args(list(words))


def test_the_commands_parse():
    served = parse("mcp", "--url", "http://x", "--token", "k")
    assert (served.url, served.token, served.handler) == ("http://x", "k", cli.run_mcp)
    run = parse("mcp", "run", "--url", "http://x", "--token", "k", "--model", "m", "--server", "a", "--server", "b", "--allow", "t", "ask this")
    assert (run.url, run.token, run.model, run.server, run.allow, run.prompt) == ("http://x", "k", "m", ["a", "b"], ["t"], "ask this")
    assert run.handler == cli.run_mcp_agent


def test_options_given_before_run_still_count():
    run = parse("mcp", "--url", "http://x", "--token", "k", "run", "--model", "m", "--server", "a", "ask")
    assert (run.url, run.token) == ("http://x", "k")


def test_run_needs_a_model_and_a_server(capsys):
    with pytest.raises(SystemExit):
        parse("mcp", "run", "ask")
    assert "--model" in capsys.readouterr().err


# --- the official client

OFFICIAL_PYTHON = os.environ.get("ISHARDED_TEST_MCP_PYTHON", "")
needs_official = pytest.mark.skipif(not (OFFICIAL_PYTHON and Path(OFFICIAL_PYTHON).is_file()), reason="needs a Python with the official mcp package")
OFFICIAL_SCRIPT = """
import asyncio, json, sys
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

async def exercise(client):
    tools = await client.list_tools()
    resources = await client.list_resources()
    route = await client.read_resource("isharded://mesh/route")
    prompts = await client.list_prompts()
    prompt = await client.get_prompt("mesh_health")
    models = await client.call_tool("list_models", {})
    bad = await client.call_tool("chat", {})
    try:
        await client.call_tool("no_such_tool", {})
        unknown = None
    except Exception as exc:
        unknown = str(exc)
    return {"protocol": client.protocol_version, "tools": [tool.name for tool in tools.tools], "models_error": models.is_error, "bad_error": bad.is_error, "bad_text": bad.content[0].text, "unknown": unknown, "resources": sorted(str(item.uri) for item in resources.resources), "route": route.contents[0].text, "prompts": [item.name for item in prompts.prompts], "prompt_text": prompt.messages[0].content.text}

async def main():
    python, cwd, url, token_file = sys.argv[1:5]
    env = {"ISHARDED_POOL_TOKEN_FILE": token_file, "PYTHONUTF8": "1", "SYSTEMROOT": sys.argv[5]}
    parameters = StdioServerParameters(command=python, args=["-m", "isharded", "mcp", "--url", url], env=env, cwd=cwd)
    results = {}
    for mode in ("auto", "legacy"):
        async with Client(stdio_client(parameters), mode=mode) as client:
            results["stdio " + mode] = await exercise(client)
        async with Client(url + "/mcp", mode=mode) as client:
            results["http " + mode] = await exercise(client)
    print(json.dumps(results))

asyncio.run(main())
"""
OFFICIAL_SERVER = """
from mcp.server import MCPServer

server = MCPServer("official-test")

@server.tool()
def multiply(a: int, b: int) -> int:
    "Multiply two numbers"
    return a * b

server.run("stdio")
"""


@needs_official
def test_the_official_client_works_in_both_eras_over_both_transports(real_socket, tmp_path):
    script = tmp_path / "official.py"
    script.write_text(OFFICIAL_SCRIPT, encoding="utf-8")
    command = [OFFICIAL_PYTHON, str(script), sys.executable, str(ROOT), real_socket, str(tmp_path / "no-token"), os.environ.get("SYSTEMROOT", "")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-1500:]
    results = json.loads(done.stdout.strip().splitlines()[-1])
    assert len(results) == 4
    for name, result in results.items():
        assert result["protocol"] == (MODERN if name.endswith("auto") else LEGACY_VERSIONS[0]), name
        assert result["tools"] == ["chat", "embed", "list_models", "mesh_status"], name
        assert result["models_error"] is False, name
        assert result["bad_error"] is True and "prompt is required" in result["bad_text"], name
        assert "Unknown tool" in result["unknown"], name
        assert result["resources"] == ["isharded://mesh/model", "isharded://mesh/route", "isharded://mesh/status"], name
        assert json.loads(result["route"])["stages"] == [], name
        assert result["prompts"] == ["mesh_health"] and "plain words" in result["prompt_text"], name


@needs_official
def test_our_client_works_with_an_official_server(tmp_path):
    script = tmp_path / "official_server.py"
    script.write_text(OFFICIAL_SERVER, encoding="utf-8")
    server = StdioServer(f'"{OFFICIAL_PYTHON}" "{script}"', server_environment([]))
    try:
        server.start()
        assert server.modern_version == MODERN
        toolbox = Toolbox([server])
        assert list(toolbox.tools) == ["multiply"]
        assert toolbox.call("multiply", {"a": 6, "b": 7}, 30) == "42"
        assert "error" in toolbox.call("multiply", {"a": "six", "b": 7}, 30).lower()
    finally:
        server.close()


# --- resources and prompts


def status_with_secrets(request: httpx.Request) -> httpx.Response:
    """What /admin/status really holds, which is more than a resource should pass on."""
    if request.url.path != "/admin/status":
        return fake_mesh(request)
    stage = {"worker_id": "a-cpu", "url": "http://10.0.0.9:8301", "start_layer": 0, "end_layer": 28, "placements": [{"unit": "x"}]}
    plan = {"architecture": "qwen3", "layers": 28, "experts": 0, "model_bytes": 1234, "mmproj": "/home/someone/projector.gguf", "draft_model": "/home/someone/draft.gguf"}
    status = {"status": "ready", "model": "small", "kind": "chat", "vision": False, "epoch": 3, "route": [stage], "active_plan": plan, "cache_url": "http://10.0.0.9:9000"}
    return httpx.Response(200, json=status)


def read_resource(target: McpHandler, uri: str) -> dict:
    return json.loads(modern(target, "resources/read", {"uri": uri})["result"]["contents"][0]["text"])


def test_the_mesh_lists_its_resources_and_prompts_in_both_eras():
    uris = {item["uri"] for item in modern(handler(), "resources/list")["result"]["resources"]}
    assert uris == {"isharded://mesh/status", "isharded://mesh/route", "isharded://mesh/model"}
    assert rpc(ready(), "resources/list")["result"]["resources"][0]["mimeType"] == "application/json"
    assert [item["name"] for item in rpc(ready(), "prompts/list")["result"]["prompts"]] == ["mesh_health"]
    assert rpc(ready(), "resources/templates/list")["result"] == {"resourceTemplates": []}
    assert modern(handler(), "prompts/list")["result"]["cacheScope"] == "private"


def test_initialize_says_resources_and_prompts_are_there():
    capabilities = rpc(handler(), "initialize", {"protocolVersion": LEGACY_VERSIONS[0]})["result"]["capabilities"]
    assert capabilities["resources"] == {"listChanged": False} and capabilities["prompts"] == {"listChanged": False}


def test_a_legacy_session_cannot_read_before_it_says_hello():
    assert rpc(handler(), "resources/read", {"uri": "isharded://mesh/status"})["error"]["code"] == INVALID_REQUEST


def test_the_status_resource_says_what_the_tool_says():
    tool = rpc(ready(), "tools/call", {"name": "mesh_status", "arguments": {}})["result"]["structuredContent"]
    assert read_resource(handler(), "isharded://mesh/status") == tool


def test_the_route_resource_names_workers_and_layers_and_no_addresses():
    route = read_resource(handler(status_with_secrets), "isharded://mesh/route")
    assert route == {"model": "small", "epoch": 3, "stages": [{"worker_id": "a-cpu", "start_layer": 0, "end_layer": 28}]}


def test_the_model_resource_describes_the_model_and_no_paths():
    model = read_resource(handler(status_with_secrets), "isharded://mesh/model")
    assert model == {"model": "small", "kind": "chat", "vision": False, "dimensions": None, "architecture": "qwen3", "layers": 28, "experts": 0, "model_bytes": 1234}


@pytest.mark.parametrize("uri", ["isharded://mesh/status", "isharded://mesh/route", "isharded://mesh/model"])
def test_no_resource_passes_on_an_address_or_a_path(uri):
    contents = modern(handler(status_with_secrets), "resources/read", {"uri": uri})["result"]["contents"][0]
    assert contents["uri"] == uri and contents["mimeType"] == "application/json"
    assert "10.0.0.9" not in contents["text"] and "/home/someone" not in contents["text"]


def test_an_unknown_or_missing_resource_is_a_protocol_error():
    assert modern(handler(), "resources/read", {"uri": "isharded://mesh/keys"})["error"]["code"] == RESOURCE_NOT_FOUND
    assert modern(handler(), "resources/read")["error"]["code"] == INVALID_PARAMS


def test_a_resource_says_when_the_mesh_will_not_answer():
    def down(request):
        raise httpx.ConnectError("refused")

    error = modern(handler(down), "resources/read", {"uri": "isharded://mesh/status"})["error"]
    assert error["code"] == INTERNAL_ERROR and "did not answer" in error["message"]


def test_the_prompt_holds_the_status_as_it_is_now():
    result = modern(handler(), "prompts/get", {"name": "mesh_health"})["result"]
    (message,) = result["messages"]
    assert message["role"] == "user" and '"coordinator": "a"' in message["content"]["text"]
    assert modern(handler(), "prompts/get", {"name": "nothing"})["error"]["code"] == INVALID_PARAMS


def test_resources_are_behind_the_same_keys_and_pay_once(site):
    key = keyed(site, requests_per_minute=1)
    assert legacy_post(site, "resources/read", {"uri": "isharded://mesh/status"}).status_code == 401
    bearer = {"Authorization": f"Bearer {key}"}
    assert json.loads(legacy_post(site, "resources/read", {"uri": "isharded://mesh/status"}, **bearer).json()["result"]["contents"][0]["text"])["status"] == "ready"
    limited = legacy_post(site, "resources/read", {"uri": "isharded://mesh/status"}, **bearer).json()["error"]
    assert limited["code"] == INTERNAL_ERROR and "429" in limited["message"]


def test_a_resource_read_mirrors_its_uri_in_the_name_header(site):
    uri = {"uri": "isharded://mesh/route"}
    assert modern_post(site, "resources/read", uri).status_code == 200
    answer = modern_post(site, "resources/read", uri, **{"Mcp-Name": "isharded://mesh/status"})
    assert answer.status_code == 400 and answer.json()["error"]["code"] == HEADER_MISMATCH


def test_a_prompt_mirrors_its_name_in_the_name_header(site):
    assert modern_post(site, "prompts/get", {"name": "mesh_health"}).status_code == 200
    assert modern_post(site, "prompts/get", {"name": "mesh_health"}, **{"Mcp-Name": "other"}).status_code == 400


# --- cancelling over HTTP, on a real socket, since a mock cannot drop a connection


@pytest.fixture
def live(site):
    """The real app on a real socket, with a model that stays silent until told to stop."""
    entered, closed = threading.Event(), threading.Event()

    async def silent_stream(session, *rest):
        entered.set()
        try:
            await asyncio.sleep(60)
            yield 7
        finally:
            closed.set()

    site.node.token_stream = silent_stream
    server = uvicorn.Server(uvicorn.Config(orchestrator.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    yield SimpleNamespace(site=site, port=port, url=f"http://127.0.0.1:{port}/mcp", entered=entered, closed=closed)
    server.should_exit = True
    thread.join(15)


CHAT_CALL = {"name": "chat", "arguments": {"prompt": "Ping?"}}


def legacy_call(identifier=7) -> dict:
    return {"jsonrpc": "2.0", "id": identifier, "method": "tools/call", "params": CHAT_CALL}


def cancel(url: str, identifier=7, **headers) -> int:
    message = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": identifier}}
    return httpx.post(url, json=message, headers={"Accept": WANTS, **headers}, timeout=15, trust_env=False).status_code


def post_in_background(url: str, message: dict, **headers) -> tuple[threading.Thread, list]:
    replies: list = []

    def run() -> None:
        replies.append(httpx.post(url, json=message, headers={"Accept": WANTS, **headers}, timeout=30, trust_env=False))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, replies


def open_and_leave(live, message: dict, headers: dict) -> socket.socket:
    """Send a request by hand, which is the only way to walk away before the reply."""
    body = json.dumps(message).encode()
    lines = ["POST /mcp HTTP/1.1", "Host: 127.0.0.1", "Content-Type: application/json", f"Accept: {WANTS}", f"Content-Length: {len(body)}", *(f"{key}: {value}" for key, value in headers.items())]
    connection = socket.create_connection(("127.0.0.1", live.port), timeout=15)
    connection.sendall("\r\n".join(lines).encode() + b"\r\n\r\n" + body)
    assert live.entered.wait(15), "the model should have been asked"
    return connection


def test_a_legacy_cancel_stops_the_model_and_answers_the_request(live):
    thread, replies = post_in_background(live.url, legacy_call())
    assert live.entered.wait(15)
    assert cancel(live.url) == 202
    assert live.closed.wait(5), "the model should stop"
    thread.join(15)
    assert replies[0].status_code == 200
    assert replies[0].json() == {"jsonrpc": "2.0", "id": 7, "error": {"code": REQUEST_CANCELLED, "message": "Request cancelled"}}
    deadline = time.monotonic() + 5
    while live.site.node.replies_in_flight and time.monotonic() < deadline:
        time.sleep(0.02)
    assert live.site.node.replies_in_flight == 0


def test_a_modern_client_cancels_by_closing_the_connection(live):
    message = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {**CHAT_CALL, "_meta": META}}
    connection = open_and_leave(live, message, {"MCP-Protocol-Version": MODERN, "Mcp-Method": "tools/call", "Mcp-Name": "chat"})
    connection.close()
    assert live.closed.wait(5), "a dropped connection is how the modern era cancels"


def test_a_dropped_connection_is_no_cancel_in_the_older_era(live):
    connection = open_and_leave(live, legacy_call(), {})
    connection.close()
    assert not live.closed.wait(1), "the older era says a dropped connection is not a cancel"
    assert cancel(live.url) == 202
    assert live.closed.wait(5)


def test_one_key_cannot_cancel_what_another_asked(live):
    site = live.site
    path = site.tmp / "keys.json"
    mine, theirs = create_key(str(path), "me", "user", 0), create_key(str(path), "them", "user", 0)
    site.node.access = AccessControl.load(None, str(path))
    thread, replies = post_in_background(live.url, legacy_call(), Authorization=f"Bearer {mine}")
    assert live.entered.wait(15)
    assert cancel(live.url, Authorization=f"Bearer {theirs}") == 202
    assert not live.closed.wait(0.5), "another caller's cancel must reach nothing"
    assert cancel(live.url, Authorization="Bearer wrong") == 401
    assert cancel(live.url, Authorization=f"Bearer {mine}") == 202
    assert live.closed.wait(5)
    thread.join(15)


def test_a_cancel_for_nothing_is_accepted_quietly(site):
    for identifier in (99, "nothing", [1], {"a": 1}, None):
        assert site.client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": identifier}}, headers={"Accept": WANTS}).status_code == 202


def test_when_two_requests_share_an_id_a_cancel_stops_neither():
    async def scene() -> None:
        running = InFlight()
        first, second = (asyncio.create_task(asyncio.sleep(30)) for _ in range(2))
        running.add("me", 1, first)
        running.add("me", 1, second)
        running.cancel("me", 1)
        await asyncio.sleep(0)
        assert not first.cancelled() and not second.cancelled()
        second.cancel()
        await asyncio.wait({second})
        await asyncio.sleep(0)
        running.cancel("me", 1)
        await asyncio.wait({first})
        assert first.cancelled() and running.all() == []

    asyncio.run(scene())


def test_stdio_survives_an_id_it_cannot_use(stdio):
    pipes, thread = stdio(httpx.MockTransport(fake_mesh))
    pipes.send({"jsonrpc": "2.0", "id": [1], "method": "ping"})
    pipes.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": {"a": 1}}})
    pipes.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert pipes.wait_for(2)["result"] == {}
    pipes.incoming.put(b"")
    thread.join(5)


# --- mcp run over HTTP


class FarServer:
    """A Streamable HTTP server, written out by hand, that keeps what it was sent."""

    def __init__(self, era: str = "legacy") -> None:
        self.era = era
        self.seen: list[tuple[str, httpx.Headers, dict | None]] = []

    def reply(self, identifier, result: dict, **headers) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": identifier, "result": result}, headers=headers)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content) if request.content else None
        self.seen.append((request.method, request.headers, message))
        if request.method == "DELETE" or message is None or "id" not in message or "method" not in message:
            return httpx.Response(202)
        identifier, method = message["id"], message["method"]
        if method == "server/discover":
            if self.era == "legacy":
                return httpx.Response(400, text="Bad Request: no session")
            return self.reply(identifier, {"supportedVersions": [MODERN], "capabilities": {"tools": {}}, "_meta": {SERVER_KEY: {"name": "far"}}, "resultType": "complete"})
        if method == "initialize":
            return self.reply(identifier, {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "far"}}, **{"Mcp-Session-Id": "session-1"})
        if method == "tools/list":
            return self.reply(identifier, {"tools": [{"name": "search", "annotations": {"readOnlyHint": True}}, {"name": "hello wörld"}]})
        events = [
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}},
            {"jsonrpc": "2.0", "id": 99, "method": "sampling/createMessage", "params": {}},
            {"jsonrpc": "2.0", "id": identifier, "result": {"content": [{"type": "text", "text": "found"}]}},
        ]
        body = "".join(f"event: message\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        return httpx.Response(200, content=b": a comment\r\n\r\nnot json\n\ndata: [1]\n\n" + body, headers={"Content-Type": "text/event-stream"})

    def calls(self, method: str) -> list:
        return [(headers, message) for _, headers, message in self.seen if message and message.get("method") == method]


def far_server(era: str = "legacy") -> tuple[FarServer, HttpServer]:
    far = FarServer(era)
    server = HttpServer("http://far.example/mcp", transport=httpx.MockTransport(far))
    server.start()
    return far, server


def test_an_http_server_of_the_older_era_gets_a_handshake_and_a_session():
    far, server = far_server("legacy")
    assert server.modern_version is None and server.name == "far"
    assert "mcp-protocol-version" not in far.calls("initialize")[0][0]
    notified = far.calls("notifications/initialized")[0][0]
    assert notified["mcp-session-id"] == "session-1" and notified["mcp-protocol-version"] == "2025-06-18"
    toolbox = Toolbox([server])
    assert toolbox.call("search", {}, 5) == "found"
    headers, message = far.calls("tools/call")[0]
    assert headers["mcp-session-id"] == "session-1" and headers["accept"] == "application/json, text/event-stream"
    assert "_meta" not in message["params"]
    server.close()
    assert far.seen[-1][0] == "DELETE" and far.seen[-1][1]["mcp-session-id"] == "session-1"


def test_an_http_server_of_the_current_era_gets_every_mirrored_header():
    far, server = far_server("modern")
    assert server.modern_version == MODERN and server.name == "far"
    assert not far.calls("initialize")
    Toolbox([server]).call("hello wörld", {}, 5)
    headers, message = far.calls("tools/call")[0]
    assert headers["mcp-protocol-version"] == MODERN and headers["mcp-method"] == "tools/call"
    assert header_value(headers["mcp-name"]) == "hello wörld"
    assert message["params"]["_meta"][VERSION_KEY] == MODERN
    server.close()
    assert far.seen[-1][0] != "DELETE", "a modern server has no session to end"


def test_a_question_on_a_stream_is_answered_with_a_no():
    far, server = far_server()
    server.request("tools/call", {"name": "search", "arguments": {}}, 5)
    answer = next(message for _, _, message in far.seen if message and message.get("id") == 99 and "error" in message)
    assert answer["error"]["code"] == -32601


def test_a_name_a_header_cannot_carry_goes_as_base64_and_a_plain_one_does_not():
    assert as_header("chat") == "chat"
    for awkward in ("héllo", " spaced", "line\nbreak"):
        assert as_header(awkward).startswith("=?base64?") and header_value(as_header(awkward)) == awkward


def test_an_http_server_that_refuses_a_tool_says_why():
    def refuses(request):
        message = json.loads(request.content)
        return httpx.Response(400, json={"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32602, "message": "no such tool"}})

    server = HttpServer("http://far.example/mcp", transport=httpx.MockTransport(refuses))
    with pytest.raises(mcp_client.ToolRefused, match="no such tool"):
        server.request("tools/call", {"name": "x"}, 5)


@pytest.mark.parametrize("answer, words", [(httpx.Response(401, text="who are you"), "401"), (httpx.Response(202), "202"), (httpx.Response(307, headers={"Location": "http://elsewhere"}), "307")])
def test_an_http_server_that_gives_no_reply_is_named(answer, words):
    server = HttpServer("http://far.example/mcp", transport=httpx.MockTransport(lambda request: answer))
    with pytest.raises(ServerError, match=words):
        server.start()


def test_a_stream_that_ends_without_a_reply_is_a_failure():
    empty = httpx.Response(200, content=b"data: {}\n\n", headers={"Content-Type": "text/event-stream"})
    server = HttpServer("http://far.example/mcp", transport=httpx.MockTransport(lambda request: empty))
    with pytest.raises(ServerError, match="without answering"):
        server.request("tools/list", {}, 5)


def test_one_enormous_event_is_refused(monkeypatch):
    monkeypatch.setattr(mcp_client, "MAX_LINE_BYTES", 100)
    big = httpx.Response(200, content=b"data: " + b"x" * 500, headers={"Content-Type": "text/event-stream"})
    server = HttpServer("http://far.example/mcp", transport=httpx.MockTransport(lambda request: big))
    with pytest.raises(ServerError, match="larger than"):
        server.request("tools/list", {}, 5)


@pytest.mark.parametrize("url, words", [("http://[nonsense", "is not a URL"), ("http://host:port/mcp", "could not reach")])
def test_a_url_that_makes_no_sense_is_said_plainly(url, words, capsys):
    assert run_agent(agent_arguments(url)) == 1
    assert words in capsys.readouterr().err


def test_a_server_that_cannot_be_reached_is_named():
    def down(request):
        raise httpx.ConnectError("refused")

    server = HttpServer("http://far.example/mcp", transport=httpx.MockTransport(down))
    with pytest.raises(ServerError, match="could not reach"):
        server.start()


@pytest.fixture
def bad_socket():
    """A real listener that takes requests and behaves badly, as ``behave(connection)`` says."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.1)
    stop = threading.Event()

    def serve(behave) -> None:
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                continue
            threading.Thread(target=behave, args=(connection,), daemon=True).start()

    def start(behave) -> str:
        threading.Thread(target=serve, args=(behave,), daemon=True).start()
        return f"http://127.0.0.1:{listener.getsockname()[1]}/mcp"

    yield start
    stop.set()
    listener.close()


def test_a_silent_server_on_a_real_socket_runs_out_of_time(bad_socket):
    def silent(connection):
        connection.recv(65536)
        time.sleep(10)

    server = HttpServer(bad_socket(silent))
    began = time.monotonic()
    with pytest.raises(ServerError, match="longer than 0.5"):
        server.request("tools/list", {}, 0.5)
    assert time.monotonic() - began < 5


def test_a_server_that_only_drips_bytes_runs_out_of_time_too(bad_socket):
    def drip(connection):
        connection.recv(65536)
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
        try:
            for _ in range(200):
                connection.sendall(b"3\r\n: x\r\n")
                time.sleep(0.05)
        except OSError:
            pass

    server = HttpServer(bad_socket(drip))
    began = time.monotonic()
    with pytest.raises(ServerError, match="longer than 0.5"):
        server.request("tools/list", {}, 0.5)
    assert time.monotonic() - began < 5


class LegacyHttp(HttpServer):
    def _probe(self):
        return None


@pytest.mark.parametrize("kind, version", [(HttpServer, MODERN), (LegacyHttp, None)])
def test_our_http_client_reaches_a_real_mesh_in_both_eras(real_socket, kind, version):
    server = kind(real_socket + "/mcp")
    try:
        server.start()
        assert server.modern_version == version and server.name == "isharded"
        toolbox = Toolbox([server])
        assert set(toolbox.tools) == {tool["name"] for tool in TOOLS}
        assert json.loads(toolbox.call("list_models", {}, 20)) == {"models": []}
    finally:
        server.close()


def test_the_command_takes_a_url_as_a_server(real_socket, monkeypatch, capsys):
    real = httpx.Client
    client, asked = scripted_model([("list_models", {})], "None are loaded.")
    monkeypatch.setattr(mcp_client.httpx, "Client", lambda **options: client if "base_url" in options else real(**options))
    assert run_agent(agent_arguments(real_socket + "/mcp")) == 0
    assert capsys.readouterr().out == "None are loaded.\n"
    assert tool_results(asked) == ['{"models": []}']


def test_a_mesh_server_and_a_stdio_server_can_work_side_by_side(real_socket, monkeypatch, capsys):
    real = httpx.Client
    client, asked = scripted_model([("add", {"a": 1, "b": 2}), ("list_models", {})], "Done.")
    monkeypatch.setattr(mcp_client.httpx, "Client", lambda **options: client if "base_url" in options else real(**options))
    assert run_agent(agent_arguments(server_command(), real_socket + "/mcp")) == 0
    assert tool_results(asked) == ["3", '{"models": []}']


def test_a_stdio_server_still_negotiates_down_to_a_strict_older_version(started):
    server = started("--only-version", "2025-06-18")
    assert server.modern_version is None and server.legacy_version == "2025-06-18"
    assert Toolbox([server]).call("add", {"a": 1, "b": 1}, 5) == "2"


def test_a_server_that_refuses_every_version_is_refused(started, tmp_path):
    with pytest.raises(mcp_client.ToolRefused, match="unsupported"):
        started("--only-version", "1999-01-01")


# --- a reply cut off at the token limit


def cut_model(text: str):
    def answer(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "length"}]})

    return httpx.Client(transport=httpx.MockTransport(answer), base_url="http://mesh")


def test_a_reply_cut_off_at_the_limit_is_not_passed_off_as_finished(started):
    with pytest.raises(CutOff, match="token limit") as caught:
        run_loop(cut_model("<think>x</think>\n\nThe answer is"), Toolbox([started()]))
    assert caught.value.answer == "The answer is"


def test_the_command_prints_what_it_has_and_says_it_is_not_finished(monkeypatch, capsys):
    client = cut_model("The answer is")
    monkeypatch.setattr(mcp_client.httpx, "Client", lambda **options: client)
    assert run_agent(agent_arguments(server_command())) == 1
    captured = capsys.readouterr()
    assert captured.out == "The answer is\n", "stdout stays for the answer alone"
    assert "cut off" in captured.err and "--max-tokens" in captured.err


def test_a_call_written_whole_before_the_limit_still_runs(started):
    calls = [{"id": "call_0", "type": "function", "function": {"name": "add", "arguments": json.dumps({"a": 1, "b": 2})}}]
    replies = [{"message": {"role": "assistant", "content": None, "tool_calls": calls}, "finish_reason": "length"}, {"message": {"role": "assistant", "content": "3"}, "finish_reason": "stop"}]

    def answer(request):
        return httpx.Response(200, json={"choices": [replies.pop(0)]})

    assert run_loop(httpx.Client(transport=httpx.MockTransport(answer), base_url="http://mesh"), Toolbox([started()])) == "3"


# --- tool names that collide


class FakeServer(McpConnection):
    """A server's answers, written down, for what does not need a process."""

    def __init__(self, name: str, tools=(), result=None) -> None:
        self.name = name
        self.names = list(tools)
        self.result = result or {}

    def list_tools(self):
        return [{"name": name} for name in self.names]

    def request(self, method, params, timeout):
        return self.result


def test_a_renamed_tool_never_replaces_the_tool_already_there(capsys):
    first, second = FakeServer("a", ["search", "b_search"]), FakeServer("b", ["search"])
    toolbox = Toolbox([first, second])
    assert set(toolbox.tools) == {"search", "b_search", "b_search_2"}
    assert toolbox.tools["b_search"][0] is first and toolbox.tools["b_search_2"][0] is second
    assert toolbox.tools["b_search_2"][1]["name"] == "search", "the server still hears its own name"
    assert '"b" offers that tool as "b_search_2"' in capsys.readouterr().err


def test_three_servers_with_one_name_all_stay(capsys):
    servers = [FakeServer("same", ["search"]) for _ in range(3)]
    toolbox = Toolbox(servers)
    assert set(toolbox.tools) == {"search", "same_search", "same_search_2"}
    assert [toolbox.tools[name][0] for name in ("search", "same_search", "same_search_2")] == servers


def test_a_rename_is_said_in_a_way_a_tool_cannot_use_to_write_to_the_terminal(capsys):
    Toolbox([FakeServer("a", ["x\x1b[2J"]), FakeServer("b\x1b[2J", ["x\x1b[2J"])])
    assert "\x1b" not in capsys.readouterr().err


# --- a tool that asks for more


def test_a_tool_that_needs_input_gets_an_honest_refusal():
    asks = {"who": {"method": "elicitation/create", "params": {"message": "Which account?"}}, "more": {"method": "sampling/createMessage"}}
    server = FakeServer("a", ["pay"], {"resultType": "input_required", "inputRequests": asks, "requestState": "opaque"})
    said = Toolbox([server]).call("pay", {}, 5)
    assert "did not finish" in said and "Which account?" in said and "sampling/createMessage" in said
    assert "tell the person" in said


@pytest.mark.parametrize("asks", [None, [], {"a": "junk"}, {"a": {"params": "junk", "method": "m"}}])
def test_a_refusal_survives_whatever_a_server_says_it_wanted(asks):
    result = {"resultType": "input_required", "inputRequests": asks}
    assert "did not finish" in Toolbox([FakeServer("a", ["pay"], result)]).call("pay", {}, 5)


def test_what_a_server_asks_cannot_flood_the_model():
    asks = {"a": {"params": {"message": "x" * 100_000}}}
    assert len(Toolbox([FakeServer("a", ["pay"], {"resultType": "input_required", "inputRequests": asks})]).call("pay", {}, 5)) < 1000


