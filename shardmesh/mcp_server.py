"""The mesh as an MCP server, over stdio and over Streamable HTTP.

MCP changed shape in revision 2026-07-28: no ``initialize``, no sessions, and every
request names its version in ``_meta``. Clients of both eras are in use, so this
serves both, which the specification calls dual-era.

Every tool here is a call to the mesh's own HTTP API, with the caller's own key.
So a reply reached through MCP takes the same path as any other, through lanes,
followers, and failover, and a key can do nothing here that it could not do there.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import sys
from typing import Any, BinaryIO, Callable, Mapping

import httpx

from . import __version__
from .toolcalls import without_reasoning

MODERN_VERSIONS = ("2026-07-28",)
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SUPPORTED_VERSIONS = (*MODERN_VERSIONS, *LEGACY_VERSIONS)
# What the specification says to assume of a client that sends no version header.
HEADERLESS_VERSION = "2025-03-26"
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
SERVER_KEY = "io.modelcontextprotocol/serverInfo"
SERVER_INFO = {"name": "shardmesh", "version": __version__}
INSTRUCTIONS = "Runs language models across the machines of a ShardMesh pool."
# The list changes only with the software, so a client may keep it for a while.
TOOL_LIST_TTL_MS = 3_600_000
CHAT_REPLY_TOKENS = 1024
STOP_GRACE_SECONDS = 5.0

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
HEADER_MISMATCH = -32020
UNSUPPORTED_VERSION = -32022


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def body(self, identifier: Any) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return {"jsonrpc": "2.0", "id": identifier, "error": error}


class ToolFailed(Exception):
    """The request was fine and the work failed. A model can read this and try again."""


def unsupported(requested: Any) -> RpcError:
    return RpcError(UNSUPPORTED_VERSION, "Unsupported protocol version", {"supported": list(SUPPORTED_VERSIONS), "requested": requested})


READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
TOOLS: list[dict[str, Any]] = [
    {
        "name": "chat",
        "title": "Ask a model",
        "description": "Ask a model running on the mesh and get its answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to ask"},
                "system": {"type": "string", "description": "Instructions for the model"},
                "model": {"type": "string", "description": "Which model; list_models shows them. Default is the main one"},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 32768, "description": f"Longest reply. Default {CHAT_REPLY_TOKENS}"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 5, "description": "0 picks the most likely token. Default 0"},
                "enable_thinking": {"type": "boolean", "description": "Let the model reason first, on models that support it"},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "list_models",
        "title": "List models",
        "description": "List the models this mesh is running now.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": READ_ONLY,
    },
    {
        "name": "mesh_status",
        "title": "Mesh status",
        "description": "Whether the mesh can answer, what it runs, and which nodes hold which stages.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": READ_ONLY,
    },
]
JSON_KINDS: dict[str, tuple[type, ...]] = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}


def check_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            raise ToolFailed(f"{name} is required")
    for name, value in arguments.items():
        if name not in properties:
            raise ToolFailed(f"{name} is not an argument of this tool; it takes {', '.join(sorted(properties)) or 'none'}")
        rule = properties[name]
        # JSON has no bool-is-an-int rule, though Python does.
        if not isinstance(value, JSON_KINDS[rule["type"]]) or (isinstance(value, bool) and rule["type"] != "boolean"):
            raise ToolFailed(f"{name} must be a {rule['type']}")
        if "minimum" in rule and value < rule["minimum"] or "maximum" in rule and value > rule["maximum"]:
            raise ToolFailed(f"{name} must be from {rule['minimum']} to {rule['maximum']}")


class MeshTools:
    """The tools, as calls to a mesh's HTTP API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def _ask(self, method: str, path: str, **options: Any) -> dict[str, Any]:
        try:
            response = await self.client.request(method, path, **options)
        except httpx.HTTPError as exc:
            raise ToolFailed(f"the mesh did not answer: {exc}. Check that the node is running and that this address is right") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            raise ToolFailed(f"the mesh refused this ({response.status_code}): {body.get('detail') or response.text[:300]}")
        return body

    async def chat(self, arguments: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "system", "content": arguments["system"]}] if "system" in arguments else []
        messages.append({"role": "user", "content": arguments["prompt"]})
        request = {key: arguments[key] for key in ("model", "temperature", "enable_thinking") if key in arguments}
        request |= {"messages": messages, "max_tokens": arguments.get("max_tokens", CHAT_REPLY_TOKENS)}
        reply = await self._ask("POST", "/v1/chat/completions", json=request)
        choice = reply["choices"][0]
        answer = without_reasoning(choice["message"].get("content") or "")
        cut = "\n\n[The reply reached max_tokens and was cut off.]" if choice["finish_reason"] == "length" else ""
        details = {"answer": answer, "model": reply.get("model"), "finish_reason": choice["finish_reason"], "usage": reply.get("usage")}
        return {"content": [{"type": "text", "text": answer + cut}], "structuredContent": details}

    async def list_models(self, arguments: dict[str, Any]) -> dict[str, Any]:
        reply = await self._ask("GET", "/v1/models")
        return as_result({"models": [item["id"] for item in reply.get("data", [])]})

    async def mesh_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        status = await self._ask("GET", "/admin/status")
        workers = status.get("workers", {})
        summary = {
            "status": status.get("status"),
            "model": status.get("model"),
            "epoch": status.get("epoch"),
            "coordinator": (status.get("mesh") or {}).get("leader"),
            "nodes": (status.get("mesh") or {}).get("members"),
            "route": [stage.get("worker_id") for stage in status.get("route", [])],
            "workers": {name: worker.get("state") for name, worker in workers.items()},
            "lanes": [{key: lane.get(key) for key in ("name", "model", "ready", "workers")} for lane in status.get("lanes", [])],
            "tokens_per_second": (status.get("metrics") or {}).get("tokens_per_second_ema"),
        }
        return as_result(summary)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = next((tool for tool in TOOLS if tool["name"] == name), None)
        if tool is None:
            raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        try:
            check_arguments(tool["inputSchema"], arguments)
            return await getattr(self, name)(arguments)
        except ToolFailed as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}


def as_result(value: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}], "structuredContent": value}


class McpHandler:
    """Answers JSON-RPC messages, whichever transport brought them.

    A connection that must say ``initialize`` first is one that keeps state, which
    stdio does. Over HTTP each request stands alone, so ``stateless`` is set.
    """

    def __init__(self, tools: MeshTools, stateless: bool = False) -> None:
        self.tools = tools
        self.ready = stateless

    async def handle(self, message: Any) -> dict[str, Any] | None:
        """The reply to one message, or None if it wants none."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return RpcError(INVALID_REQUEST, "Send one JSON-RPC 2.0 object").body(None)
        if "method" not in message or "id" not in message:
            return None  # a notification, or the answer to a request we never send
        identifier = message["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
            return RpcError(INVALID_REQUEST, "A request id is a string or a number").body(None)
        try:
            return {"jsonrpc": "2.0", "id": identifier, "result": await self._answer(message)}
        except RpcError as exc:
            return exc.body(identifier)

    async def _answer(self, message: dict[str, Any]) -> dict[str, Any]:
        method, params = message["method"], message.get("params", {})
        if not isinstance(method, str):
            raise RpcError(INVALID_REQUEST, "A method is a string")
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "params must be an object")
        meta = params.get("_meta")
        version = meta.get(VERSION_KEY) if isinstance(meta, dict) else None
        if version is not None and version not in SUPPORTED_VERSIONS:
            raise unsupported(version)
        modern = version in MODERN_VERSIONS
        if method == "initialize" and not modern:
            return self._initialize(params)
        if method == "server/discover":
            result: dict[str, Any] = {
                "supportedVersions": list(SUPPORTED_VERSIONS),
                "capabilities": {"tools": {}},
                "instructions": INSTRUCTIONS,
                "ttlMs": TOOL_LIST_TTL_MS,
                "cacheScope": "public",
            }
            return {**result, "resultType": "complete", "_meta": {SERVER_KEY: SERVER_INFO}}
        if method == "ping" and not modern:
            return {}  # removed from the modern protocol
        if not modern and not self.ready:
            raise RpcError(INVALID_REQUEST, "Send initialize first, or name a protocol version in _meta")
        if method == "tools/list":
            if params.get("cursor") is not None:
                raise RpcError(INVALID_PARAMS, "Invalid cursor: there is only one page of tools")
            result = {"tools": TOOLS}
            if modern:
                # The caller's key decides who sees this, so nothing shared may keep it.
                result |= {"ttlMs": TOOL_LIST_TTL_MS, "cacheScope": "private"}
        elif method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str):
                raise RpcError(INVALID_PARAMS, "tools/call needs a tool name")
            if not isinstance(arguments, dict):
                raise RpcError(INVALID_PARAMS, "arguments must be an object")
            result = await self.tools.call(name, arguments)
        else:
            raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
        return {**result, "resultType": "complete", "_meta": {SERVER_KEY: SERVER_INFO}} if modern else result

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if not isinstance(requested, str):
            raise RpcError(INVALID_PARAMS, "initialize needs a protocolVersion")
        self.ready = True
        return {
            # Anything unknown gets our own choice, and the client decides whether it can use that.
            "protocolVersion": requested if requested in LEGACY_VERSIONS else LEGACY_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        }


def header_value(value: str) -> str:
    """A header value, undoing the Base64 form used for what a header cannot carry."""
    if value.startswith("=?base64?") and value.endswith("?="):
        try:
            return base64.b64decode(value[9:-2], validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            raise RpcError(HEADER_MISMATCH, "A header's Base64 value could not be read") from None
    return value


def check_modern_headers(headers: Mapping[str, str], message: dict[str, Any], version: str) -> None:
    """Modern headers mirror the body, so whatever routes on one cannot be told another story."""
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
    if meta.get(VERSION_KEY) != version:
        raise RpcError(HEADER_MISMATCH, f"Header mismatch: MCP-Protocol-Version is {version}, and _meta says {meta.get(VERSION_KEY)}")
    if headers.get("mcp-method") != message.get("method"):
        raise RpcError(HEADER_MISMATCH, f"Header mismatch: Mcp-Method does not match the method {message.get('method')}")
    if message.get("method") == "tools/call" and header_value(headers.get("mcp-name", "")) != params.get("name"):
        raise RpcError(HEADER_MISMATCH, f"Header mismatch: Mcp-Name does not match the tool {params.get('name')}")


def accepts_both(accept: str) -> bool:
    kinds = {part.split(";")[0].strip().lower() for part in accept.split(",")}
    return "*/*" in kinds or {"application/json", "text/event-stream"} <= kinds


def http_status(reply: dict[str, Any], modern: bool) -> int:
    code = reply.get("error", {}).get("code")
    if code in {PARSE_ERROR, INVALID_REQUEST, HEADER_MISMATCH, UNSUPPORTED_VERSION}:
        return 400
    return 404 if modern and code == METHOD_NOT_FOUND else 200


async def answer_http(
    handler: McpHandler, headers: Mapping[str, str], body: bytes, before: Callable[[str], None] = lambda method: None
) -> tuple[int, dict[str, Any] | None]:
    """One POST to the MCP endpoint: an HTTP status, and the JSON to send, if any.

    ``before`` sees each method first, and may refuse by raising.
    """
    if not accepts_both(headers.get("accept", "")):
        return 406, RpcError(INVALID_REQUEST, "Accept must list application/json and text/event-stream").body(None)
    if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        return 415, RpcError(INVALID_REQUEST, "Content-Type must be application/json").body(None)
    try:
        message = json.loads(body)
    except ValueError:
        return 400, RpcError(PARSE_ERROR, "Parse error: the body is not JSON").body(None)
    if not isinstance(message, dict):
        return 400, RpcError(INVALID_REQUEST, "Send one JSON-RPC object per POST, not a batch").body(None)
    identifier = message.get("id")
    version = headers.get("mcp-protocol-version")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
    if version is None and meta.get(VERSION_KEY) in MODERN_VERSIONS:
        return 400, RpcError(HEADER_MISMATCH, "Header mismatch: MCP-Protocol-Version is missing").body(identifier)
    version = version or HEADERLESS_VERSION
    if version not in SUPPORTED_VERSIONS:
        return 400, unsupported(version).body(identifier)
    modern = version in MODERN_VERSIONS
    if modern and "id" in message:  # the specification sets no header rules for notifications
        try:
            check_modern_headers(headers, message, version)
        except RpcError as exc:
            return 400, exc.body(identifier)
    before(str(message.get("method")))
    reply = await handler.handle(message)
    return (202, None) if reply is None else (http_status(reply, modern), reply)


def write_message(output: BinaryIO, message: dict[str, Any]) -> None:
    output.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    output.flush()


async def serve_stdio(client: httpx.AsyncClient, source: BinaryIO, output: BinaryIO) -> None:
    """Serve one MCP client over a pair of pipes, until it closes them.

    Requests run side by side, since a long reply must not hold up a cancel.
    """
    handler = McpHandler(MeshTools(client))
    loop = asyncio.get_running_loop()
    running: dict[Any, asyncio.Task[None]] = {}

    async def answer(message: dict[str, Any]) -> None:
        reply = await handler.handle(message)
        if reply is not None:
            write_message(output, reply)

    while True:
        line = await loop.run_in_executor(None, source.readline)
        if not line:
            break
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            write_message(output, RpcError(PARSE_ERROR, "Parse error: a line is not JSON").body(None))
            continue
        if isinstance(message, dict) and message.get("method") == "notifications/cancelled":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            task = running.get(params.get("requestId"))
            if task:
                task.cancel()  # and it sends nothing more
            continue
        identifier = message.get("id") if isinstance(message, dict) else None
        task = asyncio.create_task(answer(message))
        if identifier is not None:
            running[identifier] = task
            task.add_done_callback(lambda _, key=identifier: running.pop(key, None))
    # Closed input means stop soon, and a reply nearly done is still worth sending.
    waiting = list(running.values())
    if waiting:
        await asyncio.wait(waiting, timeout=STOP_GRACE_SECONDS)
    for task in waiting:
        task.cancel()
    await asyncio.gather(*waiting, return_exceptions=True)


def mesh_client(url: str, token: str | None) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=None, trust_env=False)


async def run_server(url: str, token: str | None) -> None:
    async with mesh_client(url, token) as client:
        await serve_stdio(client, sys.stdin.buffer, sys.stdout.buffer)
