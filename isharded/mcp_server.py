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
from collections.abc import Awaitable, Callable, Mapping
import json
import sys
from typing import Any, BinaryIO

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
SERVER_INFO = {"name": "isharded", "version": __version__}
INSTRUCTIONS = "Runs language models across the machines of an iSharded pool."
# The list changes only with the software, so a client may keep it for a while.
TOOL_LIST_TTL_MS = 3_600_000
CHAT_REPLY_TOKENS = 1024
STOP_GRACE_SECONDS = 5.0

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002
HEADER_MISMATCH = -32020
UNSUPPORTED_VERSION = -32022
# The older HTTP era ends a request's stream only with a reply, so even a cancelled one gets this.
REQUEST_CANCELLED = -32800

# Which parameter a request's Mcp-Name header mirrors.
NAME_PARAMETERS = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}
# Each of these reaches the node's own API, which charges for it there.
PAID_AT_THE_API = frozenset(NAME_PARAMETERS)


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
NO_ARGUMENTS = {"type": "object", "properties": {}, "additionalProperties": False}
TOOLS: list[dict[str, Any]] = [
    {
        "name": "chat",
        "title": "Ask a model",
        "description": "Ask a model running on the mesh and get its answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to ask"},
                "image": {"type": "string", "description": "An image to ask about, as a data: URL. The model must be able to see"},
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
        "name": "embed",
        "title": "Embed text",
        "description": "Turn text into a vector with an embedding model on the mesh, for search and for comparing texts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text"},
                "model": {"type": "string", "description": "Which model; list_models shows them. Default is the main one"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "annotations": READ_ONLY,
    },
    {
        "name": "list_models",
        "title": "List models",
        "description": "List the models this mesh is running now.",
        "inputSchema": NO_ARGUMENTS,
        "annotations": READ_ONLY,
    },
    {
        "name": "mesh_status",
        "title": "Mesh status",
        "description": "Whether the mesh can answer, what it runs, and which nodes hold which stages.",
        "inputSchema": NO_ARGUMENTS,
        "annotations": READ_ONLY,
    },
]
JSON_KINDS: dict[str, tuple[type, ...]] = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}


def status_view(status: dict[str, Any]) -> dict[str, Any]:
    mesh = status.get("mesh") or {}
    return {
        "status": status.get("status"),
        "model": status.get("model"),
        "epoch": status.get("epoch"),
        "coordinator": mesh.get("leader"),
        "nodes": mesh.get("members"),
        "route": [stage.get("worker_id") for stage in status.get("route", [])],
        "workers": {name: worker.get("state") for name, worker in status.get("workers", {}).items()},
        "lanes": [{key: lane.get(key) for key in ("name", "model", "ready", "workers")} for lane in status.get("lanes", [])],
        "tokens_per_second": (status.get("metrics") or {}).get("tokens_per_second_ema"),
    }


def shared_status_view(status: dict[str, Any]) -> dict[str, Any]:
    # A member is an id and an address, and this text is made to be handed to a model.
    return status_view(status) | {"nodes": sorted((status.get("mesh") or {}).get("members") or [])}


def route_view(status: dict[str, Any]) -> dict[str, Any]:
    # A stage also holds its worker's address, which a reader of the route has no need of.
    stages = [{key: stage.get(key) for key in ("worker_id", "start_layer", "end_layer")} for stage in status.get("route", [])]
    return {"model": status.get("model"), "epoch": status.get("epoch"), "stages": stages}


def model_view(status: dict[str, Any]) -> dict[str, Any]:
    # The plan also says where the files came from, which can be a path on someone's disk.
    plan = status.get("active_plan") or {}
    details = {key: plan.get(key) for key in ("architecture", "layers", "experts", "model_bytes")}
    return {key: status.get(key) for key in ("model", "kind", "vision", "dimensions")} | details


def resource(name: str, title: str, description: str) -> dict[str, Any]:
    return {"uri": f"isharded://mesh/{name}", "name": name, "title": title, "description": description, "mimeType": "application/json"}


RESOURCES = [
    resource("status", "Mesh status", "Whether the mesh can answer, what it runs, and which nodes hold which stages."),
    resource("route", "Route", "Which worker holds which layers of the model on the main route."),
    resource("model", "Loaded model", "What the loaded model is: its kind, whether it can see, and its size."),
]
RESOURCE_VIEWS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "isharded://mesh/status": shared_status_view,
    "isharded://mesh/route": route_view,
    "isharded://mesh/model": model_view,
}
HEALTH_PROMPT = "mesh_health"
PROMPTS = [
    {
        "name": HEALTH_PROMPT,
        "title": "Check the mesh",
        "description": "The mesh's status as it is now, with a request to say in plain words whether it is healthy.",
    }
]
HEALTH_REQUEST = "Here is the status of an iSharded mesh. Say in plain words whether it can answer, and name anything that looks wrong.\n\n"


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
        if ("minimum" in rule and value < rule["minimum"]) or ("maximum" in rule and value > rule["maximum"]):
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
        content: Any = arguments["prompt"]
        if "image" in arguments:
            content = [{"type": "text", "text": content}, {"type": "image_url", "image_url": {"url": arguments["image"]}}]
        messages.append({"role": "user", "content": content})
        request = {key: arguments[key] for key in ("model", "temperature", "enable_thinking") if key in arguments}
        request |= {"messages": messages, "max_tokens": arguments.get("max_tokens", CHAT_REPLY_TOKENS)}
        reply = await self._ask("POST", "/v1/chat/completions", json=request)
        choice = reply["choices"][0]
        answer = without_reasoning(choice["message"].get("content") or "")
        cut = "\n\n[The reply reached max_tokens and was cut off.]" if choice["finish_reason"] == "length" else ""
        details = {"answer": answer, "model": reply.get("model"), "finish_reason": choice["finish_reason"], "usage": reply.get("usage")}
        return {"content": [{"type": "text", "text": answer + cut}], "structuredContent": details}

    async def embed(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = {"input": arguments["text"]} | ({"model": arguments["model"]} if "model" in arguments else {})
        reply = await self._ask("POST", "/v1/embeddings", json=request)
        return as_result({"embedding": reply["data"][0]["embedding"], "model": reply.get("model"), "usage": reply.get("usage")})

    async def list_models(self, arguments: dict[str, Any]) -> dict[str, Any]:
        reply = await self._ask("GET", "/v1/models")
        return as_result({"models": [item["id"] for item in reply.get("data", [])]})

    async def mesh_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return as_result(status_view(await self._ask("GET", "/admin/status")))

    async def _status(self) -> dict[str, Any]:
        """The status for a resource or a prompt, which have no tool error to fail with."""
        try:
            return await self._ask("GET", "/admin/status")
        except ToolFailed as exc:
            raise RpcError(INTERNAL_ERROR, str(exc)) from exc

    async def read(self, uri: str) -> dict[str, Any]:
        view = RESOURCE_VIEWS.get(uri)
        if view is None:
            raise RpcError(RESOURCE_NOT_FOUND, "Resource not found", {"uri": uri})
        text = json.dumps(view(await self._status()), ensure_ascii=False)
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": text}]}

    async def prompt(self, name: str) -> dict[str, Any]:
        if name != HEALTH_PROMPT:
            raise RpcError(INVALID_PARAMS, f"Unknown prompt: {name}")
        text = HEALTH_REQUEST + json.dumps(shared_status_view(await self._status()), ensure_ascii=False)
        return {"description": PROMPTS[0]["description"], "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}

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


def modern_result(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "resultType": "complete", "_meta": {SERVER_KEY: SERVER_INFO}}


def as_object(value: Any) -> dict[str, Any]:
    """What a stranger sent, where only an object means anything."""
    return value if isinstance(value, dict) else {}


def is_request_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


class InFlight:
    """The requests still running, so that a cancel can find one."""

    def __init__(self) -> None:
        self.tasks: dict[tuple[str, Any], list[asyncio.Task[Any]]] = {}

    def add(self, caller: str, identifier: Any, task: asyncio.Task[Any]) -> None:
        key = (caller, identifier)
        self.tasks.setdefault(key, []).append(task)
        task.add_done_callback(lambda done: self._finished(key, done))

    def _finished(self, key: tuple[str, Any], task: asyncio.Task[Any]) -> None:
        self.tasks[key].remove(task)
        if not self.tasks[key]:
            del self.tasks[key]

    def cancel(self, caller: str, identifier: Any) -> None:
        """A cancel is a request, and the specification lets us ignore it.

        Over HTTP there are no sessions, so two clients on one key can both be using
        the same id. Stopping the wrong one would drop its reply, so neither stops.
        """
        running = self.tasks.get((caller, identifier), []) if is_request_id(identifier) else []
        if len(running) == 1:
            running[0].cancel()

    def all(self) -> list[asyncio.Task[Any]]:
        return [task for tasks in self.tasks.values() for task in tasks]


LISTS: dict[str, tuple[str, list[dict[str, Any]]]] = {
    "tools/list": ("tools", TOOLS),
    "resources/list": ("resources", RESOURCES),
    "resources/templates/list": ("resourceTemplates", []),
    "prompts/list": ("prompts", PROMPTS),
}


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
        if not is_request_id(identifier):
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
        version = as_object(params.get("_meta")).get(VERSION_KEY)
        if version is not None and version not in SUPPORTED_VERSIONS:
            raise unsupported(version)
        modern = version in MODERN_VERSIONS
        if method == "initialize" and not modern:
            return self._initialize(params)
        if method == "server/discover":
            return modern_result(
                {
                    "supportedVersions": list(SUPPORTED_VERSIONS),
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "instructions": INSTRUCTIONS,
                    "ttlMs": TOOL_LIST_TTL_MS,
                    "cacheScope": "public",
                }
            )
        if method == "ping" and not modern:
            return {}  # removed from the modern protocol
        if not modern and not self.ready:
            raise RpcError(INVALID_REQUEST, "Send initialize first, or name a protocol version in _meta")
        if method in LISTS:
            result = self._list(LISTS[method], params, modern)
        elif method == "tools/call":
            result = await self._call_tool(params)
        elif method == "resources/read":
            result = await self.tools.read(self._named(params, "uri", method))
        elif method == "prompts/get":
            result = await self.tools.prompt(self._named(params, "name", method))
        else:
            raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
        return modern_result(result) if modern else result

    @staticmethod
    def _named(params: dict[str, Any], key: str, method: str) -> str:
        value = params.get(key)
        if not isinstance(value, str):
            raise RpcError(INVALID_PARAMS, f"{method} needs a {key}")
        return value

    @staticmethod
    def _list(kind: tuple[str, list[dict[str, Any]]], params: dict[str, Any], modern: bool) -> dict[str, Any]:
        key, items = kind
        if params.get("cursor") is not None:
            raise RpcError(INVALID_PARAMS, f"Invalid cursor: there is only one page of {key}")
        # The caller's key decides who sees this, so nothing shared may keep it.
        return {key: items, "ttlMs": TOOL_LIST_TTL_MS, "cacheScope": "private"} if modern else {key: items}

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name, arguments = params.get("name"), params.get("arguments", {})
        if not isinstance(name, str):
            raise RpcError(INVALID_PARAMS, "tools/call needs a tool name")
        if not isinstance(arguments, dict):
            raise RpcError(INVALID_PARAMS, "arguments must be an object")
        return await self.tools.call(name, arguments)

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if not isinstance(requested, str):
            raise RpcError(INVALID_PARAMS, "initialize needs a protocolVersion")
        self.ready = True
        return {
            # Anything unknown gets our own choice, and the client decides whether it can use that.
            "protocolVersion": requested if requested in LEGACY_VERSIONS else LEGACY_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}, "prompts": {"listChanged": False}},
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


def as_header(value: str) -> str:
    """What ``header_value`` reads back."""
    if value.isascii() and value.isprintable() and value == value.strip():
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="


def check_modern_headers(headers: Mapping[str, str], message: dict[str, Any], version: str) -> None:
    """Modern headers mirror the body, so whatever routes on one cannot be told another story."""
    params = as_object(message.get("params"))
    claimed = as_object(params.get("_meta")).get(VERSION_KEY)
    if claimed != version:
        raise RpcError(HEADER_MISMATCH, f"Header mismatch: MCP-Protocol-Version is {version}, and _meta says {claimed}")
    if headers.get("mcp-method") != message.get("method"):
        raise RpcError(HEADER_MISMATCH, f"Header mismatch: Mcp-Method does not match the method {message.get('method')}")
    named = NAME_PARAMETERS.get(str(message.get("method")))
    if named and header_value(headers.get("mcp-name", "")) != params.get(named):
        raise RpcError(HEADER_MISMATCH, f"Header mismatch: Mcp-Name does not match the {named} {params.get(named)}")


HTTP_REQUESTS = InFlight()


def accepts_both(accept: str) -> bool:
    kinds = {part.split(";")[0].strip().lower() for part in accept.split(",")}
    return "*/*" in kinds or {"application/json", "text/event-stream"} <= kinds


def http_status(reply: dict[str, Any], modern: bool) -> int:
    code = reply.get("error", {}).get("code")
    if code in {PARSE_ERROR, INVALID_REQUEST, HEADER_MISMATCH, UNSUPPORTED_VERSION}:
        return 400
    return 404 if modern and code == METHOD_NOT_FOUND else 200


async def client_left(receive: Callable[[], Awaitable[dict[str, Any]]]) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


async def answer_cancellably(
    handler: McpHandler, message: dict[str, Any], caller: str, modern: bool, receive: Callable[[], Awaitable[dict[str, Any]]] | None
) -> tuple[int, dict[str, Any] | None]:
    """Answer a request while a cancel can still reach it.

    Each era cancels its own way. The older one posts notifications/cancelled, and
    says a dropped connection is not a cancel. The modern one closes the connection,
    and that is its only signal.
    """
    identifier = message["id"]
    work = asyncio.create_task(handler.handle(message))
    if is_request_id(identifier):
        HTTP_REQUESTS.add(caller, identifier, work)
    watch = asyncio.create_task(client_left(receive)) if modern and receive else None
    try:
        await asyncio.wait({work, watch} if watch else {work}, return_when=asyncio.FIRST_COMPLETED)
        if not work.done():
            work.cancel()
            await asyncio.wait({work})
    except asyncio.CancelledError:  # the server is stopping, so this request stops with it
        work.cancel()
        raise
    finally:
        if watch:
            watch.cancel()
    if work.cancelled():
        # The modern specification forbids a reply, and no client is left to read one.
        return (202, None) if modern else (200, RpcError(REQUEST_CANCELLED, "Request cancelled").body(identifier))
    reply = work.result()
    return (202, None) if reply is None else (http_status(reply, modern), reply)


async def answer_http(
    handler: McpHandler,
    headers: Mapping[str, str],
    body: bytes,
    before: Callable[[str], None] = lambda method: None,
    caller: str = "",
    receive: Callable[[], Awaitable[dict[str, Any]]] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """One POST to the MCP endpoint: an HTTP status, and the JSON to send, if any.

    ``before`` sees each method first, and may refuse by raising, except those the
    API charges for itself. ``caller`` is who the key proved to be, since a cancel
    reaches only that caller's own requests. ``receive`` is the ASGI channel, which
    says when the client has gone.
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
    if version is None and as_object(as_object(message.get("params")).get("_meta")).get(VERSION_KEY) in MODERN_VERSIONS:
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
    method = str(message.get("method"))
    if method not in PAID_AT_THE_API:
        before(method)
    if "id" not in message and method == "notifications/cancelled":
        HTTP_REQUESTS.cancel(caller, as_object(message.get("params")).get("requestId"))
        return 202, None
    if "id" in message and "method" in message:
        return await answer_cancellably(handler, message, caller, modern, receive)
    reply = await handler.handle(message)
    return (202, None) if reply is None else (http_status(reply, modern), reply)


def encode_line(message: dict[str, Any], ensure_ascii: bool = False) -> bytes:
    return json.dumps(message, ensure_ascii=ensure_ascii, separators=(",", ":")).encode("utf-8") + b"\n"


def write_message(output: BinaryIO, message: dict[str, Any]) -> None:
    output.write(encode_line(message))
    output.flush()


async def serve_stdio(client: httpx.AsyncClient, source: BinaryIO, output: BinaryIO) -> None:
    """Serve one MCP client over a pair of pipes, until it closes them.

    Requests run side by side, since a long reply must not hold up a cancel.
    """
    handler = McpHandler(MeshTools(client))
    loop = asyncio.get_running_loop()
    running = InFlight()

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
            running.cancel("", as_object(message.get("params")).get("requestId"))  # and it sends nothing more
            continue
        identifier = message.get("id") if isinstance(message, dict) else None
        task = asyncio.create_task(answer(message))
        if is_request_id(identifier):
            running.add("", identifier, task)
    # Closed input means stop soon, and a reply nearly done is still worth sending.
    waiting = running.all()
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
