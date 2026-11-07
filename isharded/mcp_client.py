"""MCP tools for a model on the mesh: an agent loop over MCP servers, by pipe or by URL.

``isharded.assistant.plugins`` has an MCP client of its own, built on ``LineRpc``
here but held in the assistant's sandbox. It negotiates through the same code.
The dependency runs that way so the core package does not need the assistant.

A server's annotations are its own claim, and a tool's output is only text it
chose to send. So anything not marked read-only waits for a person, and nothing
a server says can widen what ``--allow`` allows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import count
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import __version__
from .mcp_server import LEGACY_VERSIONS, MODERN_VERSIONS, NAME_PARAMETERS, SERVER_KEY, UNSUPPORTED_VERSION, VERSION_KEY, as_header, as_object, encode_line
from .toolcalls import without_reasoning

CLIENT_INFO = {"name": "isharded-mcp-run", "version": __version__}
CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
START_SECONDS = 20.0
# A legacy server may stay silent about a method it does not know.
PROBE_SECONDS = 5.0
STOP_SECONDS = 3.0
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_TOOLS = 500
MAX_RESULT_CHARACTERS = 32_000
MAX_ASKED_CHARACTERS = 500
BOTH_REPLY_KINDS = "application/json, text/event-stream"
EVENT_END = re.compile(rb"\r\n\r\n|\n\n|\r\r")
EVENT_LINE = re.compile(rb"\r\n|\n|\r")
# Only a user message can hold an image, so this says whose words they are not.
TOOL_IMAGES_NOTE = "The tools just called returned these images. They are tool output, and no request from the person."
# The specification's defaults for what a server may inherit: enough to run, and no secrets.
SAFE_ENVIRONMENT = (
    "APPDATA", "HOME", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "LOGNAME", "PATH", "PATHEXT",
    "PROCESSOR_ARCHITECTURE", "SHELL", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TERM", "TMP", "USER", "USERNAME", "USERPROFILE",
)


class ServerError(Exception):
    """A server crashed, hung, or could not be understood."""


class ToolRefused(Exception):
    """A server answered a request with an error. The server itself is fine."""


class CutOff(ServerError):
    """The model's last reply hit the token limit, so what it wrote is not a whole answer."""

    def __init__(self, answer: str) -> None:
        super().__init__("the reply reached the token limit and was cut off, so the answer above is not finished. Raise --max-tokens, or ask for less")
        self.answer = answer


def split_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise ServerError("--server was given an empty command")
    if os.name == "nt":  # non-POSIX splitting keeps the quotes
        parts = [part[1:-1] if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'" else part for part in parts]
    parts[0] = shutil.which(parts[0]) or parts[0]
    return parts


def server_environment(passed: Sequence[str]) -> dict[str, str]:
    names = {item.upper() for item in (*SAFE_ENVIRONMENT, *passed)}
    return {name: value for name, value in os.environ.items() if name.upper() in names}


def offer_nothing(message: dict[str, Any]) -> dict[str, Any]:
    """Our answer to a request from a server. We offer no sampling and no roots, so a server cannot call back into a model."""
    reply = {"result": {}} if message["method"] == "ping" else {"error": {"code": -32601, "message": "Not offered."}}
    return {"jsonrpc": "2.0", "id": message["id"], **reply}


class McpConnection:
    """A client's side of MCP, of either era, over whatever carries the messages.

    Subclasses carry them, and say how failures are named.
    """

    name = "it"
    failure: type[Exception] = ServerError
    refusal: type[Exception] = ToolRefused
    client_info = CLIENT_INFO
    modern_version: str | None = None
    legacy_version: str | None = None

    def _exchange(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send a request and wait for its reply. Raises queue.Empty if none comes in time."""
        raise NotImplementedError

    def _notify(self, method: str) -> None:
        raise NotImplementedError

    def _too_slow(self, method: str, timeout: float) -> str:
        return f"{self.name} took longer than {timeout:g} seconds over {method}"

    def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self.modern_version:
            params = {**params, "_meta": self._meta(self.modern_version)}
        try:
            reply = self._exchange(method, params, timeout)
        except queue.Empty:
            raise self.failure(self._too_slow(method, timeout)) from None
        if "error" in reply:
            error = reply["error"]
            raise self.refusal(str(error.get("message", error)) if isinstance(error, dict) else str(error))
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        while len(tools) < MAX_TOOLS:
            page = self.request("tools/list", {"cursor": cursor} if cursor else {}, START_SECONDS)
            tools += [tool for tool in page.get("tools", []) if isinstance(tool, dict) and isinstance(tool.get("name"), str)]
            cursor = page.get("nextCursor")
            if not cursor:
                break
        return tools[:MAX_TOOLS]

    def _meta(self, version: str) -> dict[str, Any]:
        return {VERSION_KEY: version, CLIENT_INFO_KEY: self.client_info, CLIENT_CAPABILITIES_KEY: {}}

    def _probe(self) -> str | None:
        """Ask a server what it speaks, the way the specification says to. None means a legacy server."""
        try:
            reply = self._exchange("server/discover", {"_meta": self._meta(MODERN_VERSIONS[0])}, PROBE_SECONDS)
        except queue.Empty:
            return None
        error = reply.get("error")
        if isinstance(error, dict) and error.get("code") == UNSUPPORTED_VERSION:
            offered = (error.get("data") or {}).get("supported") or []
        elif isinstance(reply.get("result"), dict):
            offered = reply["result"].get("supportedVersions") or []
            self.name = str(((reply["result"].get("_meta") or {}).get(SERVER_KEY) or {}).get("name") or self.name)
        else:
            return None
        return next((version for version in MODERN_VERSIONS if version in offered), None)

    def _initialize(self) -> None:
        """The older handshake, newest revision first, since a strict server refuses the ones it does not speak."""
        refused: Exception | None = None
        for version in LEGACY_VERSIONS:
            # We offer no sampling and no roots: a server cannot call back into a model.
            options = {"protocolVersion": version, "capabilities": {}, "clientInfo": self.client_info}
            try:
                result = self.request("initialize", options, START_SECONDS)
            except self.refusal as exc:
                refused = exc
                continue
            self.name = str((result.get("serverInfo") or {}).get("name") or self.name)
            self.legacy_version = str(result.get("protocolVersion") or version)
            self._notify("notifications/initialized")
            return
        assert refused is not None
        raise refused

    def start(self) -> None:
        self.modern_version = self._probe()
        if not self.modern_version:
            self._initialize()


class LineRpc(McpConnection):
    """JSON-RPC to a process over its pipes, one message per line.

    Subclasses start the process.
    """

    ensure_ascii = False

    def __init__(self, stdin: Any, stdout: Any) -> None:
        self.stdin = stdin
        self.lock = threading.Lock()
        self.next_id = 0
        self.waiting: dict[int, queue.Queue[dict[str, Any] | None]] = {}
        self.alive = True
        threading.Thread(target=self._read, args=(stdout,), daemon=True).start()

    def _read(self, stream: Any) -> None:
        try:
            with stream:
                for line in iter(lambda: stream.readline(MAX_LINE_BYTES + 1), b""):
                    if len(line) > MAX_LINE_BYTES:
                        break
                    self._receive(line)
        except (OSError, ValueError):
            pass  # the pipe closed under us
        self.alive = False
        with self.lock:
            boxes = list(self.waiting.values())
        for box in boxes:
            box.put(None)

    def _receive(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            return  # some servers print banners, which cannot stand in for a reply
        if not isinstance(message, dict):
            return
        if "method" in message:
            if "id" in message:
                self._send(offer_nothing(message))
            return
        with self.lock:
            box = self.waiting.get(message.get("id"))  # type: ignore[arg-type]
        if box:
            box.put(message)

    def _send(self, message: dict[str, Any]) -> None:
        data = encode_line(message, self.ensure_ascii)
        try:
            with self.lock:
                self.stdin.write(data)
                self.stdin.flush()
        except (OSError, ValueError):
            raise self.failure(f"{self.name} stopped listening") from None

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def _exchange(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        with self.lock:
            self.next_id += 1
            number = self.next_id
            box: queue.Queue[dict[str, Any] | None] = queue.Queue(1)
            self.waiting[number] = box
        try:
            self._send({"jsonrpc": "2.0", "id": number, "method": method, "params": params})
            reply = box.get(timeout=timeout)
        finally:
            with self.lock:
                self.waiting.pop(number, None)
        if reply is None:
            raise self.failure(f"{self.name} stopped while answering")
        return reply


class StdioServer(LineRpc):
    """An MCP server this process started, of either era."""

    def __init__(self, command: str, environment: dict[str, str]) -> None:
        parts = split_command(command)
        try:
            self.process = subprocess.Popen(parts, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment, bufsize=0)
        except OSError as exc:
            raise ServerError(f"could not start {command!r}: {exc}") from None
        self.name = Path(parts[0]).stem  # until it says what it is called
        super().__init__(self.process.stdin, self.process.stdout)

    def close(self) -> None:
        """Close its input, which is the polite way, and then insist."""
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=STOP_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()


def server_events(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """The data of each server-sent event, as it arrives. A stranger cannot make us hold more than one message."""
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        *events, buffer = EVENT_END.split(buffer)
        if len(buffer) > MAX_LINE_BYTES:
            raise ServerError(f"an event was larger than {MAX_LINE_BYTES} bytes")
        for event in events:
            lines = [line[5:].removeprefix(b" ") for line in EVENT_LINE.split(event) if line.startswith(b"data:")]
            if lines:
                yield b"\n".join(lines)


class HttpServer(McpConnection):
    """An MCP server at a URL, over Streamable HTTP, of either era.

    A modern server keeps nothing between requests. An older one may give us a
    session to name in each request, and to end when we leave.
    """

    def __init__(self, url: str, transport: httpx.BaseTransport | None = None) -> None:
        self.url = url
        try:
            self.name = urlsplit(url).hostname or url  # until it says what it is called
        except ValueError:
            raise ServerError(f"{url} is not a URL") from None
        self.next_id = 0
        self.session: str | None = None
        # Redirects stay off, since a server should not be able to send us somewhere else unasked.
        self.client = httpx.Client(headers={"Accept": BOTH_REPLY_KINDS}, trust_env=False, transport=transport)

    def _headers(self, method: str, params: dict[str, Any]) -> dict[str, str]:
        headers = {"Mcp-Session-Id": self.session} if self.session else {}
        version = (params.get("_meta") or {}).get(VERSION_KEY)
        if version:  # modern headers mirror the body
            headers |= {"MCP-Protocol-Version": version, "Mcp-Method": method}
            if method in NAME_PARAMETERS:
                headers["Mcp-Name"] = as_header(str(params.get(NAME_PARAMETERS[method])))
        elif self.legacy_version:
            headers["MCP-Protocol-Version"] = self.legacy_version
        return headers

    def _post(self, message: dict[str, Any], headers: dict[str, str]) -> None:
        """Send something that wants no reply. A server that will not take it has told us nothing we can use."""
        try:
            self.client.post(self.url, json=message, headers=headers, timeout=STOP_SECONDS)
        except httpx.HTTPError:
            pass

    def _notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method}, self._headers(method, {}))

    @staticmethod
    def _chunks(response: httpx.Response, deadline: float) -> Iterator[bytes]:
        """A read timeout only sees silence, and a server that drips bytes has still not answered."""
        for chunk in response.iter_bytes():
            if time.monotonic() > deadline:
                raise queue.Empty
            yield chunk

    def _read_json(self, response: httpx.Response, deadline: float) -> dict[str, Any]:
        body = b""
        for chunk in self._chunks(response, deadline):
            body += chunk
            if len(body) > MAX_LINE_BYTES:
                raise ServerError(f"{self.name} sent more than {MAX_LINE_BYTES} bytes")
        try:
            message = json.loads(body)
        except ValueError:
            message = None
        if isinstance(message, dict) and ("result" in message or "error" in message):
            return message
        raise ServerError(f"{self.name} answered {response.status_code}, with no reply that could be read: {body[:200].decode('utf-8', 'replace')}")

    def _read_stream(self, response: httpx.Response, number: int, deadline: float, headers: dict[str, str]) -> dict[str, Any]:
        for data in server_events(self._chunks(response, deadline)):
            try:
                message = json.loads(data)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if "method" in message:
                if "id" in message:
                    self._post(offer_nothing(message), headers)
            elif message.get("id") == number:
                return message
        raise ServerError(f"{self.name} closed the stream without answering")

    def _exchange(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.next_id += 1
        number = self.next_id
        headers = self._headers(method, params)
        deadline = time.monotonic() + timeout
        message = {"jsonrpc": "2.0", "id": number, "method": method, "params": params}
        try:
            with self.client.stream("POST", self.url, json=message, headers=headers, timeout=timeout) as response:
                self.session = response.headers.get("mcp-session-id") or self.session
                kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind == "text/event-stream" and response.status_code < 300:
                    return self._read_stream(response, number, deadline, headers)
                return self._read_json(response, deadline)
        except httpx.TimeoutException:
            raise queue.Empty from None
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise ServerError(f"could not reach {self.url}: {exc}") from None

    def _probe(self) -> str | None:
        try:
            return super()._probe()
        except ServerError:
            return None  # older servers turn this away in many ways, and the handshake will show if it is more

    def close(self) -> None:
        if self.session:
            try:
                self.client.delete(self.url, headers=self._headers("", {}), timeout=STOP_SECONDS)
            except httpx.HTTPError:
                pass
        self.client.close()


def content_parts(content: Iterable[Any]) -> list[str]:
    """The text of an MCP result, saying what was left out."""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        else:
            parts.append(f"[{item.get('type', 'unknown') if isinstance(item, dict) else 'unknown'} content omitted]")
    return parts


def input_refusal(result: dict[str, Any]) -> str:
    """What to tell a model when a tool stopped to ask for something we have no way to give."""
    asks = result.get("inputRequests")
    said = []
    for ask in asks.values() if isinstance(asks, dict) else []:
        if isinstance(ask, dict):
            said.append(str(as_object(ask.get("params")).get("message") or ask.get("method")))
    wanted = f" It asked: {'; '.join(said)[:MAX_ASKED_CHARACTERS]}." if said else ""
    return f"The tool stopped to ask for more input, and this client has no way to give any, so it did not finish.{wanted} Call it again with what it needs in the arguments, or tell the person what it asked for."


def result_text(result: dict[str, Any]) -> str:
    """What a tool sent back, as the text a model will read."""
    parts = content_parts(result.get("content") or [])
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(parts)
    if len(text) > MAX_RESULT_CHARACTERS:
        return text[:MAX_RESULT_CHARACTERS] + f"\n[cut off: the result was {len(text)} characters]"
    return text


class Toolbox:
    """Every server's tools, under names unique across them."""

    def __init__(self, servers: Sequence[McpConnection]) -> None:
        self.tools: dict[str, tuple[McpConnection, dict[str, Any]]] = {}
        for server in servers:
            for tool in server.list_tools():
                self.tools[self._free_name(server, tool["name"])] = (server, tool)

    def _free_name(self, server: McpConnection, name: str) -> str:
        """Two servers may both have a "search". The model must be able to pick, and no tool may replace another."""
        if name not in self.tools:
            return name
        free = prefixed = f"{server.name}_{name}"
        for number in count(2):
            if free not in self.tools:
                break
            free = f"{prefixed}_{number}"
        print(f"The name {json.dumps(name)} is taken, so {json.dumps(server.name)} offers that tool as {json.dumps(free)}.", file=sys.stderr)
        return free

    def for_model(self) -> list[dict[str, Any]]:
        schema = {"type": "object", "properties": {}}
        return [
            {"type": "function", "function": {"name": name, "description": str(tool.get("description") or ""), "parameters": tool.get("inputSchema") or schema}}
            for name, (_, tool) in self.tools.items()
        ]

    def read_only(self, name: str) -> bool:
        annotations = self.tools[name][1].get("annotations")
        return isinstance(annotations, dict) and annotations.get("readOnlyHint") is True

    def call(self, name: str, arguments: dict[str, Any], timeout: float) -> str:
        server, tool = self.tools[name]
        try:
            result = server.request("tools/call", {"name": tool["name"], "arguments": arguments}, timeout)
        except ToolRefused as exc:
            return f"The tool could not be called: {exc}"
        except ServerError as exc:
            return f"The tool failed: {exc}"
        if result.get("resultType") == "input_required":
            return input_refusal(result)
        text = result_text(result)
        return f"The tool reported an error: {text}" if result.get("isError") else text


def ask_at_terminal(name: str, arguments: dict[str, Any]) -> bool:
    """Is a person there, and do they agree? Without a terminal, the answer is no.

    Stdout is for the answer alone. Windows calls NUL a terminal, so an empty read counts as none too.
    """
    answer = ""
    if sys.stdin.isatty():
        # json.dumps escapes control characters, so a tool cannot rewrite this question.
        print(f"Run the tool {json.dumps(name)} with {json.dumps(arguments)}?\nAllow? [y/N] ", end="", file=sys.stderr, flush=True)
        answer = sys.stdin.readline()
    if not answer:
        print(f"Refused {json.dumps(name)}: it is not marked read-only, and there is no terminal to ask at. Use --allow to permit it.", file=sys.stderr)
        return False
    return answer.strip().lower() in {"y", "yes"}


def run_call(toolbox: Toolbox, call: dict[str, Any], allowed: set[str], timeout: float, ask: Callable[[str, dict[str, Any]], bool]) -> str:
    function = call["function"]
    name = function["name"]
    if name not in toolbox.tools:
        return f"There is no tool named {name}."
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except ValueError:
        return "Those arguments are not valid JSON."
    if not isinstance(arguments, dict):
        return "Arguments must be a JSON object."
    if not (toolbox.read_only(name) or name in allowed or ask(name, arguments)):
        return "The person running this did not allow that call. Answer without it, or say what you would need."
    print(f"Calling {name} with {json.dumps(arguments, ensure_ascii=False)}", file=sys.stderr)
    return toolbox.call(name, arguments, timeout)


def ask_model(client: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/v1/chat/completions", json=body)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ServerError(f"the mesh refused this ({response.status_code}): {detail}")
    return response.json()["choices"][0]


def agent_loop(
    client: httpx.Client, toolbox: Toolbox, model: str, prompt: str, allowed: set[str], max_turns: int, max_tokens: int | None, timeout: float,
    ask: Callable[[str, dict[str, Any]], bool] = ask_at_terminal,
) -> str:
    """Let the model call tools until it answers. Returns the answer, or raises CutOff with what there is of it."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools = toolbox.for_model()
    for _ in range(max_turns):
        body: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
        if max_tokens:
            body["max_tokens"] = max_tokens
        choice = ask_model(client, body)
        message = choice["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            answer = without_reasoning(message.get("content") or "")
            if choice.get("finish_reason") == "length":
                raise CutOff(answer)
            return answer
        messages.append(message)
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": run_call(toolbox, call, allowed, timeout, ask)})
    raise ServerError(f"the model was still calling tools after {max_turns} turns. Raise --max-turns, or ask for less")


def run_agent(args: Any) -> int:
    environment = server_environment(args.pass_env or [])
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    allowed = set(args.allow or [])
    servers: list[StdioServer | HttpServer] = []
    try:
        for command in args.server:
            server = HttpServer(command) if command.lower().startswith(("http://", "https://")) else StdioServer(command, environment)
            servers.append(server)
            server.start()
        toolbox = Toolbox(servers)
        if not toolbox.tools:
            raise ServerError("no server offered any tools")
        unknown = sorted(allowed - set(toolbox.tools))
        if unknown:
            raise ServerError(f"--allow names {', '.join(unknown)}, and the servers offer {', '.join(sorted(toolbox.tools))}")
        with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=None, trust_env=False) as client:
            answer = agent_loop(client, toolbox, args.model, args.prompt, allowed, args.max_turns, args.max_tokens, args.tool_timeout)
        print(answer)
        return 0
    except CutOff as exc:
        print(exc.answer)  # all there is, and the exit code says it is not whole
        print(f"isharded mcp run: {exc}", file=sys.stderr)
        return 1
    except (ServerError, ToolRefused, httpx.HTTPError) as exc:
        print(f"isharded mcp run: {exc}", file=sys.stderr)
        return 1
    finally:
        for server in servers:
            server.close()
