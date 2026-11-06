"""MCP tools for a model on the mesh: an agent loop over MCP stdio servers.

``shardmesh.assistant.plugins`` has an MCP client of its own, built on the
assistant's sandbox and speaking only the older revisions. This one stands alone
so the core package does not depend on the assistant.

A server's annotations are its own claim, and a tool's output is only text it
chose to send. So anything not marked read-only waits for a person, and nothing
a server says can widen what ``--allow`` allows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Sequence

import httpx

from . import __version__
from .mcp_server import LEGACY_VERSIONS, MODERN_VERSIONS, SERVER_KEY, UNSUPPORTED_VERSION, VERSION_KEY
from .toolcalls import without_reasoning

CLIENT_INFO = {"name": "shardmesh-mcp-run", "version": __version__}
CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
START_SECONDS = 20.0
# A legacy server may stay silent about a method it does not know.
PROBE_SECONDS = 5.0
STOP_SECONDS = 3.0
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_TOOLS = 500
MAX_RESULT_CHARACTERS = 32_000
# The specification's defaults for what a server may inherit: enough to run, and no secrets.
SAFE_ENVIRONMENT = (
    "APPDATA", "HOME", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "LOGNAME", "PATH", "PATHEXT",
    "PROCESSOR_ARCHITECTURE", "SHELL", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TERM", "TMP", "USER", "USERNAME", "USERPROFILE",
)


class ServerError(Exception):
    """A server crashed, hung, or could not be understood."""


class ToolRefused(Exception):
    """A server answered a request with an error. The server itself is fine."""


def split_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise ServerError("--server was given an empty command")
    if os.name == "nt":  # non-POSIX splitting keeps the quotes
        parts = [part[1:-1] if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'" else part for part in parts]
    parts[0] = shutil.which(parts[0]) or parts[0]
    return parts


def server_environment(passed: Sequence[str]) -> dict[str, str]:
    names = {*SAFE_ENVIRONMENT, *passed}
    return {name: value for name, value in os.environ.items() if name.upper() in {item.upper() for item in names}}


class StdioServer:
    """An MCP server this process started: JSON-RPC, one message per line."""

    def __init__(self, command: str, environment: dict[str, str]) -> None:
        parts = split_command(command)
        try:
            self.process = subprocess.Popen(parts, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment, bufsize=0)
        except OSError as exc:
            raise ServerError(f"could not start {command!r}: {exc}") from None
        self.lock = threading.Lock()
        self.next_id = 0
        self.waiting: dict[int, queue.Queue[dict[str, Any] | None]] = {}
        self.modern_version: str | None = None
        self.name = Path(parts[0]).stem  # until it says what it is called
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        with self.process.stdout as stream:
            for line in iter(lambda: stream.readline(MAX_LINE_BYTES + 1), b""):
                if len(line) > MAX_LINE_BYTES:
                    break
                self._receive(line)
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
            if "id" in message:  # we offer no sampling and no roots, so a server cannot call back into a model
                reply = {"result": {}} if message["method"] == "ping" else {"error": {"code": -32601, "message": "Not offered."}}
                self._send({"jsonrpc": "2.0", "id": message["id"], **reply})
            return
        with self.lock:
            box = self.waiting.get(message.get("id"))  # type: ignore[arg-type]
        if box:
            box.put(message)

    def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        assert self.process.stdin is not None
        try:
            with self.lock:
                self.process.stdin.write(data)
                self.process.stdin.flush()
        except (OSError, ValueError):
            raise ServerError(f"{self.name} stopped listening") from None

    def _exchange(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send a request and wait for its reply. Raises queue.Empty if none comes in time."""
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
            raise ServerError(f"{self.name} stopped while answering")
        return reply

    def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self.modern_version:
            params = {**params, "_meta": self._meta(self.modern_version)}
        try:
            reply = self._exchange(method, params, timeout)
        except queue.Empty:
            raise ServerError(f"{self.name} took longer than {timeout:g} seconds over {method}") from None
        if "error" in reply:
            error = reply["error"]
            raise ToolRefused(str(error.get("message", error)) if isinstance(error, dict) else str(error))
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _meta(version: str) -> dict[str, Any]:
        return {VERSION_KEY: version, CLIENT_INFO_KEY: CLIENT_INFO, CLIENT_CAPABILITIES_KEY: {}}

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

    def start(self) -> None:
        self.modern_version = self._probe()
        if self.modern_version:
            return
        # We offer no sampling and no roots: a server cannot call back into a model.
        options = {"protocolVersion": LEGACY_VERSIONS[0], "capabilities": {}, "clientInfo": CLIENT_INFO}
        result = self.request("initialize", options, START_SECONDS)
        self.name = str((result.get("serverInfo") or {}).get("name") or self.name)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

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


def result_text(result: dict[str, Any]) -> str:
    """What a tool sent back, as the text a model will read, saying what was left out."""
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        else:
            parts.append(f"[{item.get('type', 'unknown') if isinstance(item, dict) else 'unknown'} content omitted]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(parts)
    if len(text) > MAX_RESULT_CHARACTERS:
        return text[:MAX_RESULT_CHARACTERS] + f"\n[cut off: the result was {len(text)} characters]"
    return text


class Toolbox:
    """Every server's tools, under names unique across them."""

    def __init__(self, servers: Sequence[StdioServer]) -> None:
        self.tools: dict[str, tuple[StdioServer, dict[str, Any]]] = {}
        for server in servers:
            for tool in server.list_tools():
                name = tool["name"]
                # Two servers may both have a "search". The model must be able to pick.
                if name in self.tools:
                    name = f"{server.name}_{name}"
                self.tools[name] = (server, tool)

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
            return "The tool asked for more input, which this client cannot give."
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
    """Let the model call tools until it answers. Returns the answer."""
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
            return without_reasoning(message.get("content") or "")
        messages.append(message)
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": run_call(toolbox, call, allowed, timeout, ask)})
    raise ServerError(f"the model was still calling tools after {max_turns} turns. Raise --max-turns, or ask for less")


def run_agent(args: Any) -> int:
    environment = server_environment(args.pass_env or [])
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    servers: list[StdioServer] = []
    try:
        for command in args.server:
            server = StdioServer(command, environment)
            servers.append(server)
            server.start()
        toolbox = Toolbox(servers)
        if not toolbox.tools:
            raise ServerError("no server offered any tools")
        unknown = sorted(set(args.allow or []) - set(toolbox.tools))
        if unknown:
            raise ServerError(f"--allow names {', '.join(unknown)}, and the servers offer {', '.join(sorted(toolbox.tools))}")
        with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=None, trust_env=False) as client:
            answer = agent_loop(client, toolbox, args.model, args.prompt, set(args.allow or []), args.max_turns, args.max_tokens, args.tool_timeout)
        print(answer)
        return 0
    except (ServerError, ToolRefused, httpx.HTTPError) as exc:
        print(f"shardmesh mcp run: {exc}", file=sys.stderr)
        return 1
    finally:
        for server in servers:
            server.close()
