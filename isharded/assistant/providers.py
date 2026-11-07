"""Where models come from: the mesh, or anything that speaks the OpenAI API.

A provider only answers. It holds no state of the assistant's and decides nothing.
Its key stays where the owner put it: the database records the name of an
environment variable or a file, and the key is read at the moment it is used.

Silence is the failure that matters here. A dead process refuses at once, while a
lost link just stops answering, so every read has a limit and a stream that ends
without saying it finished counts as a failure.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from http.cookiejar import CookieJar, DefaultCookiePolicy
import ipaddress
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from isharded.tls import client_verify, intercepted

from .contracts import Capability, ModelReply, ModelRequest, Reach, ToolCall
from .mind import actor, emit, LOCAL_USER, plainly, reach_named, table

CONNECT_SECONDS = 10.0
READ_SECONDS = 120.0  # silence this long means the link is gone; covers a long prefill
WHOLE_REPLY_SECONDS = 300.0  # a reply with no stream has nothing to read until it is done
HEALTH_SECONDS = 5.0
# Under the five seconds many servers hold an idle connection, so that this side always
# closes first, and never sends on a connection the server is dropping that same moment.
IDLE_CONNECTION_SECONDS = 2.0
MODELS_FRESH_SECONDS = 30.0
MAX_TOOL_JSON = 200_000
CHARS_PER_TOKEN = 3
ERROR_CHARS = 300
PROVIDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
UPPER_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
KEY_NOT_NAME = "That looks like a key itself. Keys stay out of the database: name an environment variable that is set, or a file"
# Thinking takes tokens, and a slow pool spends a minute on them, so only the
# roles that gain from it pay for it.
THINKING_ROLES = frozenset({"planning", "coding", "verification"})
TOOL_MODES = ("auto", "native", "prompt")
TOOL_FENCE_TAGS = frozenset({"", "tool", "tools", "tool_call", "tool_calls", "toolcall", "json", "json5", "javascript", "js"})
LOOPBACK_NAMES = frozenset({"localhost"})
NO_TOOLS_STATUSES = (400, 404, 422, 501)
# What one request can get wrong by itself. The rest of the 4xx range is about the key, the
# address, or a rate limit, and those stay wrong for everyone until someone fixes them.
REQUEST_FAULT_STATUSES = (400, 413, 422)

SCHEMA = [
    """
    CREATE TABLE providers (
        name TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        base_url TEXT NOT NULL,
        key_reference TEXT,
        reach INTEGER NOT NULL,
        tool_mode TEXT NOT NULL,
        daily_tokens INTEGER,
        read_seconds REAL NOT NULL,
        model TEXT,
        added_at REAL NOT NULL,
        added_by TEXT NOT NULL
    );
    """,
]


class ProviderError(Exception):
    """A provider failed. The message is safe to show and never holds a key."""


class RequestRefused(ProviderError):
    """A provider turned down this one request. That says nothing about the provider, so nobody else pays for it."""


class ProviderRefused(ValueError):
    pass


class ToolsNotSupported(Exception):
    pass


def estimate_tokens(text: str) -> int:
    """A rough count, used only when a server does not report one."""
    return len(text) // CHARS_PER_TOKEN + 1


def is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def api_root(url: str) -> str:
    """A server's address, with /v1 added if it was given as a bare host."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ProviderRefused(f"{url!r} is not an http or https address")
    if parts.username or parts.password or parts.query:
        raise ProviderRefused("Put the key in an environment variable or a file, not in the address")
    root = url.rstrip("/")
    return root if parts.path.strip("/") else f"{root}/v1"


def read_key(reference: str | None) -> str | None:
    """Look up a key by reference. An unset variable is a person's mistake to fix."""
    if not reference:
        return None
    kind, _, value = reference.partition(":")
    if kind == "env" and value:
        key = os.environ.get(value)
        if not key:
            raise ProviderError(f"The environment variable {value} is not set, and this provider's key should be in it")
        return key.strip()
    if kind == "file" and value:
        try:
            return Path(value).read_text(encoding="utf-8").strip()
        except OSError:
            raise ProviderError(f"This provider's key file {value} cannot be read") from None
    raise ProviderError(f"{reference!r} is not a key reference. Use env:NAME or file:PATH")


def key_reference(value: str | None) -> str | None:
    """Turn what a person typed into a reference, and refuse what looks like a key.

    A bare UPPER_CASE name is an environment variable, and anything with a path
    in it is a file. Refusing the rest keeps a pasted key out of the database.
    Any other variable name can be given as env:name once it is set.
    """
    if not value:
        return None
    if value.startswith(("env:", "file:")):
        kind, _, target = value.partition(":")
    elif UPPER_NAME.fullmatch(value):
        kind, target = "env", value
    elif "/" in value or "\\" in value or Path(value).exists():
        kind, target = "file", value
    else:
        raise ProviderRefused(KEY_NOT_NAME)
    if kind == "env" and not (UPPER_NAME.fullmatch(target) or (ENV_NAME.fullmatch(target) and os.environ.get(target))):
        raise ProviderRefused(KEY_NOT_NAME)
    return f"{kind}:{target}"


# Reading what models write


def strip_thinking(text: str) -> str:
    """The reply without its reasoning, which nobody should act on."""
    end = text.rfind("</think>")
    if end >= 0:
        return text[end + len("</think>") :].lstrip()
    start = text.find("<think>")
    return text[:start].rstrip() if start >= 0 else text


@dataclass(slots=True)
class OpenValue:
    """A JSON value as far as it has been read, so more text carries on from there."""

    size: int = 0  # characters read, which is its whole length once it closes
    depth: int = 0
    in_string: bool = False
    escaped: bool = False

    def read(self, text: str, start: int) -> bool:
        """Read on through the value opening at ``start``. True if it closes, respecting strings."""
        depth, in_string, escaped = self.depth, self.in_string, self.escaped
        for index in range(start + self.size, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character in "{[":
                depth += 1
            elif character in "}]":
                depth -= 1
                if depth == 0:
                    self.size = index + 1 - start
                    return True
        self.size, self.depth, self.in_string, self.escaped = max(len(text) - start, 0), depth, in_string, escaped
        return False


def balanced_span(text: str, start: int) -> int | None:
    """Where the JSON value opening at ``start`` closes, respecting strings."""
    value = OpenValue()
    return start + value.size if value.read(text, start) else None


def loose_json(text: str) -> Any:
    """Parse JSON as models really write it. Returns None if it cannot be trusted.

    A value cut off by the token limit is never completed. A truncated argument
    is a different argument, and it must not reach a tool.
    """
    text = text.strip()
    if not text or len(text) > MAX_TOOL_JSON:
        return None
    first = min((index for index in (text.find("{"), text.find("[")) if index >= 0), default=-1)
    if first < 0:
        return None
    end = balanced_span(text, first)
    candidate = text[first:end] if end else text[first:]
    candidate = candidate.replace("“", '"').replace("”", '"')
    for attempt in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
        try:
            return json.loads(attempt)
        except (RecursionError, ValueError):
            continue
    if end is None:
        return None
    try:  # single quotes and Python's True and None
        found = ast.literal_eval(re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", candidate))))
        json.dumps(found, sort_keys=True)  # a set, bytes, or keys of mixed types are Python, not JSON, and no tool can take them
        return found
    except (TypeError, ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def native_name(capability: str) -> str:
    """Function names may not contain the dot that capability names have."""
    return capability.replace(".", "__")


def as_calls(value: Any, offered: frozenset[str]) -> list[ToolCall]:
    """Tool calls in whatever shape a model chose. Only what was offered counts."""
    if isinstance(value, list):
        return [call for item in value for call in as_calls(item, offered)]
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("tool_calls"), list):
        return as_calls(value["tool_calls"], offered)
    if isinstance(value.get("function"), dict):
        value = {**value, **value["function"]}
    name = next((value[key] for key in ("tool", "name", "capability", "function") if isinstance(value.get(key), str)), None)
    if name is None:
        return []
    name = name.strip()
    if name not in offered:
        name = name.replace("__", ".")
    if name not in offered:
        return []
    arguments = next((value[key] for key in ("arguments", "args", "parameters", "input") if key in value), {})
    if isinstance(arguments, str):
        arguments = loose_json(arguments) if arguments.strip() else {}
    if not isinstance(arguments, dict):
        return []
    return [ToolCall(name, arguments, str(value.get("id") or ""))]


FENCED = re.compile(r"```[ \t]*([A-Za-z0-9_-]*)[ \t]*\r?\n?(.*?)(?:```|\Z)", re.DOTALL)
TAGGED = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)
CODE_SPAN = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*`", re.DOTALL)


def parse_tool_calls(text: str, offered: Sequence[Capability]) -> tuple[str, tuple[ToolCall, ...]]:
    """Find tool calls in a reply, and return the text without them.

    Text inside reasoning is ignored: a model thinking about a tool has not
    called it. A block that names nothing offered stays in the reply as text.
    """
    visible = strip_thinking(text)
    names = frozenset(capability.name for capability in offered)
    if not names:
        return visible, ()
    calls: list[ToolCall] = []

    def take(match: re.Match[str], body: str) -> str:
        found = as_calls(loose_json(body), names)
        calls.extend(found)
        return "" if found else match.group(0)

    visible = FENCED.sub(lambda match: take(match, match.group(2)) if match.group(1).lower() in TOOL_FENCE_TAGS else match.group(0), visible)
    visible = TAGGED.sub(lambda match: take(match, match.group(1)), visible)
    if not calls:
        visible, calls = bare_calls(visible, names)
    return visible.strip(), tuple(calls)


def bare_calls(text: str, names: frozenset[str]) -> tuple[str, list[ToolCall]]:
    """Calls written with no fence at all, which small models often do.

    This is the least certain way to spot a call, so it is the strictest. Code is
    ignored, since showing an example is not calling a tool, and a call must end
    the reply, since a model that is acting stops there and one explaining goes on.
    """
    if not any(name in text or native_name(name) in text for name in names):
        return text, []  # so a long reply full of code is not scanned brace by brace
    plain = CODE_SPAN.sub(lambda match: " " * len(match.group(0)), text)
    calls: list[ToolCall] = []
    kept, position, last_end = [], 0, 0
    while True:
        start = plain.find("{", position)
        if start < 0:
            break
        end = balanced_span(plain, start)
        found = as_calls(loose_json(plain[start:end]), names) if end else []
        if found:
            kept.append(text[last_end:start])
            calls += found
            position = last_end = end
        else:
            position = start + 1
    if not calls or plain[last_end:].strip():
        return text, []
    return "".join(kept), calls


class VisibleText:
    """Passes a stream through, holding back reasoning and tool-call blocks.

    Text is only released once it cannot be the start of something hidden. Code
    fences that cannot be a tool call pass straight through. One that might be is
    held until it closes: a real call disappears and anything else is released.

    A call with no fence is held the same way, and only the whole reply can settle it,
    since a call has to end the reply. So what is still held at the end comes back from
    ``flush``, and the router drops it if the reply turned out to hold a call.

    A reply can be held for a long time, so nothing is read twice. ``at`` is where the
    unread text starts, which saves copying the rest each time a little is released.
    ``looked`` is how far past ``at`` a fence or a run of calls is known to be settled.
    """

    def __init__(self, offered: Sequence[Capability] = ()) -> None:
        self.names = frozenset(capability.name for capability in offered)
        self.markers = ("<think>", "<tool_call>", "```", "{") if self.names else ("<think>",)
        self.opening = re.compile("|".join(map(re.escape, self.markers)))
        self.held = ""
        self.at = 0
        self.looked = 0
        self.value: OpenValue | None = None  # the bare call still arriving
        self.state = "text"  # or think, tool, fence, code, bare
        self.steps = {"text": self._text, "think": self._think, "tool": self._tool, "fence": self._fence, "code": self._code, "bare": self._bare}

    def feed(self, text: str) -> str:
        self.held += text
        released: list[str] = []
        while self.at < len(self.held) and self.steps[self.state](released):
            pass
        self.held, self.at = self.held[self.at :], 0
        return "".join(released)

    def _release(self, released: list[str], size: int) -> None:
        released.append(self.held[self.at : self.at + size])
        self.at += size

    def _settled(self) -> None:
        self.state, self.looked, self.value = "text", 0, None

    def _skip_past(self, marker: str) -> bool:
        end = self.held.find(marker, self.at)
        if end < 0:
            self.at = max(self.at, len(self.held) - (len(marker) - 1))
            return False
        self.held, self.at = self.held[end + len(marker) :].lstrip("\n"), 0
        self._settled()
        return True

    def _think(self, released: list[str]) -> bool:
        return self._skip_past("</think>")

    def _tool(self, released: list[str]) -> bool:
        return self._skip_past("</tool_call>")

    def _code(self, released: list[str]) -> bool:
        end = self.held.find("```", self.at)
        if end >= 0:
            self._release(released, end + 3 - self.at)
            self.state = "text"
            return True
        self._release(released, len(self.held) - self.at - self._partial("```"))
        return False

    def _fence(self, released: list[str]) -> bool:
        first_line = self.held.find("\n", self.at)
        if first_line < 0:
            return False
        if self.held[self.at + 3 : first_line].strip().lower() not in TOOL_FENCE_TAGS:
            self._release(released, first_line - self.at)
            self.state = "code"
            return True
        end = self.held.find("```", max(first_line, self.at + self.looked))
        if end < 0:
            self.looked = len(self.held) - self.at - 2  # a closing fence can still be cut in two
            return False
        block = self.held[self.at : end + 3]
        call = self.held[first_line:end]
        self.at = end + 3
        self._settled()
        if not as_calls(loose_json(call), self.names):
            released.append(block)
        return True

    def _bare(self, released: list[str]) -> bool:
        if self._may_be_calls():
            return False
        self._release(released, 1)  # only the brace, as a call can still start after it
        self._settled()
        return True

    def _may_be_calls(self) -> bool:
        """Whether what is held is nothing but tool calls so far, as ``bare_calls`` requires."""
        while True:
            if self.value is None:
                rest = self.held[self.at + self.looked :].lstrip()
                if not rest:
                    return True
                if rest[0] != "{":
                    return False
                self.looked = len(self.held) - len(rest) - self.at
                self.value = OpenValue()
            start = self.at + self.looked
            if not self.value.read(self.held, start):
                return True  # still arriving
            end = start + self.value.size
            self.value = None
            if not as_calls(loose_json(self.held[start:end]), self.names):
                return False
            self.looked = end - self.at

    def _partial(self, marker: str) -> int:
        """How much of the end of the held text could be the start of a marker."""
        for size in range(len(marker) - 1, 0, -1):
            if self.held.endswith(marker[:size], self.at):
                return size
        return 0

    def _text(self, released: list[str]) -> bool:
        hit = self.opening.search(self.held, self.at)
        if hit is None:
            self._release(released, len(self.held) - self.at - max(self._partial(marker) for marker in self.markers))
            return False
        self._release(released, hit.start() - self.at)
        marker = hit.group()
        if marker in ("```", "{"):
            self.state = "fence" if marker == "```" else "bare"
        else:
            self.at += len(marker)
            self.state = "think" if marker == "<think>" else "tool"
        return True

    def flush(self) -> str:
        held, state = self.held, self.state
        self.held = ""
        self._settled()
        return "" if state in ("think", "tool") else held


def signature(tool: Capability) -> str:
    schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
    required = set(schema.get("required", []))
    fields = []
    for name, entry in schema.get("properties", {}).items():
        kind = entry.get("type", "any") if isinstance(entry, dict) else "any"
        fields.append(f"{name}: {kind}" if name in required else f"{name}?: {kind}")
    return f"{tool.name}({', '.join(fields)})"


def example_call(tool: Capability) -> str:
    schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
    properties = schema.get("properties", {})
    arguments = {name: ["..."] if properties.get(name, {}).get("type") == "array" else "..." for name in schema.get("required", [])}
    return json.dumps({"name": tool.name, "arguments": arguments})


def tool_instructions(tools: Sequence[Capability]) -> str:
    """Tested against a real 0.6B model, because small models follow this literally.

    With each tool's JSON schema it called nothing in 10 tries. Given `{"name": "value"}`
    as the example it copied those very words. A short signature and an example built
    from a real tool got the right call in 10 of 10, on most wordings of a request.
    """
    lines = [
        "You can use tools. To use one, reply with exactly this and nothing else:",
        f"<tool_call>\n{example_call(tools[0])}\n</tool_call>",
        "The result comes back to you, and you can then use another tool or answer. If no tool is needed, just answer.",
        "Tools:",
    ]
    lines += [f"- {signature(tool)}: {tool.summary}" for tool in tools]
    return "\n".join(lines)


def with_tools_in_prompt(messages: Sequence[dict[str, Any]], tools: Sequence[Capability]) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    if not tools:
        return result
    text = tool_instructions(tools)
    if result and result[0].get("role") == "system":
        result[0]["content"] = f"{result[0]['content']}\n\n{text}"
    else:
        result.insert(0, {"role": "system", "content": text})
    return result


def plain_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Only the roles and plain strings that every server and chat template accepts."""
    return [
        {"role": message["role"] if message["role"] in ("system", "user", "assistant") else "user", "content": str(message.get("content") or "")}
        for message in messages
    ]


def seconds_to_wait(read: float) -> httpx.Timeout:
    return httpx.Timeout(read, connect=CONNECT_SECONDS)


def gather_call(streamed: dict[int, dict[str, str]], call: dict[str, Any]) -> None:
    """Add one streamed piece of a native tool call to the call it belongs to."""
    held = streamed.setdefault(int(call.get("index") or 0), {"id": "", "name": "", "arguments": ""})
    function = call.get("function") or {}
    held["id"] = call.get("id") or held["id"]
    held["name"] = function.get("name") or held["name"]
    held["arguments"] += function.get("arguments") or ""


class HeldClient:
    """One HTTP client, made on first use and kept, so a call can find a connection already open.

    It serves many threads at once, so a timeout belongs to each call, and no call
    waits for another's connection. It keeps no cookies, because a server's cookie
    must not follow later calls.
    """

    def __init__(self, options: Callable[[], dict[str, Any]]) -> None:
        self.options = options
        self.lock = threading.Lock()
        self.client: httpx.Client | None = None

    def __call__(self) -> httpx.Client:
        with self.lock:
            if self.client is None:
                limits = httpx.Limits(max_connections=None, max_keepalive_connections=20, keepalive_expiry=IDLE_CONNECTION_SECONDS)
                self.client = httpx.Client(limits=limits, cookies=CookieJar(DefaultCookiePolicy(allowed_domains=[])), **self.options())
            return self.client

    def close(self) -> None:
        with self.lock:
            client, self.client = self.client, None
        if client is not None:
            client.close()


class OpenAICompatible:
    """Any server with the OpenAI chat API."""

    kind = "openai"
    streams_tool_calls = False  # a whole reply is the safe way to read most servers' calls

    def __init__(
        self,
        name: str,
        base_url: str,
        key_env_or_file: str | None = None,
        reach: Reach = Reach.CLOUD,
        *,
        tool_mode: str = "auto",
        daily_tokens: int | None = None,
        read_seconds: float = READ_SECONDS,
        model: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = api_root(base_url)
        self.key = key_reference(key_env_or_file)
        self.reach = reach
        self.tool_mode = tool_mode
        self.daily_tokens = daily_tokens
        self.read_seconds = read_seconds
        self.default_model = model
        self._models: tuple[float, list[str]] | None = None
        self._client = HeldClient(self._client_options)

    def _client_options(self) -> dict[str, Any]:
        return {
            "verify": client_verify(),  # a pool with its own authority is checked against it
            # A proxy set for the internet has no business seeing loopback traffic.
            "trust_env": not is_loopback(self.base_url),
        }

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        key = read_key(self.key)
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _clean(self, text: str) -> str:
        """Make any server text safe to store and show."""
        short = " ".join(str(text).split())[:ERROR_CHARS]
        try:
            key = read_key(self.key)
        except ProviderError:
            key = None
        return short.replace(key, "[key]") if key else short

    def _unreachable(self, error: httpx.HTTPError, what: str = "cannot be reached") -> ProviderError:
        if intercepted(error):
            return ProviderError(
                f"{self.name}'s certificate could not be verified. If antivirus scans HTTPS on this machine, it is "
                "probably replacing certificates: exclude this address from scanning"
            )
        return ProviderError(f"{self.name} {what}: {type(error).__name__}")

    def _failed(self, response: httpx.Response) -> ProviderError:
        kind = RequestRefused if response.status_code in REQUEST_FAULT_STATUSES else ProviderError
        return kind(f"{self.name} answered HTTP {response.status_code}: {self._clean(response.text)}")

    def _post(self, path: str, body: dict[str, Any], read: float, doing: str) -> httpx.Response:
        try:
            return self._client().post(f"{self.base_url}/{path}", json=body, headers=self._headers(), timeout=seconds_to_wait(read))
        except httpx.TimeoutException:
            raise ProviderError(f"{self.name} went silent while {doing}") from None
        except httpx.HTTPError as error:
            raise self._unreachable(error) from None

    def models(self, client: httpx.Client | None = None) -> list[str]:
        cached = self._models
        now = time.monotonic()
        if cached and now - cached[0] < MODELS_FRESH_SECONDS:
            return cached[1]
        try:
            response = (client or self._client()).get(f"{self.base_url}/models", headers=self._headers(), timeout=seconds_to_wait(HEALTH_SECONDS))
        except httpx.HTTPError as error:
            raise self._unreachable(error) from None
        if response.status_code != 200:
            raise self._failed(response)
        try:
            names = [str(item["id"]) for item in response.json().get("data", [])]
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ProviderError(f"{self.name} sent a model list that could not be read") from None
        self._models = (now, names)
        return names

    def current_model(self) -> str | None:
        if self.default_model:
            return self.default_model
        names = self.models()
        return names[0] if names else None

    def health(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            self._models = None
            # A held connection can outlive the server's listener, and new calls would find that out.
            with httpx.Client(**self._client_options()) as fresh:
                names = self.models(fresh)
        except ProviderError as error:
            return {"up": False, "models": [], "error": str(error), "seconds": time.perf_counter() - started}
        return {"up": True, "models": names, "error": None, "seconds": time.perf_counter() - started}

    def _native(self, request: ModelRequest) -> bool:
        return bool(request.tools) and self.tool_mode in ("auto", "native")

    def _body(self, request: ModelRequest, model: str, stream: bool, native: bool) -> dict[str, Any]:
        messages = request.messages if native else with_tools_in_prompt(request.messages, request.tools)
        body: dict[str, Any] = {
            "model": model,
            "messages": plain_messages(messages),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.stop:
            body["stop"] = list(request.stop)
        if native:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": native_name(tool.name),
                        "description": tool.summary,
                        "parameters": tool.input_schema or {"type": "object", "properties": {}},
                    },
                }
                for tool in request.tools
            ]
        return body

    def complete(self, request: ModelRequest, model: str) -> ModelReply:
        native = self._native(request)
        try:
            return self._once(request, model, native)
        except ToolsNotSupported:
            reply = self._once(request, model, False)
            self.tool_mode = "prompt"  # it worked this way, so there is no need to ask again
            return reply

    def _once(self, request: ModelRequest, model: str, native: bool) -> ModelReply:
        started = time.perf_counter()
        body = self._body(request, model, False, native)
        response = self._post("chat/completions", body, max(self.read_seconds, WHOLE_REPLY_SECONDS), "answering")
        if native and self.tool_mode == "auto" and response.status_code in NO_TOOLS_STATUSES:
            raise ToolsNotSupported
        if response.status_code != 200:
            raise self._failed(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError(f"{self.name} sent a reply that could not be read") from None
        raw = str(message.get("content") or "")
        answered_by = payload.get("model") or model
        return self._reply(raw, message.get("tool_calls") or [], request, body, payload.get("usage") or {}, answered_by, started, choice.get("finish_reason"))

    def _reply(
        self, raw: str, native_calls: list[Any], request: ModelRequest, body: dict[str, Any], usage: dict[str, Any], model: Any, started: float, ended: Any
    ) -> ModelReply:
        text, calls = parse_tool_calls(raw, request.tools)
        found = as_calls(native_calls, frozenset(tool.name for tool in request.tools))
        return ModelReply(
            text=text,
            tool_calls=tuple(found) or calls,
            model=str(model),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens") or sum(estimate_tokens(item["content"]) for item in body["messages"])),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(raw)),
            seconds=time.perf_counter() - started,
            finish_reason="length" if ended == "length" else "stop",
        )

    def stream(self, request: ModelRequest, model: str) -> Iterator[str | ModelReply]:
        """Text as it comes, then the finished ModelReply last."""
        native = self._native(request)
        if native and not self.streams_tool_calls:
            # Native calls arrive in pieces. Only the finished reply can be parsed safely.
            reply = self.complete(request, model)
            if reply.text:
                yield reply.text
            yield reply
            return
        started = time.perf_counter()
        body = self._body(request, model, True, native)
        parts: list[str] = []
        streamed: dict[int, dict[str, str]] = {}
        refused = False
        usage: dict[str, Any] = {}
        answered_by = model
        finished = False
        ended = None
        try:
            opened = self._client().stream(
                "POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=seconds_to_wait(self.read_seconds)
            )
            with opened as response:
                refused = native and self.tool_mode == "auto" and response.status_code in NO_TOOLS_STATUSES
                if response.status_code != 200 and not refused:
                    response.read()
                    raise self._failed(response)
                for event in () if refused else self._events(response):
                    if event is None:
                        finished = True
                        break
                    answered_by = event.get("model") or answered_by
                    usage = event.get("usage") or usage
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            parts.append(piece)
                            yield piece
                        for call in delta.get("tool_calls") or []:
                            gather_call(streamed, call)
                        ended = choice.get("finish_reason") or ended
                        finished = finished or bool(ended)
        except httpx.TimeoutException:
            raise ProviderError(f"{self.name} went silent for {self.read_seconds:.0f} s") from None
        except httpx.HTTPError as error:
            raise self._unreachable(error, "dropped the connection") from None
        if refused:
            self.tool_mode = "prompt"  # this model has no tool calling of its own
            yield from self.stream(request, model)
            return
        if not finished:
            raise ProviderError(f"{self.name} stopped before finishing its reply")
        yield self._reply("".join(parts), [streamed[index] for index in sorted(streamed)], request, body, usage, answered_by, started, ended)

    def _events(self, response: httpx.Response) -> Iterator[dict[str, Any] | None]:
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                yield None
                return
            try:
                event = json.loads(data)
            except ValueError:
                continue  # a keep-alive or a comment
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                # Headers are long gone by now, so failures travel in the stream.
                error = event["error"]
                detail = error.get("message") if isinstance(error, dict) else error
                raise ProviderError(f"{self.name} failed partway: {self._clean(str(detail))}")
            yield event

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        response = self._post("embeddings", {"model": model, "input": list(texts)}, self.read_seconds, "embedding")
        if response.status_code != 200:
            raise self._failed(response)
        try:
            rows = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
            return [[float(value) for value in row["embedding"]] for row in rows]
        except (ValueError, KeyError, TypeError):
            raise ProviderError(f"{self.name} sent embeddings that could not be read") from None


class MeshProvider(OpenAICompatible):
    """The owner's own pool.

    A reply is always streamed, even when nobody is watching it: a pool can take
    minutes over a reply, and only a stream can tell that from silence. So tool calls
    are read from the stream too.
    """

    kind = "mesh"
    streams_tool_calls = True

    def __init__(self, name: str, base_url: str, key_env_or_file: str | None = None, reach: Reach = Reach.POOL, **options: Any) -> None:
        super().__init__(name, base_url, key_env_or_file, reach, **options)

    def _body(self, request: ModelRequest, model: str, stream: bool, native: bool) -> dict[str, Any]:
        body = super()._body(request, model, stream, native)
        body["enable_thinking"] = request.role in THINKING_ROLES
        return body

    def complete(self, request: ModelRequest, model: str) -> ModelReply:
        for item in self.stream(request, model):
            if isinstance(item, ModelReply):
                return item
        raise ProviderError(f"{self.name} stopped before finishing its reply")


class Providers:
    """The providers this install knows, plus any a part registers while running."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("providers", SCHEMA)
        self.registered: dict[str, Any] = {}
        self.cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self.built: list[Any] = []  # removed ones too, as a reply on one may still be running

    def close(self) -> None:
        for provider in self.built:
            provider.close()

    def add(
        self,
        name: str,
        base_url: str,
        *,
        mesh: bool = False,
        key: str | None = None,
        reach: Reach | str | None = None,
        tool_mode: str = "auto",
        daily_tokens: int | None = None,
        read_seconds: float = READ_SECONDS,
        model: str | None = None,
        by: str = LOCAL_USER,
    ) -> Any:
        if not PROVIDER_NAME.fullmatch(name):
            raise ProviderRefused(f"{name!r} is not a usable name. Use letters, numbers, dots, dashes, or underscores")
        if tool_mode not in TOOL_MODES:
            raise ProviderRefused(f"Tool mode is one of: {', '.join(TOOL_MODES)}")
        # Assume the worst unless told otherwise: an unknown server is a third party.
        limit = reach_named(reach) if reach is not None else (Reach.POOL if mesh else Reach.CLOUD)
        root = api_root(base_url)
        if limit is Reach.MACHINE and not is_loopback(root):
            raise ProviderRefused(f"{root} is not on this machine, so it cannot have reach machine. Use pool, peer, or cloud")
        with self.store.transaction():
            if self.store.one("SELECT name FROM providers WHERE name = ?", (name,)):
                raise ProviderRefused(f"{name} already exists. Remove it first to change it")
            self.store.execute(
                "INSERT INTO providers (name, kind, base_url, key_reference, reach, tool_mode, daily_tokens, read_seconds, model, added_at, added_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name, "mesh" if mesh else "openai", root, key_reference(key), int(limit), tool_mode,
                    daily_tokens, read_seconds, model, self.assistant.clock(), by,
                ),
            )
            self.assistant.audit.record(actor(by), "provider.add", name, {"reach": limit.name.lower(), "kind": "mesh" if mesh else "openai"})
        return self.get(name)

    def remove(self, name: str, by: str = LOCAL_USER) -> None:
        with self.store.transaction():
            cursor = self.store.connection.execute("DELETE FROM providers WHERE name = ?", (name,))
            if cursor.rowcount == 0:
                raise ProviderRefused(f"There is no provider {name}. `provider list` shows them")
            self.assistant.audit.record(actor(by), "provider.remove", name)
        self.cache.pop(name, None)

    def register(self, provider: Any) -> None:
        """Add a provider for as long as this process runs, such as a peer's."""
        self.registered[provider.name] = provider

    def _build(self, row: Any) -> Any:
        fingerprint = tuple(row)
        held = self.cache.get(row["name"])
        if held and held[0] == fingerprint:
            return held[1]
        kind = MeshProvider if row["kind"] == "mesh" else OpenAICompatible
        options = {"daily_tokens": row["daily_tokens"], "read_seconds": row["read_seconds"], "model": row["model"], "tool_mode": row["tool_mode"]}
        provider = kind(row["name"], row["base_url"], row["key_reference"], Reach(row["reach"]), **options)
        self.cache[row["name"]] = (fingerprint, provider)
        self.built.append(provider)
        return provider

    def get(self, name: str) -> Any | None:
        if name in self.registered:
            return self.registered[name]
        row = self.store.one("SELECT * FROM providers WHERE name = ?", (name,))
        return self._build(row) if row else None

    def all(self) -> list[Any]:
        rows = self.store.query("SELECT * FROM providers ORDER BY added_at, name")
        return [self._build(row) for row in rows] + list(self.registered.values())


def attach(assistant: Any) -> None:
    assistant.providers = Providers(assistant)


def detach(assistant: Any) -> None:
    close = getattr(assistant.providers, "close", None)  # another registry may stand in for this one
    if close is not None:
        close()


def add_commands(commands: Any) -> None:
    parser = commands.add_parser("provider", help="where models come from: the mesh, or any OpenAI-compatible server")
    actions = parser.add_subparsers(dest="provider_command", required=True)
    add = actions.add_parser("add", help="add a model provider")
    add.add_argument("name", help="what to call it")
    add.add_argument("url", help="its address, such as http://127.0.0.1:8000")
    add.add_argument("--mesh", action="store_true", help="this is an iSharded pool")
    add.add_argument("--key-env", default=None, help="the environment variable holding its key")
    add.add_argument("--key-file", default=None, help="the file holding its key")
    add.add_argument(
        "--reach", default=None, choices=[item.name.lower() for item in Reach],
        help="how far text sent to it travels; default is pool for a mesh and cloud otherwise",
    )
    add.add_argument("--tools", default="auto", choices=TOOL_MODES, help="how tools are offered; auto uses native tool calls if the server takes them")
    add.add_argument("--daily-tokens", type=int, default=None, help="stop using it after this many tokens in a day")
    add.add_argument("--read-seconds", type=float, default=READ_SECONDS, help="how long silence may last before it counts as failed")
    add.add_argument("--model", default=None, help="the model to use; default is the first it lists")
    add.set_defaults(run=plainly(run_add))
    listing = actions.add_parser("list", help="show providers, and check each one is up")
    listing.add_argument("--no-check", action="store_true", help="do not contact them")
    listing.set_defaults(run=plainly(run_list))
    remove = actions.add_parser("remove", help="remove a provider")
    remove.add_argument("name", help="its name")
    remove.set_defaults(run=plainly(run_remove))
    test = actions.add_parser("test", help="ask one provider whether it is up, on a fresh connection")
    test.add_argument("name", help="its name")
    test.set_defaults(run=plainly(run_test))


def run_add(assistant: Any, args: argparse.Namespace) -> int:
    if args.key_env and args.key_file:
        emit("Give one of --key-env and --key-file.")
        return 2
    key = f"env:{args.key_env}" if args.key_env else f"file:{args.key_file}" if args.key_file else None
    provider = assistant.providers.add(
        args.name, args.url, mesh=args.mesh, key=key, reach=args.reach, tool_mode=args.tools,
        daily_tokens=args.daily_tokens, read_seconds=args.read_seconds, model=args.model,
    )
    emit(f"Added {provider.name}, with reach {provider.reach.name.lower()}.")
    return 0


def run_list(assistant: Any, args: argparse.Namespace) -> int:
    rows = []
    for provider in assistant.providers.all():
        row = {"name": provider.name, "kind": provider.kind, "reach": provider.reach.name.lower(), "url": getattr(provider, "base_url", "")}
        if not args.no_check:
            health = provider.health()
            row["up"] = "yes" if health["up"] else "no"
            row["models"] = ", ".join(health["models"]) or (health["error"] or "none loaded")
        rows.append(row)
    if not rows:
        emit("No providers yet. Add one with `provider add`.")
        return 0
    emit(table(rows, ["name", "kind", "reach", "url"] + ([] if args.no_check else ["up", "models"])))
    return 0


def run_remove(assistant: Any, args: argparse.Namespace) -> int:
    assistant.providers.remove(args.name)
    emit(f"Removed {args.name}.")
    return 0


def run_test(assistant: Any, args: argparse.Namespace) -> int:
    provider = assistant.providers.get(args.name)
    if provider is None:
        raise ProviderRefused(f"There is no provider {args.name}. `provider list` shows them")
    health = provider.health()
    if not health["up"]:
        emit(f"{args.name} is down: {health['error']}")
        return 1
    emit(f"{args.name} is up, in {health['seconds']:.2f} s. Models: {', '.join(health['models']) or 'none loaded'}.")
    return 0
