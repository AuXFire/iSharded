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
from collections.abc import Iterator, Sequence
import ipaddress
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .contracts import Capability, ModelReply, ModelRequest, Reach, ToolCall
from .mind import actor, emit, LOCAL_USER, plainly, reach_named, table

CONNECT_SECONDS = 10.0
READ_SECONDS = 120.0  # silence this long means the link is gone; covers a long prefill
WHOLE_REPLY_SECONDS = 300.0  # a reply with no stream has nothing to read until it is done
HEALTH_SECONDS = 5.0
MODELS_FRESH_SECONDS = 30.0
MAX_TOOL_JSON = 200_000
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


class ProviderRefused(ValueError):
    pass


def estimate_tokens(text: str) -> int:
    """A rough count, used only when a server does not report one."""
    return len(text) // 3 + 1


def is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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


def balanced_span(text: str, start: int) -> int | None:
    """Where the JSON value opening at ``start`` closes, respecting strings."""
    depth, quote, escaped = 0, None, False
    for index in range(start, len(text)):
        character = text[index]
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character == '"':
            quote = character
        elif character in "{[":
            depth += 1
        elif character in "}]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


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
        except ValueError:
            continue
    if end is None:
        return None
    try:  # single quotes and Python's True and None
        return ast.literal_eval(re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", candidate))))
    except (ValueError, SyntaxError, MemoryError, RecursionError):
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
    if arguments is None or not isinstance(arguments, dict):
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
    """

    def __init__(self, offered: Sequence[Capability] = ()) -> None:
        self.names = frozenset(capability.name for capability in offered)
        self.markers = ("<think>", "<tool_call>", "```") if self.names else ("<think>",)
        self.held = ""
        self.state = "text"  # or think, tool, fence, code

    def feed(self, text: str) -> str:
        self.held += text
        released: list[str] = []
        step = {"text": self._text, "think": self._think, "tool": self._tool, "fence": self._fence, "code": self._code}
        while self.held and step[self.state](released):
            pass
        return "".join(released)

    def _skip_past(self, marker: str) -> bool:
        end = self.held.find(marker)
        if end < 0:
            self.held = self.held[-(len(marker) - 1) :]
            return False
        self.held = self.held[end + len(marker) :].lstrip("\n")
        self.state = "text"
        return True

    def _think(self, released: list[str]) -> bool:
        return self._skip_past("</think>")

    def _tool(self, released: list[str]) -> bool:
        return self._skip_past("</tool_call>")

    def _code(self, released: list[str]) -> bool:
        end = self.held.find("```")
        if end >= 0:
            released.append(self.held[: end + 3])
            self.held = self.held[end + 3 :]
            self.state = "text"
            return True
        keep = self._partial("```")
        released.append(self.held[: len(self.held) - keep])
        self.held = self.held[len(self.held) - keep :]
        return False

    def _fence(self, released: list[str]) -> bool:
        first_line = self.held.find("\n")
        if first_line < 0:
            return False
        if self.held[3:first_line].strip().lower() not in TOOL_FENCE_TAGS:
            released.append(self.held[:first_line])
            self.held = self.held[first_line:]
            self.state = "code"
            return True
        end = self.held.find("```", first_line)
        if end < 0:
            return False
        block = self.held[: end + 3]
        self.held = self.held[end + 3 :]
        self.state = "text"
        if not as_calls(loose_json(block[first_line:end]), self.names):
            released.append(block)
        return True

    def _partial(self, marker: str) -> int:
        """How much of the end of the held text could be the start of a marker."""
        for size in range(len(marker) - 1, 0, -1):
            if self.held.endswith(marker[:size]):
                return size
        return 0

    def _text(self, released: list[str]) -> bool:
        hits = [(self.held.find(marker), marker) for marker in self.markers if marker in self.held]
        if hits:
            start, marker = min(hits)
            released.append(self.held[:start])
            if marker == "```":
                self.held, self.state = self.held[start:], "fence"
            else:
                self.held = self.held[start + len(marker) :]
                self.state = "think" if marker == "<think>" else "tool"
            return True
        keep = max(self._partial(marker) for marker in self.markers)
        released.append(self.held[: len(self.held) - keep])
        self.held = self.held[len(self.held) - keep :]
        return False

    def flush(self) -> str:
        held, state = self.held, self.state
        self.held, self.state = "", "text"
        return "" if state in ("think", "tool") else held


def tool_instructions(tools: Sequence[Capability]) -> str:
    lines = [
        "You can use tools. To use one, reply with a fenced block like this and nothing after it:",
        '```tool\n{"tool": "area.verb", "arguments": {"name": "value"}}\n```',
        "The result comes back to you, and you can then use another tool or answer. If no tool is needed, just answer.",
        "Tools:",
    ]
    for tool in tools:
        schema = json.dumps(tool.input_schema, separators=(",", ":")) if tool.input_schema else "{}"
        lines.append(f"- {tool.name}: {tool.summary} Arguments: {schema}")
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


class OpenAICompatible:
    """Any server with the OpenAI chat API."""

    kind = "openai"

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

    def _client(self, read: float) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(read, connect=CONNECT_SECONDS),
            # A proxy set for the internet has no business seeing loopback traffic.
            trust_env=not is_loopback(self.base_url),
        )

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

    def _failed(self, response: httpx.Response) -> ProviderError:
        return ProviderError(f"{self.name} answered HTTP {response.status_code}: {self._clean(response.text)}")

    def models(self) -> list[str]:
        cached = self._models
        now = time.monotonic()
        if cached and now - cached[0] < MODELS_FRESH_SECONDS:
            return cached[1]
        with self._client(HEALTH_SECONDS) as client:
            try:
                response = client.get(f"{self.base_url}/models", headers=self._headers())
            except httpx.HTTPError as error:
                raise ProviderError(f"{self.name} cannot be reached: {type(error).__name__}") from None
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
            names = self.models()
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
        with self._client(max(self.read_seconds, WHOLE_REPLY_SECONDS)) as client:
            try:
                response = client.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers())
            except httpx.TimeoutException:
                raise ProviderError(f"{self.name} went silent while answering") from None
            except httpx.HTTPError as error:
                raise ProviderError(f"{self.name} cannot be reached: {type(error).__name__}") from None
        if native and self.tool_mode == "auto" and response.status_code in (400, 404, 422, 501):
            raise ToolsNotSupported
        if response.status_code != 200:
            raise self._failed(response)
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError(f"{self.name} sent a reply that could not be read") from None
        raw = str(message.get("content") or "")
        text, calls = parse_tool_calls(raw, request.tools)
        native_calls = self._native_calls(message, request.tools)
        usage = payload.get("usage") or {}
        return ModelReply(
            text=text,
            tool_calls=tuple(native_calls) or calls,
            model=str(payload.get("model") or model),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens") or sum(estimate_tokens(str(item.get("content") or "")) for item in body["messages"])),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(raw)),
            seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _native_calls(message: dict[str, Any], tools: Sequence[Capability]) -> list[ToolCall]:
        names = frozenset(tool.name for tool in tools)
        return as_calls(message.get("tool_calls") or [], names)

    def stream(self, request: ModelRequest, model: str) -> Iterator[str | ModelReply]:
        """Text as it comes, then the finished ModelReply last."""
        if self._native(request):
            # Native calls arrive in pieces. Only the finished reply can be parsed safely.
            reply = self.complete(request, model)
            if reply.text:
                yield reply.text
            yield reply
            return
        started = time.perf_counter()
        body = self._body(request, model, True, False)
        parts: list[str] = []
        usage: dict[str, Any] = {}
        answered_by = model
        finished = False
        with self._client(self.read_seconds) as client:
            try:
                with client.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers()) as response:
                    if response.status_code != 200:
                        response.read()
                        raise self._failed(response)
                    for event in self._events(response):
                        if event is None:
                            finished = True
                            break
                        answered_by = event.get("model") or answered_by
                        usage = event.get("usage") or usage
                        for choice in event.get("choices") or []:
                            piece = (choice.get("delta") or {}).get("content")
                            if piece:
                                parts.append(piece)
                                yield piece
                            finished = finished or bool(choice.get("finish_reason"))
            except httpx.TimeoutException:
                raise ProviderError(f"{self.name} went silent for {self.read_seconds:.0f} s") from None
            except httpx.HTTPError as error:
                raise ProviderError(f"{self.name} dropped the connection: {type(error).__name__}") from None
        if not finished:
            raise ProviderError(f"{self.name} stopped before finishing its reply")
        raw = "".join(parts)
        text, calls = parse_tool_calls(raw, request.tools)
        yield ModelReply(
            text=text,
            tool_calls=calls,
            model=str(answered_by),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens") or sum(estimate_tokens(item["content"]) for item in body["messages"])),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(raw)),
            seconds=time.perf_counter() - started,
        )

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
            if isinstance(event, dict) and event.get("error"):
                # Headers are long gone by now, so failures travel in the stream.
                error = event["error"]
                detail = error.get("message") if isinstance(error, dict) else error
                raise ProviderError(f"{self.name} failed partway: {self._clean(str(detail))}")
            if isinstance(event, dict):
                yield event

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        with self._client(self.read_seconds) as client:
            try:
                response = client.post(f"{self.base_url}/embeddings", json={"model": model, "input": list(texts)}, headers=self._headers())
            except httpx.TimeoutException:
                raise ProviderError(f"{self.name} went silent while embedding") from None
            except httpx.HTTPError as error:
                raise ProviderError(f"{self.name} cannot be reached: {type(error).__name__}") from None
        if response.status_code != 200:
            raise self._failed(response)
        try:
            rows = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
            return [[float(value) for value in row["embedding"]] for row in rows]
        except (ValueError, KeyError, TypeError):
            raise ProviderError(f"{self.name} sent embeddings that could not be read") from None


class ToolsNotSupported(Exception):
    pass


class MeshProvider(OpenAICompatible):
    """The owner's own pool.

    Its chat API takes plain strings and has no tool calling, so tools go in the
    prompt. A reply is always streamed, even when nobody is watching it: a pool
    can take minutes over a reply, and only a stream can tell that from silence.
    """

    kind = "mesh"

    def __init__(self, name: str, base_url: str, key_env_or_file: str | None = None, reach: Reach = Reach.POOL, **options: Any) -> None:
        options["tool_mode"] = "prompt"
        super().__init__(name, base_url, key_env_or_file, reach, **options)

    def _body(self, request: ModelRequest, model: str, stream: bool, native: bool) -> dict[str, Any]:
        body = super()._body(request, model, stream, False)
        body["enable_thinking"] = request.role in THINKING_ROLES
        return body

    def complete(self, request: ModelRequest, model: str) -> ModelReply:
        for item in self.stream(request, model):
            if isinstance(item, ModelReply):
                return item
        raise ProviderError(f"{self.name} stopped before finishing its reply")


def api_root(url: str) -> str:
    """A server's address, with /v1 added if it was given as a bare host."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ProviderRefused(f"{url!r} is not an http or https address")
    if parts.username or parts.password or parts.query:
        raise ProviderRefused("Put the key in an environment variable or a file, not in the address")
    root = url.rstrip("/")
    return root if parts.path.strip("/") else f"{root}/v1"


class Providers:
    """The providers this install knows, plus any a part registers while running."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("providers", SCHEMA)
        self.registered: dict[str, Any] = {}
        self.cache: dict[str, tuple[tuple[Any, ...], Any]] = {}

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
                    name, "mesh" if mesh else "openai", root, key_reference(key), int(limit), "prompt" if mesh else tool_mode,
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
        options = {"daily_tokens": row["daily_tokens"], "read_seconds": row["read_seconds"], "model": row["model"]}
        if row["kind"] != "mesh":
            options["tool_mode"] = row["tool_mode"]
        provider = kind(row["name"], row["base_url"], row["key_reference"], Reach(row["reach"]), **options)
        self.cache[row["name"]] = (fingerprint, provider)
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


def add_commands(commands: Any) -> None:
    parser = commands.add_parser("provider", help="where models come from: the mesh, or any OpenAI-compatible server")
    actions = parser.add_subparsers(dest="provider_command", required=True)
    add = actions.add_parser("add", help="add a model provider")
    add.add_argument("name", help="what to call it")
    add.add_argument("url", help="its address, such as http://127.0.0.1:8000")
    add.add_argument("--mesh", action="store_true", help="this is a ShardMesh pool")
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
