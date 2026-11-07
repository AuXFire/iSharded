"""Tool calling: what a model's chat template offers, and finding calls in its replies.

Each family was trained on its own way of writing a call, and the chat template
shows which, so the template picks the parser and the model's name never does.
A template that shows no format we can read is refused, because pasting tools
into a prompt the model was not trained on would look like it worked.

A reply reaches a client through ``ReplyScanner`` whether it streams or not, so
the two can never disagree about what counts as a call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import islice
import json
import re
from typing import Any
import uuid

DEFAULT_REPLY_TOKENS = 32
# A call must fit whole, and clients following the OpenAI API send no limit at all.
TOOL_REPLY_TOKENS = 4096
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# Written as special tokens by some models. Decoding drops those unless told not to.
MARKER_TOKENS = ("<tool_call>", "</tool_call>", "[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "<|python_tag|>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", THINK_OPEN, THINK_CLOSE)
OBJECT = json.JSONDecoder()
TOOL_CHOICE_RULE = 'tool_choice must be auto, none, required, or {"type": "function", "function": {"name": ...}}'


class ToolRequestError(ValueError):
    """The request itself is wrong, and the message says how."""


def tool_format(template: str | None) -> str | None:
    """Which way of writing a call a chat template teaches, if any we can read."""
    text = template or ""
    if "tools" not in text:
        return None
    if "<function=" in text and "<parameter=" in text:
        return "xml"
    if "<arg_key>" in text:  # GLM puts its own format inside <tool_call>
        return "glm"
    if "<tool_call>" in text:
        return "hermes"
    if "[TOOL_CALLS]" in text:
        return "mistral"
    if '"parameters"' in text and "tool_call" in text:
        return "llama3"
    return None


def text_of(content: Any) -> str:
    """A message's content as the plain string every chat template can handle."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    pieces = []
    for part in content:
        if part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise ToolRequestError(f"only text parts are supported here, not {part.get('type')!r}")
        pieces.append(part["text"])
    return "".join(pieces)


def _call_for_template(call: dict[str, Any]) -> dict[str, Any]:
    """OpenAI sends arguments as JSON text. Templates want the object: Qwen3.5's
    ``|items`` fails on a string, and Llama's ``|tojson`` would encode it twice."""
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ToolRequestError("each tool call needs a function with a name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            raise ToolRequestError(f"the arguments of the call to {function['name']} are not valid JSON") from None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolRequestError(f"the arguments of the call to {function['name']} must be a JSON object")
    return {**call, "type": call.get("type", "function"), "function": {**function, "arguments": arguments}}


def template_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Messages as chat templates expect them.

    A field that does not apply is left out rather than set to null, because
    templates test for it: Llama's asks ``'tool_calls' in message``.
    """
    rendered: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for message in messages:
        if message.content is None and not message.tool_calls:
            raise ToolRequestError(f"a {message.role} message needs content")
        # Only the template knows where an image goes, so a message with one keeps its parts.
        holds_image = isinstance(message.content, list) and any(part.get("type") == "image" for part in message.content)
        item: dict[str, Any] = {"role": message.role, "content": message.content if holds_image else text_of(message.content)}
        if message.tool_calls:
            item["tool_calls"] = [_call_for_template(call) for call in message.tool_calls]
            names |= {str(call.get("id")): call["function"]["name"] for call in item["tool_calls"]}
        if message.role == "tool":
            if not message.tool_call_id:
                raise ToolRequestError("a tool message needs the tool_call_id of the call it answers")
            item["tool_call_id"] = message.tool_call_id
        # Clients send only the id, and Hermes' template prints the function's name.
        name = message.name or (names.get(message.tool_call_id or "") if message.role == "tool" else None)
        if name:
            item["name"] = name
        rendered.append(item)
    return rendered


def uses_tools(tools: list[dict[str, Any]] | None, messages: list[Any]) -> bool:
    return bool(tools) or any(message.role == "tool" or message.tool_calls for message in messages)


def reply_limit(max_tokens: int | None, max_completion_tokens: int | None, tools_in_play: bool) -> int:
    return max_completion_tokens or max_tokens or (TOOL_REPLY_TOKENS if tools_in_play else DEFAULT_REPLY_TOKENS)


@dataclass(slots=True)
class ToolPlan:
    """What a request asks for, once checked."""

    format: str
    tools: list[dict[str, Any]]
    schemas: dict[str, dict[str, Any]]
    required: bool = False
    named: str | None = None
    single: bool = False
    new_style: bool = False  # Mistral's [TOOL_CALLS]name[ARGS]{...}
    offered: set[str] = field(default_factory=set)

    @property
    def forced(self) -> str | None:
        """The function a call must name: a name, an empty string for any, or None if free."""
        if self.named:
            return self.named
        if self.required:
            return next(iter(self.schemas)) if len(self.schemas) == 1 else ""
        return None


def _checked_names(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ToolRequestError('each tool must look like {"type": "function", "function": {"name": ..., "parameters": ...}}')
        schemas[function["name"]] = function.get("parameters") or {}
    return schemas


def _named_choice(choice: dict[str, Any]) -> str:
    function = choice.get("function")
    if choice.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ToolRequestError(TOOL_CHOICE_RULE)
    return function["name"]


def plan_tools(
    tools: list[dict[str, Any]] | None, tool_choice: str | dict[str, Any] | None, parallel: bool | None, template: str | None
) -> ToolPlan | None:
    """Check a request's tools. None means answer without any."""
    if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
        raise ToolRequestError(TOOL_CHOICE_RULE)
    named = _named_choice(tool_choice) if isinstance(tool_choice, dict) else None
    schemas = _checked_names(tools or [])
    if not schemas:
        if named or tool_choice == "required":
            raise ToolRequestError("tool_choice needs tools to choose from; send tools with this request")
        return None
    if named and named not in schemas:
        raise ToolRequestError(f"tool_choice names {named}, which is not among the tools sent")
    if tool_choice == "none":
        return None
    kind = tool_format(template)
    if kind is None:
        raise ToolRequestError(
            "this model's chat template has no tool format iSharded can read, so it cannot call tools here. "
            'Leave tools out, send tool_choice "none", or load a model the Tool calling section of docs/GUIDE.md lists'
        )
    return ToolPlan(
        format=kind,
        tools=tools or [],
        schemas=schemas,
        required=tool_choice == "required",
        named=named,
        # OpenAI's named choice means exactly one call, and small models repeat a forced one.
        single=parallel is False or named is not None,
        new_style="[ARGS]" in (template or ""),
        offered={named} if named else set(schemas),
    )


def opens_in_thought(prompt: str) -> bool:
    """Some templates start the reply inside <think>, so it never writes the opening tag."""
    return prompt.rstrip().endswith(THINK_OPEN)


def without_reasoning(text: str) -> str:
    """The answer in a reply, for callers who only want that."""
    _, closed, answer = text.rpartition(THINK_CLOSE)
    return answer.strip() if closed else text.strip()


def forced_opening(plan: ToolPlan) -> str:
    """The start of a call, for the prompt to end with.

    A model cannot decline a call it has already begun, so this is real control
    over the part that matters most, on any route, with no change to the workers.
    The arguments are still the model's own: nothing here constrains them to the schema.

    An opening must end where the tokenizer would end a token, or the model is
    left after a token it has never seen there. ``=get`` is one token, so stopping
    at ``<function=`` broke 4 of 8 calls that the same model wrote well unforced.
    """
    name = plan.forced
    if name is None:
        return ""
    if plan.format == "xml":
        return f"<tool_call>\n<function={name}>\n" if name else "<tool_call>\n<function"
    if plan.format == "glm":
        # GLM-4.5 and 4.6 write a newline after the name, and 4.7 does not, so the model picks.
        return f"<tool_call>{name}"
    if plan.format == "mistral":
        if plan.new_style:
            return f"[TOOL_CALLS]{name}[ARGS]" if name else "[TOOL_CALLS]"
        return f'[TOOL_CALLS] [{{"name": "{name}", "arguments":' if name else '[TOOL_CALLS] [{"name": "'
    key = "parameters" if plan.format == "llama3" else "arguments"
    start = "<tool_call>\n" if plan.format == "hermes" else ""
    return f'{start}{{"name": "{name}", "{key}":' if name else f'{start}{{"name": "'


def _json_values(text: str) -> list[Any] | None:
    """Every JSON value in a text, or None if anything else is in it."""
    values, position = [], 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return values
        try:
            value, position = OBJECT.raw_decode(text, position)
        except ValueError:
            return None
        values.append(value)


def _one_call(item: Any) -> tuple[str, dict[str, Any]] | None:
    """A call written as a JSON object. Absent arguments are a call with none."""
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        return None
    arguments = item.get("arguments", item.get("parameters", {}))
    if isinstance(arguments, str):
        values = _json_values(arguments)
        arguments = values[0] if values and len(values) == 1 else None
    return (item["name"], arguments) if isinstance(arguments, dict) else None


def _calls_from_json(text: str) -> list[tuple[str, dict[str, Any]]] | None:
    values = _json_values(text)
    if not values:
        return None
    items = values[0] if len(values) == 1 and isinstance(values[0], list) else values
    calls = [_one_call(item) for item in items]
    return calls if calls and all(calls) else None  # type: ignore[return-value]


def _closed_by(text: str, closer: str) -> int | None:
    found = text.find(closer)
    return None if found < 0 else found + len(closer)


def _value_end(text: str, start: int = 0) -> int | None:
    """Where the JSON value after any space ends, or None if it is not whole yet."""
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        return OBJECT.raw_decode(text, start)[1]
    except ValueError:
        return None


class HermesCalls:
    """``<tool_call>{"name": ..., "arguments": ...}</tool_call>``: Qwen2.5, Qwen3, Hermes."""

    opens = ("<tool_call>",)
    bare = False
    joiners: tuple[str, ...] = ()

    def end(self, opener: str, text: str) -> int | None:
        return _closed_by(text, "</tool_call>")

    def plausible(self, opener: str, text: str) -> bool:
        return text.lstrip()[:1] in {"", "{"}

    def read(self, opener: str, text: str, schemas: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]] | None:
        return _calls_from_json(text.removesuffix("</tool_call>"))


FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|\Z)", re.S)
PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|(?=</function>)|\Z)", re.S)


def _schema_kinds(schema: dict[str, Any]) -> list[str]:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        return [kinds]
    if isinstance(kinds, list):
        return [str(kind) for kind in kinds]
    return [kind for option in schema.get("anyOf", schema.get("oneOf", [])) if isinstance(option, dict) for kind in _schema_kinds(option)]


def typed_value(text: str, schema: dict[str, Any]) -> Any:
    """XML has only text, and the schema says what was meant. Anything that does
    not fit stays as the model wrote it, for the client's own checks to see."""
    kinds = _schema_kinds(schema)
    word = text.strip()
    if "null" in kinds and word.lower() in {"null", "none"}:
        return None
    if "string" in kinds or not kinds:
        return text
    for kind in kinds:
        try:
            if kind == "integer":
                return int(word)
            if kind == "number":
                number = float(word)
                return int(number) if number.is_integer() and "." not in word and "e" not in word.lower() else number
            if kind == "boolean" and word.lower() in {"true", "false"}:
                return word.lower() == "true"
            if kind in {"object", "array"}:
                value = json.loads(word)
                if isinstance(value, dict if kind == "object" else list):
                    return value
        except ValueError:
            continue
    return text


class XmlCalls:
    """``<function=name><parameter=key>value</parameter></function>``: Qwen3.5, Qwen3-Coder."""

    opens = ("<tool_call>", "<function=")
    bare = False
    joiners = ("</tool_call>",)

    def end(self, opener: str, text: str) -> int | None:
        return _closed_by(text, "</tool_call>" if opener == "<tool_call>" else "</function>")

    def plausible(self, opener: str, text: str) -> bool:
        if opener == "<function=":
            return True
        start = text.lstrip()
        return start.startswith("<function=") or "<function=".startswith(start)

    def read(self, opener: str, text: str, schemas: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]] | None:
        body = (opener if opener == "<function=" else "") + text
        calls = []
        for name, inner in FUNCTION.findall(body):
            properties = schemas.get(name, {}).get("properties") or {}
            arguments = {key: typed_value(value, properties.get(key) or {}) for key, value in PARAMETER.findall(inner)}
            calls.append((name, arguments))
        return calls or None


GLM_CALL = re.compile(r"\s*([\w.-]+)\s*(.*)\Z", re.S)
GLM_START = re.compile(r"\s*[\w.-]*\s*(?:<|\Z)")
GLM_ARGUMENT = re.compile(r"\s*<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>\s*", re.S)


class GlmCalls:
    """``<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>``: GLM-4.5 to 4.7."""

    opens = ("<tool_call>",)
    bare = False
    joiners: tuple[str, ...] = ()

    def end(self, opener: str, text: str) -> int | None:
        return _closed_by(text, "</tool_call>")

    def plausible(self, opener: str, text: str) -> bool:
        return bool(GLM_START.match(text))

    def read(self, opener: str, text: str, schemas: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]] | None:
        match = GLM_CALL.match(text.removesuffix("</tool_call>"))
        if not match:
            return None
        name, body = match.groups()
        if GLM_ARGUMENT.sub("", body).strip():
            return None  # something else is in there, so this was not a call
        properties = schemas.get(name, {}).get("properties") or {}
        return [(name, {key: typed_value(value, properties.get(key) or {}) for key, value in GLM_ARGUMENT.findall(body)})]


NEW_MISTRAL = re.compile(r"\s*([\w.-]+)(?:\[CALL_ID\][^\[]*)?\[ARGS\](.*)\Z", re.S)
MISTRAL_START = re.compile(r"\s*(?:[\[{]|[\w.-]*(?:\[|\Z))")


class MistralCalls:
    """``[TOOL_CALLS] [{"name": ..., "arguments": ...}]``, or ``[TOOL_CALLS]name[ARGS]{...}`` on newer models."""

    opens = ("[TOOL_CALLS]",)
    bare = False
    joiners: tuple[str, ...] = ()

    def end(self, opener: str, text: str) -> int | None:
        marker = text.find("[ARGS]")
        if marker >= 0:
            return _value_end(text, marker + len("[ARGS]"))
        return _value_end(text) if text.lstrip()[:1] in {"[", "{"} else None

    def plausible(self, opener: str, text: str) -> bool:
        return bool(MISTRAL_START.match(text))

    def read(self, opener: str, text: str, schemas: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]] | None:
        match = NEW_MISTRAL.match(text)
        if match:
            call = _one_call({"name": match.group(1), "arguments": match.group(2)})
            return [call] if call else None
        return _calls_from_json(text)


class LlamaCalls:
    """``{"name": ..., "parameters": ...}``, with nothing to mark it except being the whole reply."""

    opens = ("<|python_tag|>",)
    bare = True
    joiners = (";",)
    starts = ('{"name"', '{"parameters"')

    def end(self, opener: str, text: str) -> int | None:
        return _value_end(text)

    def plausible(self, opener: str, text: str) -> bool:
        # This runs on every token of a held call, so it stops as soon as it has enough.
        compact = "".join(islice((char for char in text if not char.isspace()), 16))
        return any(start.startswith(compact) or compact.startswith(start) for start in self.starts)

    def read(self, opener: str, text: str, schemas: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]] | None:
        return _calls_from_json(text)


FORMATS: dict[str, Any] = {"hermes": HermesCalls(), "xml": XmlCalls(), "mistral": MistralCalls(), "llama3": LlamaCalls(), "glm": GlmCalls()}


def new_call_id(kind: str) -> str:
    """Mistral's templates refuse ids of any other shape, and clients send ours back."""
    return uuid.uuid4().hex[:9] if kind == "mistral" else f"call_{uuid.uuid4().hex[:24]}"


def tool_deltas(call: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """A finished call in OpenAI's streaming shape: who is called, then the arguments."""
    function = call["function"]
    head = {"index": index, "id": call["id"], "type": "function", "function": {"name": function["name"], "arguments": ""}}
    return [{"tool_calls": [head]}, {"tool_calls": [{"index": index, "function": {"arguments": function["arguments"]}}]}]


@lru_cache(maxsize=None)
def _marker_starts(markers: tuple[str, ...]) -> tuple[frozenset[str], int]:
    """Every start of a marker, and the size of the longest one that can be held."""
    longest = max(map(len, markers)) - 1
    return frozenset(marker[:size] for marker in markers for size in range(1, longest + 1)), longest


def _unfinished(text: str, markers: tuple[str, ...]) -> int:
    """How much of the end of a text could be the start of a marker."""
    starts, longest = _marker_starts(markers)
    for size in range(min(len(text), longest), 0, -1):
        if text[-size:] in starts:
            return size
    return 0


class ReplyScanner:
    """Sorts a reply into text for the client and finished tool calls, as it arrives.

    Anything that may become a call is held until it is one, or clearly is not,
    and a call is only sent once it has been read whole. Text that cannot be read
    as a call is shown as it was written, since a client can use that, and a call
    made up from it could do harm.

    ``feed`` and ``finish`` return items: ``{"text": ...}`` or ``{"tool_call": ...}``.
    """

    def __init__(self, plan: ToolPlan, thinking: bool = False, opening: str = "") -> None:
        self.plan = plan
        self.forcing = plan.forced is not None
        self.truncated = False
        self.style = FORMATS[plan.format]
        self.buffer = opening
        self.opening = opening
        self.state = "think" if thinking else "text"
        self.opener = ""
        self.calls: list[dict[str, Any]] = []
        self.fresh = True  # nothing seen but reasoning, so a bare call could still start
        self.after_call = False
        self.spaces = ""
        self.held: list[str] = []
        self.markers = (THINK_OPEN, *self.style.opens)

    def feed(self, text: str) -> list[dict[str, Any]]:
        self.buffer += text
        return self._run(False)

    def finish(self, truncated: bool = False) -> list[dict[str, Any]]:
        """Settle what is left. ``truncated`` means the reply was cut off, so an open call stays unfinished."""
        self.truncated = truncated
        items = self._run(True)
        if self.forcing:
            items += self._forced_outcome()
        elif not self.calls and self.spaces:
            items.append({"text": self.spaces})
        self.spaces = ""
        return items

    def _forced_outcome(self) -> list[dict[str, Any]]:
        written = "".join(self.held)
        if self.calls:
            return [{"text": written}] if written else []
        wanted = f"a call to {self.plan.named}" if self.plan.named else "a tool call"
        said = f"This request required {wanted}, and the model did not write a usable one, so no tool was called."
        return [{"text": f"{said} It wrote: {written}" if written.strip() else said}]

    def _run(self, final: bool) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while self.STEPS[self.state](self, items, final):
            pass
        return items

    def _show(self, items: list[dict[str, Any]], text: str, visible: bool = True) -> None:
        # Space before a call is not part of what the model said, and only what
        # comes next can tell, so it waits.
        text = self.spaces + text
        kept = text.rstrip()
        self.spaces = text[len(kept) :]
        if not kept:
            return
        self.fresh = self.fresh and not visible
        if self.forcing:
            self.held.append(kept)
        else:
            items.append({"text": kept})

    def _think(self, items: list[dict[str, Any]], final: bool) -> bool:
        closed = _closed_by(self.buffer, THINK_CLOSE)
        if closed is not None:
            self._show(items, self.buffer[:closed], visible=False)
            self.buffer = self.buffer[closed:]
            self.state = "text"
            return True
        hold = 0 if final else _unfinished(self.buffer, (THINK_CLOSE,))
        self._show(items, self.buffer[: len(self.buffer) - hold], visible=False)
        self.buffer = self.buffer[len(self.buffer) - hold :]
        return False

    def _swallow_joiner(self, final: bool) -> bool | None:
        """After a call: drop what joins it to the next. None means nothing to drop."""
        start = self.buffer.lstrip()
        for joiner in self.style.joiners:
            if start.startswith(joiner):
                self.buffer = start[len(joiner) :]
                return True
            if joiner.startswith(start) and not final:
                return False
        return None

    def _text(self, items: list[dict[str, Any]], final: bool) -> bool:
        if self.after_call:
            swallowed = self._swallow_joiner(final)
            if swallowed is not None:
                return swallowed
        if self.style.bare and (self.fresh or self.after_call):
            start = self.buffer.lstrip()
            if not start and not final:
                return False
            if start[:1] == "{":
                self.fresh = self.after_call = False
                self.buffer, self.opener, self.state = start, "", "call"
                return True
        found = [(position, marker) for marker in self.markers if (position := self.buffer.find(marker)) >= 0]
        if found:
            position, marker = min(found)
            self._show(items, self.buffer[:position], visible=bool(self.buffer[:position].strip()))
            self.buffer = self.buffer[position + len(marker) :]
            if marker == THINK_OPEN:
                self._show(items, marker, visible=False)
                self.state = "think"
            else:
                self.opener, self.state = marker, "call"
            return True
        hold = 0 if final else _unfinished(self.buffer, self.markers)
        ready = self.buffer[: len(self.buffer) - hold]
        if ready.strip():
            self.after_call = False
        self._show(items, ready)
        self.buffer = self.buffer[len(self.buffer) - hold :]
        return False

    def _call(self, items: list[dict[str, Any]], final: bool) -> bool:
        end = self.style.end(self.opener, self.buffer)
        if end is None and not final:
            if self.style.plausible(self.opener, self.buffer):
                return False
            self._give_up(items)
            return True
        if end is None and self.truncated:
            self._written(items, self.opener + self.buffer)
            self.buffer, self.state = "", "text"
            return False
        body, self.buffer = (self.buffer, "") if end is None else (self.buffer[:end], self.buffer[end:])
        calls = self.style.read(self.opener, body, self.plan.schemas)
        if calls and all(name in self.plan.offered for name, _ in calls):
            for name, arguments in calls:
                self._add(items, name, arguments)
            self.after_call = True
        else:
            self._written(items, self.opener + body)
        self.state = "text"
        self.opener = ""
        return bool(self.buffer) or not final

    STEPS = {"text": _text, "think": _think, "call": _call}

    def _give_up(self, items: list[dict[str, Any]]) -> None:
        """This is not a call after all, so what held it back was only text."""
        self._written(items, self.opener)
        self.opener, self.state = "", "text"

    def _written(self, items: list[dict[str, Any]], text: str) -> None:
        # What a forced opening put in the model's mouth is not what it wrote.
        if self.opening and text.startswith(self.opening):
            text = text[len(self.opening) :]
        self.opening = ""
        self.after_call = False
        self.fresh = False
        self._show(items, text)

    def _add(self, items: list[dict[str, Any]], name: str, arguments: dict[str, Any]) -> None:
        if self.plan.single and self.calls:
            return
        call = {
            "id": new_call_id(self.plan.format),
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }
        self.calls.append(call)
        items.append({"tool_call": call})
