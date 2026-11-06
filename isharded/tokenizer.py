"""Tokenizing and chat formatting for the control plane.

Reads ``tokenizer.json`` and the chat template directly rather than through
``AutoTokenizer``, whose class lookup fails on checkpoints newer than the
installed ``transformers``. Chat templates are code that arrives with a
download, so they run in Jinja's sandbox.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from datetime import datetime
from typing import Any

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from .gguf import GgufHeader, parse_source, read_numbers, read_strings
from .weights import Reader, WeightSourceError, gguf_reader, open_reader, read_gguf_header

# GGUF names a tokenizer's text-splitting rule instead of storing it.
GGUF_SPLIT_RULES = {
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen35": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
GGUF_CONTROL_TOKEN = 3
GGUF_USER_TOKEN = 4


def _as_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(item) for item in value} if isinstance(value, list) else {int(value)}


def _token_text(value: Any) -> str:
    # tokenizer_config stores special tokens as text or as {"content": text}.
    return value.get("content", "") if isinstance(value, dict) else (value or "")


def _read_json(reader: Reader, name: str) -> dict[str, Any]:
    raw = reader.read_file(name)
    return json.loads(raw) if raw else {}


def _parse_tokenizer(text: str) -> Tokenizer:
    """Parse ``tokenizer.json`` on any ``tokenizers`` release.

    Files written since 0.20 store BPE merges as pairs, which older releases
    reject. Nodes pin different versions, so the file adapts to the library
    rather than the other way round.
    """
    try:
        return Tokenizer.from_str(text)
    except Exception as original:  # the Rust parser raises a bare Exception
        data = json.loads(text)
        merges = data.get("model", {}).get("merges")
        if not merges or not isinstance(merges[0], list):
            raise WeightSourceError(f"tokenizer.json could not be parsed: {original}") from original
        data["model"]["merges"] = [" ".join(pair) for pair in merges]
        return Tokenizer.from_str(json.dumps(data))


class MarkerDecoder:
    """Decodes as usual, but keeps the tokens a tool call is written with.

    Some models write ``[TOOL_CALLS]`` as a special token. Decoding drops those,
    and the call would then read as plain text.
    """

    def __init__(self, tokenizer: ChatTokenizer, keep: dict[int, str]) -> None:
        self.tokenizer = tokenizer
        self.keep = keep

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        run: list[int] = []
        for token in ids:
            if token in self.keep:
                pieces += [self.tokenizer.decode(run), self.keep[token]]
                run = []
            else:
                run.append(token)
        return "".join(pieces) + self.tokenizer.decode(run)


class ChatTokenizer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        chat_template: str | None,
        stop_token_ids: set[int],
        special_tokens: dict[str, str],
        marks: tuple[list[int], list[int]] = ([], []),
        tool_template: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        # Some models, Hermes among them, keep tools in a template of their own.
        self.tool_template = tool_template or chat_template
        self.stop_token_ids = stop_token_ids
        self.special_tokens = special_tokens
        self.marks = marks

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_input(self, text: str) -> list[int]:
        """Tokens for an embedding model, with the marks its GGUF asks for around them."""
        before, after = self.marks
        return [*before, *self.encode(text), *after]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def keeping(self, markers: Iterable[str]) -> MarkerDecoder:
        found = {marker: self.tokenizer.token_to_id(marker) for marker in markers}
        return MarkerDecoder(self, {index: marker for marker, index in found.items() if index is not None})

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        template = self.tool_template if tools else self.chat_template
        if not template:
            lines = [f"{item['role'].title()}: {item['content']}" for item in messages]
            return "\n".join(lines) + ("\nAssistant:" if add_generation_prompt else "\n")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        environment.filters["tojson"] = template_json

        def raise_exception(message: str) -> None:
            raise TemplateError(message)

        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = lambda spec: datetime.now().strftime(spec)
        options: dict[str, Any] = {}
        if enable_thinking is not None:
            options["enable_thinking"] = enable_thinking
        if tools:
            options["tools"] = tools
        return environment.from_string(template).render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **self.special_tokens,
            **options,
        )


def template_json(
    value: Any, indent: int | None = None, ensure_ascii: bool = False, sort_keys: bool = False, separators: tuple[str, str] | None = None
) -> str:
    """Chat templates come with the model. json.dumps runs ``default`` outside the sandbox."""
    return json.dumps(
        value, indent=None if indent is None else int(indent), ensure_ascii=bool(ensure_ascii), sort_keys=bool(sort_keys), separators=separators
    )


class IncrementalText:
    """Text so far, decoding only what is new.

    Decoding a whole reply on every token gets slower as the reply grows. Tokens
    can split a character, so a tail is only settled once it decodes cleanly.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.settled = ""
        self.pending: list[int] = []

    def add(self, token: int) -> str:
        self.pending.append(token)
        tail = self.tokenizer.decode(self.pending)
        if tail.endswith("\ufffd"):
            return self.settled + tail
        self.settled += tail
        self.pending = []
        return self.settled


def tokenizer_from_gguf(header: GgufHeader) -> ChatTokenizer:
    """Build a tokenizer from a GGUF header, which carries the whole vocabulary."""
    model = header.get("tokenizer.ggml.model")
    rule = GGUF_SPLIT_RULES.get(str(header.get("tokenizer.ggml.pre")))
    if model != "gpt2" or rule is None:
        raise WeightSourceError(
            f"GGUF tokenizer {model!r}/{header.get('tokenizer.ggml.pre')!r} is not supported yet"
        )
    tokens = read_strings(header.keys["tokenizer.ggml.tokens"])
    kinds = read_numbers(header.keys["tokenizer.ggml.token_type"])
    merges = [tuple(item.split(" ", 1)) for item in read_strings(header.keys["tokenizer.ggml.merges"])]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=merges, fuse_unk=False))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(rule), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    # add_special_tokens makes everything special whatever AddedToken says, and
    # decoding drops special tokens. <think> and <tool_call> are text a reply is
    # read by, so they go in as ordinary added tokens.
    tokenizer.add_tokens(
        [AddedToken(token, special=False, normalized=False) for token, kind in zip(tokens, kinds) if kind == GGUF_USER_TOKEN]
    )
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True, normalized=False) for token, kind in zip(tokens, kinds) if kind == GGUF_CONTROL_TOKEN]
    )
    stops = {
        int(header.get(key))
        for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id")
        if header.get(key) is not None
    }
    special = {"bos_token": "", "eos_token": "", "pad_token": "", "unk_token": ""}
    for name in ("bos", "eos", "padding", "unknown"):
        index = header.get(f"tokenizer.ggml.{name}_token_id")
        label = {"padding": "pad", "unknown": "unk"}.get(name, name)
        if index is not None and int(index) < len(tokens):
            special[f"{label}_token"] = tokens[int(index)]
    return ChatTokenizer(tokenizer, header.get("tokenizer.chat_template"), stops, special, gguf_marks(header), header.get("tokenizer.chat_template.tool_use"))


def gguf_marks(header: GgufHeader) -> tuple[list[int], list[int]]:
    """What a GGUF asks to have around a text. Models that pool the last token need this."""

    def mark(name: str) -> list[int]:
        index = header.get(f"tokenizer.ggml.{name}_token_id")
        return [int(index)] if header.get(f"tokenizer.ggml.add_{name}_token") and index is not None else []

    return mark("bos"), mark("eos")


def load_tokenizer(
    source: str, revision: str = "main", holders: list[str] | None = None, token: str | None = None
) -> ChatTokenizer:
    """Fetch tokenizer files only. Costs a few megabytes at most."""
    located = parse_source(source)
    if located:
        reader = gguf_reader(source, revision, holders, token)
        try:
            return tokenizer_from_gguf(read_gguf_header(reader, located[1]))
        finally:
            reader.close()
    reader = open_reader(source, revision)
    try:
        raw = reader.read_file("tokenizer.json")
        if raw is None:
            raise WeightSourceError(f"{source} has no tokenizer.json")
        tokenizer = _parse_tokenizer(raw.decode("utf-8"))
        settings = _read_json(reader, "tokenizer_config.json")
        generation = _read_json(reader, "generation_config.json")
        config = _read_json(reader, "config.json")
        template = settings.get("chat_template")
        if not template:
            separate = reader.read_file("chat_template.jinja")
            template = separate.decode("utf-8") if separate else None
    finally:
        reader.close()
    tool_template = None
    if isinstance(template, list):  # named templates; "default" is the chat one
        named = {item.get("name"): item.get("template") for item in template}
        template, tool_template = named.get("default"), named.get("tool_use")
    special = {
        name: _token_text(settings.get(name))
        for name in ("bos_token", "eos_token", "pad_token", "unk_token")
    }
    stops = _as_ids(generation.get("eos_token_id")) | _as_ids(config.get("eos_token_id"))
    eos_text = special["eos_token"]
    if eos_text and tokenizer.token_to_id(eos_text) is not None:
        stops.add(tokenizer.token_to_id(eos_text))
    return ChatTokenizer(tokenizer, template, stops, special, tool_template=tool_template)
