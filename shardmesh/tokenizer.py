"""Tokenizing and chat formatting for the control plane.

Reads ``tokenizer.json`` and the chat template directly rather than through
``AutoTokenizer``, whose class lookup fails on checkpoints newer than the
installed ``transformers``. Chat templates are code that arrives with a
download, so they run in Jinja's sandbox.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from jinja2.exceptions import TemplateError
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


class ChatTokenizer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        chat_template: str | None,
        stop_token_ids: set[int],
        special_tokens: dict[str, str],
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.stop_token_ids = stop_token_ids
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        enable_thinking: bool | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        if not self.chat_template:
            lines = [f"{item['role'].title()}: {item['content']}" for item in messages]
            return "\n".join(lines) + ("\nAssistant:" if add_generation_prompt else "\n")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = template_json

        def raise_exception(message: str) -> None:
            raise TemplateError(message)

        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = lambda spec: datetime.now().strftime(spec)
        options: dict[str, Any] = {}
        if enable_thinking is not None:
            options["enable_thinking"] = enable_thinking
        return environment.from_string(self.chat_template).render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **self.special_tokens,
            **options,
        )


def template_json(value: Any, indent: int | None = None, ensure_ascii: bool = False, sort_keys: bool = False) -> str:
    """Chat templates come with the model. json.dumps runs ``default`` outside the sandbox."""
    return json.dumps(value, indent=None if indent is None else int(indent), ensure_ascii=bool(ensure_ascii), sort_keys=bool(sort_keys))


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
    tokenizer.add_special_tokens(
        [
            AddedToken(token, special=kind == GGUF_CONTROL_TOKEN, normalized=False)
            for token, kind in zip(tokens, kinds)
            if kind in {GGUF_CONTROL_TOKEN, GGUF_USER_TOKEN}
        ]
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
    return ChatTokenizer(tokenizer, header.get("tokenizer.chat_template"), stops, special)


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
    if isinstance(template, list):  # named templates; "default" is the chat one
        template = next((item["template"] for item in template if item.get("name") == "default"), None)
    special = {
        name: _token_text(settings.get(name))
        for name in ("bos_token", "eos_token", "pad_token", "unk_token")
    }
    stops = _as_ids(generation.get("eos_token_id")) | _as_ids(config.get("eos_token_id"))
    eos_text = special["eos_token"]
    if eos_text and tokenizer.token_to_id(eos_text) is not None:
        stops.add(tokenizer.token_to_id(eos_text))
    return ChatTokenizer(tokenizer, template, stops, special)
