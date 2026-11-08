"""Request fields an API defines and iSharded cannot honour.

A client that asks for something and quietly gets something else has no way to find out.
So a field that would change a reply is refused with a 400 that names it, at any value but
the one that changes nothing. Clients often send that value, so it is taken: the OpenAI
library sends nothing it was not given, but other clients send ``n: 1`` and penalties of 0.
A field that does nothing to a reply, such as metadata for the client's own records, is
taken at any value. A name the API does not define is refused, as OpenAI and Anthropic
refuse one, except on Ollama's routes, where Ollama itself ignores it.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

UNSUPPORTED = "not supported"


@dataclass(frozen=True, slots=True)
class Limit:
    """The values of a field that ask for nothing iSharded leaves undone."""

    accepts: Callable[[Any], bool]
    why: str


def equal(value: Any, wanted: Any) -> bool:
    """JSON equality: 0 and 0.0 are one number, and false is not 0."""
    if isinstance(value, bool) or isinstance(wanted, bool):
        return isinstance(value, bool) and isinstance(wanted, bool) and value == wanted
    if isinstance(value, (int, float)) and isinstance(wanted, (int, float)):
        return float(value) == float(wanted)
    return bool(value == wanted)


def only(*values: Any, why: str) -> Limit:
    """Taken when left null or set to one of ``values``."""
    return Limit(lambda value: value is None or any(equal(value, wanted) for wanted in values), why)


def empty(why: str) -> Limit:
    """Taken when null or empty, such as an empty list of servers."""
    return Limit(lambda value: value is None or value in ({}, [], ""), why)


def keys_within(allowed: Collection[str], why: str) -> Limit:
    """An object taken only while every key it holds is one of these."""
    return Limit(lambda value: value is None or (isinstance(value, dict) and set(value) <= set(allowed)), why)


def items_within(allowed: Collection[str], why: str) -> Limit:
    return Limit(lambda value: value is None or (isinstance(value, list) and all(item in allowed for item in value)), why)


def anything(why: str = "it changes nothing about a reply") -> Limit:
    return Limit(lambda value: True, why)


def refused_field(fields: Mapping[str, Any], limits: Mapping[str, Limit], known: Collection[str] = (), unknown: bool = True) -> tuple[str, str] | None:
    """The first field a request cannot be served with, and why, or None. ``known`` are the fields its route reads.

    Fields are looked at in the order sent, so the first one the client wrote is the one named.
    """
    for name, value in fields.items():
        if name in known:
            continue
        limit = limits.get(name)
        if limit is None:
            if unknown:
                return name, f"{UNSUPPORTED}: {name} (this API has no such field, or iSharded does not know it)"
            continue
        if not limit.accepts(value):
            return name, f"{UNSUPPORTED}: {name} ({limit.why})"
    return None


def refusal(fields: Mapping[str, Any], limits: Mapping[str, Limit], known: Collection[str] = (), unknown: bool = True) -> str | None:
    found = refused_field(fields, limits, known, unknown)
    return None if found is None else found[1]


PENALTY = "no penalty is applied here; send 0 or leave it out"
ONE_CHOICE = "a reply here is one choice; send 1 or leave it out"
NO_LOGPROBS = "token probabilities are not returned"
NO_FORMAT = "the chat path cannot hold a reply to JSON or a schema, and asking the model nicely would only look as if it worked"
THINKING = "use enable_thinking to switch reasoning on models that have a switch"
# For the client's own records, for a provider's billing or caching, or for clearing context
# the client already sent. None changes a reply.
RECORDS = ("metadata", "user", "store", "service_tier", "safety_identifier", "prompt_cache_key", "prompt_cache_retention")

COMMON_OPENAI: dict[str, Limit] = {
    **{name: anything() for name in RECORDS},
    "n": only(1, why=ONE_CHOICE),
    "frequency_penalty": only(0, why=PENALTY),
    "presence_penalty": only(0, why=PENALTY),
    "logit_bias": empty("token biases are not applied"),
    # Usage always arrives with the last chunk, so asking for it is asking for what comes anyway.
    "stream_options": keys_within({"include_usage", "include_obfuscation"}, "only include_usage and include_obfuscation are understood"),
}

CHAT_LIMITS: dict[str, Limit] = {
    **COMMON_OPENAI,
    "logprobs": only(False, why=NO_LOGPROBS),
    "top_logprobs": only(0, why=NO_LOGPROBS),
    "response_format": Limit(lambda value: value is None or value == {"type": "text"}, NO_FORMAT),
    "modalities": Limit(lambda value: value is None or value == ["text"], "replies here are text"),
    "audio": only(why="replies here are text"),
    "prediction": only(why="predicted outputs are not used"),
    "reasoning_effort": only(why=THINKING),
    "verbosity": only("medium", why="replies are not steered towards a length; send medium or leave it out"),
    "web_search_options": only(why="nothing here searches the web"),
    "functions": empty("send tools instead"),
    "function_call": only("none", "auto", why="send tool_choice instead"),
    "chat_template_kwargs": empty(THINKING),
    # llama.cpp and vLLM sampling knobs, which some clients send to any server.
    "min_p": only(0, why="min_p sampling is not applied"),
    "repetition_penalty": only(1, why=PENALTY.replace("send 0", "send 1")),
    "repeat_penalty": only(1, why=PENALTY.replace("send 0", "send 1")),
}

COMPLETION_LIMITS: dict[str, Limit] = {
    **COMMON_OPENAI,
    "logprobs": only(0, why=NO_LOGPROBS),
    "best_of": only(1, why=ONE_CHOICE),
    "echo": only(False, why="the prompt is not repeated in the reply"),
    "suffix": empty("there is no fill in the middle"),
}

EMBEDDING_LIMITS: dict[str, Limit] = {}

CLAUDE_KNOWN = {"model", "messages", "max_tokens", "system", "stop_sequences", "stream", "temperature", "top_p", "top_k", "tools", "tool_choice", "thinking"}
CLAUDE_LIMITS: dict[str, Limit] = {
    "metadata": anything(),
    "service_tier": anything(),
    # Clears old tool results from a long conversation. Nothing here is cleared, so a
    # conversation past the context is refused as on every route, and a reply is unchanged.
    "context_management": anything("nothing is cleared, and a conversation past the context is refused"),
    # Effort steers how long a model thinks, as budget_tokens does, which is not used either.
    "output_config": keys_within({"effort"}, NO_FORMAT),
    "output_format": only(why=NO_FORMAT),
    "container": only(why="nothing here runs code for you"),
    "mcp_servers": empty("nothing here connects to MCP servers for you; run your own tools"),
}

RESPONSES_KNOWN = {
    "model", "input", "instructions", "max_output_tokens", "temperature", "top_p", "tools", "tool_choice",
    "parallel_tool_calls", "stream", "text", "previous_response_id", "conversation", "background",
}
RESPONSES_LIMITS: dict[str, Limit] = {
    **{name: anything() for name in RECORDS},
    # Effort and summaries: reasoning stays in the reply's text, so neither changes what comes back.
    "reasoning": keys_within({"effort", "summary", "generate_summary"}, "only effort and summary are understood"),
    # Codex asks for this, and there are no reasoning items to hold it.
    "include": items_within({"reasoning.encrypted_content"}, "only reasoning.encrypted_content can be included, and there is none to give"),
    "truncation": only("disabled", why="nothing is cut from a conversation; a conversation past the context is refused"),
    "max_tool_calls": only(why="calls are not counted across a reply"),
    "top_logprobs": only(0, why=NO_LOGPROBS),
    "prompt": only(why="iSharded keeps no prompts; send the text in instructions and input"),
    "stream_options": keys_within({"include_obfuscation"}, "only include_obfuscation is understood"),
}
RESPONSES_TEXT_LIMITS: dict[str, Limit] = {
    "format": Limit(lambda value: value is None or (isinstance(value, dict) and value.get("type", "text") == "text"), NO_FORMAT),
    "verbosity": only("medium", why="replies are not steered towards a length; send medium or leave it out"),
}

# Ollama ignores a field it does not know, at the top and in options, so these are only the
# ones it knows. Its own defaults count as nothing asked: 1.1 is its repeat_penalty.
OLLAMA_LIMITS: dict[str, Limit] = {
    "context": empty("the older context list is not used; send the conversation to /api/chat"),
    "logprobs": only(False, why=NO_LOGPROBS),
    "top_logprobs": only(0, why=NO_LOGPROBS),
}
OLLAMA_OPTION_LIMITS: dict[str, Limit] = {
    "repeat_penalty": only(1, 1.1, why="no repeat penalty is applied; leave it out"),
    "presence_penalty": only(0, why=PENALTY),
    "frequency_penalty": only(0, why=PENALTY),
    "min_p": only(0, why="min_p sampling is not applied"),
    "typical_p": only(1, why="typical sampling is not applied"),
    "tfs_z": only(1, why="tail-free sampling is not applied"),
    "mirostat": only(0, why="mirostat sampling is not applied"),
}
