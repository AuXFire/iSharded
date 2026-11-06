"""Other clients' API shapes, over the same replies.

Anthropic's Messages API, OpenAI's Responses API, and Ollama's API all ask for the
same thing in different words. Each request here becomes one chat request, and the
reply comes back as a stream of pieces that each shape wraps its own way, so all
of them get failover, lanes, keys, and rate limits with nothing added.

Text only. A client sending tools or images gets a plain refusal, because a reply
that silently ignored them would look like it worked.
"""
from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
import datetime
import json
import time
from typing import Any
import uuid

from .protocol import ChatCompletionRequest, ChatMessage

DEFAULT_MAX_TOKENS = 1024
MOST_TOKENS = 32768
TEXT_BLOCKS = {"text", "input_text", "output_text"}
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


class Unsupported(ValueError):
    """A request asks for something these replies cannot give."""


class StreamFailed(RuntimeError):
    def __init__(self, message: str, code: int = 500) -> None:
        super().__init__(message)
        self.code = code


def text_of(content: Any) -> str:
    """The text in a message, whichever way this client writes it."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise Unsupported("message content must be text")
    parts = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind not in TEXT_BLOCKS:
            raise Unsupported(f"only text is supported, not {kind}")
        parts.append(str(block.get("text", "")))
    return "".join(parts)


def _refuse_tools(body: dict[str, Any]) -> None:
    if body.get("tools"):
        raise Unsupported("tools are not supported on this endpoint")


def _tokens(value: Any) -> int:
    """A token limit as chat requests take it. Ollama writes -1 for no limit."""
    amount = int(value) if value is not None else DEFAULT_MAX_TOKENS
    return MOST_TOKENS if amount < 0 else max(min(amount, MOST_TOKENS), 1)


def _request(model: Any, messages: list[ChatMessage], **fields: Any) -> ChatCompletionRequest:
    chosen = {key: value for key, value in fields.items() if value is not None}
    return ChatCompletionRequest(model=model, messages=messages, stream=True, **chosen)


def from_anthropic(body: dict[str, Any]) -> ChatCompletionRequest:
    _refuse_tools(body)
    messages = [ChatMessage(role="system", content=text_of(body["system"]))] if body.get("system") else []
    messages += [ChatMessage(role=str(item["role"]), content=text_of(item["content"])) for item in body["messages"]]
    thinking = body.get("thinking")
    return _request(
        body.get("model"),
        messages,
        max_tokens=_tokens(body.get("max_tokens")),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        stop=body.get("stop_sequences") or None,
        enable_thinking=thinking.get("type") == "enabled" if isinstance(thinking, dict) else None,
    )


def from_responses(body: dict[str, Any]) -> ChatCompletionRequest:
    _refuse_tools(body)
    if body.get("previous_response_id"):
        raise Unsupported("this endpoint keeps no conversations; send the whole history in input")
    messages = [ChatMessage(role="system", content=str(body["instructions"]))] if body.get("instructions") else []
    given = body["input"]
    if isinstance(given, str):
        messages.append(ChatMessage(role="user", content=given))
    else:
        for item in given:
            if item.get("type", "message") != "message":
                raise Unsupported(f"only messages are supported, not {item.get('type')}")
            role = "system" if item["role"] == "developer" else str(item["role"])
            messages.append(ChatMessage(role=role, content=text_of(item["content"])))
    return _request(
        body.get("model"),
        messages,
        max_tokens=_tokens(body.get("max_output_tokens")),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p"),
    )


def _ollama_options(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options") or {}
    return {
        "max_tokens": _tokens(options.get("num_predict")),
        "temperature": options.get("temperature", 0.8),
        "top_p": options.get("top_p"),
        "top_k": options.get("top_k"),
        "seed": options.get("seed"),
        "stop": options.get("stop") or None,
        "enable_thinking": body.get("think") if isinstance(body.get("think"), bool) else None,
    }


def from_ollama_chat(body: dict[str, Any]) -> ChatCompletionRequest:
    _refuse_tools(body)
    if any(item.get("images") for item in body["messages"]):
        raise Unsupported("only text is supported, not images")
    messages = [ChatMessage(role=str(item["role"]), content=str(item.get("content", ""))) for item in body["messages"]]
    return _request(body.get("model"), messages, **_ollama_options(body))


def from_ollama_generate(body: dict[str, Any]) -> ChatCompletionRequest:
    if body.get("images"):
        raise Unsupported("only text is supported, not images")
    messages = [ChatMessage(role="system", content=str(body["system"]))] if body.get("system") else []
    messages.append(ChatMessage(role="user", content=str(body.get("prompt", ""))))
    return _request(body.get("model"), messages, **_ollama_options(body))


async def pieces(stream: AsyncIterable[Any]) -> AsyncIterator[dict[str, Any]]:
    """A chat stream as text, then one summary. Lanes pass bytes, split anywhere."""
    held = ""
    async for chunk in stream:
        held += chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk
        while "\n\n" in held:
            event, held = held.split("\n\n", 1)
            if not event.startswith("data: ") or event == "data: [DONE]":
                continue
            data = json.loads(event[6:])
            if "error" in data:
                raise StreamFailed(str(data["error"].get("message")), int(data["error"].get("code", 500)))
            choice = data["choices"][0]
            text = choice.get("delta", {}).get("content")
            if text:
                yield {"text": text, "model": data.get("model")}
            if choice.get("finish_reason"):
                usage = data.get("usage", {})
                yield {
                    "finish": choice["finish_reason"],
                    "model": data.get("model"),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                }


async def gather(replies: AsyncIterable[dict[str, Any]]) -> dict[str, Any]:
    """A whole reply, for a client that did not ask to stream."""
    text: list[str] = []
    final: dict[str, Any] = {}
    async for item in replies:
        if "text" in item:
            text.append(item["text"])
        else:
            final = item
    return {"finish": "stop", "prompt_tokens": 0, "completion_tokens": 0, "model": None, **final, "text": "".join(text)}


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def anthropic_message(reply: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": reply["text"]}],
        "stop_reason": STOP_REASONS.get(reply["finish"], "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": reply["prompt_tokens"], "output_tokens": reply["completion_tokens"]},
    }


async def anthropic_stream(replies: AsyncIterable[dict[str, Any]], model: str) -> AsyncIterator[str]:
    start = {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}
    yield _event("message_start", {"type": "message_start", "message": start})
    yield _event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    try:
        final: dict[str, Any] = {}
        async for item in replies:
            if "text" in item:
                yield _event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": item["text"]}})
            else:
                final = item
    except StreamFailed as exc:
        yield _event("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
        return
    yield _event("content_block_stop", {"type": "content_block_stop", "index": 0})
    delta = {"stop_reason": STOP_REASONS.get(final.get("finish", "stop"), "end_turn"), "stop_sequence": None}
    yield _event("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": final.get("completion_tokens", 0)}})
    yield _event("message_stop", {"type": "message_stop"})


def responses_object(reply: dict[str, Any], model: str, identifier: str | None = None, item_id: str | None = None) -> dict[str, Any]:
    cut_short = reply["finish"] == "length"
    return {
        "id": identifier or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if cut_short else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if cut_short else None,
        "error": None,
        "model": model,
        "output": [_responses_item(reply["text"], item_id or f"msg_{uuid.uuid4().hex}", "completed")],
        "usage": {"input_tokens": reply["prompt_tokens"], "output_tokens": reply["completion_tokens"], "total_tokens": reply["prompt_tokens"] + reply["completion_tokens"]},
    }


def _responses_item(text: str, item_id: str, status: str) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text, "annotations": []}] if status == "completed" else []
    return {"type": "message", "id": item_id, "status": status, "role": "assistant", "content": content}


async def responses_stream(replies: AsyncIterable[dict[str, Any]], model: str) -> AsyncIterator[str]:
    identifier, item_id = f"resp_{uuid.uuid4().hex}", f"msg_{uuid.uuid4().hex}"
    number = 0

    def send(kind: str, **fields: Any) -> str:
        nonlocal number
        number += 1
        return _event(kind, {"type": kind, "sequence_number": number, **fields})

    pending = {"id": identifier, "object": "response", "created_at": int(time.time()), "status": "in_progress", "model": model, "output": []}
    yield send("response.created", response=pending)
    yield send("response.output_item.added", output_index=0, item=_responses_item("", item_id, "in_progress"))
    part = {"type": "output_text", "text": "", "annotations": []}
    yield send("response.content_part.added", item_id=item_id, output_index=0, content_index=0, part=part)
    text: list[str] = []
    final: dict[str, Any] = {}
    try:
        async for item in replies:
            if "text" in item:
                text.append(item["text"])
                yield send("response.output_text.delta", item_id=item_id, output_index=0, content_index=0, delta=item["text"])
            else:
                final = item
    except StreamFailed as exc:
        yield send("response.failed", response={**pending, "status": "failed", "error": {"code": "server_error", "message": str(exc)}})
        return
    whole = "".join(text)
    yield send("response.output_text.done", item_id=item_id, output_index=0, content_index=0, text=whole)
    yield send("response.content_part.done", item_id=item_id, output_index=0, content_index=0, part={**part, "text": whole})
    yield send("response.output_item.done", output_index=0, item=_responses_item(whole, item_id, "completed"))
    reply = {"finish": "stop", "prompt_tokens": 0, "completion_tokens": 0, **final, "text": whole}
    yield send("response.completed", response=responses_object(reply, model, identifier, item_id))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _line(data: dict[str, Any]) -> str:
    return json.dumps(data) + "\n"


def _ollama_final(reply: dict[str, Any], model: str, started: float) -> dict[str, Any]:
    return {
        "model": model,
        "created_at": _now(),
        "done": True,
        "done_reason": reply.get("finish", "stop"),
        "total_duration": int((time.monotonic() - started) * 1e9),
        "prompt_eval_count": reply.get("prompt_tokens", 0),
        "eval_count": reply.get("completion_tokens", 0),
    }


def ollama_reply(reply: dict[str, Any], model: str, chat: bool, started: float) -> dict[str, Any]:
    text = {"message": {"role": "assistant", "content": reply["text"]}} if chat else {"response": reply["text"]}
    return {**_ollama_final(reply, model, started), **text}


async def ollama_stream(replies: AsyncIterable[dict[str, Any]], model: str, chat: bool, started: float) -> AsyncIterator[str]:
    empty = {"message": {"role": "assistant", "content": ""}} if chat else {"response": ""}
    final: dict[str, Any] = {}
    try:
        async for item in replies:
            if "text" in item:
                text = {"message": {"role": "assistant", "content": item["text"]}} if chat else {"response": item["text"]}
                yield _line({"model": model, "created_at": _now(), **text, "done": False})
            else:
                final = item
    except StreamFailed as exc:
        yield _line({"error": str(exc)})
        return
    yield _line({**_ollama_final(final, model, started), **empty})


def ollama_tags(names: list[str]) -> dict[str, Any]:
    return {"models": [{"name": name, "model": name, "modified_at": _now(), "size": 0, "digest": "", "details": {"format": "shardmesh"}} for name in names]}
