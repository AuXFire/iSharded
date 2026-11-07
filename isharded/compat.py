"""Claude Messages, OpenAI Responses and Ollama, on top of the chat route.

Each API here is a pair of translators: its request into a ChatCompletionRequest,
and the reply back into its own shape. Every reply goes through ``reply``, which
calls the real /v1 handler, so keys, limits, lanes, tools and images behave as they
do there and cannot drift from it. Asking for a stream every time gives lanes and
this node one way in, and a whole reply is a stream collected.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
from pydantic import ValidationError

from . import __version__
from .protocol import ChatCompletionRequest, ChatMessage, CompletionRequest, EmbeddingRequest
from .replies import render_prompts, whole_reply
from .toolcalls import ToolRequestError, plan_tools, template_messages, tool_format
from .vision import ImageRefused, chat_images

# The chat route's own limit. Clients here often ask for more, and mean "as much as there is".
MOST_REPLY_TOKENS = 32768
# Ollama's way of saying a model stays loaded for good.
NEVER = "2318-08-31T00:00:00Z"
KEY_PATHS = {"/v1/messages", "/v1/messages/count_tokens"}
STATUS_KINDS = {401: "authentication_error", 403: "permission_error", 404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error"}


@dataclass(frozen=True, slots=True)
class Node:
    """What these routes need from the orchestrator, handed over so imports stay one way."""

    chat: Callable[..., Any]
    completion: Callable[..., Any]
    embed: Callable[..., Any]
    models: Callable[..., Any]
    guard: Callable[[str | None], Any]  # checks the key and role, and spends nothing
    charge: Callable[[str | None], Any]  # the same, and spends from the caller's limit
    check_request: Callable[..., Any]
    runtime: Callable[[], Any]


class KeyHeader:
    """Claude clients send their key as x-api-key, and everything else reads Authorization.

    This runs first, and the header goes: a person's key must not leave their
    node, and forwarding only knows to drop Authorization.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["path"] in KEY_PATHS:
            sent = dict(scope["headers"])
            headers = [(name, value) for name, value in scope["headers"] if name != b"x-api-key"]
            if b"x-api-key" in sent and b"authorization" not in sent:
                headers.append((b"authorization", b"Bearer " + sent[b"x-api-key"]))
            scope = {**scope, "headers": headers}
        await self.app(scope, receive, send)


def refuse(message: str, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def error_kind(status: int) -> str:
    return STATUS_KINDS.get(status, "invalid_request_error" if status < 500 else "api_error")


def anthropic_error(status: int, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_kind(status), "message": message}, "request_id": f"req_{uuid.uuid4().hex}"}


def openai_error(status: int, message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_kind(status), "param": None, "code": None}}


def ollama_error(status: int, message: str) -> dict[str, Any]:
    return {"error": message}


async def json_body(request: Request) -> dict[str, Any]:
    """The checks FastAPI makes on a route that declares its body, for routes that shape their own errors."""
    kind = request.headers.get("content-type", "application/json").partition(";")[0].strip().lower()
    if not (kind == "application/json" or kind.endswith("+json")):
        raise refuse("send the body as JSON, with Content-Type: application/json")
    try:
        body = await request.json()
    except ValueError as exc:
        raise refuse("the body is not valid JSON") from exc
    if not isinstance(body, dict):
        raise refuse("the body must be a JSON object")
    return body


def translate(build: Callable[[dict[str, Any]], Any], body: dict[str, Any]) -> Any:
    """A client's request as ours. Anything can arrive, and a wrong shape is the client's to fix."""
    try:
        return build(body)
    except ValidationError as exc:
        raise refuse("; ".join(f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}" for problem in exc.errors())) from exc
    except (AttributeError, KeyError, TypeError) as exc:
        raise refuse(f"this request does not have the shape this API takes: {exc!r}") from exc


def reply_tokens(asked: Any) -> int:
    """Nothing asked, or Ollama's -1 and -2, means until the model stops or the context is full."""
    return min(asked, MOST_REPLY_TOKENS) if isinstance(asked, int) and asked > 0 else MOST_REPLY_TOKENS


def picked(source: dict[str, Any], *names: str) -> dict[str, Any]:
    return {name: source[name] for name in names if source.get(name) is not None}


def image_part(url: Any) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def called(identifier: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def function_tool(name: str, description: str | None, parameters: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description or "", "parameters": parameters or {}}}


async def body_of(response: Any) -> bytes:
    pieces = [piece.encode() if isinstance(piece, str) else piece async for piece in response.body_iterator]
    if response.background:
        await response.background()
    return b"".join(pieces)


async def whole_answer(result: Any) -> dict[str, Any]:
    """What a /v1 handler said, which is a stream of bytes if a lane answered."""
    if isinstance(result, dict):
        return result
    raw = await body_of(result)
    try:
        answer = json.loads(raw)
    except ValueError:
        answer = {"detail": raw.decode("utf-8", "replace")}
    if result.status_code != 200:
        raise refuse(str(answer.get("detail", answer)), result.status_code)
    return dict(answer)


async def server_events(response: Any) -> AsyncIterator[dict[str, Any]]:
    """The events of a /v1 stream. A lane's arrive as raw bytes, cut anywhere."""
    pending = b""
    try:
        async with aclosing(response.body_iterator) as source:
            async for piece in source:
                pending += piece.encode() if isinstance(piece, str) else piece
                *whole, pending = pending.split(b"\n\n")
                for event in whole:
                    payload = event.removeprefix(b"data: ")
                    if payload != b"[DONE]":
                        yield json.loads(payload)
    finally:
        if response.background:
            await response.background()


async def reply_items(response: Any) -> AsyncIterator[dict[str, Any]]:
    """A /v1 stream as ``scanned`` gave it: text, tool calls, and a closing summary."""
    heads: dict[int, dict[str, Any]] = {}
    finished = False
    try:
        async with aclosing(server_events(response)) as events:
            async for event in events:
                if "error" in event:
                    raise refuse(str(event["error"].get("message")), int(event["error"].get("code") or 500))
                choice = event["choices"][0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or choice.get("text")
                if text:
                    yield {"text": text}
                for part in delta.get("tool_calls") or []:
                    if "id" in part:
                        heads[part["index"]] = part
                    else:
                        head = heads[part["index"]]
                        yield {"tool_call": called(head["id"], head["function"]["name"], part["function"]["arguments"])}
                if choice.get("finish_reason"):
                    finished = True
                    yield {"finish_reason": choice["finish_reason"], **event.get("usage", {}), **event.get("isharded", {})}
    except httpx.HTTPError as exc:
        raise refuse(f"the reply was lost on its way here: {exc}", 503) from exc
    if not finished:
        raise refuse("the reply stopped before it was finished", 503)


async def reply(call: Callable[..., Any], request: Any, authorization: str | None) -> AsyncIterator[dict[str, Any]]:
    """Run a request through its /v1 handler. A refusal is raised here, before any reply starts."""
    response = await call(request.model_copy(update={"stream": True}), authorization)
    if response.status_code != 200:
        await whole_answer(response)
    return reply_items(response)


def answering_model(node: Node, request: Any) -> str:
    return str(request.model or node.runtime().model_id)


# --- Claude Messages


def block_text(blocks: Any) -> str:
    return blocks if isinstance(blocks, str) else "".join(block["text"] for block in blocks if block["type"] == "text")


def anthropic_image(block: dict[str, Any]) -> dict[str, Any]:
    source = block["source"]
    if source["type"] == "base64":
        return image_part(f"data:{source['media_type']};base64,{source['data']}")
    if source["type"] == "url":
        return image_part(source["url"])  # and refused in words where every image is read
    raise refuse(f"an image source of type {source['type']!r} is not supported; send it as base64")


def anthropic_result(block: dict[str, Any]) -> dict[str, Any]:
    text = block_text(block.get("content") or "")
    return {"role": "tool", "tool_call_id": block["tool_use_id"], "content": f"Error: {text}" if block.get("is_error") else text}


def anthropic_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": block_text(body["system"])}] if body.get("system") else []
    for message in body["messages"]:
        content = message["content"]
        parts: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for block in [{"type": "text", "text": content}] if isinstance(content, str) else content:
            kind = block["type"]
            if kind == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                parts.append(anthropic_image(block))
            elif kind == "tool_use":
                calls.append(called(block["id"], block["name"], json.dumps(block.get("input") or {})))
            elif kind == "tool_result":
                messages.append(anthropic_result(block))  # Claude puts these first, and they come before the words
            elif kind not in {"thinking", "redacted_thinking"}:
                raise refuse(f"a content block of type {kind!r} is not supported here; send text, image, tool_use and tool_result")
        if parts or calls:
            messages.append({"role": message["role"], "content": parts or None, **({"tool_calls": calls} if calls else {})})
    return messages


def anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type", "custom") != "custom":
        raise refuse(f"{tool['type']} is a tool Anthropic runs for you, and nothing here can; send your own tools with an input_schema")
    return function_tool(tool["name"], tool.get("description"), tool.get("input_schema"))


def anthropic_choice(choice: dict[str, Any]) -> str | dict[str, Any] | None:
    kind = choice.get("type", "auto")
    if kind == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    if kind not in {"auto", "any", "none"}:
        raise refuse("tool_choice must be auto, any, none, or a tool with a name")
    return "required" if kind == "any" else kind


def anthropic_chat(body: dict[str, Any]) -> ChatCompletionRequest:
    choice = body.get("tool_choice") or {}
    thinking = body.get("thinking")
    return ChatCompletionRequest(
        model=body.get("model"),
        messages=[ChatMessage(**message) for message in anthropic_messages(body)],
        max_tokens=reply_tokens(body.get("max_tokens")),
        stop=body.get("stop_sequences"),
        enable_thinking=thinking["type"] != "disabled" if thinking else None,
        tools=[anthropic_tool(tool) for tool in body.get("tools") or []] or None,
        tool_choice=anthropic_choice(choice),
        parallel_tool_calls=False if choice.get("disable_parallel_tool_use") else None,
        **picked(body, "temperature", "top_p", "top_k"),
    )


def stop_reason(finish_reason: str | None) -> str:
    return {"length": "max_tokens", "tool_calls": "tool_use"}.get(finish_reason or "", "end_turn")


def server_event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


async def anthropic_stream(items: AsyncIterator[dict[str, Any]], opening: dict[str, Any]) -> AsyncIterator[str]:
    index = 0
    open_text = False

    def block_stop() -> str:
        return server_event("content_block_stop", {"type": "content_block_stop", "index": index})

    def block_delta(delta: dict[str, Any]) -> str:
        return server_event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})

    def block_start(block: dict[str, Any]) -> str:
        return server_event("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})

    yield server_event("message_start", {"type": "message_start", "message": opening})
    try:
        async with aclosing(items) as source:
            async for item in source:
                if "text" in item and not open_text:
                    yield block_start({"type": "text", "text": ""})
                    open_text = True
                if "text" in item:
                    yield block_delta({"type": "text_delta", "text": item["text"]})
                    continue
                if open_text:
                    yield block_stop()
                    index, open_text = index + 1, False
                if "tool_call" in item:
                    call = item["tool_call"]
                    yield block_start({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": {}})
                    yield block_delta({"type": "input_json_delta", "partial_json": call["function"]["arguments"]})
                    yield block_stop()
                    index += 1
                    continue
                yield server_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason(item["finish_reason"]), "stop_sequence": None},
                        "usage": {"input_tokens": item["prompt_tokens"], "output_tokens": item["completion_tokens"]},
                    },
                )
                yield server_event("message_stop", {"type": "message_stop"})
    except HTTPException as exc:
        # Headers are long gone, so the failure has to travel in the stream.
        yield server_event("error", anthropic_error(exc.status_code, str(exc.detail)))


async def anthropic_reply(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] < 1:
        raise refuse("max_tokens: a whole number above 0 is required")
    chat = translate(anthropic_chat, body)
    items = await reply(node.chat, chat, authorization)
    frame = {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "model": answering_model(node, chat)}
    if body.get("stream"):
        opening = {**frame, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}
        return StreamingResponse(anthropic_stream(items, opening), media_type="text/event-stream")
    message, summary = await whole_reply(items)
    content: list[dict[str, Any]] = [{"type": "text", "text": message["content"]}] if message["content"] else []
    for call in message.get("tool_calls", []):
        content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": json.loads(call["function"]["arguments"])})
    usage = {"input_tokens": summary["prompt_tokens"], "output_tokens": summary["completion_tokens"]}
    return {**frame, "content": content, "stop_reason": stop_reason(summary["finish_reason"]), "stop_sequence": None, "usage": usage}


async def anthropic_count(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    chat = translate(anthropic_chat, body)
    caller = node.charge(authorization)
    runtime = node.runtime()
    lane = await runtime.lanes.locate(chat.model)
    if lane:
        return await runtime.lanes.forward(lane, "/v1/messages/count_tokens", body, caller)
    await node.check_request(chat.model)
    try:
        plain, _ = chat_images([message.model_dump() for message in chat.messages])
        plan = plan_tools(chat.tools, chat.tool_choice, chat.parallel_tool_calls, runtime.tokenizer.tool_template)
        messages = template_messages([message.model_copy(update={"content": item["content"]}) for message, item in zip(chat.messages, plain)])
        prompt, _ = await asyncio.to_thread(render_prompts, runtime.tokenizer, messages, chat.enable_thinking, plan.tools if plan else None)
    except (ImageRefused, ToolRequestError) as exc:
        raise refuse(str(exc)) from exc
    return {"input_tokens": len(await asyncio.to_thread(runtime.tokenizer.encode, prompt))}


# --- OpenAI Responses


def response_part(part: dict[str, Any]) -> dict[str, Any]:
    kind = part["type"]
    if kind in {"input_text", "output_text", "text"}:
        return {"type": "text", "text": part["text"]}
    if kind == "refusal":
        return {"type": "text", "text": part["refusal"]}
    if kind == "input_image" and part.get("image_url"):
        return image_part(part["image_url"])
    raise refuse(f"a part of type {kind!r} is not supported here; send input_text, and input_image with a data URL in image_url")


def responses_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": body["instructions"]}] if body.get("instructions") else []
    source = body["input"]
    for item in [{"role": "user", "content": source}] if isinstance(source, str) else source:
        kind = item.get("type", "message")
        if kind == "message":
            content = item["content"]
            role = "system" if item["role"] == "developer" else item["role"]
            messages.append({"role": role, "content": content if isinstance(content, str) else [response_part(part) for part in content]})
        elif kind == "function_call":
            # Our chat path keeps one reply's calls on one message, as OpenAI's chat API does.
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": None})
            messages[-1].setdefault("tool_calls", []).append(called(item["call_id"], item["name"], item["arguments"]))
        elif kind == "function_call_output":
            output = item["output"]
            text = output if isinstance(output, str) else "".join(part.get("text", "") for part in output)
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": text})
        elif kind != "reasoning":
            raise refuse(f"an input item of type {kind!r} is not supported here; send messages, function_call and function_call_output")
    return messages


def responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        raise refuse(f"only function tools run here, and {tool.get('type')!r} is not one")
    return function_tool(tool["name"], tool.get("description"), tool.get("parameters"))


def responses_chat(body: dict[str, Any]) -> ChatCompletionRequest:
    for name in ("previous_response_id", "conversation"):
        if body.get(name):
            raise refuse(f"{name} is refused: iSharded keeps no replies, so send the whole conversation in input")
    if body.get("background"):
        raise refuse("background is refused: iSharded keeps no replies to collect later; use stream instead")
    if (body.get("text") or {}).get("format", {}).get("type", "text") != "text":
        raise refuse("text.format is refused: the chat path cannot hold a reply to a format, so only plain text is honest")
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        choice = {"type": "function", "function": {"name": choice["name"]}}
    return ChatCompletionRequest(
        model=body.get("model"),
        messages=[ChatMessage(**message) for message in responses_messages(body)],
        max_tokens=reply_tokens(body.get("max_output_tokens")),
        tools=[responses_tool(tool) for tool in body.get("tools") or []] or None,
        tool_choice=choice,
        parallel_tool_calls=body.get("parallel_tool_calls"),
        **picked(body, "temperature", "top_p"),
    )


def output_text(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def message_item(identifier: str, status: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": identifier, "type": "message", "status": status, "role": "assistant", "content": content}


def call_item(call: dict[str, Any]) -> dict[str, Any]:
    function = call["function"]
    return {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "completed", "call_id": call["id"], "name": function["name"], "arguments": function["arguments"]}


def finished_response(frame: dict[str, Any], output: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    cut_short = summary["finish_reason"] == "length"
    prompt, completion = summary["prompt_tokens"], summary["completion_tokens"]
    return {
        **frame,
        "status": "incomplete" if cut_short else "completed",
        "completed_at": int(time.time()),
        "incomplete_details": {"reason": "max_output_tokens"} if cut_short else None,
        "output": output,
        "usage": {
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt + completion,
        },
    }


async def responses_stream(items: AsyncIterator[dict[str, Any]], frame: dict[str, Any]) -> AsyncIterator[str]:
    number = -1
    output: list[dict[str, Any]] = []
    message: str | None = None  # the id of the message being written
    written = ""

    def event(kind: str, **fields: Any) -> str:
        nonlocal number
        number += 1
        return server_event(kind, {"type": kind, "sequence_number": number, **fields})

    def where() -> dict[str, Any]:
        return {"item_id": message, "output_index": len(output), "content_index": 0}

    def end_message(status: str) -> list[str]:
        nonlocal message, written
        if message is None:
            return []
        item = message_item(message, status, [output_text(written)])
        ended = [
            event("response.output_text.done", **where(), text=written, logprobs=[]),
            event("response.content_part.done", **where(), part=output_text(written)),
            event("response.output_item.done", output_index=len(output), item=item),
        ]
        output.append(item)
        message, written = None, ""
        return ended

    yield event("response.created", response=frame)
    yield event("response.in_progress", response=frame)
    try:
        async with aclosing(items) as source:
            async for item in source:
                if "text" in item:
                    if message is None:
                        message = f"msg_{uuid.uuid4().hex}"
                        yield event("response.output_item.added", output_index=len(output), item=message_item(message, "in_progress", []))
                        yield event("response.content_part.added", **where(), part=output_text(""))
                    written += item["text"]
                    yield event("response.output_text.delta", **where(), delta=item["text"], logprobs=[])
                elif "tool_call" in item:
                    for line in end_message("completed"):
                        yield line
                    done = call_item(item["tool_call"])
                    at = {"item_id": done["id"], "output_index": len(output)}
                    yield event("response.output_item.added", output_index=len(output), item={**done, "status": "in_progress", "arguments": ""})
                    yield event("response.function_call_arguments.delta", **at, delta=done["arguments"])
                    yield event("response.function_call_arguments.done", **at, name=done["name"], arguments=done["arguments"])
                    yield event("response.output_item.done", output_index=len(output), item=done)
                    output.append(done)
                else:
                    cut_short = item["finish_reason"] == "length"
                    for line in end_message("incomplete" if cut_short else "completed"):
                        yield line
                    yield event("response.incomplete" if cut_short else "response.completed", response=finished_response(frame, output, item))
    except HTTPException as exc:
        failure = {"code": "rate_limit_exceeded" if exc.status_code == 429 else "server_error", "message": str(exc.detail)}
        yield event("response.failed", response={**frame, "status": "failed", "error": failure, "output": output})


async def responses_reply(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    chat = translate(responses_chat, body)
    items = await reply(node.chat, chat, authorization)
    frame = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "metadata": body.get("metadata") or {},
        "model": answering_model(node, chat),
        "output": [],
        "parallel_tool_calls": chat.parallel_tool_calls is not False,
        "previous_response_id": None,
        "store": False,
        "temperature": chat.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice") or "auto",
        "tools": body.get("tools") or [],
        "top_p": chat.top_p,
        "truncation": "disabled",
        "usage": None,
    }
    if body.get("stream"):
        return StreamingResponse(responses_stream(items, frame), media_type="text/event-stream")
    message, summary = await whole_reply(items)
    status = "incomplete" if summary["finish_reason"] == "length" else "completed"
    output = [message_item(f"msg_{uuid.uuid4().hex}", status, [output_text(message["content"])])] if message["content"] else []
    return finished_response(frame, output + [call_item(call) for call in message.get("tool_calls", [])], summary)


# --- Ollama


def ollama_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ollama_options(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options") or {}
    return {"max_tokens": reply_tokens(options.get("num_predict")), **picked(options, "temperature", "top_p", "top_k", "seed", "stop")}


def ollama_images(images: list[str]) -> list[dict[str, Any]]:
    # Ollama sends bare base64, and images are read by what is in them, whatever this says.
    return [image_part(f"data:image/png;base64,{image}") for image in images]


def ollama_content(text: str, images: list[str] | None) -> str | list[dict[str, Any]]:
    return [{"type": "text", "text": text}, *ollama_images(images)] if images else text


def ollama_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []  # Ollama's tool results carry a name at most, so calls are matched up here
    for message in body["messages"]:
        role = message["role"]
        item = {"role": role, "content": ollama_content(message.get("content") or "", message.get("images"))}
        if message.get("tool_calls"):
            waiting = [
                called(call.get("id") or f"call_{len(messages)}_{index}", call["function"]["name"], json.dumps(call["function"].get("arguments") or {}))
                for index, call in enumerate(message["tool_calls"])
            ]
            item["tool_calls"] = waiting
        elif role == "tool":
            named = message.get("tool_name") or message.get("name")
            match = next((call for call in waiting if call["function"]["name"] == named), waiting[0] if waiting else None)
            if match:
                waiting = [call for call in waiting if call is not match]
            item["tool_call_id"] = message.get("tool_call_id") or (match["id"] if match else f"call_{len(messages)}")
            item["name"] = named or (match["function"]["name"] if match else None)
        messages.append(item)
    return messages


def refuse_format(body: dict[str, Any]) -> None:
    if body.get("format"):
        raise refuse("format is refused: the chat path cannot hold a reply to JSON or a schema, and asking the model nicely would only look as if it worked")


def ollama_thinking(body: dict[str, Any]) -> bool | None:
    think = body.get("think")
    return None if think is None else bool(think)


def ollama_chat(body: dict[str, Any]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=body.get("model"),
        messages=[ChatMessage(**message) for message in ollama_messages(body)],
        enable_thinking=ollama_thinking(body),
        tools=body.get("tools") or None,
        **ollama_options(body),
    )


def ollama_generate_chat(body: dict[str, Any]) -> ChatCompletionRequest:
    system = [ChatMessage(role="system", content=body["system"])] if body.get("system") else []
    user = ChatMessage(role="user", content=ollama_content(body["prompt"], body.get("images")))
    return ChatCompletionRequest(model=body.get("model"), messages=[*system, user], enable_thinking=ollama_thinking(body), **ollama_options(body))


def ollama_raw(body: dict[str, Any]) -> CompletionRequest:
    if body.get("images"):
        raise refuse("images need the chat template, so they cannot go with raw")
    return CompletionRequest(model=body.get("model"), prompt=body["prompt"], **ollama_options(body))


def ollama_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": call["id"], "function": {"index": index, "name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}}
        for index, call in enumerate(calls)
    ]


def ollama_message(text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    return {"message": {**message, "tool_calls": ollama_calls(calls)} if calls else message}


def ollama_response(text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"response": text}


def ollama_timing(summary: dict[str, Any], started: float) -> dict[str, Any]:
    """The closing fields, in nanoseconds. Nothing loads for a reply here, since a model stays on its route."""
    total = int((time.perf_counter() - started) * 1e9)
    waited = min(int(summary.get("time_to_first_token_ms", 0) * 1e6), total)
    return {
        "done": True,
        "done_reason": "length" if summary["finish_reason"] == "length" else "stop",
        "total_duration": total,
        "load_duration": 0,
        "prompt_eval_count": summary["prompt_tokens"],
        "prompt_eval_duration": waited,
        "eval_count": summary["completion_tokens"],
        "eval_duration": total - waited,
    }


def json_line(data: dict[str, Any]) -> str:
    return json.dumps(data) + "\n"


async def ollama_stream(items: AsyncIterator[dict[str, Any]], frame: dict[str, Any], shape: Callable[..., dict[str, Any]], started: float) -> AsyncIterator[str]:
    try:
        async with aclosing(items) as source:
            async for item in source:
                now = {**frame, "created_at": ollama_time()}
                if "text" in item:
                    yield json_line({**now, **shape(item["text"], []), "done": False})
                elif "tool_call" in item:
                    yield json_line({**now, **shape("", [item["tool_call"]]), "done": False})
                else:
                    yield json_line({**now, **shape("", []), **ollama_timing(item, started)})
    except HTTPException as exc:
        yield json_line({"error": str(exc.detail)})


async def ollama_answer(node: Node, call: Any, request: Any, body: dict[str, Any], shape: Callable[..., dict[str, Any]], authorization: str | None) -> Any:
    started = time.perf_counter()
    items = await reply(call, request, authorization)
    frame = {"model": answering_model(node, request)}
    if body.get("stream", True):
        return StreamingResponse(ollama_stream(items, frame, shape, started), media_type="application/x-ndjson")
    message, summary = await whole_reply(items)
    return {**frame, "created_at": ollama_time(), **shape(message["content"] or "", message.get("tool_calls", [])), **ollama_timing(summary, started)}


async def ollama_load(node: Node, body: dict[str, Any], shape: Callable[..., dict[str, Any]], authorization: str | None) -> dict[str, Any]:
    """Clients send an empty request to warm a model up, and here it is warm or it is not there."""
    node.charge(authorization)
    runtime = node.runtime()
    model = body.get("model")
    if not await runtime.lanes.locate(model):
        await node.check_request(model)
    return {"model": model or runtime.model_id, "created_at": ollama_time(), **shape("", []), "done": True, "done_reason": "load"}


async def ollama_chat_reply(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    refuse_format(body)
    if body.get("messages") == []:
        return await ollama_load(node, body, ollama_message, authorization)
    return await ollama_answer(node, node.chat, translate(ollama_chat, body), body, ollama_message, authorization)


async def ollama_generate(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    refuse_format(body)
    for name, why in (("suffix", "the chat path has no fill in the middle"), ("template", "a model's template comes with the model")):
        if body.get(name):
            raise refuse(f"{name} is refused: {why}")
    if not body.get("prompt"):
        return await ollama_load(node, body, ollama_response, authorization)
    if body.get("raw"):
        return await ollama_answer(node, node.completion, translate(ollama_raw, body), body, ollama_response, authorization)
    return await ollama_answer(node, node.chat, translate(ollama_generate_chat, body), body, ollama_response, authorization)


async def ollama_vectors(node: Node, body: dict[str, Any], text: Any, authorization: str | None) -> dict[str, Any]:
    request = translate(lambda _: EmbeddingRequest(model=body.get("model"), input=text, dimensions=body.get("dimensions")), body)
    return await whole_answer(await node.embed(request, authorization))


async def ollama_embed(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    started = time.perf_counter()
    answer = await ollama_vectors(node, body, body.get("input"), authorization)
    return {
        "model": answer["model"],
        "embeddings": [row["embedding"] for row in answer["data"]],
        "total_duration": int((time.perf_counter() - started) * 1e9),
        "load_duration": 0,
        "prompt_eval_count": answer["usage"]["prompt_tokens"],
    }


async def ollama_embeddings(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    answer = await ollama_vectors(node, body, body.get("prompt"), authorization)
    return {"embedding": answer["data"][0]["embedding"]}


def model_facts(node: Node, row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The loaded target and GGUF header behind a /v1/models row. A lane's are its own, and stay there."""
    runtime = node.runtime()
    active = runtime.active if row["id"] == runtime.model_id else None
    return active, getattr(getattr(active, "adapter", None), "header", None) or {}


def ollama_row(node: Node, row: dict[str, Any]) -> dict[str, Any]:
    active, header = model_facts(node, row)
    family = str(header.get("general.architecture") or "")
    return {
        "name": row["id"],
        "model": row["id"],
        "modified_at": ollama_time(),
        "size": sum(active.file_sizes.values()) if active else 0,
        "digest": hashlib.sha256(row["id"].encode()).hexdigest(),
        "details": {
            "parent_model": "",
            "format": "gguf" if header else "",
            "family": family,
            "families": [family] if family else None,
            "parameter_size": str(header.get("general.size_label") or ""),
            "quantization_level": "",
        },
    }


async def ollama_tags(node: Node, request: Request, authorization: str | None) -> Any:
    listing = await node.models(authorization)
    return {"models": [ollama_row(node, row) for row in listing["data"]]}


async def ollama_running(node: Node, request: Request, authorization: str | None) -> Any:
    listing = await node.models(authorization)
    rows = []
    for row in listing["data"]:
        active, _ = model_facts(node, row)
        context = active.profile.context_tokens if active else None
        # Nothing here sees one machine's memory from one model's point of view, so size_vram says nothing.
        rows.append({**ollama_row(node, row), "expires_at": NEVER, "size_vram": 0, "context_length": context})
    return {"models": rows}


async def ollama_show(node: Node, request: Request, authorization: str | None) -> Any:
    body = await json_body(request)
    listing = await node.models(authorization)
    name = body.get("model") or body.get("name")
    row = next((row for row in listing["data"] if row["id"] == name), None)
    if row is None:
        raise refuse(f"model {name!r} is not loaded", 404)
    active, _ = model_facts(node, row)
    shown = ollama_row(node, row)
    family = shown["details"]["family"]
    template = getattr(node.runtime().tokenizer, "tool_template", None) if row["id"] == node.runtime().model_id else None
    capabilities = ["embedding"] if row.get("kind") == "embedding" else ["completion"]
    capabilities += ["tools"] if tool_format(template) else []
    capabilities += ["vision"] if row.get("vision") else []
    info = {"general.architecture": family, **({f"{family}.context_length": active.profile.context_tokens} if active and family else {})}
    return {
        "modelfile": "",
        "parameters": "",
        "template": template or "",
        "details": shown["details"],
        "model_info": info,
        "capabilities": capabilities,
        "modified_at": shown["modified_at"],
    }


async def ollama_version(node: Node, request: Request, authorization: str | None) -> Any:
    node.charge(authorization)
    return {"version": __version__}


async def ollama_refused(node: Node, request: Request, authorization: str | None) -> Any:
    raise refuse("iSharded does not change models through this API; load one with isharded load or POST /admin/load")


ROUTES: list[tuple[str, str, Any, Callable[[int, str], dict[str, Any]]]] = [
    ("POST", "/v1/messages", anthropic_reply, anthropic_error),
    ("POST", "/v1/messages/count_tokens", anthropic_count, anthropic_error),
    ("POST", "/v1/responses", responses_reply, openai_error),
    ("POST", "/api/chat", ollama_chat_reply, ollama_error),
    ("POST", "/api/generate", ollama_generate, ollama_error),
    ("POST", "/api/embed", ollama_embed, ollama_error),
    ("POST", "/api/embeddings", ollama_embeddings, ollama_error),
    ("GET", "/api/tags", ollama_tags, ollama_error),
    ("POST", "/api/show", ollama_show, ollama_error),
    ("GET", "/api/ps", ollama_running, ollama_error),
    ("GET", "/api/version", ollama_version, ollama_error),
    ("POST", "/api/pull", ollama_refused, ollama_error),
    ("POST", "/api/push", ollama_refused, ollama_error),
    ("POST", "/api/create", ollama_refused, ollama_error),
    ("POST", "/api/copy", ollama_refused, ollama_error),
    ("DELETE", "/api/delete", ollama_refused, ollama_error),
]


def answering(node: Node, route: Any, shape: Callable[[int, str], dict[str, Any]]) -> Callable[[Request], Any]:
    async def handle(request: Request) -> Response:
        authorization = request.headers.get("authorization")
        try:
            # First, so a stranger learns nothing about a body from how it is refused.
            node.guard(authorization)
            answer = await route(node, request, authorization)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=shape(exc.status_code, str(exc.detail)), headers=exc.headers)
        return answer if isinstance(answer, Response) else JSONResponse(answer)

    return handle


def mount(app: Any, node: Node) -> None:
    app.add_middleware(KeyHeader)
    for method, path, route, shape in ROUTES:
        app.add_api_route(path, answering(node, route, shape), methods=[method], include_in_schema=False)
