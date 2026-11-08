"""The OpenAI shape of a reply: its prompt on the way in, its events and message on the way out."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any, TypeVar

from fastapi import HTTPException
from jinja2.exceptions import TemplateError

from .toolcalls import ReplyScanner, ToolRequestError, tool_deltas


T = TypeVar("T")
Receive = Callable[[], Awaitable[dict[str, Any]]]
# Not a status anyone reads, since nobody is left to: it is how a log says the client went.
CLIENT_GONE = 499


class ClientGone(Exception):
    """The client closed its connection before its reply was ready."""


async def client_left(receive: Receive) -> None:
    """Returns once the client has gone. The body has been read by now, so nothing else arrives."""
    while (await receive())["type"] != "http.disconnect":
        pass


async def unless_client_leaves(receive: Receive | None, work: Awaitable[T]) -> T:
    """Waits for a reply that is sent whole, and stops it if its client goes first.

    A streamed reply stops when its stream is closed. A whole one has nothing to close
    until it is done, so without this it runs to the end and holds its place on the route
    for a reply nobody will read. Cancelling closes the token stream, which closes the
    entry worker's stream, which ends the generation there and frees its slot.
    """
    if receive is None:
        return await work
    task = asyncio.ensure_future(work)
    watch = asyncio.ensure_future(client_left(receive))
    try:
        await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
        if not task.done():
            task.cancel()
            await asyncio.wait({task})  # let the stream close before anything else runs
            raise ClientGone
    except asyncio.CancelledError:  # the server is stopping, and this reply stops with it
        task.cancel()
        raise
    finally:
        watch.cancel()
    return task.result()


def usage_of(summary: dict[str, Any]) -> dict[str, int]:
    prompt, completion = summary["prompt_tokens"], summary["completion_tokens"]
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def first_stop(text: str, markers: list[str]) -> str | None:
    """The stop string this text meets first, if it meets one."""
    return min((marker for marker in markers if marker in text), key=text.index, default=None)


def server_sent(payload: dict[str, Any] | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


async def event_stream(
    tokens: AsyncIterator[dict[str, Any]],
    envelope: dict[str, Any],
    to_choice: Any,
    route_now: Callable[[], dict[str, int]],
    first_delta: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Wrap generated text in OpenAI-style server-sent events.

    ``route_now`` says what the reply ended on, since a route can move during one.
    """
    opening = json.dumps(envelope)[:-1]  # the same for every token, so it is written once

    def choice_event(choice: dict[str, Any]) -> str:
        return f'data: {opening}, "choices": [{json.dumps(choice)}]}}\n\n'

    def delta_event(delta: dict[str, Any]) -> str:
        return choice_event({"index": 0, "delta": delta, "finish_reason": None})

    calls = 0
    try:
        if first_delta:
            yield delta_event(first_delta)
        async for item in tokens:
            if "text" in item:
                yield choice_event(to_choice(item["text"], None))
                continue
            if "tool_call" in item:
                for delta in tool_deltas(item["tool_call"], calls):
                    yield delta_event(delta)
                calls += 1
                continue
            timing = {
                "time_to_first_token_ms": item["time_to_first_token_ms"],
                "tokens_per_second": item["tokens_per_second"],
                **route_now(),
                **({"stop_sequence": item["stop_sequence"]} if item.get("stop_sequence") else {}),
            }
            yield server_sent(
                {
                    **envelope,
                    "choices": [to_choice("", item["finish_reason"])],
                    "usage": usage_of(item),
                    "isharded": timing,
                }
            )
    except HTTPException as exc:
        # Headers are long gone, so the failure has to travel in the stream.
        yield server_sent({"error": {"message": str(exc.detail), "code": exc.status_code}})
    yield server_sent("[DONE]")


def render_prompts(
    tokenizer: Any, messages: list[dict[str, Any]], thinking: bool | None, tools: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """The prompt, and the settled history inside it, through the model's own template."""
    options = {"tools": tools} if tools else {}
    try:
        return (
            tokenizer.render_chat(messages, thinking, **options),
            tokenizer.render_chat(messages, thinking, add_generation_prompt=False, **options),
        )
    except (TemplateError, TypeError) as exc:
        raise ToolRequestError(f"this conversation does not fit the model's chat template: {exc}") from exc


async def scanned(tokens: AsyncIterator[dict[str, Any]], scanner: ReplyScanner) -> AsyncIterator[dict[str, Any]]:
    """A reply with its tool calls taken out of the text."""
    async with aclosing(tokens) as source:
        async for item in source:
            if "text" in item:
                for found in scanner.feed(item["text"]):
                    yield found
                continue
            for found in scanner.finish(item["finish_reason"] == "length"):
                yield found
            yield {**item, "finish_reason": "tool_calls"} if scanner.calls else item


async def whole_reply(items: AsyncIterator[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The assistant message a reply comes to, and its summary."""
    pieces: list[str] = []
    thoughts: list[str] = []
    calls: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    async for item in items:
        if "text" in item:
            pieces.append(item["text"])
        elif "thinking" in item:
            thoughts.append(item["thinking"])
        elif "tool_call" in item:
            calls.append(item["tool_call"])
        else:
            summary = item
    text = "".join(pieces)
    reasoned = {"thinking": "".join(thoughts)} if thoughts else {}
    if calls:
        return {"role": "assistant", "content": text or None, "tool_calls": calls, **reasoned}, summary
    return {"role": "assistant", "content": text, **reasoned}, summary
