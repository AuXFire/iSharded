"""The OpenAI shape of a reply: its prompt on the way in, its events and message on the way out."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Any

from fastapi import HTTPException
from jinja2.exceptions import TemplateError

from .toolcalls import ReplyScanner, ToolRequestError, tool_deltas


def usage_of(summary: dict[str, Any]) -> dict[str, int]:
    prompt, completion = summary["prompt_tokens"], summary["completion_tokens"]
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


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
    calls: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    async for item in items:
        if "text" in item:
            pieces.append(item["text"])
        elif "tool_call" in item:
            calls.append(item["tool_call"])
        else:
            summary = item
    text = "".join(pieces)
    if calls:
        return {"role": "assistant", "content": text or None, "tool_calls": calls}, summary
    return {"role": "assistant", "content": text}, summary
