"""Claude Messages, OpenAI Responses and Ollama, each checked against its published shapes.

Replies are scripted under the real routes, as in tests/test_tools.py, so each test
sees what a client would. The tests that use the real anthropic, openai and ollama
libraries skip without them. They need recent versions of what they depend on, so
to keep those out of a working install, pip install --target a folder and put the
folder first on the path for the run, with pytest -o pythonpath=FOLDER.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import re
import socket
import threading
import time
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
import pytest
from starlette.testclient import TestClient
import uvicorn

from isharded import compat, orchestrator
from isharded.access import ROLE_ADMIN, ROLE_USER, AccessControl
from isharded.mesh import ControlMesh
from isharded.serving import MAX_REQUEST_BYTES
from isharded.tls import FORWARD_MARK, PEER
from isharded.vision import chat_images
from test_embeddings import serving  # noqa: F401
from test_tools import QWEN3, WEATHER, mesh  # noqa: F401

CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
SAY = {"model": "scripted", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
CLAUDE_TOOL = {"name": "get_weather", "description": "Get the weather", "input_schema": WEATHER["function"]["parameters"]}
RESPONSES_TOOL = {"type": "function", "name": "get_weather", "parameters": WEATHER["function"]["parameters"]}
PIXEL = base64.b64encode(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")).decode()


def events_of(text: str) -> list[tuple[str, dict]]:
    """Server-sent events as (name, data), checking each names its own type."""
    found = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        assert name.startswith("event: ") and data.startswith("data: ")
        body = json.loads(data[6:])
        assert body["type"] == name[7:]
        found.append((name[7:], body))
    return found


def lines_of(text: str) -> list[dict]:
    return [json.loads(line) for line in text.strip().split("\n")]


def claude(mesh, path="/v1/messages", **body):
    return mesh.client.post(path, json={**SAY, **body})


def failing_stream(mesh):
    async def token_stream(session, *rest):
        yield mesh.node.tokenizer.script("Hi", 2)[0]
        raise HTTPException(status_code=503, detail="the route is gone")

    mesh.node.token_stream = token_stream


@pytest.fixture
def plain(mesh):
    mesh.use(QWEN3, "Hello there")
    return mesh


@pytest.fixture
def keyed(plain):
    """The same node, on a mesh with keys."""
    plain.node.access = AccessControl()
    plain.key = plain.node.access.add_key("alice", ROLE_USER)
    return plain


# --- Claude Messages


def test_a_claude_reply_has_the_published_shape(plain):
    answer = claude(plain).json()
    assert re.fullmatch(r"msg_\w+", answer.pop("id"))
    usage = answer.pop("usage")
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0
    assert (usage["cache_creation_input_tokens"], usage["cache_read_input_tokens"]) == (0, 0)
    assert answer == {
        "type": "message",
        "role": "assistant",
        "model": "scripted",
        "content": [{"type": "text", "text": "Hello there"}],
        "container": None,
        "diagnostics": None,
        "stop_details": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }


def test_max_tokens_is_required_and_a_limit_is_honoured(plain):
    body = {key: value for key, value in SAY.items() if key != "max_tokens"}
    refused = plain.client.post("/v1/messages", json=body)
    assert refused.status_code == 400 and refused.json()["error"]["type"] == "invalid_request_error"
    answer = claude(plain, max_tokens=2).json()
    assert answer["stop_reason"] == "max_tokens"
    assert plain.seen[-1]["max_tokens"] == 2


def test_a_limit_past_the_chat_route_is_held_to_it(plain):
    assert claude(plain, max_tokens=64000).status_code == 200
    assert plain.seen[-1]["max_tokens"] == compat.MOST_REPLY_TOKENS


def test_claude_fields_reach_the_chat_request():
    chat = compat.anthropic_chat(
        {
            **SAY,
            "system": [{"type": "text", "text": "Be "}, {"type": "text", "text": "brief."}],
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 7,
            "stop_sequences": ["END"],
            "thinking": {"type": "disabled"},
            "tools": [CLAUDE_TOOL],
            "tool_choice": {"type": "tool", "name": "get_weather", "disable_parallel_tool_use": True},
        }
    )
    assert chat.messages[0].model_dump(exclude_none=True) == {"role": "system", "content": "Be brief."}
    assert (chat.temperature, chat.top_p, chat.top_k, chat.stop, chat.enable_thinking) == (0.5, 0.9, 7, ["END"], False)
    assert chat.tools == [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": CLAUDE_TOOL["input_schema"]}}]
    assert chat.tool_choice == {"type": "function", "function": {"name": "get_weather"}}
    assert chat.parallel_tool_calls is False


@pytest.mark.parametrize("choice, expected", [({"type": "auto"}, "auto"), ({"type": "any"}, "required"), ({"type": "none"}, "none")])
def test_claude_tool_choices(choice, expected):
    assert compat.anthropic_chat({**SAY, "tools": [CLAUDE_TOOL], "tool_choice": choice}).tool_choice == expected


def test_a_claude_image_reaches_the_image_reader():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXEL}}
    chat = compat.anthropic_chat({**SAY, "messages": [{"role": "user", "content": [{"type": "text", "text": "what?"}, image]}]})
    plain, images = chat_images([message.model_dump() for message in chat.messages])
    assert plain[0]["content"] == [{"type": "text", "text": "what?"}, {"type": "image"}]
    assert images == [base64.b64decode(PIXEL)]


def test_an_image_url_is_refused_like_everywhere_else(plain):
    image = {"type": "image", "source": {"type": "url", "url": "http://169.254.169.254/x.png"}}
    answer = claude(plain, messages=[{"role": "user", "content": [image]}])
    assert answer.status_code == 400 and "remote image URLs are refused" in answer.json()["error"]["message"]


def test_an_image_needs_a_projector_as_on_the_chat_route(plain):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXEL}}
    answer = claude(plain, messages=[{"role": "user", "content": [image]}])
    assert answer.status_code == 400 and "no image projector" in answer.json()["error"]["message"]


def test_a_claude_tool_turn_goes_through_the_template(plain):
    turns = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "17C"}]}, {"type": "text", "text": "and now?"}]},
    ]
    assert claude(plain, messages=turns, tools=[CLAUDE_TOOL]).status_code == 200
    prompt = plain.prompt
    assert '"city": "Paris"' in prompt
    assert prompt.index("<tool_response>17C</tool_response>") < prompt.index("and now?")


def test_a_failed_tool_says_so():
    block = {"type": "tool_result", "tool_use_id": "t", "content": "no such city", "is_error": True}
    assert compat.anthropic_result(block)["content"] == "Error: no such city"


def test_a_claude_reply_with_a_call(mesh):
    mesh.use(QWEN3, "Let me look." + CALL)
    answer = claude(mesh, tools=[CLAUDE_TOOL]).json()
    assert answer["stop_reason"] == "tool_use"
    text, call = answer["content"]
    assert text == {"type": "text", "text": "Let me look."}
    assert call["type"] == "tool_use" and call["name"] == "get_weather" and call["input"] == {"city": "Paris"} and call["id"]


def test_a_tool_result_holding_more_than_text_is_refused_rather_than_lost(plain):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXEL}}
    turns = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [image]}]}]
    reply = claude(plain, messages=turns)
    assert reply.status_code == 400 and "text only" in reply.json()["error"]["message"]


def test_tools_anthropic_runs_are_refused(plain):
    answer = claude(plain, tools=[{"type": "web_search_20250305", "name": "web_search"}])
    assert answer.status_code == 400 and "web_search_20250305" in answer.json()["error"]["message"]


def test_a_claude_stream_is_the_real_event_sequence(plain):
    events = events_of(claude(plain, stream=True).text)
    names = [name for name, _ in events]
    deltas = names.count("content_block_delta")
    assert deltas > 1
    assert names == ["message_start", "content_block_start", *["content_block_delta"] * deltas, "content_block_stop", "message_delta", "message_stop"]
    start = events[0][1]["message"]
    assert start["role"] == "assistant" and start["content"] == [] and start["stop_reason"] is None and start["usage"]["output_tokens"] == 0
    assert events[1][1]["content_block"] == {"type": "text", "text": ""} and events[1][1]["index"] == 0
    assert "".join(body["delta"]["text"] for name, body in events if name == "content_block_delta") == "Hello there"
    closing = events[-2][1]
    assert closing["delta"] == {"container": None, "stop_details": None, "stop_reason": "end_turn", "stop_sequence": None}
    assert closing["usage"]["output_tokens"] > 0 and closing["usage"]["input_tokens"] > 0


def test_a_claude_stream_with_a_call(mesh):
    mesh.use(QWEN3, "Let me look." + CALL)
    events = events_of(claude(mesh, stream=True, tools=[CLAUDE_TOOL]).text)
    starts = [body for name, body in events if name == "content_block_start"]
    stops = [body["index"] for name, body in events if name == "content_block_stop"]
    assert [start["index"] for start in starts] == stops == [0, 1]
    tool = starts[1]["content_block"]
    assert tool["type"] == "tool_use" and tool["name"] == "get_weather" and tool["input"] == {} and tool["id"]
    written = "".join(body["delta"]["partial_json"] for name, body in events if name == "content_block_delta" and body["delta"]["type"] == "input_json_delta")
    assert json.loads(written) == {"city": "Paris"}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_a_failure_in_a_claude_stream_travels_as_an_error_event(plain):
    failing_stream(plain)
    events = events_of(claude(plain, stream=True).text)
    assert events[-1] == ("error", {**events[-1][1], "type": "error", "error": {"type": "api_error", "message": "the route is gone"}})


def test_a_failure_with_no_stream_is_a_status(plain):
    failing_stream(plain)
    answer = claude(plain)
    assert answer.status_code == 503 and answer.json()["error"]["message"] == "the route is gone"


def test_count_tokens(plain):
    answer = plain.client.post("/v1/messages/count_tokens", json={"model": "scripted", "messages": SAY["messages"], "system": "x" * 400})
    assert answer.status_code == 200 and answer.json()["input_tokens"] >= 100
    assert not plain.seen, "counting runs nothing"


def test_counting_a_model_that_is_not_there_is_refused(plain):
    answer = plain.client.post("/v1/messages/count_tokens", json={"model": "nope", "messages": SAY["messages"]})
    assert answer.status_code == 404 and answer.json()["error"]["type"] == "not_found_error"


def test_a_claude_error_has_the_published_shape(plain):
    answer = claude(plain, model="nope")
    assert answer.status_code == 404
    body = answer.json()
    assert body["type"] == "error" and body["request_id"].startswith("req_")
    assert body["error"] == {"type": "not_found_error", "message": "model 'nope' is not loaded"}


@pytest.mark.parametrize(
    "status, kind",
    [(400, "invalid_request_error"), (401, "authentication_error"), (403, "permission_error"), (404, "not_found_error"), (413, "request_too_large"), (429, "rate_limit_error"), (500, "api_error"), (503, "api_error")],
)
def test_error_kinds(status, kind):
    assert compat.anthropic_error(status, "x")["error"]["type"] == kind


@pytest.mark.parametrize(
    "body, words",
    [
        ({"messages": "hi"}, "shape"),
        ({"messages": [{"role": "user", "content": [{"type": "document"}]}]}, "'document'"),
        ({"temperature": 99}, "temperature"),
        ({"tool_choice": {"type": "sometimes"}}, "tool_choice"),
    ],
)
def test_a_bad_claude_request_is_told_why(plain, body, words):
    answer = claude(plain, **body)
    assert answer.status_code == 400 and words in answer.json()["error"]["message"]


# --- OpenAI Responses


def responses(mesh, **body):
    return mesh.client.post("/v1/responses", json={"model": "scripted", "input": "hi", **body})


def test_a_response_has_the_published_shape(plain):
    answer = responses(plain, instructions="Be brief.", max_output_tokens=64).json()
    assert answer["id"].startswith("resp_") and answer["object"] == "response" and answer["status"] == "completed"
    assert answer["model"] == "scripted" and answer["error"] is None and answer["instructions"] == "Be brief."
    (item,) = answer["output"]
    assert item["type"] == "message" and item["role"] == "assistant" and item["status"] == "completed" and item["id"].startswith("msg_")
    assert item["content"] == [{"type": "output_text", "text": "Hello there", "annotations": [], "logprobs": []}]
    usage = answer["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert plain.seen[-1]["max_tokens"] == 64
    assert "Be brief." in plain.prompt


def test_a_response_with_no_limit_runs_until_the_model_stops(plain):
    assert responses(plain).json()["status"] == "completed"
    assert plain.seen[-1]["max_tokens"] == compat.MOST_REPLY_TOKENS


@pytest.mark.parametrize("limit", [0, -1, "64", True, 1.5])
def test_a_reply_limit_that_is_no_number_is_refused_rather_than_unlimited(plain, limit):
    reply = plain.client.post("/v1/responses", json={"model": "scripted", "input": "hi", "max_output_tokens": limit})
    assert reply.status_code == 400 and "max_output_tokens" in reply.json()["error"]["message"]
    assert claude(plain, max_tokens=limit).status_code == 400


def test_a_format_left_null_is_plain_text(plain):
    assert plain.client.post("/v1/responses", json={"model": "scripted", "input": "hi", "text": {"format": None}}).status_code == 200


def test_a_function_output_holding_more_than_text_is_refused_rather_than_lost(plain):
    output = [{"type": "input_image", "image_url": f"data:image/png;base64,{PIXEL}"}]
    items = [{"type": "function_call_output", "call_id": "c1", "output": output}]
    reply = plain.client.post("/v1/responses", json={"model": "scripted", "input": items})
    assert reply.status_code == 400 and "text only" in reply.json()["error"]["message"]
    items[0]["output"] = [{"type": "input_text", "text": "17C"}]
    call = {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"}
    assert plain.client.post("/v1/responses", json={"model": "scripted", "input": [{"role": "user", "content": "hi"}, call, *items]}).status_code == 200


def test_a_response_cut_short_says_so(plain):
    answer = responses(plain, max_output_tokens=2).json()
    assert answer["status"] == "incomplete" and answer["incomplete_details"] == {"reason": "max_output_tokens"}
    assert answer["output"][0]["status"] == "incomplete"


def test_a_tool_turn_in_items_goes_through_the_template(plain):
    items = [
        {"role": "developer", "content": "Be brief."},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "17C"},
    ]
    assert responses(plain, input=items, tools=[RESPONSES_TOOL]).status_code == 200
    assert "<system>Be brief." in plain.prompt and "weather?" in plain.prompt
    assert '"city": "Paris"' in plain.prompt and "<tool_response>17C</tool_response>" in plain.prompt


def test_two_calls_of_one_reply_share_a_message():
    item = {"type": "function_call", "name": "f", "arguments": "{}"}
    messages = compat.responses_messages({"input": [{"role": "user", "content": "x"}, {**item, "call_id": "a"}, {**item, "call_id": "b"}]})
    assert [call["id"] for call in messages[-1]["tool_calls"]] == ["a", "b"]


def test_responses_fields_reach_the_chat_request():
    chat = compat.responses_chat(
        {"input": "hi", "temperature": 0.5, "top_p": 0.9, "tools": [RESPONSES_TOOL], "tool_choice": {"type": "function", "name": "get_weather"}, "parallel_tool_calls": False}
    )
    assert (chat.temperature, chat.top_p, chat.parallel_tool_calls) == (0.5, 0.9, False)
    assert chat.tool_choice == {"type": "function", "function": {"name": "get_weather"}}
    assert chat.tools[0]["function"]["parameters"] == RESPONSES_TOOL["parameters"]


def test_a_response_image_reaches_the_image_reader():
    url = f"data:image/png;base64,{PIXEL}"
    part = compat.response_part({"type": "input_image", "image_url": url})
    assert chat_images([{"role": "user", "content": [part]}])[1] == [base64.b64decode(PIXEL)]


def test_a_response_with_a_call_gives_a_function_call_item(mesh):
    mesh.use(QWEN3, CALL)
    answer = responses(mesh, tools=[RESPONSES_TOOL]).json()
    (item,) = answer["output"]
    assert item["type"] == "function_call" and item["name"] == "get_weather" and item["status"] == "completed"
    assert item["id"].startswith("fc_") and item["call_id"] and json.loads(item["arguments"]) == {"city": "Paris"}


@pytest.mark.parametrize(
    "field, words",
    [
        ({"previous_response_id": "resp_1"}, "keeps no replies"),
        ({"conversation": "conv_1"}, "keeps no replies"),
        ({"background": True}, "keeps no replies"),
        ({"text": {"format": {"type": "json_object"}}}, "text.format"),
        ({"tools": [{"type": "web_search"}]}, "only function tools"),
        ({"input": [{"type": "item_reference", "id": "x"}]}, "'item_reference'"),
    ],
)
def test_what_responses_cannot_do_is_refused_in_openai_shape(plain, field, words):
    answer = responses(plain, **field)
    assert answer.status_code == 400
    error = answer.json()["error"]
    assert words in error["message"] and error["type"] == "invalid_request_error" and set(error) == {"message", "type", "param", "code"}


def test_a_response_stream_is_the_real_event_sequence(plain):
    events = events_of(responses(plain, stream=True).text)
    names = [name for name, _ in events]
    deltas = names.count("response.output_text.delta")
    assert deltas > 1
    assert names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        *["response.output_text.delta"] * deltas,
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [body["sequence_number"] for _, body in events] == list(range(len(events)))
    assert events[0][1]["response"]["status"] == "in_progress" and events[0][1]["response"]["output"] == []
    added = events[2][1]
    assert added["output_index"] == 0 and added["item"]["status"] == "in_progress" and added["item"]["content"] == []
    delta = events[4][1]
    assert delta["item_id"] == added["item"]["id"] and delta["output_index"] == 0 and delta["content_index"] == 0
    assert "".join(body["delta"] for name, body in events if name.endswith("text.delta")) == "Hello there"
    final = events[-1][1]["response"]
    assert final["status"] == "completed" and final["output"][0]["content"][0]["text"] == "Hello there" and final["usage"]["output_tokens"] > 0


def test_a_response_stream_with_a_call(mesh):
    mesh.use(QWEN3, "Let me look." + CALL)
    events = events_of(responses(mesh, stream=True, tools=[RESPONSES_TOOL]).text)
    names = [name for name, _ in events]
    at = names.index("response.function_call_arguments.delta")
    assert names[at - 1 : at + 3] == ["response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.output_item.done"]
    assert names[at - 2] == "response.output_item.done", "the message closes before the call opens"
    added = events[at - 1][1]
    assert added["output_index"] == 1 and added["item"]["type"] == "function_call" and added["item"]["arguments"] == ""
    assert json.loads(events[at + 1][1]["arguments"]) == {"city": "Paris"}
    output = events[-1][1]["response"]["output"]
    assert [item["type"] for item in output] == ["message", "function_call"]


def test_a_cut_stream_ends_incomplete(plain):
    events = events_of(responses(plain, stream=True, max_output_tokens=2).text)
    assert events[-1][0] == "response.incomplete" and events[-1][1]["response"]["status"] == "incomplete"


def test_a_failure_in_a_response_stream_ends_failed(plain):
    failing_stream(plain)
    name, body = events_of(responses(plain, stream=True).text)[-1]
    assert name == "response.failed"
    assert body["response"]["status"] == "failed" and body["response"]["error"] == {"code": "server_error", "message": "the route is gone"}


# --- Ollama


def ollama(mesh, path="/api/chat", **body):
    return mesh.client.post(path, json={"model": "scripted", **body})


def chat_ollama(mesh, **body):
    return ollama(mesh, messages=[{"role": "user", "content": "hi"}], **body)


def test_ollama_streams_by_default_as_json_lines(plain):
    answer = chat_ollama(plain)
    assert answer.headers["content-type"].startswith("application/x-ndjson")
    lines = lines_of(answer.text)
    assert all(line["done"] is False and line["model"] == "scripted" and line["created_at"].endswith("Z") for line in lines[:-1])
    assert "".join(line["message"]["content"] for line in lines) == "Hello there"
    assert lines[0]["message"]["role"] == "assistant"
    final = lines[-1]
    assert final["done"] is True and final["done_reason"] == "stop" and final["message"]["content"] == ""
    assert final["prompt_eval_count"] > 0 and final["eval_count"] > 0
    assert final["total_duration"] >= final["prompt_eval_duration"] and final["load_duration"] == 0 and final["eval_duration"] >= 0


def test_ollama_streams_when_stream_is_null(plain):
    reply = plain.client.post("/api/chat", json={"model": "scripted", "messages": SAY["messages"], "stream": None})
    assert reply.headers["content-type"].startswith("application/x-ndjson") and lines_of(reply.text)[-1]["done"]


def test_a_whole_ollama_chat(plain):
    answer = chat_ollama(plain, stream=False).json()
    assert answer["message"] == {"role": "assistant", "content": "Hello there"}
    assert answer["done"] is True and answer["done_reason"] == "stop" and answer["eval_count"] > 0 and "total_duration" in answer


def test_ollama_options_reach_the_chat_request():
    options = {"temperature": 0.5, "top_p": 0.9, "top_k": 7, "seed": 11, "num_predict": 9, "stop": ["END"], "num_ctx": 4096}
    chat = compat.ollama_chat({"messages": [{"role": "user", "content": "hi"}], "options": options, "think": False})
    assert (chat.temperature, chat.top_p, chat.top_k, chat.seed, chat.max_tokens, chat.stop, chat.enable_thinking) == (0.5, 0.9, 7, 11, 9, ["END"], False)


@pytest.mark.parametrize("asked", [None, -1, -2])
def test_ollama_with_no_limit_runs_until_the_model_stops(plain, asked):
    assert chat_ollama(plain, options={"num_predict": asked}, stream=False).json()["done_reason"] == "stop"
    assert plain.seen[-1]["max_tokens"] == compat.MOST_REPLY_TOKENS


def test_a_cut_ollama_reply_says_length(plain):
    assert chat_ollama(plain, options={"num_predict": 2}, stream=False).json()["done_reason"] == "length"


@pytest.mark.parametrize("wanted", ["json", {"type": "object"}])
@pytest.mark.parametrize("path, field", [("/api/chat", {"messages": [{"role": "user", "content": "hi"}]}), ("/api/generate", {"prompt": "hi"})])
def test_format_is_refused_rather_than_pretended(plain, path, field, wanted):
    answer = ollama(plain, path, format=wanted, **field)
    assert answer.status_code == 400 and "format is refused" in answer.json()["error"]


def test_ollama_images_reach_the_image_reader():
    chat = compat.ollama_chat({"messages": [{"role": "user", "content": "what?", "images": [PIXEL]}]})
    assert chat_images([message.model_dump() for message in chat.messages])[1] == [base64.b64decode(PIXEL)]


def test_ollama_tools_and_a_call_back(mesh):
    mesh.use(QWEN3, "Let me look." + CALL)
    answer = chat_ollama(mesh, tools=[WEATHER], stream=False).json()
    assert answer["message"]["content"] == "Let me look."
    (call,) = answer["message"]["tool_calls"]
    assert call["function"]["name"] == "get_weather" and call["function"]["arguments"] == {"city": "Paris"}
    assert answer["done_reason"] == "stop"


def test_an_ollama_stream_with_a_call(mesh):
    mesh.use(QWEN3, CALL)
    lines = lines_of(chat_ollama(mesh, tools=[WEATHER]).text)
    calls = [call for line in lines for call in line["message"].get("tool_calls", [])]
    assert [call["function"]["arguments"] for call in calls] == [{"city": "Paris"}]
    assert lines[-1]["done"] is True


def test_an_ollama_tool_result_finds_its_call(plain):
    turns = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_name": "get_weather", "content": "17C"},
    ]
    assert ollama(plain, messages=turns, tools=[WEATHER], stream=False).status_code == 200
    assert "<tool_response>17C</tool_response>" in plain.prompt and '"city": "Paris"' in plain.prompt


def test_a_failure_in_an_ollama_stream_is_a_last_line(plain):
    failing_stream(plain)
    assert lines_of(chat_ollama(plain).text)[-1] == {"error": "the route is gone"}


def test_generate_goes_through_the_template(plain):
    lines = lines_of(ollama(plain, "/api/generate", prompt="hi", system="Be brief.").text)
    assert "".join(line["response"] for line in lines) == "Hello there"
    assert lines[-1]["done"] is True and lines[-1]["response"] == "" and lines[-1]["eval_count"] > 0
    assert "<system>Be brief." in plain.prompt and "<user>hi" in plain.prompt


def test_raw_generate_skips_the_template(plain):
    answer = ollama(plain, "/api/generate", prompt="hi", raw=True, stream=False).json()
    assert answer["response"] == "Hello there" and answer["done"] is True
    assert not hasattr(plain, "prompt"), "no template ran"


@pytest.mark.parametrize("field", [{"suffix": "x"}, {"template": "x"}, {"raw": True, "images": [PIXEL]}])
def test_generate_refuses_what_it_cannot_do(plain, field):
    answer = ollama(plain, "/api/generate", prompt="hi", **field)
    assert answer.status_code == 400 and answer.json()["error"]


@pytest.mark.parametrize("path, field, shown", [("/api/generate", {}, "response"), ("/api/chat", {"messages": []}, "message")])
def test_an_empty_request_warms_a_model_up(plain, path, field, shown):
    answer = ollama(plain, path, **field).json()
    assert answer["done"] is True and answer["done_reason"] == "load" and shown in answer
    assert not plain.seen


def test_warming_up_a_model_that_is_not_there_is_refused(plain):
    answer = plain.client.post("/api/generate", json={"model": "nope"})
    assert answer.status_code == 404 and answer.json() == {"error": "model 'nope' is not loaded"}


def test_api_embed(serving):
    client, seen = serving
    answer = client.post("/api/embed", json={"model": "embed.gguf", "input": ["one", "two words"]})
    assert answer.status_code == 200
    body = answer.json()
    assert body["model"] == "embed.gguf" and len(body["embeddings"]) == 2 and body["prompt_eval_count"] > 0 and body["total_duration"] > 0
    assert seen


def test_api_embeddings_is_the_older_shape(serving):
    client, _ = serving
    answer = client.post("/api/embeddings", json={"model": "embed.gguf", "prompt": "one"}).json()
    assert list(answer) == ["embedding"] and isinstance(answer["embedding"][0], float)


def test_embedding_on_a_chat_model_is_refused_the_same_way(plain):
    answer = ollama(plain, "/api/embed", input="hi")
    assert answer.status_code == 400 and "gives no embeddings" in answer.json()["error"]


def active_model():
    return SimpleNamespace(
        embedding=False,
        projector=None,
        file_sizes={"a.gguf": 1000, "b.gguf": 24},
        profile=SimpleNamespace(context_tokens=8192),
        adapter=SimpleNamespace(header={"general.architecture": "qwen3", "general.size_label": "8B"}),
    )


def test_api_tags_lists_every_running_model(plain):
    plain.node.active = active_model()
    plain.node.lanes.seen["small"] = {"ready": True, "model": "lane-model", "kind": "chat", "vision": False}
    rows = plain.client.get("/api/tags").json()["models"]
    assert [row["model"] for row in rows] == ["scripted", "lane-model"]
    assert rows[0]["size"] == 1024 and rows[0]["name"] == "scripted" and len(rows[0]["digest"]) == 64
    assert rows[0]["details"]["family"] == "qwen3" and rows[0]["details"]["parameter_size"] == "8B" and rows[0]["details"]["format"] == "gguf"


def test_api_ps(plain):
    plain.node.active = active_model()
    (row,) = plain.client.get("/api/ps").json()["models"]
    assert row["model"] == "scripted" and row["size"] == 1024 and row["context_length"] == 8192 and row["expires_at"]


def test_api_show(plain):
    plain.node.active = active_model()
    answer = plain.client.post("/api/show", json={"model": "scripted"}).json()
    assert answer["details"]["family"] == "qwen3"
    assert answer["model_info"] == {"general.architecture": "qwen3", "qwen3.context_length": 8192}
    assert answer["capabilities"] == ["completion", "tools"]
    assert answer["template"] == QWEN3


def test_api_show_refuses_what_is_not_running(plain):
    answer = plain.client.post("/api/show", json={"model": "nope"})
    assert answer.status_code == 404 and answer.json() == {"error": "model 'nope' is not loaded"}


def test_api_version_is_a_version_an_ollama_client_accepts(plain):
    from isharded import __version__

    answer = plain.client.get("/api/version").json()
    assert answer == {"version": compat.OLLAMA_VERSION, "isharded": __version__}
    # Clients gate on the release they need: thinking came with 0.9.
    assert tuple(map(int, answer["version"].split("."))) >= (0, 9, 0)


@pytest.mark.parametrize("method, path", [("POST", "/api/pull"), ("POST", "/api/push"), ("POST", "/api/create"), ("POST", "/api/copy"), ("DELETE", "/api/delete")])
def test_changing_models_is_refused_with_where_to_go(plain, method, path):
    answer = plain.client.request(method, path, json={"model": "llama3"})
    assert answer.status_code == 400 and "isharded load" in answer.json()["error"]


# --- the same model rules on every route


@pytest.mark.parametrize(
    "path, body",
    [
        ("/v1/messages", SAY),
        ("/v1/responses", {"input": "hi"}),
        ("/api/chat", {"messages": SAY["messages"]}),
        ("/api/generate", {"prompt": "hi"}),
    ],
)
def test_the_model_field_works_as_it_does_on_the_chat_route(plain, path, body):
    assert plain.client.post(path, json={**body, "model": "scripted", "stream": False}).status_code == 200
    assert plain.client.post(path, json={**{k: v for k, v in body.items() if k != "model"}, "stream": False}).status_code == 200, "no model means the loaded one"
    refused = plain.client.post(path, json={**body, "model": "nope", "stream": False})
    assert refused.status_code == 404 and "model 'nope' is not loaded" in refused.text


@pytest.mark.parametrize("path, body", [("/v1/messages", SAY), ("/v1/responses", {"input": "hi"}), ("/api/chat", {"messages": SAY["messages"]})])
def test_with_nothing_loaded_is_refused_as_on_the_chat_route(mesh, path, body):
    mesh.node.model_id = None
    mesh.node.route = []
    answer = mesh.client.post(path, json={**body, "model": None})
    assert answer.status_code == 409 and "no model is ready" in answer.text


# --- the same doors as /v1/chat/completions


SHUT_ROUTES = [(method, path) for method, path, _, _ in compat.ROUTES]


def call(client, method, path, headers=None, content=b"{}"):
    return client.request(method, path, headers={"content-type": "application/json", **(headers or {})}, content=None if method == "GET" else content)


@pytest.mark.parametrize("method, path", SHUT_ROUTES)
def test_no_key_on_a_mesh_with_keys_is_refused(keyed, method, path):
    answer = call(keyed.client, method, path)
    assert answer.status_code == 401
    assert "a valid key is required" in answer.text
    assert not keyed.seen


@pytest.mark.parametrize("method, path", SHUT_ROUTES)
def test_a_wrong_key_is_refused(keyed, method, path):
    assert call(keyed.client, method, path, {"Authorization": "Bearer isk-wrong"}).status_code == 401


@pytest.mark.parametrize("method, path", SHUT_ROUTES)
def test_a_node_certificate_is_not_a_user_key(keyed, method, path, monkeypatch):
    keyed.node.access = AccessControl(identities=True)

    async def with_identity(scope, receive, send):
        PEER.set("node-a")
        await orchestrator.app(scope, receive, send)

    assert call(TestClient(with_identity), method, path).status_code == 403


def test_ollama_with_no_key_gets_its_own_error_shape(keyed):
    answer = keyed.client.post("/api/chat", json={"model": "scripted", "messages": SAY["messages"]})
    assert answer.status_code == 401 and answer.json() == {"error": "a valid key is required"}


def test_ollama_with_the_header_gets_in(keyed):
    answer = keyed.client.post("/api/chat", json={"messages": SAY["messages"], "stream": False}, headers={"Authorization": f"Bearer {keyed.key}"})
    assert answer.status_code == 200


def test_a_stranger_learns_nothing_from_a_broken_body(keyed):
    answer = keyed.client.post("/v1/messages", content=b"{not json", headers={"content-type": "application/json"})
    assert answer.status_code == 401


@pytest.mark.parametrize(
    "headers",
    [lambda key: {"x-api-key": key}, lambda key: {"Authorization": f"Bearer {key}"}, lambda key: {"x-api-key": "isk-wrong", "Authorization": f"Bearer {key}"}],
    ids=["x-api-key", "bearer", "bearer-wins"],
)
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_claude_routes_take_the_key_either_way(keyed, headers, path):
    assert keyed.client.post(path, json=SAY, headers=headers(keyed.key)).status_code == 200


def test_x_api_key_means_nothing_elsewhere(keyed):
    assert keyed.client.post("/v1/responses", json={"input": "hi"}, headers={"x-api-key": keyed.key}).status_code == 401
    assert keyed.client.post("/api/chat", json={"messages": SAY["messages"]}, headers={"x-api-key": keyed.key}).status_code == 401


def test_a_claude_auth_error_has_the_claude_shape(keyed):
    body = keyed.client.post("/v1/messages", json=SAY, headers={"x-api-key": "isk-wrong"}).json()
    assert body["type"] == "error" and body["error"]["type"] == "authentication_error"


@pytest.mark.parametrize(
    "path, body, shape",
    [
        ("/v1/messages", SAY, lambda reply: reply["error"]["type"] == "rate_limit_error"),
        ("/v1/responses", {"input": "hi"}, lambda reply: reply["error"]["type"] == "rate_limit_error"),
        ("/api/chat", {"messages": SAY["messages"], "stream": False}, lambda reply: "rate limit" in reply["error"]),
    ],
)
def test_the_rate_limit_counts_each_reply_once_and_holds(plain, path, body, shape):
    plain.node.access = AccessControl()
    key = plain.node.access.add_key("slow", ROLE_USER, requests_per_minute=2)
    headers = {"Authorization": f"Bearer {key}"}
    assert [plain.client.post(path, json=body, headers=headers).status_code for _ in range(2)] == [200, 200]
    refused = plain.client.post(path, json=body, headers=headers)
    assert refused.status_code == 429 and int(refused.headers["retry-after"]) >= 1 and shape(refused.json())


@pytest.mark.parametrize("method, path", [(method, path) for method, path in SHUT_ROUTES if method == "POST"])
def test_the_body_limit_holds(keyed, method, path):
    answer = keyed.client.post(path, content=b"x" * (MAX_REQUEST_BYTES + 1), headers={"Authorization": f"Bearer {keyed.key}"})
    assert answer.status_code == 413


@pytest.mark.parametrize("kind", ["text/plain", "application/x-www-form-urlencoded"])
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses", "/api/chat", "/api/generate"])
def test_a_page_cannot_post_here_with_a_simple_form(plain, path, kind):
    """No preflight guards those two, and a mesh with no keys has only this."""
    answer = plain.client.post(path, content=json.dumps({**SAY, "prompt": "hi", "input": "hi"}), headers={"content-type": kind})
    assert answer.status_code == 400 and "Content-Type: application/json" in answer.text
    assert not plain.seen


@pytest.mark.parametrize("method, path", [(method, path) for method, path in SHUT_ROUTES if method != "GET"])
def test_every_request_that_is_not_a_get_is_audited(keyed, method, path, monkeypatch):
    seen = []
    monkeypatch.setattr(keyed.node.audit, "record", lambda *entry: seen.append(entry))
    call(keyed.client, method, path, {"Authorization": f"Bearer {keyed.key}"})
    assert seen and seen[0][0] == "alice" and seen[0][1:3] == (method, path)


def test_the_audit_names_a_caller_who_used_x_api_key(keyed, monkeypatch):
    seen = []
    monkeypatch.setattr(keyed.node.audit, "record", lambda *entry: seen.append(entry))
    keyed.client.post("/v1/messages", json=SAY, headers={"x-api-key": keyed.key})
    assert seen[0][:3] == ("alice", "POST", "/v1/messages")


def test_the_audit_keeps_prompts_out(keyed, monkeypatch):
    seen = []
    monkeypatch.setattr(keyed.node.audit, "record", lambda *entry: seen.append(entry))
    keyed.client.post("/api/chat", json={"messages": [{"role": "user", "content": "a private thought"}], "stream": False}, headers={"Authorization": f"Bearer {keyed.key}"})
    assert seen and "private" not in str(seen)


class Clock:
    def __call__(self):
        return 100.0


def streamed(status, body):
    """What a node sends another: forwarding reads replies as a stream."""
    return httpx.Response(status, stream=httpx.ByteStream(json.dumps(body).encode()), headers={"content-type": "application/json"})


@pytest.fixture
def follower(monkeypatch):
    """A node with no route, which passes requests to its coordinator."""
    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["url"] = str(request.url)
        return streamed(200, {"from": "coordinator"})

    node = orchestrator.Orchestrator()
    node.access = AccessControl()
    node.key = node.access.add_key("alice", ROLE_ADMIN)
    node.mesh = ControlMesh("follower", "http://follower:1", (), Clock())
    node.mesh.observe({"node_id": "boss", "url": "http://boss:9", "term": 1, "leader_id": "boss"})
    node.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(orchestrator, "runtime", node)
    return node, seen, TestClient(orchestrator.app)


@pytest.mark.parametrize("method, path", [("POST", "/v1/messages"), ("POST", "/v1/responses"), ("POST", "/api/chat"), ("GET", "/api/tags")])
def test_a_node_with_no_route_passes_these_on(follower, method, path):
    node, seen, client = follower
    answer = client.request(method, path, headers={"Authorization": f"Bearer {node.key}"})
    assert answer.json() == {"from": "coordinator"} and seen["url"] == f"http://boss:9{path}"
    assert seen[FORWARD_MARK.lower()] == "follower"


def test_x_api_key_never_leaves_the_node_it_was_given_to(follower):
    """Forwarding drops Authorization where it signs for the caller, and knows nothing of x-api-key."""
    node, seen, client = follower
    client.post("/v1/messages", json=SAY, headers={"x-api-key": node.key})
    assert seen["url"] == "http://boss:9/v1/messages"
    assert "x-api-key" not in seen


@pytest.mark.parametrize("path", ["/v1/models", "/v1/models/scripted", "/admin/status", "/api/tags", "/v1/chat/completions"])
def test_x_api_key_goes_on_every_path_that_forwards(follower, path):
    """Only some paths read this header, and every path that forwards must drop it."""
    node, seen, client = follower
    client.get(path, headers={"x-api-key": node.key, "Authorization": f"Bearer {node.key}"})
    assert seen["url"] == f"http://boss:9{path}"
    assert "x-api-key" not in seen


def lane_stream(*chunks):
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return body.encode()


LANE_REPLY = lane_stream(
    {"model": "lane-model", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"model": "lane-model", "choices": [{"index": 0, "delta": {"content": "from the lane é"}, "finish_reason": None}]},
    {"model": "lane-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, "isharded": {"time_to_first_token_ms": 1.0}},
)


@pytest.fixture
def lane(plain):
    """A front door whose lane serves a model, and sends its bytes cut anywhere."""
    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        if request.url.path == "/v1/messages/count_tokens":
            return streamed(200, {"input_tokens": 12})
        pieces = [LANE_REPLY[start : start + 7] for start in range(0, len(LANE_REPLY), 7)]
        return httpx.Response(200, stream=httpx.ByteStream(b"".join(pieces)), headers={"content-type": "text/event-stream"})

    plain.node.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    plain.node.mesh = ControlMesh("boss", "http://boss:9", (), Clock())
    plain.node.mesh.leader_id = "boss"
    plain.node.mesh.lanes = {"small": {"slot": 1, "budgets": {}}}
    plain.node.lanes.path = Path("unused")
    plain.node.lanes.seen["small"] = {"ready": True, "model": "lane-model", "leader": "boss", "leader_url": "http://lane-boss:1009"}
    plain.lane_seen = seen
    return plain


def test_a_reply_from_a_lane_comes_back_translated(lane):
    answer = lane.client.post("/v1/messages", json={**SAY, "model": "lane-model"}, headers={"x-api-key": "isk-private"}).json()
    assert answer["content"] == [{"type": "text", "text": "from the lane é"}] and answer["model"] == "lane-model"
    assert lane.lane_seen["url"].endswith("/v1/chat/completions") and lane.lane_seen["body"]["stream"] is True
    assert "isk-private" not in str(lane.lane_seen)


def test_a_stream_from_a_lane_comes_back_translated(lane):
    lines = lines_of(ollama(lane, model="lane-model", messages=SAY["messages"]).text)
    assert lines[0]["message"]["content"] == "from the lane é" and lines[-1]["done"] is True and lines[-1]["eval_count"] == 2


def test_counting_for_a_lane_is_asked_of_the_lane(lane):
    answer = lane.client.post("/v1/messages/count_tokens", json={"model": "lane-model", "messages": SAY["messages"]})
    assert answer.json() == {"input_tokens": 12} and lane.lane_seen["url"].endswith("/v1/messages/count_tokens")


def test_a_lane_that_refuses_is_refused_in_the_clients_shape(lane):
    lane.node.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: streamed(400, {"detail": "too long"})))
    answer = claude(lane, model="lane-model")
    assert answer.status_code == 400 and answer.json()["error"]["message"] == "too long"


def test_a_lane_that_stops_early_is_not_a_finished_reply(lane):
    half = lane_stream({"model": "m", "choices": [{"index": 0, "delta": {"content": "half"}, "finish_reason": None}]})[: -len(b"data: [DONE]\n\n")]
    lane.node.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=httpx.ByteStream(half), headers={"content-type": "text/event-stream"})))
    assert claude(lane, model="lane-model").status_code == 503
    assert events_of(claude(lane, model="lane-model", stream=True).text)[-1][0] == "error"


# --- a client that leaves


@pytest.mark.parametrize(
    "stream",
    [
        lambda items: compat.anthropic_stream(items, {}),
        lambda items: compat.responses_stream(items, {}),
        lambda items: compat.ollama_stream(items, {}, compat.ollama_response, 0.0),
    ],
    ids=["claude", "responses", "ollama"],
)
def test_leaving_a_stream_closes_the_reply_behind_it(stream):
    closed = []

    async def items():
        try:
            yield {"text": "one"}
            yield {"text": "two"}
        finally:
            closed.append(True)

    async def leave():
        events = stream(items())
        async for line in events:
            if "one" in line:
                break
        await events.aclose()

    asyncio.run(leave())
    assert closed


# --- how a reply ended, and what it reasoned


THOUGHT = "<think>\nLet me see.\n</think>\n\nHello there"
OPENS_THINKING = "{% for message in messages %}{{ message.content }}{% endfor %}<think>\n"


def blocks_of(events: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    """Each content block as its type and all that was written into it."""
    blocks = []
    for name, body in events:
        if name == "content_block_start":
            blocks.append([body["content_block"]["type"], ""])
        elif name == "content_block_delta":
            delta = body["delta"]
            blocks[-1][1] += delta.get("text", "") + delta.get("thinking", "")
    return [tuple(block) for block in blocks]


def test_a_stop_sequence_is_named_in_a_claude_reply(mesh):
    mesh.use(QWEN3, "Hello END there STOP")
    answer = claude(mesh, stop_sequences=["STOP", "END"]).json()
    assert answer["content"] == [{"type": "text", "text": "Hello "}]
    assert (answer["stop_reason"], answer["stop_sequence"]) == ("stop_sequence", "END")


def test_a_stop_sequence_is_named_in_a_claude_stream(mesh):
    mesh.use(QWEN3, "Hello END there")
    events = events_of(claude(mesh, stream=True, stop_sequences=["END"]).text)
    assert blocks_of(events) == [("text", "Hello ")]
    assert events[-2][1]["delta"] == {"container": None, "stop_details": None, "stop_reason": "stop_sequence", "stop_sequence": "END"}


def test_a_stop_sequence_that_never_came_is_not_named(plain):
    answer = claude(plain, stop_sequences=["NOPE"]).json()
    assert (answer["stop_reason"], answer["stop_sequence"]) == ("end_turn", None)


def test_a_call_ends_a_claude_reply_as_a_call_even_with_stop_sequences(mesh):
    mesh.use(QWEN3, CALL + "END")
    answer = claude(mesh, tools=[CLAUDE_TOOL], stop_sequences=["END"]).json()
    assert (answer["stop_reason"], answer["stop_sequence"]) == ("tool_use", None)


def test_the_chat_stream_says_which_stop_string_ended_it(mesh):
    mesh.use(QWEN3, "Hello END")
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}], "stream": True, "stop": ["END"], "max_tokens": 64}
    lines = [json.loads(line[6:]) for line in mesh.client.post("/v1/chat/completions", json=body).text.split("\n\n") if line.startswith("data: {")]
    assert lines[-1]["isharded"]["stop_sequence"] == "END" and lines[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("size", [1, 3, 100])
def test_a_claude_reply_gives_its_reasoning_a_block(mesh, size):
    mesh.use(QWEN3, THOUGHT, size=size)
    content = claude(mesh).json()["content"]
    assert content[0]["type"] == "thinking" and content[0]["thinking"] == "Let me see." and content[0]["signature"]
    assert content[1:] == [{"type": "text", "text": "Hello there"}]


def test_a_claude_stream_gives_reasoning_its_own_block_with_a_signature_last(mesh):
    mesh.use(QWEN3, THOUGHT)
    events = events_of(claude(mesh, stream=True).text)
    assert blocks_of(events) == [("thinking", "Let me see."), ("text", "Hello there")]
    start = next(body for name, body in events if name == "content_block_start")
    assert start["content_block"] == {"type": "thinking", "thinking": "", "signature": ""} and start["index"] == 0
    first = [(name, body) for name, body in events if body.get("index") == 0]
    assert first[-1][0] == "content_block_stop" and first[-2][1]["delta"]["type"] == "signature_delta"
    assert first[-2][1]["delta"]["signature"] == claude(mesh).json()["content"][0]["signature"]
    assert [body["index"] for name, body in events if name == "content_block_stop"] == [0, 1]


def test_a_template_that_opens_reasoning_still_gets_a_block(mesh):
    mesh.use(OPENS_THINKING, "Let me see.\n</think>\n\nHello there")
    assert [block["type"] for block in claude(mesh).json()["content"]] == ["thinking", "text"]
    assert blocks_of(events_of(claude(mesh, stream=True).text)) == [("thinking", "Let me see."), ("text", "Hello there")]


def test_only_a_template_that_opens_reasoning_marks_the_chat_stream(mesh):
    body = {"model": "scripted", "messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 64}
    for template, marked in ((OPENS_THINKING, True), (QWEN3, False)):
        mesh.use(template, "Hello")
        first = json.loads(mesh.client.post("/v1/chat/completions", json=body).text.split("\n\n")[0][6:])
        assert first["choices"][0]["delta"] == {"role": "assistant", **({"isharded": {"in_thought": True}} if marked else {})}


def test_a_reply_that_only_mentions_the_tag_is_all_text(mesh):
    mesh.use(QWEN3, "Models write <think> first, then </think> after.")
    assert claude(mesh).json()["content"] == [{"type": "text", "text": "Models write <think> first, then </think> after."}]


def test_reasoning_cut_off_by_the_limit_is_still_reasoning(mesh):
    mesh.use(QWEN3, "<think>\nThis goes on and on and on", size=2)
    answer = claude(mesh, max_tokens=6).json()
    assert [block["type"] for block in answer["content"]] == ["thinking"] and answer["stop_reason"] == "max_tokens"


def test_reasoning_comes_before_a_call(mesh):
    mesh.use(QWEN3, "<think>\nI should look.\n</think>\n\n" + CALL)
    answer = claude(mesh, tools=[CLAUDE_TOOL]).json()
    assert [block["type"] for block in answer["content"]] == ["thinking", "tool_use"] and answer["content"][0]["thinking"] == "I should look."
    assert blocks_of(events_of(claude(mesh, stream=True, tools=[CLAUDE_TOOL]).text))[0] == ("thinking", "I should look.")


def test_omitted_thinking_keeps_the_block_and_loses_the_words(mesh):
    mesh.use(QWEN3, THOUGHT)
    asked = {"thinking": {"type": "enabled", "budget_tokens": 1024, "display": "omitted"}}
    block = claude(mesh, **asked).json()["content"][0]
    assert block["type"] == "thinking" and block["thinking"] == "" and block["signature"]
    events = events_of(claude(mesh, stream=True, **asked).text)
    assert blocks_of(events) == [("thinking", ""), ("text", "Hello there")]
    assert [body["delta"]["type"] for name, body in events if name == "content_block_delta" and body["index"] == 0] == ["thinking_delta", "signature_delta"]


def test_a_lane_reply_is_sorted_and_named_like_any_other():
    stream = lane_stream(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "isharded": {"in_thought": True}}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "hmm</think>\n\nHi"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, "isharded": {"stop_sequence": "END"}},
    )

    async def collect() -> list[dict]:
        async def chunks():
            yield stream

        return [item async for item in compat.reply_items(SimpleNamespace(body_iterator=chunks(), background=None), compat.ThoughtSplitter())]

    items = asyncio.run(collect())
    assert items[:2] == [{"thinking": "hmm"}, {"text": "Hi"}] and items[2]["stop_sequence"] == "END"
    assert compat.stop_fields(items[2]) == {"stop_reason": "stop_sequence", "stop_sequence": "END"}


def test_thinking_in_history_reaches_the_template(mesh):
    mesh.use("{% for message in messages %}[{{ message.reasoning_content }}|{{ message.content }}]{% endfor %}", "Hello")
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "I thought", "signature": "abc"}, {"type": "redacted_thinking", "data": "sealed"}, {"type": "text", "text": "said"}]},
        {"role": "user", "content": "more"},
    ]
    assert claude(mesh, messages=turns).status_code == 200
    assert "[I thought|said]" in mesh.prompt and "sealed" not in mesh.prompt


def test_a_message_of_only_thinking_is_no_message(mesh):
    mesh.use(QWEN3, "Hello")
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "sealed"}]}, {"role": "user", "content": "more"}]
    assert claude(mesh, messages=turns).status_code == 200


@pytest.mark.parametrize("cut", range(1, len(THOUGHT)))
def test_reasoning_is_found_wherever_the_text_is_cut(cut):
    splitter = compat.ThoughtSplitter()
    items = splitter.feed(THOUGHT[:cut]) + splitter.feed(THOUGHT[cut:]) + splitter.finish()
    assert "".join(item.get("thinking", "") for item in items) == "Let me see."
    assert "".join(item.get("text", "") for item in items) == "Hello there"


def test_with_splitting_off_a_reply_is_left_alone():
    splitter = compat.ThoughtSplitter(False)
    splitter.open()
    assert splitter.feed(THOUGHT) == [{"text": THOUGHT}] and splitter.finish() == []


def test_a_reply_of_nothing_but_space_comes_out_as_it_went_in():
    splitter = compat.ThoughtSplitter()
    assert splitter.feed("  ") == [] and splitter.finish() == [{"text": "  "}]


# --- Ollama's reasoning, seed and version


def test_ollama_chat_gives_reasoning_its_own_field(mesh):
    mesh.use(QWEN3, THOUGHT)
    answer = mesh.client.post("/api/chat", json={"model": "scripted", "messages": SAY["messages"], "stream": False}).json()
    assert answer["message"] == {"role": "assistant", "content": "Hello there", "thinking": "Let me see."}


def test_an_ollama_stream_sends_reasoning_first(mesh):
    mesh.use(QWEN3, THOUGHT)
    lines = lines_of(mesh.client.post("/api/chat", json={"model": "scripted", "messages": SAY["messages"]}).text)
    assert lines[0]["message"]["thinking"] and lines[0]["message"]["content"] == "" and lines[0]["done"] is False
    assert "".join(line["message"].get("thinking", "") for line in lines) == "Let me see."
    assert "".join(line["message"]["content"] for line in lines) == "Hello there"
    assert lines[-1]["done"] is True


def test_ollama_generate_gives_reasoning_its_own_field(mesh):
    mesh.use(QWEN3, THOUGHT)
    answer = mesh.client.post("/api/generate", json={"model": "scripted", "prompt": "hi", "stream": False}).json()
    assert (answer["response"], answer["thinking"]) == ("Hello there", "Let me see.")


def test_raw_ollama_generate_leaves_the_text_as_it_came(mesh):
    mesh.use(QWEN3, THOUGHT)
    answer = mesh.client.post("/api/generate", json={"model": "scripted", "prompt": "hi", "raw": True, "stream": False}).json()
    assert answer["response"] == THOUGHT and "thinking" not in answer


def test_ollama_history_passes_its_thinking_on():
    messages = compat.ollama_messages({"messages": [{"role": "assistant", "content": "said", "thinking": "I thought"}]})
    assert messages == [{"role": "assistant", "content": "said", "reasoning_content": "I thought"}]


@pytest.mark.parametrize("options, seed", [({"seed": -1}, None), ({"seed": 0}, 0), ({"seed": 42}, 42), ({}, None)])
def test_ollama_seed_minus_one_means_no_seed(options, seed):
    assert compat.ollama_chat({"messages": [{"role": "user", "content": "hi"}], "options": options}).seed == seed


def test_ollama_seed_minus_one_is_answered(plain):
    answer = plain.client.post("/api/generate", json={"model": "scripted", "prompt": "hi", "stream": False, "options": {"seed": -1, "temperature": 0.5}})
    assert answer.status_code == 200 and answer.json()["response"] == "Hello there"


def test_api_show_says_a_model_can_think_where_its_template_can(plain):
    template = QWEN3 + "<think></think>"
    plain.use(template, "Hello", template)
    capabilities = plain.client.post("/api/show", json={"model": "scripted"}).json()["capabilities"]
    assert "thinking" in capabilities


def test_a_model_row_reads_its_quantization_from_the_file():
    from isharded.gguf import quantization_name

    assert quantization_name({"general.file_type": 15}) == "Q4_K_M" and quantization_name({"general.file_type": 7}) == "Q8_0"
    assert quantization_name({}) is None


def test_blob_uploads_are_refused_in_ollamas_shape(plain):
    answer = plain.client.post("/api/blobs/sha256:abc")
    assert answer.status_code == 400 and "isharded load" in answer.json()["error"]


# --- models, for Claude and OpenAI clients alike


def test_v1_models_takes_x_api_key(keyed):
    good = keyed.client.get("/v1/models", headers={"x-api-key": keyed.key})
    assert good.status_code == 200 and good.json()["data"][0]["id"] == "scripted"
    assert keyed.client.get("/v1/models", headers={"x-api-key": "isk-wrong"}).status_code == 401
    assert keyed.client.get("/v1/models").status_code == 401


def test_one_model_takes_x_api_key_and_follows_the_same_rules(keyed):
    good = keyed.client.get("/v1/models/scripted", headers={"x-api-key": keyed.key})
    assert good.status_code == 200 and good.json()["id"] == "scripted"
    refused = keyed.client.get("/v1/models/scripted", headers={"x-api-key": "isk-wrong"})
    assert refused.status_code == 401 and refused.json()["error"]["type"] == "authentication_error"
    assert keyed.client.get("/v1/models/scripted").status_code == 401


def test_the_model_list_has_both_published_shapes(plain):
    answer = plain.client.get("/v1/models").json()
    row = answer["data"][0]
    assert answer["object"] == "list" and (row["object"], row["owned_by"]) == ("model", "isharded")
    assert (answer["first_id"], answer["last_id"], answer["has_more"]) == ("scripted", "scripted", False)
    assert (row["type"], row["display_name"], row["lifecycle"]) == ("model", "scripted", "active")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", row["created_at"])
    assert all(row[name] is None for name in ("capabilities", "deprecated_at", "line", "max_input_tokens", "max_tokens", "retires_at"))


def test_a_display_name_leaves_folders_out():
    assert compat.model_row(r"C:\models\Qwen3-8B.gguf", {}, 0)["display_name"] == "Qwen3-8B.gguf"
    assert compat.model_row("org/Qwen3-8B", {}, 0)["display_name"] == "Qwen3-8B"


def test_an_empty_list_of_models_has_no_ids():
    assert compat.model_listing({}, 0) == {"object": "list", "data": [], "first_id": None, "has_more": False, "last_id": None}


def test_a_model_that_is_not_here_is_a_404_in_claudes_shape(plain):
    answer = plain.client.get("/v1/models/nope")
    assert answer.status_code == 404 and answer.json()["error"]["type"] == "not_found_error"


@pytest.mark.parametrize("status, kind", [(409, "conflict_error"), (504, "timeout_error"), (529, "overloaded_error"), (503, "api_error"), (422, "invalid_request_error")])
def test_errors_use_the_published_names(status, kind):
    assert compat.anthropic_error(status, "no")["error"]["type"] == kind


# --- what the guide says


def test_every_route_here_is_in_the_guide():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "GUIDE.md").read_text(encoding="utf-8")
    assert not [path for _, path, _, _ in compat.ROUTES if f"`{path}`" not in guide]


# --- real client libraries


@pytest.fixture
def served(mesh):
    """The API on a real socket, for clients that bring their own HTTP library."""
    server = uvicorn.Server(uvicorn.Config(orchestrator.app, lifespan="off", log_level="warning"))
    listener = socket.create_server(("127.0.0.1", 0))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    server.should_exit = True
    thread.join(10)
    listener.close()


def test_the_real_anthropic_client(mesh, served):
    anthropic = pytest.importorskip("anthropic")
    mesh.use(QWEN3, "Hello there")
    client = anthropic.Anthropic(base_url=served, api_key="unused", max_retries=0)
    reply = client.messages.create(model="scripted", max_tokens=64, messages=SAY["messages"])
    assert reply.content[0].text == "Hello there" and reply.stop_reason == "end_turn" and reply.usage.output_tokens > 0
    assert client.messages.count_tokens(model="scripted", messages=SAY["messages"]).input_tokens > 0
    mesh.use(QWEN3, "Hello there")
    with client.messages.stream(model="scripted", max_tokens=64, messages=SAY["messages"]) as stream:
        assert "".join(stream.text_stream) == "Hello there"
        assert stream.get_final_message().usage.input_tokens > 0
    mesh.use(QWEN3, "Let me look." + CALL)
    with client.messages.stream(model="scripted", max_tokens=64, messages=SAY["messages"], tools=[CLAUDE_TOOL]) as stream:
        final = stream.get_final_message()
    assert [block.type for block in final.content] == ["text", "tool_use"]
    assert final.content[1].input == {"city": "Paris"} and final.stop_reason == "tool_use"
    with pytest.raises(anthropic.NotFoundError) as refused:
        client.messages.create(model="nope", max_tokens=64, messages=SAY["messages"])
    assert refused.value.body["error"]["type"] == "not_found_error"


def test_the_real_anthropic_client_reads_how_a_reply_ended_and_what_it_reasoned(mesh, served):
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(base_url=served, api_key="unused", max_retries=0)
    mesh.use(QWEN3, "Hello END there")
    reply = client.messages.create(model="scripted", max_tokens=64, messages=SAY["messages"], stop_sequences=["END"])
    assert (reply.stop_reason, reply.stop_sequence, reply.content[0].text) == ("stop_sequence", "END", "Hello ")
    mesh.use(QWEN3, THOUGHT)
    reply = client.messages.create(model="scripted", max_tokens=64, messages=SAY["messages"])
    assert [block.type for block in reply.content] == ["thinking", "text"] and reply.content[0].thinking == "Let me see." and reply.content[0].signature
    mesh.use(QWEN3, THOUGHT)
    with client.messages.stream(model="scripted", max_tokens=64, messages=SAY["messages"]) as stream:
        final = stream.get_final_message()
    assert final.content[0].thinking == "Let me see." and final.content[0].signature and final.content[1].text == "Hello there"
    mesh.use(QWEN3, "Sure")
    history = [*SAY["messages"], {"role": "assistant", "content": [block.model_dump() for block in reply.content]}, {"role": "user", "content": "more"}]
    assert client.messages.create(model="scripted", max_tokens=64, messages=history).content[0].text == "Sure"


def test_the_real_anthropic_client_lists_models(keyed, served):
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(base_url=served, api_key=keyed.key, max_retries=0)
    assert [model.id for model in client.models.list()] == ["scripted"]
    assert client.models.retrieve("scripted").display_name == "scripted"


def test_the_real_ollama_client_reads_reasoning_and_takes_seed_minus_one(mesh, served):
    ollama_library = pytest.importorskip("ollama")
    mesh.use(QWEN3, THOUGHT)
    reply = ollama_library.Client(host=served).chat(model="scripted", messages=SAY["messages"], options={"seed": -1})
    assert reply.message.thinking == "Let me see." and reply.message.content == "Hello there"


def test_the_real_anthropic_client_sends_its_key_as_x_api_key(keyed, served):
    anthropic = pytest.importorskip("anthropic")
    assert anthropic.Anthropic(base_url=served, api_key=keyed.key, max_retries=0).messages.create(model="scripted", max_tokens=64, messages=SAY["messages"]).content
    with pytest.raises(anthropic.AuthenticationError):
        anthropic.Anthropic(base_url=served, api_key="isk-wrong", max_retries=0).messages.create(model="scripted", max_tokens=64, messages=SAY["messages"])


def test_the_real_openai_client(mesh):
    openai = pytest.importorskip("openai", minversion="1.66")  # the first with Responses
    mesh.use(QWEN3, "Hello there")
    client = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=mesh.client, max_retries=0)
    assert client.responses.create(model="scripted", input="hi").output_text == "Hello there"
    mesh.use(QWEN3, "Hello there")
    events = list(client.responses.create(model="scripted", input="hi", stream=True))
    assert events[-1].type == "response.completed" and events[-1].response.output_text == "Hello there"
    assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == "Hello there"
    mesh.use(QWEN3, CALL)
    with client.responses.stream(model="scripted", input="hi", tools=[RESPONSES_TOOL]) as stream:
        final = stream.get_final_response()
    assert final.output[0].type == "function_call" and json.loads(final.output[0].arguments) == {"city": "Paris"}
    with pytest.raises(openai.BadRequestError, match="keeps no replies"):
        client.responses.create(model="scripted", input="hi", previous_response_id="resp_1")


def test_the_real_ollama_client(mesh):
    ollama_library = pytest.importorskip("ollama")
    mesh.use(QWEN3, "Hello there")
    client = ollama_library.Client(host="http://testserver")
    client._client = mesh.client
    assert client.chat(model="scripted", messages=SAY["messages"]).message.content == "Hello there"
    parts = list(client.chat(model="scripted", messages=SAY["messages"], stream=True))
    assert "".join(part.message.content for part in parts) == "Hello there" and parts[-1].done and parts[-1].eval_count > 0
    assert client.generate(model="scripted", prompt="hi").response == "Hello there"
    mesh.use(QWEN3, CALL)
    reply = client.chat(model="scripted", messages=SAY["messages"], tools=[WEATHER])
    assert reply.message.tool_calls[0].function.arguments == {"city": "Paris"}
    mesh.node.active = active_model()
    assert [row.model for row in client.list().models] == ["scripted"]
    assert client.show("scripted").capabilities == ["completion", "tools"]
    assert client.ps().models[0].context_length == 8192
    with pytest.raises(ollama_library.ResponseError, match="isharded load"):
        client.pull("llama3")


def test_the_real_ollama_client_embeds(serving):
    ollama_library = pytest.importorskip("ollama")
    client = ollama_library.Client(host="http://testserver")
    client._client = serving[0]
    assert len(client.embed(model="embed.gguf", input="hello").embeddings) == 1


# --- What made a reply


@pytest.fixture
def fingerprinted(plain):
    plain.node.active = SimpleNamespace(exact=False, request=None, manifest=SimpleNamespace(revision="r1"), embedding=False, projector=None, adapter=None)
    return plain


def test_every_api_says_what_made_a_reply(fingerprinted):
    """OpenAI's replies have a field for it, and every API's replies carry the same in a header."""
    node = fingerprinted.node
    wanted = node.fingerprint()["id"]
    assert re.fullmatch(r"fp_[0-9a-f]{12}", wanted)
    chat = fingerprinted.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert chat.json()["system_fingerprint"] == wanted and chat.headers["x-isharded-fingerprint"] == wanted
    streamed = fingerprinted.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    chunks = [json.loads(line[6:]) for line in streamed.text.split("\n") if line.startswith("data: {")]
    assert chunks and all(chunk["system_fingerprint"] == wanted for chunk in chunks)
    assert streamed.headers["x-isharded-fingerprint"] == wanted
    for stream in (False, True):
        completion = fingerprinted.client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4, "stream": stream})
        assert completion.headers["x-isharded-fingerprint"] == wanted
        if not stream:
            assert completion.json()["system_fingerprint"] == wanted
        answers = [
            claude(fingerprinted, stream=stream),
            fingerprinted.client.post("/v1/responses", json={"model": "scripted", "input": "hi", "stream": stream}),
            fingerprinted.client.post("/api/chat", json={"model": "scripted", "messages": [{"role": "user", "content": "hi"}], "stream": stream}),
            fingerprinted.client.post("/api/generate", json={"model": "scripted", "prompt": "hi", "stream": stream}),
        ]
        for answer in answers:
            assert answer.status_code == 200, answer.text
            assert answer.headers["x-isharded-fingerprint"] == wanted, answer.request.url


def test_a_new_split_or_precision_gives_a_new_fingerprint_and_status_can_still_read_the_old(fingerprinted):
    node = fingerprinted.node
    first = node.fingerprint()
    assert first["split"] == "0-1" and first["precision"] == "unknown" and first["exact"] is False
    assert "llama.cpp" in first["build"]
    node.route = [orchestrator.StageAssignment(worker_id="w", url="http://worker", start_layer=0, end_layer=1), orchestrator.StageAssignment(worker_id="v", url="http://other", start_layer=1, end_layer=2)]
    node.workers = {"w": SimpleNamespace(health={"dtype": "bfloat16"}), "v": SimpleNamespace(health={"dtype": "float32"})}
    second = node.fingerprint()
    assert second["split"] == "0-1/1-2" and second["precision"] == "bf16/fp32" and second["id"] != first["id"]
    assert node.fingerprint()["id"] == second["id"], "the same layout gives the same fingerprint"
    assert [item["id"] for item in node.fingerprints.values()] == [first["id"], second["id"]]
    assert "w" not in json.dumps(second["split"]), "a fingerprint names layers, not nodes"


def test_no_route_means_no_fingerprint(plain):
    assert plain.node.fingerprint() is None
    answer = plain.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert answer.json()["system_fingerprint"] is None and "x-isharded-fingerprint" not in answer.headers
