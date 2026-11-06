"""Other clients' shapes: requests in, replies out, and honest refusals."""
from __future__ import annotations

import asyncio
import json

import pytest

from shardmesh import formats


async def collect(stream):
    return [item async for item in stream]


def run(stream):
    return asyncio.run(collect(stream))


def sse(*chunks):
    async def stream():
        for chunk in chunks:
            yield chunk

    return stream()


def chat_stream(*words, finish="stop"):
    events = [f"data: {json.dumps({'model': 'm', 'choices': [{'delta': {'content': word}, 'finish_reason': None}]})}\n\n" for word in words]
    last = {"model": "m", "choices": [{"delta": {}, "finish_reason": finish}], "usage": {"prompt_tokens": 7, "completion_tokens": len(words)}}
    return [*events, f"data: {json.dumps(last)}\n\n", "data: [DONE]\n\n"]


def replies(*words, finish="stop"):
    return formats.pieces(sse(*chat_stream(*words, finish=finish)))


def test_anthropic_system_and_blocks_become_chat_messages() -> None:
    chat = formats.from_anthropic({"model": "m", "max_tokens": 50, "system": [{"type": "text", "text": "Be brief."}], "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": " there"}]}], "stop_sequences": ["END"], "thinking": {"type": "enabled"}})
    assert [(item.role, item.content) for item in chat.messages] == [("system", "Be brief."), ("user", "Hi there")]
    assert chat.max_tokens == 50 and chat.stop == ["END"] and chat.enable_thinking is True and chat.stream


def test_responses_takes_plain_text_or_items() -> None:
    assert formats.from_responses({"input": "Hello"}).messages[0].content == "Hello"
    chat = formats.from_responses({"instructions": "Be kind.", "input": [{"role": "developer", "content": "Rules"}, {"role": "user", "content": [{"type": "input_text", "text": "Go"}]}], "max_output_tokens": 9})
    assert [(item.role, item.content) for item in chat.messages] == [("system", "Be kind."), ("system", "Rules"), ("user", "Go")]
    assert chat.max_tokens == 9


def test_ollama_options_map_and_no_limit_is_the_most_we_allow() -> None:
    chat = formats.from_ollama_chat({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "options": {"temperature": 0.1, "num_predict": -1, "seed": 4, "stop": ["x"]}, "think": False})
    assert chat.temperature == 0.1 and chat.max_tokens == formats.MOST_TOKENS and chat.seed == 4 and chat.stop == ["x"] and chat.enable_thinking is False
    made = formats.from_ollama_generate({"prompt": "Hi", "system": "Sys"})
    assert [item.role for item in made.messages] == ["system", "user"]


@pytest.mark.parametrize(
    "translate,body",
    [
        (formats.from_anthropic, {"messages": [{"role": "user", "content": "x"}], "tools": [{"name": "t"}]}),
        (formats.from_anthropic, {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}),
        (formats.from_responses, {"input": "x", "tools": [{"type": "function"}]}),
        (formats.from_responses, {"input": "x", "previous_response_id": "resp_1"}),
        (formats.from_responses, {"input": [{"type": "function_call_output", "output": "x"}]}),
        (formats.from_ollama_chat, {"messages": [{"role": "user", "content": "x", "images": ["abc"]}]}),
        (formats.from_ollama_chat, {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}),
        (formats.from_ollama_generate, {"prompt": "x", "images": ["abc"]}),
    ],
)
def test_what_a_reply_could_only_ignore_is_refused(translate, body) -> None:
    with pytest.raises(formats.Unsupported):
        translate(body)


def test_pieces_survive_chunks_split_anywhere() -> None:
    whole = "".join(chat_stream("Hel", "lo"))
    split = [whole[index : index + 7].encode() for index in range(0, len(whole), 7)]
    items = run(formats.pieces(sse(*split)))
    assert [item["text"] for item in items if "text" in item] == ["Hel", "lo"]
    assert items[-1]["finish"] == "stop" and items[-1]["prompt_tokens"] == 7


def test_a_failure_in_the_stream_is_raised_with_its_code() -> None:
    bad = sse('data: {"error": {"message": "stage lost", "code": 503}}\n\n')
    with pytest.raises(formats.StreamFailed, match="stage lost") as caught:
        run(formats.pieces(bad))
    assert caught.value.code == 503


def events(lines):
    found = []
    for block in "".join(lines).strip().split("\n\n"):
        name, data = block.split("\n", 1)
        found.append((name[7:], json.loads(data[6:])))
    return found


def test_anthropic_stream_follows_the_published_order() -> None:
    sent = events(run(formats.anthropic_stream(replies("Hi", " you", finish="length"), "m")))
    assert [name for name, _ in sent] == ["message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert sent[2][1]["delta"] == {"type": "text_delta", "text": "Hi"}
    assert sent[5][1]["delta"]["stop_reason"] == "max_tokens" and sent[5][1]["usage"]["output_tokens"] == 2


def test_anthropic_whole_reply() -> None:
    message = formats.anthropic_message(asyncio.run(formats.gather(replies("Hi", " you"))), "m")
    assert message["content"] == [{"type": "text", "text": "Hi you"}] and message["stop_reason"] == "end_turn"
    assert message["usage"] == {"input_tokens": 7, "output_tokens": 2}


def test_responses_stream_ends_with_the_whole_reply() -> None:
    sent = events(run(formats.responses_stream(replies("Hi", " you"), "m")))
    names = [name for name, _ in sent]
    assert names[0] == "response.created" and names[-1] == "response.completed"
    assert names.count("response.output_text.delta") == 2
    assert [data["sequence_number"] for _, data in sent] == list(range(1, len(sent) + 1))
    final = sent[-1][1]["response"]
    assert final["status"] == "completed" and final["output"][0]["content"][0]["text"] == "Hi you"
    assert final["id"] == sent[0][1]["response"]["id"]


def test_a_cut_off_response_says_so() -> None:
    reply = asyncio.run(formats.gather(replies("Hi", finish="length")))
    answer = formats.responses_object(reply, "m")
    assert answer["status"] == "incomplete" and answer["incomplete_details"] == {"reason": "max_output_tokens"}


def test_ollama_streams_lines_and_finishes_with_counts() -> None:
    lines = [json.loads(line) for line in run(formats.ollama_stream(replies("Hi", " you"), "m", True, 0.0))]
    assert [line["message"]["content"] for line in lines] == ["Hi", " you", ""]
    assert lines[-1]["done"] is True and lines[-1]["eval_count"] == 2 and lines[-1]["prompt_eval_count"] == 7
    assert "response" in json.loads(run(formats.ollama_stream(replies("x"), "m", False, 0.0))[0])


def test_a_failure_mid_stream_reaches_each_client_in_its_own_form() -> None:
    def broken():
        return formats.pieces(sse('data: {"error": {"message": "lost", "code": 503}}\n\n'))

    assert events(run(formats.anthropic_stream(broken(), "m")))[-1][0] == "error"
    assert events(run(formats.responses_stream(broken(), "m")))[-1][0] == "response.failed"
    assert json.loads(run(formats.ollama_stream(broken(), "m", True, 0.0))[-1]) == {"error": "lost"}
