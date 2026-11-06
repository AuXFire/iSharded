"""Tool calling on /v1/chat/completions, from a chat template to OpenAI's response shape.

Replies are scripted under the real route, scanner and token loop, so each test
sees what a client would. Tests that need a real model read SHARDMESH_TEST_LLAMA_DIR
and SHARDMESH_TEST_GGUF, as tests/test_gguf.py does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel

from shardmesh import orchestrator, tokenizer as tokenizer_module
from shardmesh.llamacpp import step_random
from shardmesh.protocol import ChatCompletionRequest, ChatMessage, StageAssignment
from shardmesh.tokenizer import ChatTokenizer, IncrementalText, tokenizer_from_gguf
from shardmesh.toolcalls import (
    DEFAULT_REPLY_TOKENS,
    MARKER_TOKENS,
    TOOL_REPLY_TOKENS,
    ReplyScanner,
    ToolRequestError,
    forced_opening,
    new_call_id,
    opens_in_thought,
    plan_tools,
    template_messages,
    tool_format,
    typed_value,
    without_reasoning,
)

STOP = 0
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}, "tags": {"type": "array"}},
            "required": ["city"],
        },
    },
}
OTHER = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
# Short forms of the real templates, keeping what each one does with tools.
QWEN3 = """{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>
For each call return <tool_call>{"name": ..., "arguments": ...}</tool_call>{% endif %}
{% for m in messages %}{% if m.role == 'tool' %}<tool_response>{{ m.content }}</tool_response>{% else %}<{{ m.role }}>{{ m.content }}{% if m.tool_calls %}{% for call in m.tool_calls %}<tool_call>{"name": "{{ call.function.name }}", "arguments": {% if call.function.arguments is string %}{{ call.function.arguments }}{% else %}{{ call.function.arguments | tojson }}{% endif %}}</tool_call>{% endfor %}{% endif %}{% endif %}{% endfor %}<assistant>"""
QWEN35 = """{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools> <tool_call><function=x><parameter=y>{% endif %}
{% for m in messages %}{% if m.role == 'tool' %}<tool_response>{{ m.content }}</tool_response>{% else %}<{{ m.role }}>{{ m.content }}{% if m.tool_calls %}{% for call in m.tool_calls %}{% set call = call.function %}<tool_call><function={{ call.name }}>{% for key, value in call.arguments|items %}<parameter={{ key }}>{{ value }}</parameter>{% endfor %}</function></tool_call>{% endfor %}{% endif %}{% endif %}{% endfor %}<assistant>{% if enable_thinking %}<think>
{% else %}<think>

</think>

{% endif %}"""
LLAMA3 = """{% if tools %}Respond in the format {"name": function name, "parameters": dictionary}. {% for t in tools %}{{ t | tojson(indent=4) }}{% endfor %}{% endif %}
{% for m in messages %}{% if 'tool_calls' in m %}{% if not m.tool_calls|length == 1 %}{{ raise_exception('This model only supports single tool-calls at once!') }}{% endif %}{% set tool_call = m.tool_calls[0].function %}{"name": "{{ tool_call.name }}", "parameters": {{ tool_call.arguments | tojson }}}{% else %}<{{ m.role }}>{{ m.content }}{% endif %}{% endfor %}<assistant>"""
MISTRAL = """{% if tools %}[AVAILABLE_TOOLS]{{ tools | tojson }}[/AVAILABLE_TOOLS]{% endif %}{% for m in messages %}{% if m.tool_calls is defined and m.tool_calls is not none %}[TOOL_CALLS] [{% for call in m.tool_calls %}{% if call.id|length != 9 %}{{ raise_exception('Tool call IDs should be alphanumeric strings with length 9!') }}{% endif %}{{ call.function | tojson }}{% endfor %}]{% else %}[INST]{{ m.content }}[/INST]{% endif %}{% endfor %}"""
HERMES_USE = """{% for tool in tools %}{{ tool | tojson }}{% endfor %}<tool_call>{% for m in messages %}{{ m.role }}:{{ m.content }}{% if m.role == 'tool' %}({{ m.name }}){% endif %}|{% endfor %}<assistant>"""
PLAIN = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>"


class Scripted(ChatTokenizer):
    """A real chat template, over a reply written out in advance."""

    def __init__(self, template: str | None, tool_template: str | None = None) -> None:
        super().__init__(None, template, {STOP}, {}, tool_template)  # type: ignore[arg-type]
        self.pieces = ["<stop>"]

    def script(self, reply: str, size: int = 3) -> list[int]:
        start = len(self.pieces)
        self.pieces += [reply[index : index + size] for index in range(0, len(reply), size)]
        return [*range(start, len(self.pieces)), STOP]

    def encode(self, text: str) -> list[int]:
        return [1] * max(len(text) // 4, 1)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[token] for token in ids)

    def keeping(self, markers):
        return self


@pytest.fixture
def mesh(monkeypatch):
    """A node with one route, whose model says whatever a test scripts."""
    node = orchestrator.Orchestrator()
    node.model_id = "scripted"
    node.route = [StageAssignment(worker_id="w", url="http://worker", start_layer=0, end_layer=1)]
    state = SimpleNamespace(node=node, client=TestClient(orchestrator.app), seen=[], tokens=[])

    def use(template: str | None, reply: str = "", tool_template: str | None = None, size: int = 3) -> None:
        node.tokenizer = Scripted(template, tool_template)
        state.tokens = node.tokenizer.script(reply, size)

    async def token_stream(session, max_tokens, *rest):
        state.seen.append({"prompt_tokens": len(session.tokens), "max_tokens": max_tokens})
        for token in state.tokens[:max_tokens]:
            yield token

    real_render = ChatTokenizer.render_chat

    def render(self, messages, *args, **options):
        rendered = real_render(self, messages, *args, **options)
        if options.get("add_generation_prompt", True):
            state.prompt = rendered
        return rendered

    node.token_stream = token_stream
    monkeypatch.setattr(ChatTokenizer, "render_chat", render)
    monkeypatch.setattr(orchestrator, "runtime", node)
    state.use = use
    return state


def ask(mesh, **body):
    body.setdefault("messages", [{"role": "user", "content": "weather in Paris?"}])
    return mesh.client.post("/v1/chat/completions", json=body)


def stream_events(mesh, **body):
    text = ask(mesh, stream=True, **body).text
    lines = [line[6:] for line in text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def scan(kind: str, text: str, step: int | None = None, tools=(WEATHER,), choice=None, parallel=None, thinking=False, opening="", truncated=False, template=None):
    templates = {"hermes": QWEN3, "xml": QWEN35, "llama3": LLAMA3, "mistral": MISTRAL, "mistral-new": MISTRAL + "[ARGS]"}
    plan = plan_tools(list(tools), choice, parallel, template or templates[kind])
    scanner = ReplyScanner(plan, thinking, opening)
    items = []
    size = step or max(len(text), 1)
    for start in range(0, len(text), size):
        items += scanner.feed(text[start : start + size])
    items += scanner.finish(truncated)
    words = "".join(item["text"] for item in items if "text" in item)
    calls = [(item["tool_call"]["function"]["name"], json.loads(item["tool_call"]["function"]["arguments"])) for item in items if "tool_call" in item]
    return words, calls


# --- which format a template teaches


@pytest.mark.parametrize(
    "template, expected",
    [
        (QWEN3, "hermes"),
        (QWEN35, "xml"),
        (LLAMA3 + "tool_call", "llama3"),
        (MISTRAL, "mistral"),
        (HERMES_USE, "hermes"),
        (PLAIN, None),
        (None, None),
        ("tools <tool_call>{{ name }}<arg_key>k</arg_key>", None),
        ("tools, in a format of its own", None),
    ],
)
def test_the_template_picks_the_format(template, expected):
    assert tool_format(template) == expected


def test_a_model_named_like_one_family_is_read_by_its_template():
    assert plan_tools([WEATHER], None, None, QWEN35).format == "xml"
    assert plan_tools([WEATHER], None, None, QWEN3).format == "hermes"


def test_a_template_with_no_tools_is_refused_rather_than_prompted_around():
    with pytest.raises(ToolRequestError, match="no tool format"):
        plan_tools([WEATHER], None, None, PLAIN)


# --- what a template is given


def message(**fields):
    return ChatMessage(**fields)


def test_arguments_reach_templates_as_objects():
    call = {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    rendered = template_messages([message(role="user", content="hi"), message(role="assistant", content=None, tool_calls=[call])])
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert rendered[1]["content"] == ""


def test_fields_that_do_not_apply_are_absent_rather_than_null():
    # Llama's template asks ``'tool_calls' in message``, which null would answer yes to.
    assert template_messages([message(role="user", content="hi")]) == [{"role": "user", "content": "hi"}]


def test_text_parts_become_a_string():
    parts = [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}]
    assert template_messages([message(role="user", content=parts)])[0]["content"] == "one two"


def test_a_tool_result_gets_its_function_name_from_the_call():
    call = {"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}
    rendered = template_messages([message(role="assistant", tool_calls=[call]), message(role="tool", tool_call_id="c1", content="ok")])
    assert rendered[1] == {"role": "tool", "content": "ok", "tool_call_id": "c1", "name": "get_weather"}


@pytest.mark.parametrize(
    "fields, words",
    [
        ({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}, "only text"),
        ({"role": "user"}, "needs content"),
        ({"role": "tool", "content": "x"}, "tool_call_id"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{nope"}}]}, "not valid JSON"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}, "JSON object"),
        ({"role": "assistant", "tool_calls": [{"function": {}}]}, "with a name"),
    ],
)
def test_a_message_that_cannot_be_used_says_why(fields, words):
    with pytest.raises(ToolRequestError, match=words):
        template_messages([message(**fields)])


@pytest.mark.parametrize("template", [QWEN3, QWEN35, LLAMA3, MISTRAL])
def test_a_whole_tool_conversation_renders_through_each_template(template):
    call = {"id": new_call_id("mistral"), "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    rendered = template_messages(
        [message(role="user", content="hi"), message(role="assistant", content=None, tool_calls=[call]), message(role="tool", tool_call_id=call["id"], content="17C")]
    )
    prompt = Scripted(template).render_chat(rendered, tools=[WEATHER])
    assert "Paris" in prompt and "get_weather" in prompt
    assert '\\"city\\"' not in prompt  # encoded twice is what a string would give


def test_tools_use_the_template_kept_for_them():
    chat = Scripted("plain:{{ messages[0].content }}", HERMES_USE)
    messages = [{"role": "user", "content": "hi"}]
    assert chat.render_chat(messages) == "plain:hi"
    assert chat.render_chat(messages, tools=[WEATHER]).startswith('{"type": "function"')


def test_named_templates_are_read_from_the_model_config(tmp_path):
    from tokenizers.models import WordLevel as Word

    from shardmesh.tokenizer import load_tokenizer

    Tokenizer(Word({"a": 0}, unk_token="a")).save(str(tmp_path / "tokenizer.json"))
    named = [{"name": "default", "template": "chat"}, {"name": "tool_use", "template": "tools"}]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": named}), encoding="utf-8")
    loaded = load_tokenizer(str(tmp_path))
    assert (loaded.chat_template, loaded.tool_template) == ("chat", "tools")


def test_a_gguf_keeps_markers_and_hides_control_tokens(monkeypatch):
    tokens = ["a", "b", "<think>", "</think>", "<tool_call>", "<|im_end|>"]
    kinds = [1, 1, 4, 4, 4, 3]
    values = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2", "tokenizer.chat_template": "x", "tokenizer.chat_template.tool_use": "y", "tokenizer.ggml.eos_token_id": 5}
    header = SimpleNamespace(get=values.get, keys={"tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": kinds, "tokenizer.ggml.merges": []})
    monkeypatch.setattr(tokenizer_module, "read_strings", lambda value: value)
    monkeypatch.setattr(tokenizer_module, "read_numbers", lambda value: value)
    chat = tokenizer_from_gguf(header)
    ids = chat.encode("<think>a</think><tool_call>b<|im_end|>")
    assert ids == [2, 0, 3, 4, 1, 5]
    assert chat.decode(ids) == "<think>a</think><tool_call>b"
    assert chat.tool_template == "y"


def test_special_tokens_a_call_is_written_with_survive_decoding():
    word = Tokenizer(WordLevel({"hi": 0, "[TOOL_CALLS]": 1, "there": 2}, unk_token="hi"))
    word.add_special_tokens([AddedToken("[TOOL_CALLS]", special=True, normalized=False)])
    chat = ChatTokenizer(word, None, set(), {})
    assert "[TOOL_CALLS]" not in chat.decode([0, 1, 2])
    reader = chat.keeping(MARKER_TOKENS)
    assert "[TOOL_CALLS]" in reader.decode([0, 1, 2])
    text = IncrementalText(reader)
    for token in (0, 1, 2):
        settled = text.add(token)
    assert "[TOOL_CALLS]" in settled


# --- reading replies

GOOD = [
    ("hermes", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', {"city": "Paris"}),
    ("hermes", '<tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}</tool_call>', {"city": "Paris"}),
    ("hermes", '<tool_call>\n{"name": "get_weather", "parameters": {"city": "Paris"}}\n</tool_call>', {"city": "Paris"}),
    ("xml", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", {"city": "Paris"}),
    ("xml", "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>", {"city": "Paris"}),
    ("llama3", '{"name": "get_weather", "parameters": {"city": "Paris"}}', {"city": "Paris"}),
    ("llama3", '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}', {"city": "Paris"}),
    ("mistral", '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]', {"city": "Paris"}),
    ("mistral-new", '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}', {"city": "Paris"}),
    ("mistral-new", '[TOOL_CALLS]get_weather[CALL_ID]abc123def[ARGS]{"city": "Paris"}', {"city": "Paris"}),
]


@pytest.mark.parametrize("kind, text, arguments", GOOD)
def test_every_format_gives_the_same_call_however_the_text_arrives(kind, text, arguments):
    for step in [None, *range(1, 14)]:
        assert scan(kind, text, step) == ("", [("get_weather", arguments)]), step


@pytest.mark.parametrize("kind, text", [(kind, text) for kind, text, _ in GOOD])
def test_no_markup_reaches_a_client_before_a_call(kind, text):
    plan = plan_tools([WEATHER], None, None, {"hermes": QWEN3, "xml": QWEN35, "llama3": LLAMA3, "mistral": MISTRAL, "mistral-new": MISTRAL + "[ARGS]"}[kind])
    scanner = ReplyScanner(plan)
    shown = [item for start in range(len(text)) for item in scanner.feed(text[start])]
    assert not [item for item in shown if "text" in item]


@pytest.mark.parametrize(
    "kind, text",
    [
        ("hermes", '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}}</tool_call>\n<tool_call>{"name": "get_weather", "arguments": {"city": "B"}}</tool_call>'),
        ("hermes", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "A"}}\n{"name": "get_weather", "arguments": {"city": "B"}}\n</tool_call>'),
        ("xml", "<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>"),
        ("xml", "<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>\n</tool_call>\n<function=get_weather>\n<parameter=city>\nB\n</parameter>\n</function>"),
        ("llama3", '{"name": "get_weather", "parameters": {"city": "A"}}; {"name": "get_weather", "parameters": {"city": "B"}}'),
        ("mistral", '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "A"}}, {"name": "get_weather", "arguments": {"city": "B"}}]'),
        ("mistral-new", '[TOOL_CALLS]get_weather[ARGS]{"city": "A"}[TOOL_CALLS]get_weather[ARGS]{"city": "B"}'),
    ],
)
def test_parallel_calls_come_back_in_order_or_one_if_asked(kind, text):
    for step in (None, 1, 5):
        assert [arguments["city"] for _, arguments in scan(kind, text, step)[1]] == ["A", "B"]
        assert [arguments["city"] for _, arguments in scan(kind, text, step, parallel=False)[1]] == ["A"]


def test_text_before_a_call_stays_as_text():
    words, calls = scan("hermes", 'Let me check.\n\n<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>', 4)
    assert words == "Let me check."
    assert calls == [("get_weather", {"city": "Rome"})]


def test_plain_answers_pass_through_unchanged():
    for kind in ("hermes", "xml", "llama3", "mistral"):
        for step in (None, 1, 3):
            assert scan(kind, "It is sunny.\n\nEnjoy it. 3 < 4 and {x}", step) == ("It is sunny.\n\nEnjoy it. 3 < 4 and {x}", [])


def test_a_call_inside_reasoning_is_only_a_thought():
    text = '<think>I could write <tool_call>{"name": "get_weather", "arguments": {"city": "X"}}</tool_call> but no.</think>\n\nIt is sunny.'
    for step in (None, 1, 7):
        assert scan("hermes", text, step) == (text, [])


def test_a_real_call_after_reasoning_counts():
    reply = '<think>\nNeed weather.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    for step in (None, 1, 7):
        assert scan("hermes", reply, step) == ("<think>\nNeed weather.\n</think>", [("get_weather", {"city": "Paris"})])


def test_a_reply_that_starts_inside_reasoning_has_no_opening_tag_to_find():
    reply = 'mention <tool_call>{"name": "get_weather", "arguments": {}}</tool_call> here\n</think>\n\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>'
    words, calls = scan("xml", reply, 5, thinking=True)
    assert calls == [("get_weather", {"city": "Oslo"})]
    assert words.endswith("here\n</think>")


def test_reasoning_that_never_ends_holds_no_calls():
    text = '<think>maybe <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    assert scan("hermes", text, 3) == (text, [])


def test_which_prompts_open_in_reasoning():
    assert opens_in_thought("<|im_start|>assistant\n<think>\n")
    assert not opens_in_thought("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not opens_in_thought("<|im_start|>assistant\n")


def test_without_reasoning():
    assert without_reasoning("<think>a</think>\n\nanswer") == "answer"
    assert without_reasoning("answer") == "answer"


@pytest.mark.parametrize(
    "kind, text",
    [
        ("hermes", '<tool_call>{"name": "get_weather", "arguments": {broken}}</tool_call>'),
        ("hermes", '<tool_call>{"name": "not_offered", "arguments": {}}</tool_call>'),
        ("hermes", '<tool_call>{"arguments": {"city": "X"}}</tool_call>'),
        ("hermes", '<tool_call>{"name": "get_weather", "arguments": 3}</tool_call>'),
        ("hermes", "I would use a <tool_call> tag for that."),
        ("xml", "<tool_call>\n<function=not_offered>\n</function>\n</tool_call>"),
        ("xml", "Here is <tool_call> as a word."),
        ("llama3", '{"name": "not_offered", "parameters": {}}'),
        ("llama3", '{"answer": 42}'),
        ("llama3", '{"name": "Bob", "age": 3}'),
        ("mistral", "[TOOL_CALLS] nonsense !"),
        ("mistral", '[TOOL_CALLS] [{"name": "not_offered", "arguments": {}}]'),
    ],
)
def test_what_cannot_be_read_as_a_call_comes_back_as_written(kind, text):
    for step in (None, 1, 6):
        assert scan(kind, text, step) == (text, []), step


def test_a_call_cut_off_by_the_limit_is_not_guessed_at():
    text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar"
    assert scan("xml", text, truncated=True) == (text, [])
    text = '<tool_call>{"name": "get_weather", "arguments": {"ci'
    assert scan("hermes", text, truncated=True) == (text, [])


def test_a_model_that_stops_without_the_closing_tag_has_still_made_its_call():
    assert scan("hermes", '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}}')[1] == [("get_weather", {"city": "A"})]
    assert scan("xml", "<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>")[1] == [("get_weather", {"city": "A"})]


@pytest.mark.parametrize(
    "text, schema, expected",
    [
        ("3", {"type": "integer"}, 3),
        ("3.5", {"type": "number"}, 3.5),
        ("3", {"type": "number"}, 3),
        ("True", {"type": "boolean"}, True),
        ("false", {"type": "boolean"}, False),
        ('["a"]', {"type": "array"}, ["a"]),
        ('{"a": 1}', {"type": "object"}, {"a": 1}),
        ("null", {"type": ["string", "null"]}, None),
        ("7", {"anyOf": [{"type": "integer"}, {"type": "null"}]}, 7),
        ("7", {"type": "string"}, "7"),
        ("7", {}, "7"),
        ("many", {"type": "integer"}, "many"),
        ("maybe", {"type": "boolean"}, "maybe"),
        ("[1", {"type": "array"}, "[1"),
    ],
)
def test_xml_values_become_what_the_schema_says_or_stay_as_written(text, schema, expected):
    assert typed_value(text, schema) == expected


def test_xml_values_keep_their_own_newlines():
    text = "<function=get_weather>\n<parameter=city>\nline one\nline two\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>"
    assert scan("xml", text)[1] == [("get_weather", {"city": "line one\nline two", "days": 3})]


def test_ids_are_what_each_family_will_take_back():
    assert re.fullmatch(r"[0-9a-f]{9}", new_call_id("mistral"))
    assert new_call_id("hermes").startswith("call_")


# --- forcing a call


@pytest.mark.parametrize(
    "template, choice, expected",
    [
        (QWEN3, {"type": "function", "function": {"name": "get_time"}}, '<tool_call>\n{"name": "get_time", "arguments":'),
        (QWEN3, "required", '<tool_call>\n{"name": "'),
        (QWEN35, {"type": "function", "function": {"name": "get_time"}}, "<tool_call>\n<function=get_time>\n"),
        (QWEN35, "required", "<tool_call>\n<function="),
        (LLAMA3 + "tool_call", {"type": "function", "function": {"name": "get_time"}}, '{"name": "get_time", "parameters":'),
        (MISTRAL, {"type": "function", "function": {"name": "get_time"}}, '[TOOL_CALLS] [{"name": "get_time", "arguments":'),
        (MISTRAL + "[ARGS]", {"type": "function", "function": {"name": "get_time"}}, "[TOOL_CALLS]get_time[ARGS]"),
    ],
)
def test_a_forced_call_starts_in_the_prompt(template, choice, expected):
    assert forced_opening(plan_tools([WEATHER, OTHER], choice, None, template)) == expected


def test_required_with_one_tool_names_it():
    assert "get_weather" in forced_opening(plan_tools([WEATHER], "required", None, QWEN3))


def test_auto_forces_nothing():
    assert forced_opening(plan_tools([WEATHER], "auto", None, QWEN3)) == ""


def test_the_model_finishes_a_forced_call():
    plan = plan_tools([WEATHER, OTHER], {"type": "function", "function": {"name": "get_weather"}}, None, QWEN3)
    opening = forced_opening(plan)
    words, calls = scan("hermes", ' {"city": "Paris"}}\n</tool_call>', 3, tools=[WEATHER, OTHER], choice={"type": "function", "function": {"name": "get_weather"}}, opening=opening)
    assert (words, calls) == ("", [("get_weather", {"city": "Paris"})])


def test_a_forced_call_the_model_ruins_is_reported_and_never_invented():
    choice = {"type": "function", "function": {"name": "get_weather"}}
    opening = forced_opening(plan_tools([WEATHER], choice, None, QWEN3))
    words, calls = scan("hermes", " I would rather not.", 3, choice=choice, opening=opening)
    assert calls == []
    assert "required a call to get_weather" in words and "no tool was called" in words
    assert words.endswith("I would rather not.") and "<tool_call>" not in words


def test_a_required_call_the_model_ruins_is_reported():
    opening = forced_opening(plan_tools([WEATHER, OTHER], "required", None, QWEN35))
    words, calls = scan("xml", "nothing useful", tools=[WEATHER, OTHER], choice="required", opening=opening)
    assert calls == [] and "required a tool call" in words


def test_named_choice_will_not_accept_another_tool():
    choice = {"type": "function", "function": {"name": "get_time"}}
    words, calls = scan("hermes", '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>', tools=[WEATHER, OTHER], choice=choice)
    assert calls == [] and "required a call to get_time" in words


# --- requests


@pytest.mark.parametrize(
    "tools, choice, words",
    [
        ([WEATHER], "sometimes", "tool_choice must be"),
        ([WEATHER], {"type": "function"}, "tool_choice must be"),
        ([WEATHER], {"type": "function", "function": {"name": "nope"}}, "not among the tools"),
        (None, "required", "needs tools"),
        ([], {"type": "function", "function": {"name": "x"}}, "needs tools"),
        ([{"type": "retrieval"}], None, "each tool must"),
        ([{"type": "function", "function": {}}], None, "each tool must"),
    ],
)
def test_bad_tool_requests_are_refused_with_a_reason(mesh, tools, choice, words):
    mesh.use(QWEN3)
    answer = ask(mesh, tools=tools, tool_choice=choice)
    assert answer.status_code == 400 and words in answer.json()["detail"]


def test_tools_on_a_model_with_no_tool_format_are_refused(mesh):
    mesh.use(PLAIN)
    answer = ask(mesh, tools=[WEATHER])
    assert answer.status_code == 400 and "no tool format" in answer.json()["detail"]


def test_tool_choice_none_leaves_the_tools_out(mesh):
    mesh.use(PLAIN, "Sunny.")
    answer = ask(mesh, tools=[WEATHER], tool_choice="none")
    assert answer.status_code == 200 and answer.json()["choices"][0]["message"]["content"] == "Sunny."
    assert "get_weather" not in mesh.prompt


def test_tool_choice_none_reads_nothing_as_a_call(mesh):
    text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    mesh.use(QWEN3, text)
    choice = ask(mesh, tools=[WEATHER], tool_choice="none").json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": text} and choice["finish_reason"] == "stop"


def test_a_conversation_the_template_rejects_is_a_bad_request(mesh):
    mesh.use(LLAMA3 + "tool_call")
    call = {"id": "a", "function": {"name": "get_weather", "arguments": "{}"}}
    answer = ask(mesh, tools=[WEATHER], messages=[{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [call, call]}])
    assert answer.status_code == 400 and "single tool-calls" in answer.json()["detail"]


def test_replies_get_room_for_a_call_when_no_limit_is_sent(mesh):
    # OpenAI clients send no limit, and the usual default would cut a call in half.
    mesh.use(QWEN3, "hello")
    ask(mesh, tools=[WEATHER])
    ask(mesh, tools=[WEATHER], max_tokens=50)
    ask(mesh, tools=[WEATHER], max_completion_tokens=60)
    ask(mesh)
    result = [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "a", "content": "ok"}]
    ask(mesh, messages=result)
    assert [item["max_tokens"] for item in mesh.seen] == [TOOL_REPLY_TOKENS, 50, 60, DEFAULT_REPLY_TOKENS, TOOL_REPLY_TOKENS]


def test_a_tool_reply_has_the_openai_shape(mesh):
    mesh.use(QWEN3, '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')
    choice = ask(mesh, tools=[WEATHER]).json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}


def test_tools_go_through_the_template_and_not_into_the_text(mesh):
    mesh.use(QWEN3, "ok")
    ask(mesh, tools=[WEATHER])
    assert '"name": "get_weather"' in mesh.prompt and "<tools>" in mesh.prompt


def test_a_stream_has_the_openai_shape(mesh):
    mesh.use(QWEN3, 'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')
    events = stream_events(mesh, tools=[WEATHER])
    deltas = [event["choices"][0]["delta"] for event in events]
    assert deltas[0] == {"role": "assistant"}
    words = "".join(delta.get("content", "") for delta in deltas)
    assert words == "Checking."
    parts = [part for delta in deltas for part in delta.get("tool_calls", [])]
    assert parts[0]["index"] == 0 and parts[0]["type"] == "function" and parts[0]["id"].startswith("call_")
    assert parts[0]["function"] == {"name": "get_weather", "arguments": ""}
    assert parts[1] == {"index": 0, "function": {"arguments": '{"city": "Paris"}'}}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["completion_tokens"] > 0


class HidesSpecialTokens(Scripted):
    """Like a real tokenizer: decoding drops [TOOL_CALLS] unless told to keep it."""

    def decode(self, ids):
        return super().decode(ids).replace("[TOOL_CALLS]", "")

    def keeping(self, markers):
        assert "[TOOL_CALLS]" in markers
        return SimpleNamespace(decode=lambda ids: Scripted.decode(self, ids))


def special_token_reply(mesh, *pieces: str) -> None:
    """A reply in whole tokens, since a special token is never split."""
    mesh.use(MISTRAL)
    chat = mesh.node.tokenizer
    chat.__class__ = HidesSpecialTokens
    chat.pieces += pieces
    mesh.tokens = [*range(1, len(pieces) + 1), STOP]


def test_a_call_written_with_a_special_token_is_seen_through_the_real_token_loop(mesh):
    special_token_reply(mesh, "[TOOL_CALLS]", ' [{"name": "get_weather", ', '"arguments": {"city": "Paris"}}]')
    choice = ask(mesh, tools=[WEATHER]).json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None


def test_replies_without_tools_decode_as_they_always_did(mesh):
    special_token_reply(mesh, "before ", "[TOOL_CALLS]", "after")
    assert ask(mesh).json()["choices"][0]["message"]["content"] == "before after"


def test_a_stream_numbers_parallel_calls(mesh):
    one = '<tool_call>{"name": "get_weather", "arguments": {"city": "%s"}}</tool_call>'
    mesh.use(QWEN3, one % "A" + "\n" + one % "B")
    parts = [part for event in stream_events(mesh, tools=[WEATHER]) for part in event["choices"][0]["delta"].get("tool_calls", [])]
    assert [part["index"] for part in parts] == [0, 0, 1, 1]
    assert len({part["id"] for part in parts if "id" in part}) == 2


def test_a_stream_of_a_plain_answer_arrives_as_it_is_written(mesh):
    mesh.use(QWEN3, "It is sunny in Paris today.", size=2)
    events = stream_events(mesh, tools=[WEATHER])
    words = [event["choices"][0]["delta"].get("content") for event in events]
    assert len([word for word in words if word]) > 5
    assert "".join(word or "" for word in words) == "It is sunny in Paris today."
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_cut_off_call_streams_as_text_with_finish_length(mesh):
    mesh.use(QWEN3, '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>', size=1)
    events = stream_events(mesh, tools=[WEATHER], max_tokens=20)
    assert events[-1]["choices"][0]["finish_reason"] == "length"
    words = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    assert words.startswith("<tool_call>") and not any("tool_calls" in event["choices"][0]["delta"] for event in events)


def test_required_says_so_in_the_stream_too(mesh):
    mesh.use(QWEN3, " I cannot do that.")
    events = stream_events(mesh, tools=[WEATHER], tool_choice="required")
    words = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    assert "no tool was called" in words and "I cannot do that." in words
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_prompt_that_opens_in_reasoning_makes_the_reply_reasoning_until_it_closes(mesh):
    reply = "maybe <function=get_weather>\n<parameter=city>\nX\n</parameter>\n</function> or not\n</think>\n\nSunny."
    mesh.use(QWEN35, reply)
    choice = ask(mesh, tools=[WEATHER], enable_thinking=True).json()["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"] == {"role": "assistant", "content": reply}


def test_a_forced_call_is_written_after_the_template_prompt(mesh):
    mesh.use(QWEN35, "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
    tokens_before = []

    async def spy(prompt, *args, **options):
        tokens_before.append(prompt)
        yield {"text": "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"}
        yield {"finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1, "time_to_first_token_ms": 1, "tokens_per_second": 1}

    mesh.node.stream_tokens = spy
    named = {"type": "function", "function": {"name": "get_weather"}}
    choice = ask(mesh, tools=[WEATHER], tool_choice=named, enable_thinking=True).json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert tokens_before[0].endswith("<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n")
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}


def test_the_official_client_reads_both_kinds_of_reply(mesh):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=mesh.client)
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    mesh.use(QWEN3, text)
    reply = client.chat.completions.create(model="scripted", messages=[{"role": "user", "content": "x"}], tools=[WEATHER])
    assert reply.choices[0].finish_reason == "tool_calls"
    call = reply.choices[0].message.tool_calls[0]
    assert (call.function.name, json.loads(call.function.arguments)) == ("get_weather", {"city": "Paris"})
    mesh.use(QWEN3, text)
    chunks = list(client.chat.completions.create(model="scripted", messages=[{"role": "user", "content": "x"}], tools=[WEATHER], stream=True))
    parts = [part for chunk in chunks for part in chunk.choices[0].delta.tool_calls or []]
    assert "".join(part.function.arguments or "" for part in parts) == '{"city": "Paris"}'
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_the_full_loop_through_each_template(mesh):
    for template, reply in ((QWEN3, GOOD[0][1]), (QWEN35, GOOD[3][1]), (LLAMA3 + "tool_call", GOOD[5][1]), (MISTRAL, GOOD[7][1])):
        mesh.use(template, reply)
        message = ask(mesh, tools=[WEATHER]).json()["choices"][0]["message"]
        call = message["tool_calls"][0]
        mesh.use(template, "It is 17C.")
        result = {"role": "tool", "tool_call_id": call["id"], "content": "17C"}
        conversation = [{"role": "user", "content": "weather?"}, message, result]
        answer = ask(mesh, tools=[WEATHER], messages=conversation)
        assert answer.status_code == 200, answer.text
        assert answer.json()["choices"][0]["message"]["content"] == "It is 17C."


# --- sampling


def test_nearby_seeds_do_not_share_their_draws():
    # Adding seed to position made this 39 values, and one rare draw struck every
    # reply with a nearby seed, which looked like a model that could not call tools.
    draws = {step_random(seed, position).random() for seed in range(20) for position in range(20)}
    assert len(draws) == 400


def test_a_replay_repeats_its_draws():
    assert step_random(5, 9).random() == step_random(5, 9).random()


# --- on the normal path


def test_a_request_passed_to_a_lane_keeps_everything_tools_need():
    schema = {"type": "object", "properties": {"unit": {"type": ["string", "null"], "default": None, "enum": ["c", None]}}}
    tool = {"type": "function", "function": {"name": "f", "parameters": schema}}
    call = {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    body = ChatCompletionRequest(
        messages=[{"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "ok"}]}],
        tools=[tool],
        tool_choice="required",
        parallel_tool_calls=False,
    )
    passed = body.model_dump(exclude_none=True)
    assert passed["tools"] == [tool], "null is part of a schema and must survive"
    assert "max_tokens" not in passed, "the lane must see that no limit was asked for"
    again = ChatCompletionRequest(**json.loads(json.dumps(passed)))
    assert again == body


def test_tools_reach_the_lane_that_serves_the_model(mesh):
    seen = {}

    async def locate(model):
        return "small" if model == "elsewhere" else None

    async def forward(name, path, payload, caller):
        seen.update(name=name, path=path, payload=payload)
        return {"forwarded": True}

    mesh.use(QWEN3)
    mesh.node.lanes.locate = locate
    mesh.node.lanes.forward = forward
    answer = ask(mesh, model="elsewhere", tools=[WEATHER], tool_choice="auto")
    assert answer.json() == {"forwarded": True}
    assert seen["name"] == "small" and seen["payload"]["tools"] == [WEATHER] and seen["payload"]["tool_choice"] == "auto"


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_survives_the_entry_worker_failing_half_way_through_it(mesh, stream):
    mesh.use(QWEN3, GOOD[0][1], size=2)
    tokens, node = mesh.tokens, mesh.node
    half = len(tokens) // 2
    assert 0 < mesh.node.tokenizer.decode(tokens[:half]).find("get_weather") < len(mesh.node.tokenizer.decode(tokens[:half])), "the failure must land inside the call"

    def entry_worker(request):
        lines = [json.dumps({"token_id": token}) for token in tokens[:half]] + [json.dumps({"error": "a stage died"})]
        return httpx.Response(200, content="\n".join(lines).encode() + b"\n")

    remaining = iter(tokens[half:])

    async def next_token(session, *rest):
        return next(remaining)

    del node.token_stream  # the real loop, recovery and all
    node.client = httpx.AsyncClient(transport=httpx.MockTransport(entry_worker))
    node.next_token = next_token
    if stream:
        events = stream_events(mesh, tools=[WEATHER])
        parts = [part for event in events for part in event["choices"][0]["delta"].get("tool_calls", [])]
        assert "".join(part["function"].get("arguments", "") for part in parts) == '{"city": "Paris"}'
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    else:
        choice = ask(mesh, tools=[WEATHER]).json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls" and choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert node.stream_fallbacks == 1


# --- with a real model

BUILD = os.environ.get("SHARDMESH_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("SHARDMESH_TEST_GGUF", "")
needs_real_model = pytest.mark.skipif(not (MODEL and Path(MODEL).is_file()), reason="needs a real GGUF")


@needs_real_model
def test_a_real_gguf_keeps_tool_markers_and_renders_tools():
    from shardmesh.tokenizer import load_tokenizer

    chat = load_tokenizer(MODEL)
    assert tool_format(chat.tool_template) in {"hermes", "xml"}
    assert chat.decode(chat.encode("<think>a</think><tool_call>b</tool_call>")) == "<think>a</think><tool_call>b</tool_call>"
    call = {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    messages = template_messages(
        [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", tool_calls=[call]), ChatMessage(role="tool", tool_call_id="a", content="17C")]
    )
    prompt = chat.render_chat(messages, tools=[WEATHER])
    assert "get_weather" in prompt and "Paris" in prompt and "<tool_response>" in prompt
