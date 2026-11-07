"""Providers: speaking to the mesh and to OpenAI-compatible servers, over real HTTP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import time

import httpx
import pytest

from isharded.assistant import conversation, providers
from isharded.assistant.app import Assistant
from isharded.assistant.cli import add_arguments, run
from isharded.assistant.contracts import Capability, Effect, ModelReply, ModelRequest, Reach, ToolCall
from isharded.assistant.providers import (
    MeshProvider,
    OpenAICompatible,
    OpenValue,
    ProviderError,
    ProviderRefused,
    VisibleText,
    api_root,
    estimate_tokens,
    key_reference,
    loose_json,
    parse_tool_calls,
    read_key,
    strip_thinking,
)
from test_assistant_fakes import CountingServer, FakeModelServer, Reply

READ_FILE = Capability("files.read", "Read a file.", frozenset({Effect.READ}), input_schema={"type": "object", "properties": {"path": {"type": "string"}}})
SEND_MAIL = Capability("mail.send", "Send mail.", frozenset({Effect.EXTERNAL}))
OFFERED = (READ_FILE, SEND_MAIL)


def request(text: str = "Hello", tools: tuple[Capability, ...] = (), role: str = "conversation", **options) -> ModelRequest:
    return ModelRequest(role=role, messages=[{"role": "system", "content": "Be kind."}, {"role": "user", "content": text}], tools=tools, **options)


@pytest.fixture
def assistant(tmp_path):
    with Assistant(tmp_path) as instance:
        yield instance


def run_cli(home: Path, *arguments: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return run(parser.parse_args(["--home", str(home), *arguments]))


def closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


# Parsing what models write


@pytest.mark.parametrize(
    "text",
    [
        '```tool\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```',
        '```json\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```',
        '```\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```',
        '``` tool_call\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```',
        '{"tool": "files.read", "arguments": {"path": "a.txt"}}',
        '<tool_call>{"name": "files.read", "arguments": {"path": "a.txt"}}</tool_call>',
        '```tool\n{"name": "files.read", "parameters": {"path": "a.txt"}}\n```',
        '```tool\n{"capability": "files.read", "args": {"path": "a.txt"}}\n```',
        '```tool\n{"tool": "files.read", "input": {"path": "a.txt"}}\n```',
        '```tool\n{"tool": "files.read", "arguments": "{\\"path\\": \\"a.txt\\"}"}\n```',
        '```tool\n{"tool": "files__read", "arguments": {"path": "a.txt"}}\n```',
        '```tool\n{"tool": "files.read", "arguments": {"path": "a.txt",},}\n```',
        "```tool\n{'tool': 'files.read', 'arguments': {'path': 'a.txt'}}\n```",
        '```tool\n{“tool”: “files.read”, “arguments”: {“path”: “a.txt”}}\n```',
        '```tool\n{"type": "function", "function": {"name": "files.read", "arguments": "{\\"path\\": \\"a.txt\\"}"}}\n```',
        '```tool\n[{"tool": "files.read", "arguments": {"path": "a.txt"}}]\n```',
        '```tool\n{"tool_calls": [{"tool": "files.read", "arguments": {"path": "a.txt"}}]}\n```',
        'Sure, here goes:\n```tool\nCalling now:\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\nDone\n```',
        '<think>Let me look.</think>```tool\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```',
        '```tool\n{"tool": "files.read", "arguments": {"path": "a.txt"}}',
    ],
)
def test_a_tool_call_is_understood_however_it_is_written(text):
    _, calls = parse_tool_calls(text, OFFERED)
    assert [(call.capability, call.arguments) for call in calls] == [("files.read", {"path": "a.txt"})]


def test_the_call_is_removed_and_the_words_around_it_stay():
    text, calls = parse_tool_calls('I will check.\n```tool\n{"tool": "files.read", "arguments": {}}\n```\nOne moment.', OFFERED)
    assert len(calls) == 1 and text == "I will check.\n\nOne moment."


def test_several_calls_in_one_reply_are_all_found_in_order():
    text = '```tool\n{"tool": "files.read", "arguments": {"path": "a"}}\n```\n```tool\n{"tool": "mail.send", "arguments": {"to": "b"}}\n```'
    assert [call.capability for call in parse_tool_calls(text, OFFERED)[1]] == ["files.read", "mail.send"]


def test_a_python_style_call_keeps_its_booleans_and_nulls():
    calls = parse_tool_calls("```tool\n{'tool': 'files.read', 'arguments': {'a': true, 'b': false, 'c': null}}\n```", OFFERED)[1]
    assert calls[0].arguments == {"a": True, "b": False, "c": None}


@pytest.mark.parametrize(
    "text",
    [
        "Just an answer.",
        'The format is {"tool": "unknown.thing", "arguments": {}}',
        '```python\nprint({"tool": "files.read", "arguments": {}})\n```',
        'Call it like `{"tool": "files.read", "arguments": {}}` when you need to.',
        'You could use {"tool": "files.read", "arguments": {}} to do that, but it is not needed.',
        "```json\n{\"name\": \"Ada\", \"age\": 36}\n```",
        "```tool\nnot json at all\n```",
        '```tool\n{"tool": 7}\n```',
        '```tool\n{"tool": "files.read", "arguments": 7}\n```',
        '```tool\n{"tool": "files.read", "arguments": "not json"}\n```',
    ],
)
def test_text_that_is_not_a_real_call_stays_text(text):
    visible, calls = parse_tool_calls(text, OFFERED)
    assert calls == () and visible == text


def test_a_call_cut_off_by_the_token_limit_is_never_run():
    """A truncated argument is a different argument: the path below is not the real one."""
    text = '```tool\n{"tool": "mail.send", "arguments": {"to": "ada@exa'
    visible, calls = parse_tool_calls(text, OFFERED)
    assert calls == () and "mail.send" in visible


def test_reasoning_about_a_tool_is_not_calling_it():
    text = '<think>Maybe ```tool\n{"tool": "mail.send", "arguments": {}}\n``` but no.</think>The answer is 4.'
    assert parse_tool_calls(text, OFFERED) == ("The answer is 4.", ())


def test_reasoning_cut_off_by_the_token_limit_is_never_a_call():
    assert parse_tool_calls('Hmm <think>{"tool": "mail.send", "arguments": {}}', OFFERED) == ("Hmm", ())


def test_nothing_is_a_call_when_no_tools_were_offered():
    text = '```tool\n{"tool": "files.read", "arguments": {}}\n```'
    assert parse_tool_calls(text, ()) == (text, ())


def test_a_tool_that_was_not_offered_cannot_be_called():
    assert parse_tool_calls('```tool\n{"tool": "mail.send", "arguments": {}}\n```', (READ_FILE,))[1] == ()


def test_thinking_is_stripped():
    assert strip_thinking("<think>private</think>\n\nThe answer.") == "The answer."
    assert strip_thinking("private</think>The answer.") == "The answer."
    assert strip_thinking("No thinking here.") == "No thinking here."
    assert strip_thinking("<think>never finished") == ""


@pytest.mark.parametrize("text", ["", "   ", "no json", "{unclosed", "[1, 2"])
def test_loose_json_refuses_what_it_cannot_trust(text):
    assert loose_json(text) is None


@pytest.mark.parametrize("text", ['{{"a": 1}}', "{{}}", "[{[1]: 2}]", '{"a", "b"}', "{'a': {1}}", "[b'x']", "[1j]", "{1: 2, 'a': 3}", "[{None: 1, 'a': 2}]"])
def test_python_that_is_not_json_reads_as_not_json(text):
    """It would raise, or reach a tool as something no tool can take, and fail the whole turn."""
    assert loose_json(text) is None


def test_nesting_too_deep_to_read_reads_as_not_json():
    assert loose_json("[" * 50_000 + "]" * 50_000) is None
    assert loose_json('{"a":' * 20_000 + "1" + "}" * 20_000) is None


def test_a_reply_in_python_words_still_parses():
    assert loose_json("{'a': True, 'b': None, 'c': [1, 2]}") == {"a": True, "b": None, "c": [1, 2]}


def test_a_token_is_counted_one_way(monkeypatch):
    assert conversation.CHARS_PER_TOKEN is providers.CHARS_PER_TOKEN
    monkeypatch.setattr(providers, "CHARS_PER_TOKEN", 10)
    assert estimate_tokens("x" * 100) == 11


def test_loose_json_refuses_something_absurdly_long():
    assert loose_json('{"a": "' + "x" * 300_000 + '"}') is None


def test_a_long_reply_of_code_is_scanned_quickly():
    text = "void f() { if (a) { b(); } }\n" * 20_000
    started = time.perf_counter()
    assert parse_tool_calls(text, OFFERED)[1] == ()
    assert time.perf_counter() - started < 1.0


# Streaming what is safe to show


def stream_through(text: str, size: int, offered=OFFERED) -> str:
    visible = VisibleText(offered)
    pieces = [visible.feed(text[start : start + size]) for start in range(0, len(text), size)]
    tail = visible.flush()
    return "".join(pieces) + ("" if parse_tool_calls(text, offered)[1] else tail)  # as the router does


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 16, 1000])
@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("Plain words.", "Plain words."),
        ("<think>secret</think>The answer.", "The answer."),
        ('Checking.```tool\n{"tool": "files.read", "arguments": {}}\n```', "Checking."),
        ('<tool_call>{"name": "files.read", "arguments": {}}</tool_call>Done', "Done"),
        ("Code:\n```python\nprint(1)\n```\nAfter it.", "Code:\n```python\nprint(1)\n```\nAfter it."),
        ('Data:\n```json\n{"name": "Ada"}\n```\nAfter it.', 'Data:\n```json\n{"name": "Ada"}\n```\nAfter it.'),
        ("A < b and c > d, with `code`.", "A < b and c > d, with `code`."),
        ("<think>never ends", ""),
        ("Unfinished ```python\nprint(1)", "Unfinished ```python\nprint(1)"),
        # A call with no fence, which is what a real small model wrote.
        ('{"name": "files.read", "arguments": {"path": "a.txt"}}', ""),
        ('Reading it. {"name": "files.read", "arguments": {}}', "Reading it. "),
        ('{"name": "files.read", "arguments": {}}\n{"name": "files.read", "arguments": {}}', ""),
        ('{"name": "files.read", "arguments": {}} is how you would call it.', '{"name": "files.read", "arguments": {}} is how you would call it.'),
        ('{"name": "something.else", "arguments": {}}', '{"name": "something.else", "arguments": {}}'),
        ('A set {1, 2} and a call {"name": "files.read", "arguments": {}}', "A set {1, 2} and a call "),
        ("A set {1, 2} and more.", "A set {1, 2} and more."),
        ("An open { that never closes.", "An open { that never closes."),
        ('Cut off: {"name": "files.read", "arguments": {"pa', 'Cut off: {"name": "files.read", "arguments": {"pa'),
        ('Shown as `{"name": "files.read", "arguments": {}}`', 'Shown as `{"name": "files.read", "arguments": {}}`'),
    ],
)
def test_a_stream_hides_the_same_things_however_it_is_split(text, shown, size):
    assert stream_through(text, size) == shown


def test_what_is_not_a_call_is_released_as_soon_as_that_is_known():
    visible = VisibleText(OFFERED)
    assert visible.feed("A set {1, 2}") == "A set {1, 2}"
    assert visible.feed(" and more words.") == " and more words."


@pytest.mark.parametrize("text", ['{"name": "files.read", "arguments": {}}', 'I will read it. {"name": "files.read", "arguments": {"path": "a.txt"}}'])
def test_a_stream_and_the_final_reply_agree_on_what_a_person_sees(text):
    assert stream_through(text, 3).strip() == parse_tool_calls(text, OFFERED)[0]


def test_ordinary_code_is_not_held_until_the_end():
    visible = VisibleText(OFFERED)
    assert visible.feed("Here:\n```python\nprint(1)\n") == "Here:\n```python\nprint(1)\n"
    assert visible.feed("```\nThen more words.") == "```\nThen more words."


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
@pytest.mark.parametrize(
    ("text", "shown"),
    [
        # A fence held across feeds, closing where a feed can cut it.
        ('```json\n{"a": "x` y``z"}\n```After', '```json\n{"a": "x` y``z"}\n```After'),
        ('```tool\n{"tool": "files.read"}\n```\n```tool\n{"tool": "files.read"}\n```', "\n"),
        # A string that holds what would close a value, and an escape cut from its quote.
        ('{"name": "files.read", "arguments": {"path": "}{\\"]"}}', ""),
        # A run of calls, then something that shows it never was one.
        ('{"name": "files.read", "arguments": {}}  \n {"name": "files.read", "arguments": {}} ok', '{"name": "files.read", "arguments": {}}  \n {"name": "files.read", "arguments": {}} ok'),
        ('{"a": {"name": "files.read", "arguments": {}}}', '{"a": {"name": "files.read", "arguments": {}}}'),
        # What comes after a value that was let go of.
        ("{1} <think>hidden</think>shown", "{1} shown"),
        ('{"a": [{"b": 1}, {"c": 2}]} <tool_call>{"name": "files.read"}</tool_call>end', '{"a": [{"b": 1}, {"c": 2}]} end'),
    ],
)
def test_held_text_is_settled_the_same_however_it_arrives(text, shown, size):
    assert stream_through(text, size) == shown


def test_a_long_value_is_read_once_however_many_pieces_it_comes_in(monkeypatch):
    read: list[int] = []
    original = OpenValue.read

    def counted(self, text, start):
        before = self.size
        closed = original(self, text, start)
        read.append(self.size - before)
        return closed

    monkeypatch.setattr(OpenValue, "read", counted)
    rows = '{"id": 12, "name": "an entry"}, ' * 400
    visible = VisibleText(OFFERED)
    assert visible.feed('{"items": [') == ""
    for start in range(0, len(rows), 3):
        assert visible.feed(rows[start : start + 3]) == ""
    assert sum(read) == len('{"items": [') + len(rows)
    assert visible.flush() == '{"items": [' + rows


def test_a_stream_is_ready_for_another_reply_after_a_flush():
    visible = VisibleText(OFFERED)
    assert visible.feed('{"name": "files.re') == ""
    assert visible.flush() == '{"name": "files.re'
    assert visible.feed("A set {1, 2}") == "A set {1, 2}"
    assert visible.feed("```json\n{") == ""
    assert visible.flush() == "```json\n{"
    assert visible.feed('{"name": "files.read", "arguments": {}}') == ""


def test_with_no_tools_only_reasoning_is_hidden():
    text = '```tool\n{"tool": "files.read", "arguments": {}}\n```'
    assert stream_through(text, 4, offered=()) == text


# The mesh


def test_the_mesh_receives_only_what_its_real_schema_accepts():
    with FakeModelServer(mesh_shape=True) as server:
        provider = MeshProvider("mesh", server.url, tool_mode="prompt")
        tools_request = ModelRequest(
            role="conversation",
            messages=[
                {"role": "system", "content": "Be kind."},
                {"role": "tool", "content": "tool output"},
                {"role": "user", "content": "Hello"},
            ],
            tools=OFFERED,
            max_tokens=300,
            temperature=0.5,
            stop=("END",),
        )
        reply = provider.complete(tools_request, "fake-model")
    assert server.bad_requests == []
    body = server.requests[0]["body"]
    assert reply.text == "Hello." and reply.provider == "mesh"
    assert body["max_tokens"] == 300 and body["temperature"] == 0.5 and body["stop"] == ["END"]
    assert "tools" not in body and body["enable_thinking"] is False
    assert all(isinstance(message["content"], str) for message in body["messages"])
    assert [message["role"] for message in body["messages"]] == ["system", "user", "user"]


def test_tools_can_be_offered_to_the_mesh_in_the_system_prompt():
    with FakeModelServer(mesh_shape=True) as server:
        MeshProvider("mesh", server.url, tool_mode="prompt").complete(request(tools=OFFERED), "fake-model")
    system = server.requests[0]["body"]["messages"][0]["content"]
    assert system.startswith("Be kind.") and "files.read" in system and "mail.send" in system and "<tool_call>" in system
    assert server.requests[0]["body"]["messages"].count({"role": "system", "content": system}) == 1


def test_a_tool_call_from_the_mesh_is_parsed():
    text = 'Reading.\n```tool\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```'
    with FakeModelServer([Reply(text=text)], mesh_shape=True) as server:
        reply = MeshProvider("mesh", server.url).complete(request(tools=OFFERED), "fake-model")
    assert reply.tool_calls == (ToolCall("files.read", {"path": "a.txt"}),) and reply.text == "Reading."


@pytest.mark.parametrize(("role", "thinking"), [("conversation", False), ("worker", False), ("planning", True), ("coding", True)])
def test_thinking_is_spent_only_where_it_helps(role, thinking):
    with FakeModelServer(mesh_shape=True) as server:
        MeshProvider("mesh", server.url).complete(request(role=role), "fake-model")
    assert server.requests[0]["body"]["enable_thinking"] is thinking


def test_the_mesh_lists_its_loaded_model_and_says_when_none_is():
    with FakeModelServer(models=("Qwen/Qwen3-8B",)) as server:
        assert MeshProvider("mesh", server.url).current_model() == "Qwen/Qwen3-8B"
    with FakeModelServer(models=()) as server:
        assert MeshProvider("mesh", server.url).current_model() is None


def test_a_reply_from_the_mesh_is_always_streamed_so_silence_can_be_told_apart():
    with FakeModelServer() as server:
        MeshProvider("mesh", server.url).complete(request(), "fake-model")
    assert server.requests[0]["body"]["stream"] is True


def test_the_mesh_reports_real_token_counts_from_the_end_of_the_stream():
    with FakeModelServer([Reply(usage={"prompt_tokens": 40, "completion_tokens": 12})]) as server:
        reply = MeshProvider("mesh", server.url).complete(request(), "fake-model")
    assert (reply.prompt_tokens, reply.completion_tokens) == (40, 12)


def test_tokens_are_estimated_when_a_server_does_not_count_them():
    with FakeModelServer([Reply(text="A reply.", usage=None)]) as server:
        reply = MeshProvider("mesh", server.url).complete(request(), "fake-model")
    assert reply.prompt_tokens > 0 and reply.completion_tokens > 0


# Failing silently, and failing partway


def test_a_failure_sent_inside_a_stream_is_seen():
    """After the headers are sent the mesh can only report failure in the stream."""
    with FakeModelServer([Reply(text="Partial answer here", error_after=2)]) as server:
        with pytest.raises(ProviderError, match="a stage was lost"):
            MeshProvider("mesh", server.url).complete(request(), "fake-model")


def test_a_stream_that_is_cut_is_a_failure_and_not_a_short_answer():
    with FakeModelServer([Reply(text="Partial answer here", cut_after=2)]) as server:
        with pytest.raises(ProviderError, match="before finishing"):
            MeshProvider("mesh", server.url).complete(request(), "fake-model")


def test_a_stream_that_ends_without_finishing_is_a_failure():
    with FakeModelServer([Reply(text="Partial", no_finish=True)]) as server:
        with pytest.raises(ProviderError, match="before finishing"):
            MeshProvider("mesh", server.url).complete(request(), "fake-model")


def test_a_link_that_goes_silent_before_any_reply_is_detected():
    with FakeModelServer([Reply(silent=True)]) as server:
        started = time.monotonic()
        with pytest.raises(ProviderError, match="silent"):
            MeshProvider("mesh", server.url, read_seconds=0.4).complete(request(), "fake-model")
        assert time.monotonic() - started < 5


def test_a_link_that_goes_silent_partway_through_a_reply_is_detected():
    with FakeModelServer([Reply(text="Partial answer here", silent_after=2)]) as server:
        started = time.monotonic()
        with pytest.raises(ProviderError, match="silent"):
            MeshProvider("mesh", server.url, read_seconds=0.4).complete(request(), "fake-model")
        assert time.monotonic() - started < 5


def test_a_slow_reply_is_not_mistaken_for_silence():
    """Silence is judged per read, so a long reply that keeps moving is fine."""
    with FakeModelServer([Reply(text="x" * 200)]) as server:
        reply = MeshProvider("mesh", server.url, read_seconds=0.5).complete(request(), "fake-model")
    assert len(reply.text) == 200


def test_a_refused_connection_says_who_cannot_be_reached():
    with pytest.raises(ProviderError, match="mesh"):
        MeshProvider("mesh", closed_port_url()).complete(request(), "fake-model")
    with pytest.raises(ProviderError, match="other cannot be reached"):
        OpenAICompatible("other", closed_port_url()).complete(request(), "fake-model")


def tls_server(folder: Path) -> FakeModelServer:
    import ssl

    from test_assistant_gateway import self_signed

    server = FakeModelServer()
    certificate, key = self_signed(folder)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.server.socket = context.wrap_socket(server.server.socket, server_side=True)
    return server


def test_a_certificate_that_cannot_be_verified_says_what_to_look_at(tmp_path):
    """On this owner's machine antivirus replaces certificates, and "ConnectError" says nothing."""
    with tls_server(tmp_path) as server:
        url = server.url.replace("http://", "https://")
        for provider in (OpenAICompatible("other", url), MeshProvider("mesh", url)):
            with pytest.raises(ProviderError, match="antivirus"):
                provider.complete(request(), "m")


def test_a_pool_with_its_own_authority_is_verified_against_it(tmp_path, monkeypatch):
    from test_assistant_gateway import self_signed

    certificate, _ = self_signed(tmp_path)
    monkeypatch.setenv("ISHARDED_CA_FILE", str(certificate))
    monkeypatch.delenv("ISHARDED_TLS_CERT", raising=False)
    with httpx_client_of(MeshProvider("mesh", "https://127.0.0.1:1")) as client:
        names = [item["subject"] for item in client._transport._pool._ssl_context.get_ca_certs()]
    assert any("assistant-test" in str(name) for name in names)


def httpx_client_of(provider):
    return provider._client()


def test_calls_to_a_provider_share_a_connection_and_no_cookies():
    with CountingServer() as server:
        provider = OpenAICompatible("other", server.url)
        for _ in range(4):
            assert provider._post("chat/completions", {}, 5.0, "answering").status_code == 200
        provider.models()
        assert server.connections == 1
        assert provider.health()["up"]
        provider.close()
        assert server.connections == 2, "a health check has to show a new connection works"
        assert server.cookies == [None] * 6


def test_each_call_keeps_its_own_timeout_on_the_shared_client():
    with CountingServer(slow_seconds=0.6) as server:
        provider = OpenAICompatible("other", server.url)

        def patient() -> int:
            return provider._post("slow", {}, 10.0, "answering").status_code

        with ThreadPoolExecutor(2) as pool:
            waiting = pool.submit(patient)
            with pytest.raises(ProviderError, match="went silent while answering"):
                provider._post("slow", {}, 0.1, "answering")
            assert waiting.result() == 200
        assert provider._post("fast", {}, 0.5, "answering").status_code == 200
        provider.close()


def test_many_threads_can_use_a_provider_at_once():
    with CountingServer() as server:
        provider = OpenAICompatible("other", server.url)
        with ThreadPoolExecutor(8) as pool:
            statuses = list(pool.map(lambda _: provider._post("chat/completions", {}, 5.0, "answering").status_code, range(40)))
        assert statuses == [200] * 40
        assert 1 <= server.connections <= 8
        provider.close()


def test_a_connection_the_server_hung_up_is_not_held_against_the_next_call():
    with CountingServer(hang_up=True) as server:
        provider = OpenAICompatible("other", server.url)
        for _ in range(3):
            assert provider._post("chat/completions", {}, 5.0, "answering").status_code == 200
            time.sleep(0.05)
        provider.close()
        assert server.connections == 3


def test_a_client_lets_go_of_an_idle_connection_before_a_server_does():
    # Uvicorn drops one after five seconds, and a call sent that moment would fail.
    client = OpenAICompatible("other", "http://127.0.0.1:1")._client()
    assert client._transport._pool._keepalive_expiry < 5
    client.close()


def test_no_call_waits_for_a_connection_another_call_holds():
    client = OpenAICompatible("other", "http://127.0.0.1:1")._client()
    assert client._transport._pool._max_connections > 10_000
    client.close()


def test_a_provider_works_again_after_being_closed():
    with CountingServer() as server:
        provider = OpenAICompatible("other", server.url)
        provider._post("chat/completions", {}, 5.0, "answering")
        provider.close()
        provider.close()
        assert provider._post("chat/completions", {}, 5.0, "answering").status_code == 200
        provider.close()


def test_shutting_down_closes_every_client_and_removing_one_leaves_it_running(tmp_path):
    with CountingServer() as server, Assistant(tmp_path / "home") as assistant:
        assistant.providers.add("kept", server.url)
        assistant.providers.add("removed", server.url)
        kept, removed = assistant.providers.get("kept")._client(), assistant.providers.get("removed")._client()
        assistant.providers.remove("removed")
        assert not removed.is_closed  # a reply may still be coming through it
        assistant.close()
        assert kept.is_closed and removed.is_closed


def test_an_http_failure_says_what_the_server_said():
    with FakeModelServer([Reply(status=409, error_body='{"detail": "no model is ready; load one first"}')]) as server:
        with pytest.raises(ProviderError, match="409.*load one first"):
            MeshProvider("mesh", server.url).complete(request(), "fake-model")


def test_a_reply_that_is_not_json_is_a_failure_with_a_plain_message():
    with FakeModelServer([Reply(status=200, text="fine")]) as server:
        provider = OpenAICompatible("other", server.url)
        original = FakeModelServer.send_json
        FakeModelServer.send_json = staticmethod(lambda handler, status, payload: original(handler, status, ["not", "a", "reply"]))
        try:
            with pytest.raises(ProviderError, match="could not be read"):
                provider.complete(request(), "fake-model")
        finally:
            FakeModelServer.send_json = staticmethod(original)


def test_a_stalled_model_list_is_bounded(monkeypatch):
    monkeypatch.setattr("isharded.assistant.providers.HEALTH_SECONDS", 0.3)
    with FakeModelServer() as server:
        server.model_list_silent = True
        started = time.monotonic()
        health = MeshProvider("mesh", server.url).health()
        assert health["up"] is False and time.monotonic() - started < 4


def test_health_reports_what_is_there():
    with FakeModelServer(models=("a", "b")) as server:
        health = OpenAICompatible("other", server.url).health()
    assert health["up"] is True and health["models"] == ["a", "b"] and health["error"] is None
    down = OpenAICompatible("other", closed_port_url()).health()
    assert down["up"] is False and "cannot be reached" in down["error"]


# Streaming


def test_a_stream_arrives_in_pieces_then_the_finished_reply():
    with FakeModelServer([Reply(text="A longer answer than one piece.")]) as server:
        items = list(MeshProvider("mesh", server.url).stream(request(), "fake-model"))
    assert isinstance(items[-1], ModelReply) and len(items) > 3
    assert "".join(items[:-1]) == "A longer answer than one piece." == items[-1].text


# OpenAI-compatible servers and native tools


def native_call(name: str = "files__read", arguments: str = '{"path": "a.txt"}') -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_native_tools_are_sent_with_names_the_api_allows():
    with FakeModelServer([Reply(text="", tool_calls=[native_call()])]) as server:
        reply = OpenAICompatible("other", server.url, tool_mode="native").complete(request(tools=OFFERED), "m")
    tools = server.requests[0]["body"]["tools"]
    assert [tool["function"]["name"] for tool in tools] == ["files__read", "mail__send"]
    assert tools[0]["function"]["parameters"]["properties"]["path"] == {"type": "string"}
    assert reply.tool_calls == (ToolCall("files.read", {"path": "a.txt"}, "call_1"),)
    assert "<tool_call>" not in server.requests[0]["body"]["messages"][0]["content"]


def test_a_native_call_to_something_not_offered_is_dropped():
    with FakeModelServer([Reply(text="Sorry.", tool_calls=[native_call("mail__send")])]) as server:
        reply = OpenAICompatible("other", server.url, tool_mode="native").complete(request(tools=(READ_FILE,)), "m")
    assert reply.tool_calls == ()


def test_a_server_without_native_tools_gets_them_in_the_prompt_and_is_remembered():
    with FakeModelServer([Reply(status=400), Reply(text="ok"), Reply(text="ok")]) as server:
        provider = OpenAICompatible("other", server.url)
        assert provider.complete(request(tools=OFFERED), "m").text == "ok"
        provider.complete(request(tools=OFFERED), "m")
    bodies = [item["body"] for item in server.requests]
    assert "tools" in bodies[0] and "tools" not in bodies[1] and "tools" not in bodies[2]
    assert "files.read" in bodies[1]["messages"][0]["content"]


def test_a_failed_fallback_does_not_change_how_a_server_is_treated():
    with FakeModelServer([Reply(status=400), Reply(status=500)]) as server:
        provider = OpenAICompatible("other", server.url)
        with pytest.raises(ProviderError):
            provider.complete(request(tools=OFFERED), "m")
    assert provider.tool_mode == "auto"


def test_prompt_mode_never_sends_native_tools():
    with FakeModelServer() as server:
        OpenAICompatible("other", server.url, tool_mode="prompt").complete(request(tools=OFFERED), "m")
    assert "tools" not in server.requests[0]["body"]


def test_a_fenced_call_is_still_understood_from_a_server_with_native_tools():
    text = '```tool\n{"tool": "files.read", "arguments": {"path": "a.txt"}}\n```'
    with FakeModelServer([Reply(text=text)]) as server:
        reply = OpenAICompatible("other", server.url).complete(request(tools=OFFERED), "m")
    assert reply.tool_calls[0].capability == "files.read"


def test_streaming_with_native_tools_returns_the_finished_call():
    with FakeModelServer([Reply(text="", tool_calls=[native_call()])]) as server:
        items = list(OpenAICompatible("other", server.url, tool_mode="native").stream(request(tools=OFFERED), "m"))
    assert items[-1].tool_calls[0].capability == "files.read"


def test_embeddings_come_back_in_order():
    with FakeModelServer() as server:
        vectors = OpenAICompatible("other", server.url).embed("m", ["a", "bbb"])
    assert vectors == [[1.0, 1.0], [3.0, 1.0]]


# Keys


def test_a_key_in_the_environment_is_sent_and_read_when_used(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_KEY", "first-secret")
    with FakeModelServer() as server:
        provider = OpenAICompatible("other", server.url, "TEST_MODEL_KEY")
        provider.complete(request(), "m")
        monkeypatch.setenv("TEST_MODEL_KEY", "rotated-secret")
        provider.complete(request(), "m")
    assert [item["headers"]["Authorization"] for item in server.requests] == ["Bearer first-secret", "Bearer rotated-secret"]


def test_a_key_in_a_file_is_sent(tmp_path):
    (tmp_path / "key.txt").write_text("file-secret\n", encoding="utf-8")
    with FakeModelServer() as server:
        OpenAICompatible("other", server.url, str(tmp_path / "key.txt")).complete(request(), "m")
    assert server.requests[0]["headers"]["Authorization"] == "Bearer file-secret"


def test_no_key_sends_no_header():
    with FakeModelServer() as server:
        OpenAICompatible("other", server.url).complete(request(), "m")
    assert "Authorization" not in server.requests[0]["headers"]


def test_a_missing_key_says_what_to_set(monkeypatch):
    monkeypatch.delenv("TEST_MODEL_KEY", raising=False)
    with pytest.raises(ProviderError, match="TEST_MODEL_KEY is not set"):
        read_key("env:TEST_MODEL_KEY")
    with pytest.raises(ProviderError, match="cannot be read"):
        read_key("file:/nowhere/at/all")


@pytest.mark.parametrize("pasted", ["sk-abc123def456", "abcdef0123456789abcdef", "ghp_AbCd1234EfGh5678", "env:sk-abc123", "env:lowercasekey123"])
def test_a_pasted_key_is_refused_with_what_to_do_instead(pasted, monkeypatch):
    monkeypatch.delenv("lowercasekey123", raising=False)
    with pytest.raises(ProviderRefused, match="looks like a key"):
        key_reference(pasted)


def test_a_lowercase_variable_that_really_exists_is_accepted(monkeypatch):
    monkeypatch.setenv("my_key", "secret")
    assert key_reference("env:my_key") == "env:my_key"


def test_the_database_never_holds_a_key(tmp_path, monkeypatch):
    secret = "ultra-secret-key-value"
    monkeypatch.setenv("TEST_MODEL_KEY", secret)
    key_file = tmp_path / "key.txt"
    key_file.write_text("another-ultra-secret", encoding="utf-8")
    with FakeModelServer() as server, Assistant(tmp_path / "home") as assistant:
        assistant.providers.add("one", server.url, key="env:TEST_MODEL_KEY")
        assistant.providers.add("two", server.url, key=f"file:{key_file}")
        assistant.providers.get("one").complete(request(), "m")
        assistant.providers.get("two").complete(request(), "m")
        path = assistant.store.path
        stored = b"".join(file.read_bytes() for file in path.parent.glob(path.name + "*"))
        assert b"TEST_MODEL_KEY" in stored
        assert secret.encode() not in stored and b"another-ultra-secret" not in stored
        assert server.requests[0]["headers"]["Authorization"] == f"Bearer {secret}"


def test_a_server_echoing_the_key_does_not_put_it_in_an_error(monkeypatch):
    secret = "ultra-secret-key-value"
    monkeypatch.setenv("TEST_MODEL_KEY", secret)
    with FakeModelServer([Reply(status=401, error_body=f"bad key: {secret}")]) as server:
        with pytest.raises(ProviderError) as caught:
            OpenAICompatible("other", server.url, "TEST_MODEL_KEY").complete(request(), "m")
    assert secret not in str(caught.value) and "[key]" in str(caught.value)


@pytest.mark.parametrize("url", ["https://user:pass@example.com/v1", "https://example.com/v1?api_key=abc", "ftp://example.com", "example.com", ""])
def test_an_address_holding_a_key_or_no_scheme_is_refused(url):
    with pytest.raises(ProviderRefused):
        api_root(url)


@pytest.mark.parametrize(
    ("url", "root"),
    [
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000/v1"),
        ("http://127.0.0.1:9000/", "http://127.0.0.1:9000/v1"),
        ("http://127.0.0.1:9000/v1", "http://127.0.0.1:9000/v1"),
        ("https://api.example.com/openai/v1/", "https://api.example.com/openai/v1"),
    ],
)
def test_an_address_is_normalised(url, root):
    assert api_root(url) == root


# The registry


def test_reach_defaults_to_the_worst_case_except_for_ones_own_mesh(assistant):
    assert assistant.providers.add("cloud", "https://api.example.com/v1").reach is Reach.CLOUD
    assert assistant.providers.add("pool", "http://100.64.0.1:8000", mesh=True).reach is Reach.POOL


def test_reach_machine_is_only_for_this_machine(assistant):
    assert assistant.providers.add("local", "http://127.0.0.1:9000", reach="machine").reach is Reach.MACHINE
    assert assistant.providers.add("local6", "http://[::1]:9000", reach="machine").reach is Reach.MACHINE
    with pytest.raises(ProviderRefused, match="not on this machine"):
        assistant.providers.add("liar", "https://api.example.com/v1", reach="machine")


def test_providers_survive_a_restart(tmp_path):
    with Assistant(tmp_path) as first:
        first.providers.add("pool", "http://127.0.0.1:9000", mesh=True, daily_tokens=500, read_seconds=30.0, model="qwen")
    with Assistant(tmp_path) as second:
        provider = second.providers.get("pool")
        assert isinstance(provider, MeshProvider)
        assert (provider.reach, provider.daily_tokens, provider.read_seconds, provider.current_model()) == (Reach.POOL, 500, 30.0, "qwen")


def test_a_registered_provider_is_listed_for_as_long_as_it_runs(assistant):
    class Peer:
        name, reach, kind = "peer-a", Reach.PEER, "peer"

    assistant.providers.register(Peer())
    assert assistant.providers.get("peer-a").reach is Reach.PEER
    assert [provider.name for provider in assistant.providers.all()] == ["peer-a"]


def test_names_and_duplicates_and_removal_say_what_is_wrong(assistant):
    with pytest.raises(ProviderRefused, match="usable name"):
        assistant.providers.add("has/slash", "http://127.0.0.1:9000")
    assistant.providers.add("one", "http://127.0.0.1:9000")
    with pytest.raises(ProviderRefused, match="already exists"):
        assistant.providers.add("one", "http://127.0.0.1:9000")
    assistant.providers.remove("one")
    assert assistant.providers.get("one") is None
    with pytest.raises(ProviderRefused, match="provider list"):
        assistant.providers.remove("one")


def test_adding_and_removing_is_audited(assistant):
    assistant.providers.add("one", "http://127.0.0.1:9000")
    assistant.providers.remove("one")
    assert [row["action"] for row in assistant.audit.read()] == ["provider.add", "provider.remove"]


# Commands


def test_commands_add_list_remove(tmp_path, capsys):
    with FakeModelServer(models=("the-model",)) as server:
        assert run_cli(tmp_path, "provider", "add", "pool", server.url, "--mesh") == 0
        assert run_cli(tmp_path, "provider", "list") == 0
    shown = capsys.readouterr().out
    assert "reach pool" in shown and "the-model" in shown and "yes" in shown
    assert run_cli(tmp_path, "provider", "remove", "pool") == 0
    capsys.readouterr()
    assert run_cli(tmp_path, "provider", "list") == 0
    assert "No providers yet" in capsys.readouterr().out


def test_an_explicit_reach_is_respected_even_for_a_mesh(tmp_path, capsys):
    assert run_cli(tmp_path, "provider", "add", "pool", "http://127.0.0.1:9000", "--mesh", "--reach", "cloud") == 0
    assert "reach cloud" in capsys.readouterr().out


def test_a_mesh_reads_tool_calls_from_the_stream_and_falls_back_when_it_has_none(monkeypatch) -> None:
    import httpx

    from isharded.assistant import providers
    from isharded.assistant.contracts import Capability, ModelRequest  # noqa: F401

    pieces = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "files__read", "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.txt"}'}}]}, "finish_reason": "tool_calls"}]},
    ]
    stream = "".join(f"data: {__import__('json').dumps(piece)}\n\n" for piece in pieces) + "data: [DONE]\n\n"
    seen = []

    def serve(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        seen.append("tools" in body)
        if "tools" in body and refuse["now"]:
            return httpx.Response(400, json={"detail": "no tool calling here"})
        return httpx.Response(200, text=stream if "tools" in body else 'data: {"choices":[{"delta":{"content":"plain"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

    refuse = {"now": False}
    mesh = providers.MeshProvider("mesh", "http://mesh/v1", None)
    monkeypatch.setattr(mesh, "_client", lambda: httpx.Client(transport=httpx.MockTransport(serve), timeout=5))
    tool = type("Tool", (), {"name": "files.read", "summary": "Read a file", "input_schema": {"type": "object"}})()
    request = type("Request", (), {"tools": (tool,), "messages": [{"role": "user", "content": "go"}], "max_tokens": 8, "temperature": 0, "stop": (), "role": "conversation"})()

    reply = mesh.complete(request, "m")
    assert [(call.capability, call.arguments) for call in reply.tool_calls] == [("files.read", {"path": "a.txt"})]

    refuse["now"] = True
    reply = mesh.complete(request, "m")  # this model has no tool calling, so the prompt takes over
    assert reply.text == "plain" and mesh.tool_mode == "prompt"
# How a reply ended


@pytest.mark.parametrize("kind", [OpenAICompatible, MeshProvider])
@pytest.mark.parametrize("said, read", [("length", "length"), ("stop", "stop"), ("tool_calls", "stop"), ("content_filter", "stop")])
def test_a_reply_says_whether_the_limit_ended_it(kind, said, read):
    with FakeModelServer([Reply(finish=said)]) as server:
        assert kind("p", server.url).complete(request(), "m").finish_reason == read


def test_a_stream_says_so_too():
    with FakeModelServer([Reply(finish="length")]) as server:
        assert list(OpenAICompatible("p", server.url).stream(request(), "m"))[-1].finish_reason == "length"


