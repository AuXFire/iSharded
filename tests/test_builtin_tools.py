"""The mesh's own tools: what each does, what it refuses, and a chat that uses them through the real route."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from isharded import builtin_tools, orchestrator
from isharded.replies import whole_reply, with_builtin_tools
from test_tools import QWEN3, WEATHER, Scripted, ask, mesh, stream_events  # noqa: F401

CALL = '<tool_call>\n{"name": "calculate", "arguments": {"expression": "17 * 23"}}\n</tool_call>'


def result_of(name, arguments, facts=None):
    return json.loads(builtin_tools.run(name, arguments, facts))


def test_the_calculator_works_out_arithmetic_exactly():
    assert result_of("calculate", {"expression": "17 * 23"})["result"] == 391
    assert result_of("calculate", '{"expression": "(1234.5 * 0.0725) + 19"}')["result"] == pytest.approx(108.50125)
    assert result_of("calculate", {"expression": "2 ^ 10"})["result"] == 1024, "people write powers with a caret"
    assert result_of("calculate", {"expression": "round(sqrt(2), 4) + max(1, 2) + pi // 1"})["result"] == pytest.approx(6.4142)
    assert result_of("calculate", {"expression": "factorial(20)"})["result"] == 2432902008176640000


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('x').read()", "x + 1", "(1).__class__", "[1, 2][0]", "lambda: 1", "1 if 1 else 2", "'a' * 3", "9 ** 9 ** 9", "factorial(100000)", "1 / 0", "", "1 +", "1" * 400],
)  # fmt: skip
def test_the_calculator_runs_nothing_but_arithmetic(expression):
    answer = result_of("calculate", {"expression": expression})
    assert set(answer) == {"error"}, answer


def test_the_time_is_given_in_utc_or_a_named_zone():
    moment = datetime(2026, 11, 10, 22, 5, 9, tzinfo=timezone.utc)
    assert json.loads(builtin_tools.current_time(None, moment)) == {"date": "2026-11-10", "time": "22:05:09", "weekday": "Tuesday", "timezone": "UTC", "utc_offset": "+00:00"}
    now = result_of("current_time", {})
    assert now["timezone"] == "UTC" and len(now["date"]) == 10
    try:
        paris = json.loads(builtin_tools.current_time("Europe/Paris", moment))
    except builtin_tools.ToolRefusal:
        pytest.skip("this machine has no time zone data")
    assert paris["time"] == "23:05:09" and paris["utc_offset"] == "+01:00"
    assert "not a time zone" in result_of("current_time", {"timezone": "Mars/Olympus"})["error"]


def test_a_call_that_cannot_be_answered_says_why_and_never_raises():
    assert "no tool named" in result_of("delete_everything", {})["error"]
    assert "not valid JSON" in result_of("calculate", "{broken")["error"]
    assert "JSON object" in result_of("calculate", "[1]")["error"]
    assert result_of("mesh_facts", {}, lambda: {"model": "m.gguf", "stages": 3}) == {"model": "m.gguf", "stages": 3}
    assert len(builtin_tools.run("mesh_facts", {}, lambda: {"text": "x" * 9000})) == builtin_tools.RESULT_LIMIT


def test_every_built_in_tool_is_named_and_described_for_a_model():
    assert builtin_tools.NAMES == ["current_time", "calculate", "mesh_facts"]
    for item in builtin_tools.DEFINITIONS:
        assert item["type"] == "function" and len(item["function"]["description"]) > 30
        assert item["function"]["parameters"]["type"] == "object"


def scripted(mesh, replies):  # noqa: F811
    """The model writes each of these in turn, one for each time it is asked."""
    mesh.use(QWEN3, replies[0])
    scripts = [mesh.node.tokenizer.script(reply) for reply in replies]
    asked = iter(scripts)

    async def token_stream(session, max_tokens, *rest):
        mesh.seen.append({"prompt_tokens": len(session.tokens), "max_tokens": max_tokens})
        for token in next(asked)[:max_tokens]:
            yield token

    mesh.node.token_stream = token_stream


def test_a_chat_that_asks_gets_the_answer_with_the_tool_already_run(mesh):  # noqa: F811
    scripted(mesh, [CALL, "17 times 23 is 391."])
    body = ask(mesh, builtin_tools=True, messages=[{"role": "user", "content": "What is 17 times 23?"}]).json()
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "17 times 23 is 391."} and choice["finish_reason"] == "stop"
    use = body["isharded"]["tool_uses"][0]
    assert use["name"] == "calculate" and json.loads(use["result"])["result"] == 391
    # The second pass saw the call and what it returned, in the model's own format.
    assert '<tool_response>{"expression": "17 * 23", "result": 391}</tool_response>' in mesh.prompt
    assert len(mesh.seen) == 2 and body["usage"]["completion_tokens"] > 0


def test_a_stream_shows_the_tool_it_used_and_then_the_answer(mesh):  # noqa: F811
    scripted(mesh, ["Let me work that out. " + CALL, "It is 391."])
    events = stream_events(mesh, builtin_tools=True)
    deltas = [event["choices"][0]["delta"] for event in events]
    assert "".join(delta.get("content", "") for delta in deltas).replace("\n", "") == "Let me work that out.It is 391."
    uses = [delta["isharded"]["tool_use"] for delta in deltas if "tool_use" in delta.get("isharded", {})]
    assert [use["name"] for use in uses] == ["calculate"]
    assert not any("tool_calls" in delta for delta in deltas), "the call was the node's to run, not the client's"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_model_that_keeps_calling_is_asked_to_answer_without_tools(mesh):  # noqa: F811
    scripted(mesh, [CALL] * builtin_tools.ROUNDS + ["Here is my answer."])
    body = ask(mesh, builtin_tools=True).json()
    assert body["choices"][0]["message"]["content"] == "Here is my answer."
    assert len(body["isharded"]["tool_uses"]) == builtin_tools.ROUNDS and len(mesh.seen) == builtin_tools.ROUNDS + 1
    assert "<tools>" not in mesh.prompt, "the last pass offers no tools, so it can only answer"


def test_a_chat_that_does_not_ask_is_left_as_it_was(mesh):  # noqa: F811
    mesh.use(QWEN3, "Hello.")
    body = ask(mesh).json()
    assert body["choices"][0]["message"]["content"] == "Hello." and "isharded" not in body
    assert "<tools>" not in mesh.prompt


def test_a_chat_that_asks_and_cannot_have_them_is_told_why(mesh):  # noqa: F811
    mesh.use(QWEN3, "Hello.")
    both = ask(mesh, builtin_tools=True, tools=[WEATHER])
    assert both.status_code == 400 and "no tools of its own" in both.json()["detail"]
    forced = ask(mesh, builtin_tools=True, tool_choice="required")
    assert forced.status_code == 400 and "tool_choice" in forced.json()["detail"]
    mesh.node.builtin_tools = False
    off = ask(mesh, builtin_tools=True)
    assert off.status_code == 400 and "--no-builtin-tools" in off.json()["detail"]
    mesh.node.builtin_tools = True
    mesh.use("{% for m in messages %}{{ m.content }}{% endfor %}", "Hello.")
    plain = ask(mesh, builtin_tools=True)
    assert plain.status_code == 400 and "no tool format" in plain.json()["detail"]


def test_a_node_says_which_built_in_tools_its_model_can_use(mesh):  # noqa: F811
    mesh.use(QWEN3, "")
    assert mesh.node.model_kind()["builtin_tools"] == builtin_tools.NAMES
    mesh.node.builtin_tools = False
    assert mesh.node.model_kind()["builtin_tools"] == []
    mesh.node.builtin_tools = True
    mesh.use("{% for m in messages %}{{ m.content }}{% endfor %}", "")
    assert mesh.node.model_kind()["builtin_tools"] == []


def test_the_flag_turns_built_in_tools_off_for_a_control_plane(monkeypatch):
    monkeypatch.setenv("ISHARDED_BUILTIN_TOOLS", "0")
    assert orchestrator.Orchestrator().builtin_tools is False
    monkeypatch.delenv("ISHARDED_BUILTIN_TOOLS")
    assert orchestrator.Orchestrator().builtin_tools is True
    assert orchestrator.build_parser().parse_args(["--no-builtin-tools"]).no_builtin_tools is True


def test_what_a_model_is_told_about_its_mesh_names_no_machine(mesh):  # noqa: F811
    mesh.use(QWEN3, "")
    facts = mesh.node.mesh_facts()
    assert facts["model"] == "scripted" and facts["stages"] == 1 and facts["machines_sharing_the_model"] == 1
    assert "http" not in json.dumps(facts) and "worker" not in json.dumps(facts)


def test_rounds_of_tool_calls_count_every_token_and_end_with_one_summary():
    import asyncio

    async def turn(items):
        for item in items:
            yield item

    call = {"id": "call_1", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "1 + 1"}'}}
    first = [{"text": "Working. "}, {"tool_call": call}, {"finish_reason": "tool_calls", "prompt_tokens": 10, "completion_tokens": 5, "time_to_first_token_ms": 7}]
    second = [{"text": "Two."}, {"finish_reason": "stop", "prompt_tokens": 30, "completion_tokens": 2, "time_to_first_token_ms": 99, "tokens_per_second": 4.0}]
    seen = []

    async def again(extra, final):
        seen.append((extra, final))
        return turn(second)

    message, summary = asyncio.run(whole_reply(with_builtin_tools(turn(first), again, builtin_tools.run, 4)))
    assert message == {"role": "assistant", "content": "Working. Two."}
    assert (summary["prompt_tokens"], summary["completion_tokens"], summary["time_to_first_token_ms"], summary["finish_reason"]) == (40, 7, 7, "stop")
    assert json.loads(summary["tool_uses"][0]["result"])["result"] == 2
    extra, final = seen[0]
    assert final is False and [item["role"] for item in extra] == ["assistant", "tool"] and extra[1]["tool_call_id"] == "call_1"
