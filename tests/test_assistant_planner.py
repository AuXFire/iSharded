"""The planner, with a scripted model. Nothing a model writes is believed."""
from __future__ import annotations

import argparse
import json

import pytest

from isharded.assistant import cli
from isharded.assistant import planner as planner_part
from isharded.assistant.app import Assistant
from isharded.assistant.contracts import Capability, Effect, ModelReply, Reach
from isharded.assistant.planner import MAX_ASKS, PlanRejected, parse_reply, triage


class ScriptedRouter:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return ModelReply(text, (), "fake", "fake", 1, 1, 0.0)


@pytest.fixture
def assistant(tmp_path):
    with Assistant(tmp_path / "home") as made:
        yield made


def script(assistant, *replies):
    assistant.router = ScriptedRouter(*replies)
    return assistant.router


def read_step(step_id="a", path="/tmp/x", **fields):
    return {"id": step_id, "kind": "tool", "capability": "files.read", "arguments": {"path": path}, **fields}


GOOD = {"title": "Read it", "steps": [read_step()]}


def test_it_asks_the_planning_role_for_json_with_the_real_capabilities(assistant):
    router = script(assistant, GOOD)
    assistant.planner.plan("ann", "Read the file")
    request = router.requests[0]
    assert request.role == "planning"
    system = request.messages[0]["content"]
    assert "JSON" in system and "files.read(path, max_bytes?)" in system and "git.push" in system
    assert request.messages[1] == {"role": "user", "content": "Read the file"}


def test_a_good_plan_comes_back_ready_for_the_engine(assistant):
    script(assistant, GOOD)
    plan = assistant.planner.plan("ann", "Read the file")
    assert plan["title"] == "Read it" and plan["goal"] == "Read the file"
    assert assistant.tasks.submit("ann", plan).startswith("t-")


def test_a_fence_around_the_json_is_forgiven():
    assert parse_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_reply('  {"a": 1}  ') == {"a": 1}


@pytest.mark.parametrize(
    "bad, words",
    [
        ("this is not json", "not JSON"),
        ('["a", "list"]', "JSON object"),
        ({"steps": []}, "list of steps"),
        ({"steps": [read_step(depends_on=["ghost"])]}, "not in this plan"),
        ({"steps": [{"id": "a", "kind": "tool", "capability": "bank.transfer", "arguments": {}}]}, "not a capability"),
        ({"steps": [{"id": "a", "kind": "model", "prompt": "{{steps.ghost.output}}"}]}, "not in this plan"),
        ({"steps": [{"id": "a", "kind": "model", "role": "wizard", "prompt": "x"}]}, "role"),
        ({"steps": [{"id": "a", "kind": "model", "prompt": "x", "escalate_to": ["wizard"]}]}, "role"),
        (
            {"steps": [read_step("a", depends_on=["b"]), read_step("b", depends_on=["a"])]},
            "circle",
        ),
        ({"steps": [read_step(str(number)) for number in range(51)]}, "at most 50"),
        (
            {"steps": [{"id": "s", "kind": "subtask", "plan": {"steps": [{"id": "x", "kind": "tool", "capability": "nope.nope", "arguments": {}}]}}]},
            "not a capability",
        ),
    ],
)
def test_a_bad_plan_is_sent_back_with_what_is_wrong(assistant, bad, words):
    router = script(assistant, bad, GOOD)
    plan = assistant.planner.plan("ann", "Do it")
    assert plan["title"] == "Read it"
    assert len(router.requests) == 2
    assert words in router.requests[1].messages[-1]["content"]


def test_nested_steps_count_toward_the_limit(assistant):
    nested = {"steps": [read_step(str(number)) for number in range(30)]}
    plan = {"steps": [{"id": "s1", "kind": "subtask", "plan": nested}, {"id": "s2", "kind": "subtask", "plan": nested}]}
    router = script(assistant, plan, GOOD)
    assistant.planner.plan("ann", "Do it")
    assert "at most 50" in router.requests[1].messages[-1]["content"]


def test_it_re_asks_at_most_twice(assistant):
    router = script(assistant, "no", "still no", "never", GOOD)
    with pytest.raises(PlanRejected, match="refused 3 times") as refused:
        assistant.planner.plan("ann", "Do it")
    assert len(router.requests) == MAX_ASKS == 3
    assert refused.value.errors


def test_each_re_ask_carries_the_conversation(assistant):
    router = script(assistant, "no", "still no", GOOD)
    assistant.planner.plan("ann", "Do it")
    assert [message["role"] for message in router.requests[2].messages] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_a_model_cannot_schedule_or_clear_taint_or_widen_reach(assistant):
    sneaky = {**GOOD, "every_seconds": 1, "run_at": 1, "priority": 99, "tainted": False, "reach": "cloud", "inputs": {"a": 1}}
    script(assistant, sneaky)
    plan = assistant.planner.plan("ann", "Read it", reach=Reach.MACHINE, tainted=True)
    assert plan["tainted"] is True and plan["reach"] == "machine"
    assert not {"every_seconds", "run_at", "priority", "inputs"} & set(plan)


def test_a_model_cannot_do_those_inside_a_subtask_either(assistant):
    nested = {"steps": [read_step()], "reach": "cloud", "every_seconds": 1}
    script(assistant, {"steps": [{"id": "s", "kind": "subtask", "plan": nested}]})
    plan = assistant.planner.plan("ann", "Do it")
    assert set(plan["steps"][0]["plan"]) == {"steps"}


def test_the_reach_given_limits_where_the_plan_is_asked_for(assistant):
    router = script(assistant, GOOD)
    assistant.planner.plan("ann", "A private goal", reach=Reach.MACHINE)
    assert router.requests[0].reach is Reach.MACHINE


def test_planning_says_when_there_is_no_router(assistant):
    assistant.router = None
    with pytest.raises(ValueError, match="router"):
        assistant.planner.plan("ann", "Do it")


def test_attempts_are_audited_without_the_goal(assistant):
    script(assistant, "no", GOOD)
    assistant.planner.plan("ann", "my private goal")
    rows = assistant.audit.read(actor="planner")
    assert [row["detail"]["accepted"] for row in rows] == [False, True]
    assert "private" not in json.dumps(rows)


# Checks.


def write_step(step_id="w", path="/tmp/out.txt"):
    return {"id": step_id, "kind": "tool", "capability": "files.write", "arguments": {"path": path, "content": "x"}}


def by_id(plan):
    return {step["id"]: step for step in plan["steps"]}


def test_a_check_is_added_after_a_consequential_step(assistant):
    script(assistant, {"steps": [write_step()]})
    plan = assistant.planner.plan("ann", "Write it")
    assert plan["checks_added"] == ["check-w"]
    assert [step["id"] for step in plan["steps"]] == ["w", "check-w"]
    added = by_id(plan)["check-w"]
    assert added["kind"] == "check" and added["depends_on"] == ["w"]
    assert added["check"] == {"type": "file_exists", "path": "/tmp/out.txt"}


def test_a_shell_step_gets_an_exit_code_check(assistant):
    step = {"id": "s", "kind": "tool", "capability": "shell.run", "arguments": {"command": ["x"], "cwd": "/tmp"}}
    script(assistant, {"steps": [step]})
    assert by_id(assistant.planner.plan("ann", "Run it"))["check-s"]["check"] == {"type": "exit_code", "step": "s"}


def test_any_other_consequential_step_gets_an_ok_check(assistant):
    assistant.broker.register(Capability("mail.send", "x", frozenset({Effect.EXTERNAL})), lambda a, c: "sent")
    step = {"id": "m", "kind": "tool", "capability": "mail.send", "arguments": {}}
    script(assistant, {"steps": [step]})
    assert by_id(assistant.planner.plan("ann", "Send it"))["check-m"]["check"] == {"type": "ok", "step": "m"}


def test_a_step_that_reads_gets_no_check(assistant):
    script(assistant, GOOD)
    assert assistant.planner.plan("ann", "Read it")["checks_added"] == []


@pytest.mark.parametrize(
    "extra",
    [
        {"id": "c", "kind": "check", "check": {"type": "ok", "step": "w"}},
        {"id": "c", "kind": "check", "check": {"type": "regex", "text": "{{steps.w.output}}", "pattern": "x"}},
        {"id": "c", "kind": "check", "depends_on": ["w"], "check": {"type": "file_exists", "path": "/tmp/out.txt"}},
    ],
)
def test_a_step_that_has_a_check_keeps_only_that(assistant, extra):
    script(assistant, {"steps": [write_step(), extra]})
    assert assistant.planner.plan("ann", "Write it")["checks_added"] == []


def test_a_step_with_its_own_expectation_needs_no_extra_check(assistant):
    step = {**write_step(), "expect": {"type": "regex", "pattern": "Wrote"}}
    script(assistant, {"steps": [step]})
    assert assistant.planner.plan("ann", "Write it")["checks_added"] == []


def test_a_check_inside_a_subtask_is_added_there(assistant):
    script(assistant, {"steps": [{"id": "s", "kind": "subtask", "plan": {"steps": [write_step()]}}]})
    plan = assistant.planner.plan("ann", "Write it")
    assert [step["id"] for step in plan["steps"][0]["plan"]["steps"]] == ["w", "check-w"]


def test_what_followed_the_step_now_waits_for_its_check(assistant):
    follower = {"id": "next", "kind": "model", "prompt": "After {{steps.w.output}}"}
    script(assistant, {"steps": [write_step(), follower]})
    plan = assistant.planner.plan("ann", "Write it")
    assert by_id(plan)["next"]["depends_on"] == ["check-w"]


def test_an_added_check_does_not_clash_with_an_id_already_used(assistant):
    taken = {"id": "check-w", "kind": "wait", "seconds": 1}
    script(assistant, {"steps": [write_step(), taken]})
    assert assistant.planner.plan("ann", "Write it")["checks_added"] == ["check-w-2"]


def test_an_added_check_really_stops_the_plan(assistant, tmp_path):
    """Through the engine, on a real file: the follower must not run after a failed check."""
    folder = tmp_path / "work"
    folder.mkdir()
    target = str(folder / "out.txt")
    assistant.broker.add_grant("ann", "files.write", str(folder), "ann")
    ran = []
    assistant.broker.register(Capability("demo.next", "x", frozenset({Effect.READ})), lambda a, c: ran.append(1) or "ok")
    assistant.broker.add_grant("ann", "demo.next", None, "ann")
    bad_check_target = str(folder / "elsewhere.txt")
    follower = {"id": "next", "kind": "tool", "capability": "demo.next", "arguments": {}, "depends_on": ["w"]}
    script(assistant, {"steps": [write_step(path=target), follower]})
    plan = assistant.planner.plan("ann", "Write it")
    by_id(plan)["check-w"]["check"]["path"] = bad_check_target  # as if the write had gone astray
    task = assistant.tasks.submit("ann", plan)
    assert assistant.tasks.run_until_idle(30)
    assert assistant.tasks.status(task)["status"] == "failed"
    assert ran == []


# Grants up front.


def test_missing_grants_are_listed_up_front(assistant):
    script(assistant, {"steps": [read_step("a", "/data/x"), read_step("b", "/data/x"), write_step("w", "/out/y")]})
    plan = assistant.planner.plan("ann", "Do it")
    listed = {entry["capability"]: entry for entry in plan["missing_grants"]}
    assert listed["files.read"]["steps"] == ["a", "b"] and listed["files.read"]["scope"] == "/data/x"
    assert listed["files.write"]["effects"] == ["write_local"]
    assert listed["files.read"]["command"] == "isharded assistant grant add files.read /data/x --user ann"


def test_what_is_already_granted_is_not_listed(assistant, tmp_path):
    assistant.broker.add_grant("ann", "files.read", str(tmp_path), "ann")
    script(assistant, {"steps": [read_step(path=str(tmp_path / "x"))]})
    assert assistant.planner.plan("ann", "Read it")["missing_grants"] == []


def test_another_users_grant_does_not_count(assistant, tmp_path):
    assistant.broker.add_grant("bob", "files.read", str(tmp_path), "bob")
    script(assistant, {"steps": [read_step(path=str(tmp_path / "x"))]})
    assert len(assistant.planner.plan("ann", "Read it")["missing_grants"]) == 1


def test_a_scope_known_only_at_run_time_is_said_to_be(assistant):
    step = {"id": "b", "kind": "tool", "capability": "files.read", "arguments": {"path": "{{steps.a.output}}"}}
    script(assistant, {"steps": [{"id": "a", "kind": "model", "prompt": "Which file?"}, step]})
    entry = assistant.planner.plan("ann", "Do it")["missing_grants"][0]
    assert entry["scope"] is None and "only when it runs" in entry["note"]
    assert "SCOPE" in entry["command"]


def test_physical_is_listed_as_asking_every_time(assistant):
    assistant.broker.register(Capability("door.open", "x", frozenset({Effect.PHYSICAL})), lambda a, c: "open")
    script(assistant, {"steps": [{"id": "d", "kind": "tool", "capability": "door.open", "arguments": {}}]})
    entry = assistant.planner.plan("ann", "Open it")["missing_grants"][0]
    assert entry["command"] == "" and "each action" in entry["note"]


def test_one_yes_covers_what_a_plan_needs_and_no_more(assistant, tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "x.txt").write_text("hello", encoding="utf-8")
    target = str(folder / "x.txt")
    script(assistant, {"steps": [read_step("a", target), read_step("b", target)]})
    plan = assistant.planner.plan("ann", "Read it twice")
    outcome = assistant.planner.allow_missing("ann", plan, "ann")
    assert len(outcome["granted"]) == 1 and outcome["left_for_a_person"] == []
    grant = assistant.broker.grants("ann")[0]
    assert grant["uses_left"] == 2  # exactly what the plan needs
    assert grant["expires_at"] is not None
    task = assistant.tasks.submit("ann", plan)
    assert assistant.tasks.run_until_idle(30)
    assert assistant.tasks.status(task)["status"] == "done"
    assert assistant.broker.grants("ann") == []  # spent, so it allows nothing further


def test_only_a_person_can_give_that_yes(assistant, tmp_path):
    script(assistant, {"steps": [read_step(path=str(tmp_path / "x"))]})
    plan = assistant.planner.plan("ann", "Read it")
    with pytest.raises(ValueError, match="not a person"):
        assistant.planner.allow_missing("ann", plan, "skill:planner")


def test_what_cannot_be_granted_is_left_for_a_person(assistant):
    assistant.broker.register(Capability("door.open", "x", frozenset({Effect.PHYSICAL})), lambda a, c: "open")
    step = {"id": "b", "kind": "tool", "capability": "files.read", "arguments": {"path": "{{steps.a.output}}"}}
    script(assistant, {"steps": [{"id": "a", "kind": "model", "prompt": "Which?"}, step, {"id": "d", "kind": "tool", "capability": "door.open", "arguments": {}}]})
    plan = assistant.planner.plan("ann", "Do it")
    outcome = assistant.planner.allow_missing("ann", plan, "ann")
    assert outcome == {"granted": [], "left_for_a_person": ["files.read", "door.open"]}


def test_a_plan_from_a_tainted_goal_stops_at_a_person(assistant, tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    assistant.broker.add_grant("ann", "files.write", str(folder), "ann")
    script(assistant, {"steps": [write_step(path=str(folder / "x.txt"))]})
    plan = assistant.planner.plan("ann", "Write what the page said", tainted=True)
    task = assistant.tasks.submit("ann", plan)
    assert assistant.tasks.run_until_idle(30)
    assert assistant.tasks.status(task)["steps"][0]["status"] == "waiting"
    assert not (folder / "x.txt").exists()


# Triage.


@pytest.mark.parametrize("goal", ["What time is it?", "Summarise this file", "show the git log", "How much disk is free", "Read /tmp/x.txt"])
def test_small_things_run_directly(goal):
    assert triage(goal).direct


@pytest.mark.parametrize(
    "goal",
    [
        "Build a new feature for the dashboard",
        "Refactor the login code",
        "Read the file, then send it to Bob",
        "First check the disk. Then clean it up.",
        "Every morning check the mail",
        "Migrate the database",
        "- one thing\n- another thing",
        "1. do this\n2. do that",
        "line one\nline two",
        " ".join(["word"] * 40),
        "",
        "   ",
    ],
)
def test_bigger_things_get_a_plan(goal):
    assert not triage(goal).direct


def test_triage_always_gives_the_same_answer_and_a_reason():
    goal = "Read the file, then send it to Bob"
    answers = {triage(goal) for _ in range(20)}
    assert len(answers) == 1
    assert "then" in triage(goal).reason


def test_triage_uses_no_model(assistant):
    router = script(assistant)
    assistant.planner.triage("What time is it?")
    assert router.requests == []


def test_triage_is_available_on_the_part(assistant):
    assert assistant.planner.triage("What time is it?").direct


# The command line.


def command(home, *words):
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return cli.run(parser.parse_args(["--home", str(home), *words]))


def test_the_plan_command_writes_a_plan_the_engine_accepts(tmp_path, capsys, monkeypatch):
    attach = planner_part.attach

    def attach_with_router(assistant):
        attach(assistant)
        assistant.router = ScriptedRouter(GOOD)

    monkeypatch.setattr(planner_part, "attach", attach_with_router)
    home, out = tmp_path / "home", tmp_path / "plan.json"
    assert command(home, "plan", "Read the file", "--user", "ann", "--out", str(out)) == 0
    shown = capsys.readouterr().out
    assert "Small enough to run directly" in shown and "Missing: isharded assistant grant add files.read" in shown
    assert command(home, "task", "run", str(out), "--detach") == 0


def test_the_plan_command_prints_a_plan_and_says_when_it_cannot(tmp_path, capsys, monkeypatch):
    attach = planner_part.attach

    def attach_with_router(assistant):
        attach(assistant)
        assistant.router = ScriptedRouter(GOOD)

    home = tmp_path / "home"
    assert command(home, "plan", "Read the file") == 1
    assert "provider add" in capsys.readouterr().out
    monkeypatch.setattr(planner_part, "attach", attach_with_router)
    assert command(home, "plan", "Build a new feature") == 0
    assert "Needs a plan" in capsys.readouterr().out


# One yes, and only for what the steps do


def test_one_yes_is_worked_out_from_the_steps_and_not_from_the_plans_own_list(assistant, tmp_path):
    script(assistant, {"steps": [read_step(path=str(tmp_path / "x"))]})
    plan = assistant.planner.plan("ann", "Read it")
    plan["missing_grants"] = [{"capability": "shell.run", "scope": "*", "uses": 1000, "steps": ["a"]}]
    outcome = assistant.planner.allow_missing("ann", plan, "ann")
    assert len(outcome["granted"]) == 1
    assert [(row["capability"], row["uses_left"]) for row in assistant.broker.grants("ann")] == [("files.read", 1)]


def test_what_is_already_allowed_gets_no_second_grant(assistant, tmp_path):
    assistant.broker.add_grant("ann", "files.read", str(tmp_path), "ann")
    script(assistant, {"steps": [read_step(path=str(tmp_path / "x"))]})
    plan = assistant.planner.plan("ann", "Read it")
    assert assistant.planner.allow_missing("ann", plan, "ann") == {"granted": [], "left_for_a_person": []}


def glob_scoped(assistant):
    capability = Capability("notes.tag", "x", frozenset({Effect.WRITE_LOCAL}), scope_field="tag", input_schema={"type": "object", "properties": {"tag": {"type": "string"}}})
    assistant.broker.register(capability, lambda arguments, context: "tagged")


@pytest.mark.parametrize("scope", ["*", "work*", "wor?", "[a-z]"])
def test_a_scope_a_model_wrote_as_a_pattern_is_left_for_a_person(assistant, scope):
    """One yes is for exact scopes, so a plan must not turn it into everything."""
    glob_scoped(assistant)
    script(assistant, {"steps": [{"id": "a", "kind": "tool", "capability": "notes.tag", "arguments": {"tag": scope}}]})
    plan = assistant.planner.plan("ann", "Tag it")
    assert assistant.planner.allow_missing("ann", plan, "ann") == {"granted": [], "left_for_a_person": ["notes.tag"]}
    assert assistant.broker.grants("ann") == []


def test_an_exact_scope_of_that_kind_is_granted(assistant):
    glob_scoped(assistant)
    script(assistant, {"steps": [{"id": "a", "kind": "tool", "capability": "notes.tag", "arguments": {"tag": "work"}}]})
    plan = assistant.planner.plan("ann", "Tag it")
    assert len(assistant.planner.allow_missing("ann", plan, "ann")["granted"]) == 1
    assert assistant.broker.grants("ann")[0]["scope"] == "work"


def test_a_plan_that_could_not_run_gets_no_grants(assistant):
    plan = {"steps": [{"id": "a", "kind": "tool", "capability": "nothing.here", "arguments": {}}]}
    with pytest.raises(PlanRejected, match="not a capability"):
        assistant.planner.allow_missing("ann", plan, "ann")


@pytest.mark.parametrize("by", ["skill:planner", "task:t1", "peer:ann", "model:x", "plugin:y", "conversation"])
def test_no_machine_can_give_that_yes_even_when_nothing_is_missing(assistant, by):
    with pytest.raises(ValueError, match="not a person"):
        assistant.planner.allow_missing("ann", {"steps": []}, by)


def plan_command(home, *words):
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    return cli.run(parser.parse_args(["--home", str(home), "plan-allow", *words]))


def test_the_command_gives_one_yes_for_a_saved_plan(tmp_path, capsys):
    home, file = tmp_path / "home", tmp_path / "plan.json"
    with Assistant(home) as assistant:
        script(assistant, {"steps": [read_step(path=str(tmp_path / "x"), retries=2)]})
        file.write_text(json.dumps(assistant.planner.plan("ann", "Read it")), encoding="utf-8")
    assert plan_command(home, str(file), "--user", "ann", "--by", "ann") == 0
    assert "Granted 1" in capsys.readouterr().out
    with Assistant(home) as assistant:
        grant = assistant.broker.grants("ann")[0]
    assert (grant["capability"], grant["uses_left"], grant["granted_by"]) == ("files.read", 3, "ann")


def test_the_command_says_what_it_left_for_a_person(tmp_path, capsys):
    home, file = tmp_path / "home", tmp_path / "plan.json"
    steps = [{"id": "a", "kind": "wait", "seconds": 1}, {"id": "b", "kind": "tool", "capability": "files.read", "arguments": {"path": "{{steps.a.output}}"}}]
    file.write_text(json.dumps({"steps": steps}), encoding="utf-8")
    assert plan_command(home, str(file)) == 0
    assert "Left for a person: files.read" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("content", "words"),
    [(None, "Could not read"), ("not json", "Expecting value"), ('{"steps": [{"id": "a", "kind": "tool", "capability": "nothing.here", "arguments": {}}]}', "not a capability")],
)
def test_the_command_refuses_in_words(tmp_path, capsys, content, words):
    file = tmp_path / "plan.json"
    if content is not None:
        file.write_text(content, encoding="utf-8")
    assert plan_command(tmp_path / "home", str(file)) == 1
    assert words in capsys.readouterr().out


def test_the_command_refuses_a_machine_as_the_person(tmp_path, capsys):
    file = tmp_path / "plan.json"
    file.write_text('{"steps": []}', encoding="utf-8")
    assert plan_command(tmp_path / "home", str(file), "--by", "skill:x") == 1
    assert "not a person" in capsys.readouterr().out
