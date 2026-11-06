"""The guide to the hands stays true: every option and setting is in it, and its examples work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pytest

from shardmesh.assistant import cli
from shardmesh.assistant.app import Assistant
from shardmesh.assistant.contracts import Effect
from shardmesh.assistant.tasks import plan_errors

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs" / "assistant-parts" / "hands.md").read_text(encoding="utf-8")
PARTS = ("broker", "tools", "plugins", "sandbox", "tasks", "planner", "schema")
COMMANDS = ("grant", "approvals", "task", "plan", "plugin", "sandbox", "undo")


def assistant_commands():
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)

    def walk(node, path):
        yield path, node
        for action in node._actions:
            if isinstance(action, argparse._SubParsersAction):
                for word, child in action.choices.items():
                    yield from walk(child, f"{path} {word}".strip())

    for path, node in walk(parser, ""):
        if path.split(" ")[0] in COMMANDS:
            yield path, node


def leaves():
    for path, node in assistant_commands():
        if not any(isinstance(action, argparse._SubParsersAction) for action in node._actions):
            yield path, node


@pytest.mark.parametrize("path", sorted(path for path, _ in leaves()))
def test_every_command_has_a_heading(path):
    assert f"`{path}" in GUIDE, f"{path} is missing from the guide"


@pytest.mark.parametrize(
    "path, flag",
    sorted(
        (path, flag)
        for path, node in leaves()
        for action in node._actions
        if not isinstance(action, argparse._HelpAction)
        for flag in action.option_strings
    ),
)
def test_every_option_is_documented(path, flag):
    assert f"`{flag}`" in GUIDE, f"{path} {flag} is missing from the guide"


def settings():
    found = set()
    for part in PARTS:
        found |= set(re.findall(r"SHARDMESH_[A-Z_]+", (ROOT / "shardmesh" / "assistant" / f"{part}.py").read_text(encoding="utf-8")))
    return sorted(found)


@pytest.mark.parametrize("setting", settings())
def test_every_setting_is_documented(setting):
    assert f"`{setting}`" in GUIDE, f"{setting} is missing from the guide"


def json_examples():
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", GUIDE, re.DOTALL)]


def test_the_example_plan_is_a_valid_plan():
    plan = next(example for example in json_examples() if "steps" in example)
    assert plan_errors(plan) == []


def test_the_example_plan_shows_every_kind_of_step():
    plan = next(example for example in json_examples() if "steps" in example)
    assert {step["kind"] for step in plan["steps"]} == {"tool", "check", "model", "approval", "wait", "subtask"}


def test_the_example_plan_is_accepted_by_the_engine(tmp_path):
    plan = next(example for example in json_examples() if "steps" in example)
    with Assistant(tmp_path / "home") as assistant:
        assert assistant.tasks.submit("ann", plan).startswith("t-")


def test_the_example_manifests_install(tmp_path, monkeypatch):
    manifests = [example for example in json_examples() if "command" in example]
    plain = next(example for example in manifests if "capabilities" in example)
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(plain), encoding="utf-8")
    with Assistant(tmp_path / "home") as assistant:
        assert assistant.plugins.add(path) == ["plugin.weather.forecast"]
        assert assistant.broker.capability("plugin.weather.forecast").effects == {Effect.NETWORK}


def test_the_table_of_tools_matches_what_is_registered(tmp_path):
    rows = re.findall(r"^\| `([a-z]+\.[a-z]+)` \| ([a-z_, ]+) \| (`[a-z]+`|none) \| (yes|no) \|$", GUIDE, re.MULTILINE)
    assert len(rows) == 17
    with Assistant(tmp_path / "home") as assistant:
        for name, effects, scope, repeat in rows:
            capability = assistant.broker.capability(name)
            assert {effect.value for effect in capability.effects} == set(effects.split(", ")), name
            assert (capability.scope_field or "none") == scope.strip("`"), name
            assert capability.idempotent == (repeat == "yes"), name
