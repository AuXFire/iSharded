"""docs/ASSISTANT.md is the whole manual. This fails if it stops being true.

It walks the real command line and the real app with every part attached, so a new
command, option, route, or setting cannot arrive without saying what it does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex

import pytest

from shardmesh.assistant import cli, context, conversation, gateway, gateway_cli, knowledge, memory, providers, router
from shardmesh.assistant.app import Assistant
from shardmesh.assistant.contracts import Effect
from shardmesh.assistant.gateway import create_app
from shardmesh.assistant.mind import LOCAL_USER
from shardmesh.assistant.tasks import plan_errors

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs" / "ASSISTANT.md").read_text(encoding="utf-8")
LINES = DOCS.splitlines()
PACKAGE = ROOT / "shardmesh" / "assistant"
# Handed to a child process by the assistant itself. Nobody sets these.
INTERNAL_SETTINGS = {"SHARDMESH_VOICE_OUT", "SHARDMESH_VOICE_TEXT", "SHARDMESH_VOICE_NAME"}


def root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shardmesh assistant")
    cli.add_arguments(parser)
    return parser


def walk(parser: argparse.ArgumentParser, path: str = ""):
    yield path, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for word, child in action.choices.items():
                yield from walk(child, f"{path} {word}".strip())


COMMANDS = dict(walk(root_parser()))
LEAVES = {path: parser for path, parser in COMMANDS.items() if path and not any(isinstance(action, argparse._SubParsersAction) for action in parser._actions)}


def arguments_of(parser: argparse.ArgumentParser):
    return [action for action in parser._actions if not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))]


def where(command: str) -> str:
    """The part of the manual about one command: its table row, or its heading's section."""
    found = []
    for index, line in enumerate(LINES):
        if f"`{command}" not in line:
            continue
        found.append(line)
        if line.startswith("#"):
            for following in LINES[index + 1 :]:
                if following.startswith("#"):
                    break
                found.append(following)
    return "\n".join(found)


@pytest.mark.parametrize("command", sorted(LEAVES))
def test_every_command_is_documented(command):
    assert where(command), f"`{command}` is missing from docs/ASSISTANT.md"


@pytest.mark.parametrize(
    "command, flag",
    sorted((path, flag) for path, parser in LEAVES.items() for action in arguments_of(parser) for flag in action.option_strings),
)
def test_every_option_is_documented_with_its_command(command, flag):
    if flag == "--user" and "`--user NAME`" in DOCS and command.split()[0] not in ("peer", "task", "grant", "approvals", "plan", "undo"):
        return  # stated once for every command that acts for a person
    assert flag in where(command), f"{command} {flag} is not explained where {command} is"


def test_the_shared_options_are_documented_once_for_all():
    assert "`--user NAME`" in DOCS and LOCAL_USER in DOCS
    assert "`--home`" in DOCS


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_argument_explains_itself(command):
    assert [action.dest for action in arguments_of(COMMANDS[command]) if not action.help] == []


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_subcommand_explains_itself(command):
    for action in COMMANDS[command]._actions:
        if isinstance(action, argparse._SubParsersAction):
            assert [item.dest for item in action._choices_actions if not item.help] == []


def real_routes(tmp_path: Path) -> list[str]:
    with Assistant(tmp_path) as assistant:
        app = create_app(assistant)
    return sorted({f"{method} {route.path}" for route in app.routes for method in getattr(route, "methods", None) or () if method != "HEAD"})


def test_every_route_is_documented(tmp_path):
    assert [route for route in real_routes(tmp_path) if f"`{route}`" not in DOCS] == []


def test_every_documented_route_exists(tmp_path):
    real = set(real_routes(tmp_path))
    named = {route for route in re.findall(r"`((?:GET|POST|DELETE) /[^`]*)`", DOCS)}
    assert sorted(named - real) == []


def test_the_routes_left_open_are_exactly_the_ones_the_manual_says(tmp_path):
    from shardmesh.assistant import federation

    for method, path in federation.PUBLIC_ROUTES:
        assert f"`{method} {path}`" in DOCS
    said = next(line for line in LINES if line.startswith("Every path needs a key except"))
    assert "federation routes" in said
    for path in gateway.PUBLIC_PATHS:
        assert f"`GET {path}`" in said


def real_settings() -> list[str]:
    found: set[str] = set()
    for path in PACKAGE.glob("*.py"):
        found |= set(re.findall(r"\bSHARDMESH_[A-Z0-9_]+", path.read_text(encoding="utf-8")))
    return sorted(found - INTERNAL_SETTINGS)


@pytest.mark.parametrize("setting", real_settings())
def test_every_setting_is_documented(setting):
    assert f"`{setting}`" in DOCS


@pytest.mark.parametrize(
    ("name", "value", "module"),
    [
        ("DEFAULT_PORT", gateway_cli.DEFAULT_PORT, "gateway_cli.py"),
        ("BUDGET_TOKENS", conversation.BUDGET_TOKENS, "conversation.py"),
        ("REPLY_TOKENS", conversation.REPLY_TOKENS, "conversation.py"),
        ("MAX_ROUNDS", conversation.MAX_ROUNDS, "conversation.py"),
        ("MAX_LEARNED", conversation.MAX_LEARNED, "conversation.py"),
        ("HISTORY_MESSAGES", context.HISTORY_MESSAGES, "context.py"),
        ("READ_SECONDS", int(providers.READ_SECONDS), "providers.py"),
        ("WHOLE_REPLY_SECONDS", int(providers.WHOLE_REPLY_SECONDS), "providers.py"),
        ("MEASURED_RUNS", router.MEASURED_RUNS, "router.py"),
        ("BACKOFF_SECONDS", int(router.BACKOFF_SECONDS), "router.py"),
        ("MAX_BACKOFF_SECONDS", int(router.MAX_BACKOFF_SECONDS), "router.py"),
        ("RECENCY_HALF_LIFE_DAYS", int(memory.RECENCY_HALF_LIFE_DAYS), "memory.py"),
    ],
)
def test_every_constant_says_its_real_value(name, value, module):
    assert f"| `{name}` | {value} | `{module}` |" in DOCS


def test_the_roles_and_kinds_are_the_real_ones():
    assert all(f"`{role}`" in DOCS for role in router.ROLES)
    assert all(f"`{kind}`" in DOCS for kind in memory.KINDS)


def test_the_documented_skips_are_real():
    for pattern in (".env*", "*.key", "*.pem", "id_rsa*", "credentials*"):
        assert f"`{pattern}`" in DOCS and pattern in knowledge.SECRET_NAMES


def json_examples() -> list[dict]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", DOCS, re.DOTALL)]


def test_the_example_plan_is_a_real_plan_that_shows_every_kind_of_step(tmp_path):
    plan = next(example for example in json_examples() if "steps" in example and len(example["steps"]) > 3)
    assert plan_errors(plan) == []
    assert {step["kind"] for step in plan["steps"]} == {"tool", "check", "model", "approval", "wait", "subtask"}
    with Assistant(tmp_path / "home") as assistant:
        assert assistant.tasks.submit("ann", plan).startswith("t-")


def test_the_example_manifest_installs(tmp_path):
    plain = next(example for example in json_examples() if "command" in example and "capabilities" in example)
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(plain), encoding="utf-8")
    with Assistant(tmp_path / "home") as assistant:
        assert assistant.plugins.add(path) == ["plugin.weather.forecast"]
        assert assistant.broker.capability("plugin.weather.forecast").effects == {Effect.NETWORK}


def test_the_table_of_tools_matches_what_is_registered(tmp_path):
    rows = re.findall(r"^\| `([a-z]+\.[a-z]+)` \| ([a-z_, ]+) \| (`[a-z_]+`|none) \| (yes|no) \|$", DOCS, re.MULTILINE)
    with Assistant(tmp_path / "home") as assistant:
        documented = {name for name, *_ in rows}
        assert sorted({capability.name for capability in assistant.broker.capabilities()} - documented) == []
        for name, effects, scope, repeat in rows:
            capability = assistant.broker.capability(name)
            assert {effect.value for effect in capability.effects} == set(effects.split(", ")), name
            assert (capability.scope_field or "none") == scope.strip("`"), name
            assert capability.idempotent == (repeat == "yes"), name


def shell_lines() -> list[str]:
    blocks = re.findall(r"```(?:text|sh|bash|powershell)?\n(.*?)```", DOCS, re.DOTALL)
    return [line.strip() for block in blocks for line in block.splitlines() if line.strip().startswith("shardmesh assistant ")]


@pytest.mark.parametrize("line", shell_lines())
def test_every_command_shown_really_parses(line, capsys):
    words = shlex.split(line.split(" #")[0], posix=True)[2:]
    try:
        root_parser().parse_args(words)
    except SystemExit as stop:
        pytest.fail(f"{line}: {capsys.readouterr().err}")
        raise stop


def test_the_quick_start_comes_first_and_covers_the_whole_path():
    start = DOCS.split("## Quick start")[1].split("\n## ")[0]
    assert DOCS.index("## Quick start") < DOCS.index("## What keeps it safe")
    for step in ("pip install", "provider add", "assistant say", "grant add", "task run"):
        assert step in start, step


def test_what_is_not_built_or_proven_is_said_plainly():
    honest = DOCS.split("## What is not built or proven")[1].lower()
    for subject in ("avatar", "wake word", "vision", "home", "robot", "`process`"):
        assert subject in honest, subject


def test_the_taint_rule_for_memories_is_stated():
    assert "A memory proposed on a tainted turn is stored as EXTERNAL, not MODEL" in DOCS


def test_the_parts_folder_is_gone_and_nothing_points_at_it():
    assert not (ROOT / "docs" / "assistant-parts").exists()
    for path in PACKAGE.glob("*.py"):
        assert "assistant-parts" not in path.read_text(encoding="utf-8"), path.name


def test_the_page_ships_with_the_package():
    assert '"shardmesh.assistant" = ["assistant.html"]' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
