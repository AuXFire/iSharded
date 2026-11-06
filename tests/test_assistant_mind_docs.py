"""docs/assistant-parts/mind.md has to say everything these parts add."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pytest

from shardmesh.assistant import context, conversation, gateway_cli, identity, knowledge, memory, providers, router
from shardmesh.assistant.app import Assistant
from shardmesh.assistant.gateway import create_app
from shardmesh.assistant.mind import LOCAL_USER

DOCS = (Path(__file__).parent.parent / "docs" / "assistant-parts" / "mind.md").read_text(encoding="utf-8")
PARTS = (identity, memory, knowledge, providers, router, context, conversation)


def my_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    for part in PARTS:
        part.add_commands(commands)
    return parser


def walk(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield (*path, name), child
                yield from walk(child, (*path, name))


COMMANDS = dict(walk(my_parser()))
OPTIONS = [
    (name, flag, action)
    for name, parser in COMMANDS.items()
    for action in parser._actions
    for flag in action.option_strings
    if flag not in ("-h", "--help")
]


@pytest.mark.parametrize("name", [" ".join(name) for name in COMMANDS])
def test_every_command_is_documented(name):
    assert f"`{name}" in DOCS


@pytest.mark.parametrize(("flag", "command"), sorted({(flag, " ".join(name)) for name, flag, _ in OPTIONS if flag != "--user"}))
def test_every_option_is_documented(flag, command):
    assert flag in DOCS, f"{command} {flag}"


def test_the_user_option_is_documented_once_for_all():
    assert "`--user NAME`" in DOCS and LOCAL_USER in DOCS


@pytest.mark.parametrize("name", [" ".join(name) for name in COMMANDS])
def test_every_command_explains_itself(name):
    parents = dict(walk(my_parser()))
    for path, parser in parents.items():
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                assert all(item.help for item in action._choices_actions)


def test_every_option_explains_itself():
    assert [f"{' '.join(name)} {flag}" for name, flag, action in OPTIONS if not action.help] == []


def routes(tmp_path) -> list[tuple[str, str]]:
    with Assistant(tmp_path) as assistant:
        app = create_app(assistant)
    return sorted({(method, route.path) for route in app.routes for method in getattr(route, "methods", None) or () if method != "HEAD"})


def test_every_route_is_documented(tmp_path):
    missing = [f"{method} {path}" for method, path in routes(tmp_path) if f"`{method} {path}`" not in DOCS]
    assert missing == []


def test_every_documented_route_exists(tmp_path):
    real = {f"{method} {path}" for method, path in routes(tmp_path)}
    named = set(re.findall(r"`((?:GET|POST|DELETE) /[^`]*)`", DOCS))
    assert named <= real


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
def test_every_setting_says_its_real_value(name, value, module):
    assert f"| `{name}` | {value} | `{module}` |" in DOCS


def test_the_roles_and_kinds_are_the_real_ones():
    assert all(f"`{role}`" in DOCS for role in router.ROLES)
    assert all(f"`{kind}`" in DOCS for kind in memory.KINDS)


def test_the_documented_skips_are_real():
    for pattern in (".env*", "*.key", "*.pem", "id_rsa*", "credentials*"):
        assert f"`{pattern}`" in DOCS and pattern in knowledge.SECRET_NAMES
