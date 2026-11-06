"""The README is the manual, so anything a person can set must be in it."""

import argparse
from pathlib import Path
import re

import pytest

from shardmesh import cli, orchestrator, worker

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CODE = {path: path.read_text(encoding="utf-8") for path in (ROOT / "shardmesh").glob("*.py")}


def parsers():
    def walk(parser, name):
        yield name, parser
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for command, child in action.choices.items():
                    yield from walk(child, f"{name} {command}")

    yield from walk(cli.build_parser(), "shardmesh")
    yield "orchestrator", orchestrator.build_parser()
    yield "worker", worker.build_parser()


def options():
    for name, parser in parsers():
        for action in parser._actions:
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            yield name, action


@pytest.mark.parametrize("command, flag", sorted({(name, flag) for name, action in options() for flag in action.option_strings}))
def test_every_option_is_documented(command, flag):
    assert f"`{flag}" in README or f" {flag} " in README, f"{command} {flag} is missing from the README"


def test_every_option_explains_itself():
    missing = [f"{name} {action.option_strings or action.dest}" for name, action in options() if not action.help]
    assert not missing


def test_every_command_has_a_section():
    for command in ("demo", "node", "load", "cordon", "model hash", "key", "pool"):
        assert re.search(rf"^### shardmesh {command}", README, re.M), command


@pytest.mark.parametrize("name", ["init", "add-node", "request-join", "requests", "approve", "deny", "revoke"])
def test_every_pool_command_is_documented(name):
    assert f"`pool {name} " in README


def test_every_setting_is_documented():
    found = set()
    for text in CODE.values():
        found |= set(re.findall(r"\b(?:SHARDMESH|HF|HUGGING_FACE)_[A-Z0-9_]+", text))
    missing = sorted(name for name in found if name not in README)
    assert not missing


def test_every_public_route_is_documented():
    routes = set(re.findall(r'@app\.(?:get|post|delete)\("(/[^"]+)"', CODE[ROOT / "shardmesh" / "orchestrator.py"]))
    public = {route for route in routes if not route.startswith("/internal")}
    missing = sorted(route for route in public if f"`{route}`" not in README and route not in README)
    assert not missing


def test_every_metric_is_documented():
    names = set(re.findall(r'add\(\s*"([a-z_]+)"', CODE[ROOT / "shardmesh" / "orchestrator.py"]))
    assert names
    assert not sorted(name for name in names if f"`{name}`" not in README)


def test_every_load_field_is_documented():
    from shardmesh.protocol import LoadModelRequest

    assert not sorted(name for name in LoadModelRequest.model_fields if f"`{name}`" not in README)


def test_every_chat_field_is_documented():
    from shardmesh.protocol import ChatCompletionRequest, CompletionRequest

    fields = {*ChatCompletionRequest.model_fields, *CompletionRequest.model_fields}
    assert not sorted(name for name in fields if f"`{name}`" not in README)


def test_links_and_images_point_at_real_files():
    targets = re.findall(r'\]\((?!http|#)([^)#]+)', README) + re.findall(r'(?:src|srcset)="([^"]+)"', README)
    assert targets
    missing = sorted(target for target in targets if not (ROOT / target).exists())
    assert not missing


def test_links_inside_the_page_have_a_heading():
    def slug(heading):
        return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")

    headings = {slug(text) for text in re.findall(r"^#+ (.+)$", README, re.M)}
    missing = sorted(anchor for anchor in re.findall(r"\]\(#([^)]+)\)", README) if anchor not in headings)
    assert not missing


def test_scripts_named_in_the_readme_exist():
    for name in re.findall(r"`(scripts/[\w./-]+|packaging/[\w./-]+|shardmesh/\w+\.py)`", README):
        assert (ROOT / name).exists(), name
