"""docs/assistant-parts/growth.md is the manual for federation, skills, coding, and voice."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pytest

from shardmesh.assistant import coder, federation, skills, voice

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "assistant-parts" / "growth.md").read_text(encoding="utf-8")
PARTS = (federation, skills, coder, voice)


def walk(parser: argparse.ArgumentParser, name: str):
    yield name, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for command, child in action.choices.items():
                yield from walk(child, f"{name} {command}")


def parsers():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers()
    for part in PARTS:
        part.add_commands(commands)
    yield from walk(root, "")


def options():
    for name, parser in parsers():
        for action in parser._actions:
            if not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                yield name.strip(), action


@pytest.mark.parametrize("command, flag", sorted({(name, flag) for name, action in options() for flag in action.option_strings}))
def test_every_option_is_documented(command, flag):
    assert flag in DOC, f"{command} {flag} is missing from growth.md"


@pytest.mark.parametrize("command", sorted({name.strip() for name, _ in parsers() if len(name.split()) == 2}))
def test_every_command_is_documented(command):
    assert f"`{command}" in DOC, f"{command} is missing from growth.md"


def test_every_option_explains_itself():
    assert not [f"{name} {action.option_strings or action.dest}" for name, action in options() if not action.help]


def test_every_setting_is_documented():
    found: set[str] = set()
    for part in PARTS:
        found |= set(re.findall(r"\bSHARDMESH_[A-Z0-9_]+", Path(part.__file__).read_text(encoding="utf-8")))
    internal = {"SHARDMESH_VOICE_OUT", "SHARDMESH_VOICE_TEXT", "SHARDMESH_VOICE_NAME"}  # how the script is handed its input
    assert not sorted(name for name in found - internal if name not in DOC)


@pytest.mark.parametrize("part, prefix", [(federation, "/federation"), (voice, "/assistant/voice")])
def test_every_route_is_documented(part, prefix):
    text = Path(part.__file__).read_text(encoding="utf-8")
    paths = re.findall(r'@router\.(?:get|post)\("(/[^"]+)"', text)
    assert paths
    assert not [path for path in paths if f"{prefix}{path}" not in DOC]


def test_the_protocol_is_written_down():
    for needed in ("X25519", "HKDF", "AES-256-GCM", "Ed25519", "nonce", "expires_at", "audience", "request_hash", "notice"):
        assert needed.lower() in DOC.lower(), needed
