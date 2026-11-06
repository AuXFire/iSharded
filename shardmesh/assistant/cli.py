"""``shardmesh assistant``: every part adds its own commands."""
from __future__ import annotations

import argparse
from typing import Any

from .app import Assistant, installed_parts


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", default=None, help="the assistant's folder; default is ShardMesh/assistant in your data folder")
    commands = parser.add_subparsers(dest="assistant_command", required=True)
    audit = commands.add_parser("audit", help="show what the assistant did, and check the record is intact")
    audit.add_argument("--limit", type=int, default=50, help="how many entries to show")
    audit.add_argument("--actor", default=None, help="only this actor")
    audit.set_defaults(run=run_audit)
    for part in installed_parts():
        add_commands = getattr(part, "add_commands", None)
        if add_commands:
            add_commands(commands)


def run_audit(assistant: Assistant, args: Any) -> int:
    broken = assistant.audit.verify()
    for row in assistant.audit.read(actor=args.actor)[-args.limit:]:
        print(f"{row['id']:>6}  {row['actor']:<20}  {row['action']:<20}  {row['subject']}")
    print("The record is intact." if broken is None else f"The record was changed at entry {broken}.")
    return 0 if broken is None else 1


def run(args: argparse.Namespace) -> int:
    with Assistant(args.home) as assistant:
        return args.run(assistant, args)
