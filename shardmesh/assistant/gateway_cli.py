"""The server's commands, kept apart from the server.

Every ``shardmesh`` command builds this parser, including ones that have nothing to
do with the assistant. Importing FastAPI here would add most of a second to all of
them, so the gateway is only imported by a command that needs it.
"""
from __future__ import annotations

import argparse
from typing import Any

from .mind import plainly

DEFAULT_PORT = 8470


def lazily(name: str) -> Any:
    def run(assistant: Any, args: argparse.Namespace) -> int:
        from . import gateway

        return getattr(gateway, name)(assistant, args)

    return plainly(run)


def add_commands(commands: Any) -> None:
    serve = commands.add_parser("serve", help="run the web page and API; loopback only unless told otherwise")
    serve.add_argument("--host", default="127.0.0.1", help="address to listen on; default is this machine only")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on; default is {DEFAULT_PORT}")
    serve.add_argument("--tls-cert", default=None, help="TLS certificate file")
    serve.add_argument("--tls-key", default=None, help="TLS key file")
    serve.add_argument("--trusted-network", action="store_true", help="serve without TLS, because nobody else can read this network")
    serve.set_defaults(run=lazily("run_serve"))

    user = commands.add_parser("user", help="who may connect, each with their own key")
    actions = user.add_subparsers(dest="user_command", required=True)
    add = actions.add_parser("add", help="make a user, and write their key to a private file")
    add.add_argument("name", help="their name")
    add.add_argument("--role", default=None, choices=["owner", "member"], help="default is owner for the first user and member after")
    add.add_argument("--key-file", default=None, help="where to write the key; default is keys/NAME.key in the assistant's folder")
    add.add_argument("--reclaim", action="store_true", help="give a removed name, and the data it left, to this person")
    add.set_defaults(run=lazily("run_user_add"))
    listing = actions.add_parser("list", help="show who may connect")
    listing.set_defaults(run=lazily("run_user_list"))
    remove = actions.add_parser("remove", help="end a user's access now; their data stays")
    remove.add_argument("name", help="their name")
    remove.set_defaults(run=lazily("run_user_remove"))

    make = commands.add_parser("backup", help="write an encrypted copy of everything to one file")
    make.add_argument("file", help="where to write it")
    make.add_argument("--passphrase-file", required=True, help="a file holding the passphrase")
    make.set_defaults(run=lazily("run_backup"))
    back = commands.add_parser("restore", help="replace this install's state with a backup")
    back.add_argument("file", help="the backup")
    back.add_argument("--passphrase-file", required=True, help="a file holding the passphrase")
    back.add_argument("--force", action="store_true", help="replace what is already here")
    back.set_defaults(run=lazily("run_restore"))
