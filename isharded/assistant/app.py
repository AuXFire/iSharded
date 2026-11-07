"""The assistant as one object, built from whichever parts are installed.

A part is a module in this package with an ``attach(assistant)`` function. It
may also define ``add_commands(subparsers)`` for the command line and
``routes(assistant)`` returning a FastAPI router. Parts attach in PARTS order,
so a later part can use an earlier one. Nothing here knows what a part does,
which keeps any one of them replaceable.
"""
from __future__ import annotations

from collections.abc import Callable
import importlib
from pathlib import Path
import time
from types import ModuleType
from typing import Any

from .store import AuditTrail, Store, default_home

PARTS = (
    "identity",
    "memory",
    "knowledge",
    "broker",
    "tools",
    "plugins",
    "sandbox",
    "providers",
    "router",
    "context",
    "conversation",
    "tasks",
    "planner",
    "skills",
    "coder",
    "federation",
    "voice",
)


def installed_parts() -> list[ModuleType]:
    parts = []
    for name in PARTS:
        try:
            parts.append(importlib.import_module(f"{__package__}.{name}"))
        except ModuleNotFoundError as error:
            if error.name != f"{__package__}.{name}":  # a part that exists but lacks something
                raise
    return parts


class Assistant:
    """Owns the state. Models, tools, and peers only provide capabilities to it."""

    def __init__(self, home: Path | str | None = None, clock: Callable[[], float] = time.time) -> None:
        self.home = Path(home) if home else default_home()
        self.clock = clock
        self.store = Store(self.home / "assistant.db")
        self.audit = AuditTrail(self.store, clock)
        self.open_folders: list[Path] = []  # work kept under home that file and git tools may reach
        self.parts = installed_parts()
        for part in self.parts:
            attach = getattr(part, "attach", None)
            if attach:
                attach(self)

    def close(self) -> None:
        for part in reversed(self.parts):
            detach = getattr(part, "detach", None)
            if detach:
                detach(self)
        self.store.close()

    def __enter__(self) -> Assistant:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
