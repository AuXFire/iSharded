"""Small things the memory, model, and conversation parts share.

This is not a part: it attaches nothing and decides nothing.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import functools
import json
import os
import re
import sqlite3
import sys
from typing import Any, TextIO

from .contracts import Reach

# Who the command line acts for. Over HTTP a person's key says who they are.
LOCAL_USER = "owner"

WORD = re.compile(r"\w+", re.UNICODE)
# Dropped from searches, or every memory with the word "my" answers every question.
STOP_WORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in is it its me my "
    "of on or our so that the their them then there these they this to us was we were what when where "
    "which who why will with would you your".split()
)
MAX_SEARCH_WORDS = 32
JSON_ATTEMPTS = 64  # bounded, so a reply of nested braces costs little
MACHINE_ACTORS = ("skill:", "peer:", "task:", "model")  # none of these may commit, push, or promote


def actor(user: str) -> str:
    return f"user:{user}"


def owner_of(assistant: Any) -> str:
    identity = getattr(assistant, "identity", None)
    return getattr(identity, "owner", None) or os.environ.get("ISHARDED_ASSISTANT_USER") or "owner"


def emit(text: str = "", end: str = "\n", stream: TextIO | None = None) -> None:
    """Print anything. Redirected output on Windows uses a code page that cannot
    write every character a model produces, and that once crashed a model load."""
    target = stream or sys.stdout
    try:
        target.write(text + end)
    except UnicodeEncodeError:
        encoding = target.encoding or "ascii"
        target.write((text + end).encode(encoding, "backslashreplace").decode(encoding))
    target.flush()


def plainly(run: Callable[[Any, argparse.Namespace], int]) -> Callable[[Any, argparse.Namespace], int]:
    """A command's own refusals are for a person, so they print as words.

    KeyError and IndexError are left alone: those are bugs, and a bug should be loud.
    """

    @functools.wraps(run)
    def wrapped(assistant: Any, args: argparse.Namespace) -> int:
        try:
            return run(assistant, args)
        except (KeyError, IndexError):
            raise
        except (ValueError, LookupError) as error:
            emit(str(error.args[0] if error.args else error), stream=sys.stderr)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error):
                raise
            emit("Something else is busy writing to the assistant's data. Try again in a moment.", stream=sys.stderr)
        return 1

    return wrapped


def printing(refusal: type[Exception]) -> Callable[[Callable[[Any, Any], int]], Callable[[Any, Any], int]]:
    """A command that prints this kind of refusal, and exits 1, instead of raising it."""

    def wrap(run: Callable[[Any, Any], int]) -> Callable[[Any, Any], int]:
        def guarded(assistant: Any, args: Any) -> int:
            try:
                return run(assistant, args)
            except refusal as error:
                print(error)
                return 1

        return guarded

    return wrap


def reach_named(name: str | Reach) -> Reach:
    if isinstance(name, Reach):
        return name
    try:
        return Reach[str(name).upper()]
    except KeyError:
        choices = ", ".join(item.name.lower() for item in Reach)
        raise ValueError(f"{name!r} is not a reach. Use one of: {choices}") from None


def add_user_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user", default=LOCAL_USER, help=f"whose data this acts on; default is {LOCAL_USER}")


def add_reach_option(parser: argparse.ArgumentParser, default: str, what: str) -> None:
    parser.add_argument(
        "--reach", default=default, choices=[item.name.lower() for item in Reach], help=f"how far {what} may travel; default is {default}"
    )


def search_terms(text: str) -> str:
    """Turn what a person typed into a safe FTS5 query.

    Each word is quoted, so punctuation in a question can never be read as query
    syntax. Common words are dropped unless they are all there is.
    """
    words = list(dict.fromkeys(word.casefold() for word in WORD.findall(text)))
    useful = [word for word in words if word not in STOP_WORDS] or words
    return " OR ".join(f'"{word}"' for word in useful[:MAX_SEARCH_WORDS])


def first_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's reply, fenced or not."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    for _ in range(JSON_ATTEMPTS):
        if start == -1:
            break
        try:
            return decoder.raw_decode(text, start)[0]
        except (RecursionError, ValueError):
            start = text.find("{", start + 1)
    return None


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Plain aligned columns for the command line."""
    if not rows:
        return ""
    cells = [[str(row.get(column, "")) for column in columns] for row in rows]
    widths = [max(len(column), *(len(line[index]) for line in cells)) for index, column in enumerate(columns)]
    lines = ["  ".join(column.ljust(width) for column, width in zip(columns, widths)).rstrip()]
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in cells]
    return "\n".join(lines)
