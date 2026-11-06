"""Small things the memory, model, and conversation parts share.

This is not a part: it attaches nothing and decides nothing.
"""
from __future__ import annotations

import argparse
import re
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


def actor(user: str) -> str:
    return f"user:{user}"


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


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Plain aligned columns for the command line."""
    if not rows:
        return ""
    cells = [[str(row.get(column, "")) for column in columns] for row in rows]
    widths = [max(len(column), *(len(line[index]) for line in cells)) for index, column in enumerate(columns)]
    lines = ["  ".join(column.ljust(width) for column, width in zip(columns, widths)).rstrip()]
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in cells]
    return "\n".join(lines)
