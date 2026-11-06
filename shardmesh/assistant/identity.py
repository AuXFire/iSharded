"""Who the assistant is: a name, a way of speaking, and what it has promised.

All of this is presentation. It shapes how the assistant sounds and never what
it may do, so this module knows nothing about permissions and stays that way.
The same words reach every model, so changing models changes nobody's assistant.
"""
from __future__ import annotations

import argparse
from typing import Any

from .mind import actor, add_user_option, emit, LOCAL_USER, plainly

DEFAULT_NAME = "Mesh"
DEFAULT_STYLE = "Plain, direct, and warm. Says what it knows, what it does not, and how sure it is."
DEFAULT_VOICE = ""
MAX_NAME = 64
MAX_TEXT = 2000
SETTINGS = {"name": DEFAULT_NAME, "style": DEFAULT_STYLE, "voice": DEFAULT_VOICE}

SCHEMA = [
    """
    CREATE TABLE identity_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        changed_at REAL NOT NULL,
        changed_by TEXT NOT NULL
    );
    CREATE TABLE identity_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        old TEXT,
        new TEXT NOT NULL,
        changed_at REAL NOT NULL,
        changed_by TEXT NOT NULL
    );
    CREATE TABLE commitments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        text TEXT NOT NULL,
        made_at REAL NOT NULL,
        made_by TEXT NOT NULL,
        withdrawn_at REAL,
        withdrawn_by TEXT
    );
    CREATE INDEX commitments_user ON commitments (user, withdrawn_at);
    CREATE TABLE relationships (
        user TEXT PRIMARY KEY,
        note TEXT NOT NULL,
        changed_at REAL NOT NULL,
        changed_by TEXT NOT NULL
    );
    CREATE TABLE relationship_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT NOT NULL,
        note TEXT NOT NULL,
        changed_at REAL NOT NULL,
        changed_by TEXT NOT NULL
    );
    CREATE INDEX relationship_history_user ON relationship_history (user, id);
    """,
]


class Identity:
    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("identity", SCHEMA)

    def _setting(self, key: str) -> str:
        row = self.store.one("SELECT value FROM identity_settings WHERE key = ?", (key,))
        return row["value"] if row else SETTINGS[key]

    @property
    def name(self) -> str:
        return self._setting("name")

    @property
    def style(self) -> str:
        return self._setting("style")

    @property
    def voice(self) -> str:
        return self._setting("voice")

    def set(self, by: str, **changes: str | None) -> None:
        """Change the name, style, or voice. Every change is kept."""
        unknown = set(changes) - set(SETTINGS)
        if unknown:
            raise ValueError(f"Only {', '.join(sorted(SETTINGS))} can be set, not {', '.join(sorted(unknown))}")
        with self.store.transaction():
            for key, value in changes.items():
                if value is not None:
                    self._change(key, _checked(key, value), by)

    def _change(self, key: str, value: str, by: str) -> None:
        old = self._setting(key)
        if value == old:
            return
        now = self.assistant.clock()
        self.store.execute(
            "INSERT INTO identity_settings (key, value, changed_at, changed_by) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, changed_at = excluded.changed_at, changed_by = excluded.changed_by",
            (key, value, now, by),
        )
        self.store.execute(
            "INSERT INTO identity_history (key, old, new, changed_at, changed_by) VALUES (?, ?, ?, ?, ?)", (key, old, value, now, by)
        )
        self.assistant.audit.record(actor(by), "identity.set", key)

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM identity_history ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def commit(self, text: str, by: str, user: str | None = None) -> int:
        """Record a promise, to one person or to everyone. Withdrawing keeps the record."""
        text = _text(text, "A commitment")
        with self.store.transaction():
            row_id = self.store.execute(
                "INSERT INTO commitments (user, text, made_at, made_by) VALUES (?, ?, ?, ?)", (user, text, self.assistant.clock(), by)
            )
            self.assistant.audit.record(actor(by), "identity.commit", f"commitment:{row_id}", {"to": user or "everyone"})
        return row_id

    def withdraw(self, commitment_id: int, by: str) -> None:
        with self.store.transaction():
            cursor = self.store.connection.execute(
                "UPDATE commitments SET withdrawn_at = ?, withdrawn_by = ? WHERE id = ? AND withdrawn_at IS NULL",
                (self.assistant.clock(), by, commitment_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"There is no open commitment {commitment_id}. `identity show` lists them")
            self.assistant.audit.record(actor(by), "identity.withdraw", f"commitment:{commitment_id}")

    def commitments(self, user: str | None = None, include_withdrawn: bool = False) -> list[dict[str, Any]]:
        """Promises that apply to this person: their own, and those made to everyone."""
        withdrawn = "" if include_withdrawn else "AND withdrawn_at IS NULL"
        rows = self.store.query(f"SELECT * FROM commitments WHERE (user IS NULL OR user = ?) {withdrawn} ORDER BY id", (user,))
        return [dict(row) for row in rows]

    def relationship(self, user: str) -> str:
        row = self.store.one("SELECT note FROM relationships WHERE user = ?", (user,))
        return row["note"] if row else ""

    def set_relationship(self, user: str, note: str, by: str) -> None:
        note = _text(note, "A relationship note")
        now = self.assistant.clock()
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO relationships (user, note, changed_at, changed_by) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user) DO UPDATE SET note = excluded.note, changed_at = excluded.changed_at, changed_by = excluded.changed_by",
                (user, note, now, by),
            )
            self.store.execute(
                "INSERT INTO relationship_history (user, note, changed_at, changed_by) VALUES (?, ?, ?, ?)", (user, note, now, by)
            )
            self.assistant.audit.record(actor(by), "identity.relationship", f"user:{user}")

    def relationship_history(self, user: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM relationship_history WHERE user = ? ORDER BY id DESC LIMIT ?", (user, limit))
        return [dict(row) for row in rows]

    def system_prompt(self, user: str) -> str:
        """Who the assistant is to this person. Nothing here names a model."""
        lines = [
            f"You are {self.name}, a persistent assistant. You are the same {self.name} whichever model is running you, "
            "and you carry your memory, your promises, and your history from one conversation to the next.",
            f"How you speak: {self.style}",
            f"You are talking with {user}.",
        ]
        note = self.relationship(user)
        if note:
            lines.append(f"What you know of your relationship with {user}: {note}")
        promises = [row["text"] for row in self.commitments(user)]
        if promises:
            lines.append("Promises you have made and keep:")
            lines += [f"- {promise}" for promise in promises]
        return "\n".join(lines)

    def describe(self, user: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "style": self.style,
            "voice": self.voice,
            "commitments": [{"id": row["id"], "text": row["text"], "to": row["user"] or "everyone"} for row in self.commitments(user)],
            "relationship": self.relationship(user),
        }


def _text(value: str, what: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{what} cannot be empty")
    if len(value) > MAX_TEXT:
        raise ValueError(f"{what} is limited to {MAX_TEXT} characters")
    return value


def _checked(key: str, value: str) -> str:
    if key == "voice":  # a voice can be cleared
        return value.strip()
    value = _text(value, f"The {key}")
    if key == "name" and (len(value) > MAX_NAME or "\n" in value or "\r" in value):
        raise ValueError(f"A name is one line of at most {MAX_NAME} characters")
    return value


def attach(assistant: Any) -> None:
    assistant.identity = Identity(assistant)


def add_commands(commands: Any) -> None:
    parser = commands.add_parser("identity", help="see or change who the assistant is; this never changes a permission")
    actions = parser.add_subparsers(dest="identity_command", required=True)
    show = actions.add_parser("show", help="show the name, style, promises, and relationship")
    add_user_option(show)
    show.add_argument("--prompt", action="store_true", help="also show the exact text models receive")
    show.set_defaults(run=plainly(run_show))
    change = actions.add_parser("set", help="change the name, style, voice, promises, or a relationship note")
    add_user_option(change)
    change.add_argument("--name", default=None, help="what the assistant is called")
    change.add_argument("--style", default=None, help="how it speaks")
    change.add_argument("--voice", default=None, help="which voice it uses, if speech is installed")
    change.add_argument("--commit", default=None, help="add a promise, kept with its history")
    change.add_argument("--to-user", action="store_true", help="make --commit a promise to --user alone, not to everyone")
    change.add_argument("--withdraw", type=int, default=None, help="withdraw the promise with this number; it stays in the history")
    change.add_argument("--relationship", default=None, help="set the relationship note for --user")
    change.set_defaults(run=plainly(run_set))


def run_show(assistant: Any, args: argparse.Namespace) -> int:
    identity = assistant.identity
    emit(f"Name:  {identity.name}")
    emit(f"Style: {identity.style}")
    if identity.voice:
        emit(f"Voice: {identity.voice}")
    for row in identity.commitments(args.user):
        emit(f"Promise {row['id']}: {row['text']} (to {row['user'] or 'everyone'})")
    note = identity.relationship(args.user)
    if note:
        emit(f"With {args.user}: {note}")
    if args.prompt:
        emit("\n" + identity.system_prompt(args.user))
    return 0


def run_set(assistant: Any, args: argparse.Namespace) -> int:
    identity = assistant.identity
    identity.set(LOCAL_USER, name=args.name, style=args.style, voice=args.voice)
    if args.commit:
        emit(f"Promise {identity.commit(args.commit, LOCAL_USER, args.user if args.to_user else None)} recorded.")
    if args.withdraw is not None:
        identity.withdraw(args.withdraw, LOCAL_USER)
        emit(f"Promise {args.withdraw} withdrawn.")
    if args.relationship:
        identity.set_relationship(args.user, args.relationship, LOCAL_USER)
    emit(f"{identity.name} is up to date.")
    return 0
