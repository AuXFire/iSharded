"""What the assistant remembers, and how far each memory can be believed.

Two tables. ``messages`` is what was actually said, word for word. ``memories``
is what was kept from it, and every row says who it came from, how far it can be
believed, and how far it may travel. A fact can only be replaced by something at
least as trustworthy, so a model's guess never overwrites what a person said.

Each query below names the user itself. Nothing is filtered afterwards, so a
missing clause shows up as a bug in one place and not as a leak somewhere else.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from .contracts import ContextItem, Provenance, Reach, Trust
from .mind import actor, add_reach_option, add_user_option, emit, plainly, reach_named, search_terms

KINDS = ("episodic", "semantic", "procedural", "user", "system")
RANK = {Trust.EXTERNAL: 0, Trust.MODEL: 1, Trust.OWNER: 2, Trust.VERIFIED: 3}
TRUST_WEIGHT = {Trust.VERIFIED: 1.3, Trust.OWNER: 1.2, Trust.MODEL: 0.8, Trust.EXTERNAL: 0.5}
RECENCY_HALF_LIFE_DAYS = 30.0
SECONDS_PER_DAY = 86_400.0
# bm25 cannot see trust or age, so ask it for more than will be returned.
SEARCH_POOL = 4

SCHEMA = [
    """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        turn_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        model TEXT,
        provider TEXT,
        tool TEXT,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        tainted INTEGER NOT NULL DEFAULT 0,
        reach INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL
    );
    CREATE INDEX messages_conversation ON messages (user, conversation_id, id);
    CREATE INDEX messages_turn ON messages (user, turn_id, id);
    CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT NOT NULL,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        subject TEXT,
        source TEXT NOT NULL,
        source_id TEXT NOT NULL,
        author TEXT NOT NULL,
        trust TEXT NOT NULL,
        reach INTEGER NOT NULL,
        captured_at REAL NOT NULL,
        superseded_by INTEGER,
        expires_at REAL,
        confirmed_at REAL
    );
    CREATE INDEX memories_user ON memories (user, superseded_by, id);
    CREATE UNIQUE INDEX memories_one_current ON memories (user, subject) WHERE subject IS NOT NULL AND superseded_by IS NULL;
    CREATE VIRTUAL TABLE memories_fts USING fts5(text, subject, content='memories', content_rowid='id', tokenize='porter unicode61');
    CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='id', tokenize='porter unicode61');
    CREATE TRIGGER memories_added AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts (rowid, text, subject) VALUES (new.id, new.text, coalesce(new.subject, ''));
    END;
    CREATE TRIGGER memories_removed AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts (memories_fts, rowid, text, subject) VALUES ('delete', old.id, old.text, coalesce(old.subject, ''));
    END;
    CREATE TRIGGER memories_changed AFTER UPDATE OF text, subject ON memories BEGIN
        INSERT INTO memories_fts (memories_fts, rowid, text, subject) VALUES ('delete', old.id, old.text, coalesce(old.subject, ''));
        INSERT INTO memories_fts (rowid, text, subject) VALUES (new.id, new.text, coalesce(new.subject, ''));
    END;
    CREATE TRIGGER messages_added AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, content) VALUES (new.id, new.content);
    END;
    CREATE TRIGGER messages_removed AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts (messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    END;
    """,
]


class MemoryNotFound(LookupError):
    """Also what another user's memory looks like, so its existence is not revealed."""


class MemoryRefused(ValueError):
    pass


def memory_item(row: Any) -> ContextItem:
    return ContextItem(
        row["text"],
        Provenance(
            source="memory",
            source_id=f"memory:{row['id']}",
            author=row["author"],
            trust=Trust(row["trust"]),
            captured_at=row["captured_at"],
            reach=Reach(row["reach"]),
        ),
    )


def _subject(subject: str | None) -> str | None:
    cleaned = " ".join(subject.split()).casefold() if subject else ""
    return cleaned or None


class MemoryStore:
    """Implements contracts.Memory, and keeps the conversation record beside it."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("memory", SCHEMA)

    # What was said

    def record_message(
        self,
        user: str,
        conversation_id: str,
        role: str,
        content: str,
        *,
        turn_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        tool: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tainted: bool = False,
        reach: Reach = Reach.POOL,
    ) -> int:
        return self.store.execute(
            "INSERT INTO messages (user, conversation_id, turn_id, role, content, model, provider, tool, "
            "prompt_tokens, completion_tokens, tainted, reach, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user, conversation_id, turn_id, role, content, model, provider, tool,
                prompt_tokens, completion_tokens, int(tainted), int(reach), self.assistant.clock(),
            ),
        )

    def recent(self, user: str, conversation_id: str, limit: int = 40) -> list[dict[str, Any]]:
        """The latest messages of one conversation, oldest first."""
        rows = self.store.query(
            "SELECT * FROM messages WHERE user = ? AND conversation_id = ? ORDER BY id DESC LIMIT ?", (user, conversation_id, limit)
        )
        return [dict(row) for row in reversed(rows)]

    def turn_messages(self, user: str, turn_id: str) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM messages WHERE user = ? AND turn_id = ? ORDER BY id", (user, turn_id))
        return [dict(row) for row in rows]

    def message(self, user: str, message_id: int) -> dict[str, Any] | None:
        row = self.store.one("SELECT * FROM messages WHERE user = ? AND id = ?", (user, message_id))
        return dict(row) if row else None

    def conversations(self, user: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.query(
            "SELECT conversation_id, count(*) AS messages, max(created_at) AS last_at FROM messages "
            "WHERE user = ? GROUP BY conversation_id ORDER BY last_at DESC LIMIT ?",
            (user, limit),
        )
        return [dict(row) for row in rows]

    def search_messages(self, user: str, query: str, limit: int = 8, conversation_id: str | None = None) -> list[dict[str, Any]]:
        terms = search_terms(query)
        if not terms:
            return []
        clause, parameters = "", [terms, user]
        if conversation_id:
            clause = "AND m.conversation_id = ?"
            parameters.append(conversation_id)
        rows = self.store.query(
            "SELECT m.* FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
            f"WHERE messages_fts MATCH ? AND m.user = ? {clause} ORDER BY bm25(messages_fts) LIMIT ?",
            [*parameters, limit],
        )
        return [dict(row) for row in rows]

    def forget_conversation(self, user: str, conversation_id: str) -> int:
        self._secure()
        with self.store.transaction():
            cursor = self.store.connection.execute("DELETE FROM messages WHERE user = ? AND conversation_id = ?", (user, conversation_id))
            removed = cursor.rowcount
            if removed:
                self.assistant.audit.record(actor(user), "memory.forget_conversation", f"conversation:{conversation_id}", {"messages": removed})
        if not removed:
            raise MemoryNotFound(f"There is no conversation {conversation_id}")
        self._scrub("messages_fts")
        return removed

    # What was kept

    def remember(
        self,
        user: str,
        kind: str,
        text: str,
        provenance: Provenance,
        subject: str | None = None,
        expires_at: float | None = None,
    ) -> str:
        """Keep a fact and return its id.

        With a subject, it replaces the current fact on that subject if it is at
        least as trustworthy. If it is not, it is kept for history already
        superseded, so a person can look at it and confirm it, and search never
        shows it.
        """
        if kind not in KINDS:
            raise ValueError(f"{kind!r} is not a kind of memory. Use one of: {', '.join(KINDS)}")
        text = text.strip()
        if not text:
            raise ValueError("A memory cannot be empty")
        subject = _subject(subject)
        with self.store.transaction():
            current = self._current(user, subject)
            row_id = self.store.execute(
                "INSERT INTO memories (user, kind, text, subject, source, source_id, author, trust, reach, captured_at, superseded_by, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user, kind, text, subject, provenance.source, provenance.source_id, provenance.author,
                    provenance.trust.value, int(provenance.reach), provenance.captured_at,
                    current["id"] if current else None, expires_at,
                ),
            )
            kept = current is None or self._outranks(provenance.trust, current)
            if current and kept:
                self._replace(user, current["id"], row_id)
            self.assistant.audit.record(
                actor(user), "memory.remember", f"memory:{row_id}",
                {"kind": kind, "trust": provenance.trust.value, "author": provenance.author, "kept": kept},
            )
        return str(row_id)

    def _current(self, user: str, subject: str | None) -> Any:
        if subject is None:
            return None
        return self.store.one("SELECT * FROM memories WHERE user = ? AND subject = ? AND superseded_by IS NULL", (user, subject))

    def _outranks(self, trust: Trust, current: Any) -> bool:
        expired = current["expires_at"] is not None and current["expires_at"] <= self.assistant.clock()
        return expired or RANK[trust] >= RANK[Trust(current["trust"])]

    def _replace(self, user: str, old_id: int, new_id: int) -> None:
        # In this order, so one current row per subject holds at every step.
        self.store.execute("UPDATE memories SET superseded_by = ? WHERE id = ? AND user = ?", (new_id, old_id, user))
        self.store.execute("UPDATE memories SET superseded_by = NULL WHERE id = ? AND user = ?", (new_id, user))

    def supersede(self, user: str, memory_id: str | int, text: str, provenance: Provenance) -> str:
        """Replace one named memory, which only something as trustworthy may do."""
        with self.store.transaction():
            old = self._row(user, memory_id)
            if old["superseded_by"] is not None:
                raise MemoryRefused(f"Memory {old['id']} was already replaced by {old['superseded_by']}. Replace that one instead")
            if not self._outranks(provenance.trust, old):
                raise MemoryRefused(
                    f"Memory {old['id']} has trust {old['trust']} and a {provenance.trust.value} fact cannot replace it. "
                    "Say it yourself, or `confirm` the new fact"
                )
            if old["subject"] is not None:
                return self.remember(user, old["kind"], text, provenance, subject=old["subject"], expires_at=old["expires_at"])
            new_id = self.remember(user, old["kind"], text, provenance, expires_at=old["expires_at"])
            self.store.execute("UPDATE memories SET superseded_by = ? WHERE id = ? AND user = ?", (int(new_id), old["id"], user))
            return new_id

    def confirm(self, user: str, memory_id: str | int) -> None:
        """A person vouches for what a model inferred, which makes it theirs."""
        with self.store.transaction():
            row = self._row(user, memory_id)
            if row["trust"] != Trust.MODEL.value:
                raise MemoryRefused(
                    f"Memory {row['id']} has trust {row['trust']}. Only a model's inference can be confirmed. "
                    "To vouch for text from a file or a page, `remember` it in your own words"
                )
            current = self._current(user, row["subject"])
            if row["superseded_by"] is not None and row["subject"] is None:
                raise MemoryRefused(f"Memory {row['id']} was replaced by {row['superseded_by']}, so there is nothing to confirm")
            if current is not None and current["id"] != row["id"]:
                self._replace(user, current["id"], row["id"])
            self.store.execute(
                "UPDATE memories SET trust = ?, confirmed_at = ? WHERE id = ? AND user = ?",
                (Trust.OWNER.value, self.assistant.clock(), row["id"], user),
            )
            self.assistant.audit.record(actor(user), "memory.confirm", f"memory:{row['id']}", {"from": Trust.MODEL.value, "to": Trust.OWNER.value})

    def forget(self, user: str, memory_id: str | int) -> int:
        """Delete a memory and every older version of it, from the row to the disk.

        A deleted row leaves its words in old index segments, free pages, and the
        write-ahead log, so all three are cleared as well.
        """
        self._secure()
        with self.store.transaction():
            row = self._row(user, memory_id)
            ids = [
                item["id"]
                for item in self.store.query(
                    "WITH RECURSIVE chain (id) AS ("
                    "SELECT id FROM memories WHERE id = ? AND user = ? "
                    "UNION SELECT m.id FROM memories m JOIN chain c ON m.superseded_by = c.id WHERE m.user = ?"
                    ") SELECT id FROM chain",
                    (row["id"], user, user),
                )
            ]
            marks = ",".join("?" * len(ids))
            self.store.execute(f"DELETE FROM memories WHERE user = ? AND id IN ({marks})", [user, *ids])
            self.assistant.audit.record(actor(user), "memory.forget", f"memory:{row['id']}", {"rows": len(ids)})
        self._scrub("memories_fts")
        return len(ids)

    def _secure(self) -> None:
        self.store.connection.execute("PRAGMA secure_delete=ON")

    def _scrub(self, index: str) -> None:
        self.store.execute(f"INSERT INTO {index} ({index}) VALUES ('optimize')")
        self.store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()

    def _row(self, user: str, memory_id: str | int) -> Any:
        try:
            number = int(memory_id)
        except (TypeError, ValueError):
            raise MemoryNotFound(f"There is no memory {memory_id}") from None
        row = self.store.one("SELECT * FROM memories WHERE id = ? AND user = ?", (number, user))
        if row is None:
            raise MemoryNotFound(f"There is no memory {memory_id}")
        return row

    def holds(self, user: str, text: str) -> bool:
        """Whether this exact fact is already current."""
        return self.store.one("SELECT id FROM memories WHERE user = ? AND text = ? AND superseded_by IS NULL", (user, text)) is not None

    def get(self, user: str, memory_id: str | int) -> dict[str, Any]:
        return dict(self._row(user, memory_id))

    def history(self, user: str, memory_id: str | int) -> list[dict[str, Any]]:
        """A memory and what it replaced, newest first."""
        row = self._row(user, memory_id)
        rows = self.store.query(
            "WITH RECURSIVE chain (id) AS ("
            "SELECT id FROM memories WHERE id = ? AND user = ? "
            "UNION SELECT m.id FROM memories m JOIN chain c ON m.superseded_by = c.id WHERE m.user = ?"
            ") SELECT memories.* FROM memories JOIN chain ON chain.id = memories.id WHERE memories.user = ? ORDER BY memories.id DESC",
            (row["id"], user, user, user),
        )
        return [dict(item) for item in rows]

    def entries(self, user: str, limit: int = 100, offset: int = 0, include_superseded: bool = False) -> list[dict[str, Any]]:
        current = "" if include_superseded else "AND superseded_by IS NULL AND (expires_at IS NULL OR expires_at > ?)"
        parameters: list[Any] = [user] if include_superseded else [user, self.assistant.clock()]
        rows = self.store.query(f"SELECT * FROM memories WHERE user = ? {current} ORDER BY id DESC LIMIT ? OFFSET ?", [*parameters, limit, offset])
        return [dict(row) for row in rows]

    def search(self, user: str, query: str, limit: int = 8, kinds: tuple[str, ...] | None = None) -> list[ContextItem]:
        """The best current memories, by relevance, trust, and how recent they are."""
        terms = search_terms(query)
        if not terms:
            return []
        now = self.assistant.clock()
        clause, parameters = "", [terms, user, now]
        if kinds:
            clause = f"AND m.kind IN ({','.join('?' * len(kinds))})"
            parameters += list(kinds)
        rows = self.store.query(
            "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.user = ? AND m.superseded_by IS NULL AND (m.expires_at IS NULL OR m.expires_at > ?) "
            f"{clause} ORDER BY rank LIMIT ?",
            [*parameters, limit * SEARCH_POOL],
        )
        scored = sorted(((_score(row, now), row["id"], row) for row in rows), key=lambda entry: (-entry[0], -entry[1]))
        return [replace(memory_item(row), score=score) for score, _, row in scored[:limit]]

    def export(self, user: str) -> dict[str, Any]:
        """Everything held for one person, in a form they can take away."""
        memories = self.store.query("SELECT * FROM memories WHERE user = ? ORDER BY id", (user,))
        messages = self.store.query("SELECT * FROM messages WHERE user = ? ORDER BY id", (user,))
        return {
            "user": user,
            "exported_at": self.assistant.clock(),
            "memories": [dict(row) for row in memories],
            "messages": [dict(row) for row in messages],
        }


def _score(row: Any, now: float) -> float:
    relevance = -row["rank"]  # bm25 is negative here, and lower means better
    age_days = max(now - row["captured_at"], 0.0) / SECONDS_PER_DAY
    recency = 0.5 + 0.5 * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return relevance * TRUST_WEIGHT[Trust(row["trust"])] * recency


def attach(assistant: Any) -> None:
    assistant.memory = MemoryStore(assistant)


def add_commands(commands: Any) -> None:
    remember = commands.add_parser("remember", help="keep a fact, as something you said")
    remember.add_argument("text", help="the fact")
    remember.add_argument("--kind", default="user", choices=KINDS, help="what sort of memory this is; default is user")
    remember.add_argument("--subject", default=None, help="what it is about; a newer fact on the same subject replaces it")
    remember.add_argument("--expires-days", type=float, default=None, help="stop using it after this many days")
    add_reach_option(remember, "pool", "this fact")
    add_user_option(remember)
    remember.set_defaults(run=plainly(run_remember))

    recall = commands.add_parser("recall", help="search what is remembered, showing where each memory came from")
    recall.add_argument("query", help="what to look for")
    recall.add_argument("--limit", type=int, default=8, help="how many results to show")
    recall.add_argument("--said", action="store_true", help="also search what was actually said in conversations")
    add_user_option(recall)
    recall.set_defaults(run=plainly(run_recall))

    forget = commands.add_parser("forget", help="delete a memory for good, with its older versions")
    forget.add_argument("id", nargs="?", default=None, help="the memory's number, from recall")
    forget.add_argument("--conversation", default=None, help="delete this whole conversation instead")
    add_user_option(forget)
    forget.set_defaults(run=plainly(run_forget))

    confirm = commands.add_parser("confirm", help="vouch for something a model inferred, so it counts as yours")
    confirm.add_argument("id", help="the memory's number, from recall")
    add_user_option(confirm)
    confirm.set_defaults(run=plainly(run_confirm))

    export = commands.add_parser("export", help="write everything held for one person as JSON")
    export.add_argument("file", help="where to write it")
    add_user_option(export)
    export.set_defaults(run=plainly(run_export))


def run_remember(assistant: Any, args: argparse.Namespace) -> int:
    now = assistant.clock()
    provenance = Provenance("user", "command line", args.user, Trust.OWNER, now, reach_named(args.reach))
    expires = now + args.expires_days * SECONDS_PER_DAY if args.expires_days is not None else None
    memory_id = assistant.memory.remember(args.user, args.kind, args.text, provenance, subject=args.subject, expires_at=expires)
    emit(f"Remembered as {memory_id}.")
    return 0


def run_recall(assistant: Any, args: argparse.Namespace) -> int:
    items = assistant.memory.search(args.user, args.query, args.limit)
    for item in items:
        origin = item.provenance
        emit(f"{origin.source_id.split(':')[-1]:>6}  [{origin.trust.value}, {origin.reach.name.lower()}, by {origin.author}]  {item.text}")
    if not items:
        emit("Nothing remembered matches that.")
    if args.said:
        for row in assistant.memory.search_messages(args.user, args.query, args.limit):
            emit(f"said   [{row['role']}, conversation {row['conversation_id']}]  {row['content']}")
    return 0


def run_forget(assistant: Any, args: argparse.Namespace) -> int:
    if args.conversation:
        emit(f"Deleted {assistant.memory.forget_conversation(args.user, args.conversation)} messages.")
        return 0
    if args.id is None:
        emit("Name a memory to forget, or pass --conversation.")
        return 2
    emit(f"Forgotten: {assistant.memory.forget(args.user, args.id)} rows deleted.")
    return 0


def run_confirm(assistant: Any, args: argparse.Namespace) -> int:
    assistant.memory.confirm(args.user, args.id)
    emit(f"Memory {args.id} now counts as yours.")
    return 0


def run_export(assistant: Any, args: argparse.Namespace) -> int:
    Path(args.file).write_text(json.dumps(assistant.memory.export(args.user), indent=2, ensure_ascii=False), encoding="utf-8")
    emit(f"Wrote {args.file}.")
    return 0
