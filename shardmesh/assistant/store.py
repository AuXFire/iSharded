"""The assistant's state: one SQLite file, and a record that shows tampering.

Each part owns its own tables and migrates them here, so parts can change
without touching one another. Everything durable lives in this file, which makes
backup and restore a single copy.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

BUSY_MILLISECONDS = 10_000  # how long to wait on another process
TURN_SECONDS = 60.0  # how long to wait behind other writers here; every write is short, so this means something is stuck
GENESIS = "0" * 64


def default_home() -> Path:
    configured = os.environ.get("SHARDMESH_ASSISTANT_HOME")
    if configured:
        return Path(configured)
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "ShardMesh" / "assistant" if base else Path.home() / ".local" / "share" / "shardmesh" / "assistant"


class Turns:
    """First come, first served.

    SQLite gives a waiting writer no turn. It sleeps and retries, and loses to anyone
    writing back to back: with six conversations, ten tasks, and an index running,
    a writer waited 11 s and two failed as locked. A plain lock would not help, since
    the thread that just let go can take it again before the next one wakes.
    """

    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.queue: deque[threading.Event] = deque()
        self.taken = False

    def take(self, seconds: float) -> bool:
        with self.guard:
            if not self.taken:
                self.taken = True
                return True
            mine = threading.Event()
            self.queue.append(mine)
        if mine.wait(seconds):
            return True
        with self.guard:
            if mine.is_set():  # handed over just as the time ran out
                return True
            self.queue.remove(mine)
            return False

    def give_back(self) -> None:
        with self.guard:
            if self.queue:
                self.queue.popleft().set()  # handed straight over, so nobody can step in between
            else:
                self.taken = False


class Store:
    """One connection per thread, in WAL mode so readers never wait on a writer.

    Writers in this process queue in ``turns``, then wait on SQLite only for another
    process. A write goes through ``transaction`` or ``execute``, or it skips the queue.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.turns = Turns()
        self.execute("CREATE TABLE IF NOT EXISTS schema_versions (module TEXT PRIMARY KEY, version INTEGER NOT NULL)")

    @property
    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=BUSY_MILLISECONDS / 1000, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")  # a finished step must survive a power cut
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA busy_timeout={BUSY_MILLISECONDS}")
            self._local.connection = connection
            self._local.depth = 0
        return connection

    @contextmanager
    def _turn(self) -> Iterator[None]:
        if not self.turns.take(TURN_SECONDS):
            raise sqlite3.OperationalError("database is locked: too many writers were waiting")
        try:
            yield
        finally:
            self.turns.give_back()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All or nothing. Nested calls join the outer transaction."""
        connection = self.connection
        if self._local.depth > 0:
            self._local.depth += 1
            try:
                yield connection
            finally:
                self._local.depth -= 1
            return
        with self._turn():
            connection.execute("BEGIN IMMEDIATE")
            self._local.depth = 1
            try:
                yield connection
            except BaseException:
                self._local.depth = 0
                connection.execute("ROLLBACK")
                raise
            self._local.depth = 0
            connection.execute("COMMIT")

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        """Run a statement and return the new row's id."""
        connection = self.connection
        if self._local.depth > 0:
            return connection.execute(sql, parameters).lastrowid or 0
        with self._turn():
            return connection.execute(sql, parameters).lastrowid or 0

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, parameters).fetchall()

    def one(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, parameters).fetchone()

    def migrate(self, module: str, steps: Sequence[str]) -> None:
        """Bring a part's tables up to date. Only ever add to the end of steps."""
        with self.transaction() as connection:
            row = connection.execute("SELECT version FROM schema_versions WHERE module = ?", (module,)).fetchone()
            version = row["version"] if row else 0
            if version > len(steps):
                raise RuntimeError(f"{self.path} has {module} at version {version}, newer than this code")
            for step in steps[version:]:
                for statement in split_statements(step):
                    connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_versions (module, version) VALUES (?, ?) "
                "ON CONFLICT(module) DO UPDATE SET version = excluded.version",
                (module, len(steps)),
            )

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None


def split_statements(script: str) -> Iterator[str]:
    """Run a script inside a transaction, which executescript would commit."""
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                yield statement
            statement = ""
    if statement.strip():
        yield statement


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


AUDIT_SCHEMA = [
    """
    CREATE TABLE audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        at REAL NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        subject TEXT NOT NULL,
        detail TEXT NOT NULL,
        task_id TEXT,
        previous TEXT NOT NULL,
        hash TEXT NOT NULL
    );
    CREATE INDEX audit_actor ON audit (actor, id);
    CREATE INDEX audit_task ON audit (task_id, id);
    """,
]


class AuditTrail:
    """Who did what. Each row's hash covers the one before, so edits show."""

    def __init__(self, store: Store, clock: Any = time.time) -> None:
        self.store = store
        self.clock = clock
        store.migrate("audit", AUDIT_SCHEMA)

    @staticmethod
    def _hash(previous: str, at: float, actor: str, action: str, subject: str, detail: str, task_id: str | None) -> str:
        body = canonical([previous, at, actor, action, subject, detail, task_id])
        return hashlib.sha256(body.encode()).hexdigest()

    def record(self, actor: str, action: str, subject: str = "", detail: dict[str, Any] | None = None, task_id: str | None = None) -> int:
        text = canonical(detail or {})
        at = float(self.clock())
        with self.store.transaction():
            row = self.store.one("SELECT hash FROM audit ORDER BY id DESC LIMIT 1")
            previous = row["hash"] if row else GENESIS
            return self.store.execute(
                "INSERT INTO audit (at, actor, action, subject, detail, task_id, previous, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (at, actor, action, subject, text, task_id, previous, self._hash(previous, at, actor, action, subject, text, task_id)),
            )

    def verify(self) -> int | None:
        """The first row that does not match its chain, or None if all is intact."""
        previous = GENESIS
        for row in self.store.query("SELECT * FROM audit ORDER BY id"):
            expected = self._hash(previous, row["at"], row["actor"], row["action"], row["subject"], row["detail"], row["task_id"])
            if row["previous"] != previous or row["hash"] != expected:
                return row["id"]
            previous = row["hash"]
        return None

    def read(self, since: int = 0, actor: str | None = None, task_id: str | None = None, limit: int = 200, newest: bool = False) -> list[dict[str, Any]]:
        """Oldest first. ``newest`` takes the last ``limit`` rows instead of the first."""
        clauses, parameters = ["id > ?"], [since]
        if actor:
            clauses.append("actor = ?")
            parameters.append(actor)
        if task_id:
            clauses.append("task_id = ?")
            parameters.append(task_id)
        order = "DESC" if newest else "ASC"
        rows = self.store.query(f"SELECT * FROM audit WHERE {' AND '.join(clauses)} ORDER BY id {order} LIMIT ?", [*parameters, limit])
        ordered = reversed(rows) if newest else rows
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in ordered]
