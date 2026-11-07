"""The library: folders the owner authorized, searchable without rereading them.

Everything here is EXTERNAL, because a file can say anything. A search returns
the words with a citation someone can go and check. The walk never follows a
link and never reads a secret, and a folder that has gone missing is left alone,
since an unplugged drive is not a request to delete an index.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
import fnmatch
import hashlib
import os
from pathlib import Path
import re
import stat as stat_module
import time
from typing import Any

import numpy as np

from .contracts import ContextItem, Provenance, Reach, Trust
from .mind import actor, add_reach_option, add_user_option, emit, plainly, reach_named, search_terms

SKIPPED_FOLDERS = frozenset(
    {".git", ".hg", ".svn", ".ssh", ".gnupg", ".aws", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"}
)
SECRET_NAMES = (
    ".env*", "*.key", "*.pem", "id_rsa*", "credentials*",
    "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.p12", "*.pfx", "*.kdbx", ".netrc", ".npmrc", ".pypirc",
)
MARKDOWN = frozenset({".md", ".markdown", ".mdx"})
CODE = frozenset(
    ".py .pyi .js .jsx .ts .tsx .c .h .cc .cpp .hpp .cs .java .kt .go .rs .rb .php .swift .lua .sh .bash .ps1 .bat .sql "
    ".html .css .scss .json .toml .yaml .yml .ini .cfg .xml .gradle .cmake".split()
)
MAX_FILE_BYTES = 32 * 1024 * 1024
SNIFF_BYTES = 8192
MAX_CHUNK_CHARS = 1500
# Writes go in batches: one per file would wait on the disk thousands of times,
# and one for a whole folder would hold up every conversation until it finished.
BATCH_FILES = 200
EMBED_BATCH = 64
RRF_K = 60  # the constant from the paper, which works well without tuning
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
# Windows reparse points that lead somewhere else. Others, such as cloud file
# placeholders, are ordinary files.
LINK_TAGS = frozenset({0xA0000003, 0xA000000C})

SCHEMA = [
    """
    CREATE TABLE library_roots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT NOT NULL,
        path TEXT NOT NULL,
        reach INTEGER NOT NULL,
        added_at REAL NOT NULL,
        added_by TEXT NOT NULL,
        indexed_at REAL,
        UNIQUE (user, path)
    );
    CREATE TABLE library_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        root_id INTEGER NOT NULL REFERENCES library_roots (id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        size INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        kind TEXT NOT NULL,
        indexed_at REAL NOT NULL,
        UNIQUE (root_id, path)
    );
    CREATE TABLE library_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES library_files (id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        heading TEXT NOT NULL DEFAULT '',
        line INTEGER,
        end_line INTEGER,
        page INTEGER,
        text TEXT NOT NULL
    );
    CREATE INDEX library_chunks_file ON library_chunks (file_id);
    CREATE VIRTUAL TABLE library_fts USING fts5(text, heading, content='library_chunks', content_rowid='id', tokenize='porter unicode61');
    CREATE TRIGGER library_added AFTER INSERT ON library_chunks BEGIN
        INSERT INTO library_fts (rowid, text, heading) VALUES (new.id, new.text, new.heading);
    END;
    CREATE TRIGGER library_removed AFTER DELETE ON library_chunks BEGIN
        INSERT INTO library_fts (library_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading);
    END;
    CREATE TABLE library_vectors (
        chunk_id INTEGER PRIMARY KEY REFERENCES library_chunks (id) ON DELETE CASCADE,
        model TEXT NOT NULL,
        vector BLOB NOT NULL
    );
    """,
]


class LibraryRefused(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str = ""
    line: int | None = None
    end_line: int | None = None
    page: int | None = None


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    chunks: int = 0
    embedded: int = 0
    seconds: float = 0.0
    problems: list[str] = field(default_factory=list)

    def merge(self, other: "IndexReport") -> None:
        for name in ("added", "updated", "unchanged", "removed", "skipped", "chunks", "embedded"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.seconds += other.seconds
        self.problems += other.problems


def is_secret(name: str) -> bool:
    lowered = name.casefold()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in SECRET_NAMES)


def is_link(status: os.stat_result) -> bool:
    if stat_module.S_ISLNK(status.st_mode):
        return True
    return getattr(status, "st_reparse_tag", 0) in LINK_TAGS


def inside(root: str, candidate: str) -> bool:
    """Whether a path, with every link and ``..`` resolved, is really in the root."""
    real_root = os.path.normcase(os.path.realpath(root))
    real_path = os.path.normcase(os.path.realpath(candidate))
    try:
        return os.path.commonpath([real_root, real_path]) == real_root
    except ValueError:  # on different drives
        return False


def walk(root: str, unreadable: list[str]) -> Iterator[tuple[str, os.stat_result]]:
    """Every indexable file under a root, as (path relative to it, status).

    Links are never entered. Unreadable folders are reported, so a later step
    does not mistake their contents for deleted files.
    """
    pending = [root]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as scan:
                entries = list(scan)
        except OSError:
            unreadable.append(os.path.relpath(folder, root).replace(os.sep, "/"))
            continue
        for entry in entries:
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if is_link(status) or is_secret(entry.name):
                continue
            if stat_module.S_ISDIR(status.st_mode):
                if entry.name.casefold() not in SKIPPED_FOLDERS:
                    pending.append(entry.path)
            elif stat_module.S_ISREG(status.st_mode):
                yield os.path.relpath(entry.path, root).replace(os.sep, "/"), status


def kind_of(name: str) -> str:
    suffix = os.path.splitext(name)[1].casefold()
    if suffix in MARKDOWN:
        return "markdown"
    if suffix in CODE:
        return "code"
    return "pdf" if suffix == ".pdf" else "text"


def decode(data: bytes) -> str | None:
    """The file as text, or None if it is binary."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if b"\0" in data[:SNIFF_BYTES]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        return text if text.count("�") * 100 <= len(text) else None


def lines_of(text: str) -> list[str]:
    # Not splitlines, which also splits on form feeds and so disagrees with editors.
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def split_long(lines: list[str], first_line: int, limit: int) -> Iterator[tuple[str, int, int]]:
    """Cut a block that is too long at line boundaries, or inside a very long line."""
    group: list[str] = []
    start, size = first_line, 0
    for offset, line in enumerate(lines):
        number = first_line + offset
        if size + len(line) + 1 > limit and group:
            yield "\n".join(group), start, number - 1
            group, start, size = [], number, 0
        while len(line) > limit:
            yield line[:limit], number, number
            line = line[limit:]
        group.append(line)
        size += len(line) + 1
    if "".join(group).strip():
        yield "\n".join(group), start, first_line + len(lines) - 1


def blocks_of(lines: list[str]) -> Iterator[tuple[int, int]]:
    """Runs of lines between blank lines, as 1-based (first, last)."""
    start = None
    for number, line in enumerate(lines, 1):
        if line.strip():
            start = number if start is None else start
        elif start is not None:
            yield start, number - 1
            start = None
    if start is not None:
        yield start, len(lines)


def chunk_blocks(text: str, limit: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Code and plain text: whole blocks, gathered up to a size, with line numbers."""
    lines = lines_of(text)
    chunks: list[Chunk] = []
    first = last = None

    def flush() -> None:
        if first is not None:
            for body, start, end in split_long(lines[first - 1 : last], first, limit):
                chunks.append(Chunk(body, line=start, end_line=end))

    for start, end in blocks_of(lines):
        if first is not None and sum(len(line) + 1 for line in lines[first - 1 : end]) > limit:
            flush()
            first = None
        first = start if first is None else first
        last = end
    flush()
    return chunks


def chunk_markdown(text: str, limit: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """One chunk per heading, ignoring lines in code fences that only look like one."""
    lines = lines_of(text)
    sections: list[tuple[str, int, int]] = []
    heading, start, fenced = "", 1, False
    for number, line in enumerate(lines, 1):
        if FENCE.match(line):
            fenced = not fenced
        match = None if fenced else HEADING.match(line)
        if match:
            if number > start:
                sections.append((heading, start, number - 1))
            heading, start = match.group(1), number
    sections.append((heading, start, len(lines)))
    chunks = []
    for title, first, last in sections:
        for body, begin, end in split_long(lines[first - 1 : last], first, limit):
            if body.strip():
                lead = len(body) - len(body.lstrip("\n"))  # a citation points at the first real line
                chunks.append(Chunk(body.strip("\n"), heading=title, line=begin + lead, end_line=end))
    return chunks


def chunk_pdf(path: str, limit: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    from pypdf import PdfReader

    chunks = []
    for number, page in enumerate(PdfReader(path).pages, 1):
        text = (page.extract_text() or "").strip()
        for start in range(0, len(text), limit):
            chunks.append(Chunk(text[start : start + limit], page=number))
    return chunks


def citation(root: str, relative: str, line: int | None, page: int | None) -> str:
    path = os.path.join(root, relative.replace("/", os.sep))
    if page is not None:
        return f"{path}#page={page}"
    return f"{path}:{line}" if line is not None else path


def fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: two rankings agree without their scores being comparable."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return scores


class Library:
    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("knowledge", SCHEMA)

    # Roots

    def add(self, user: str, path: str | Path, reach: Reach | str = Reach.POOL, by: str | None = None) -> int:
        """Authorize a folder. Links are resolved now, so the root is a real place."""
        try:
            root = Path(path).resolve(strict=True)
        except OSError:
            raise LibraryRefused(f"{path} does not exist") from None
        if not root.is_dir():
            raise LibraryRefused(f"{path} is not a folder. Add the folder that holds it")
        limit = reach_named(reach)
        with self.store.transaction():
            if self.store.one("SELECT id FROM library_roots WHERE user = ? AND path = ?", (user, str(root))):
                raise LibraryRefused(f"{root} is already in the library")
            root_id = self.store.execute(
                "INSERT INTO library_roots (user, path, reach, added_at, added_by) VALUES (?, ?, ?, ?, ?)",
                (user, str(root), int(limit), self.assistant.clock(), by or user),
            )
            self.assistant.audit.record(actor(by or user), "library.add", f"root:{root_id}", {"user": user, "reach": limit.name.lower()})
        return root_id

    def remove(self, user: str, root: int | str) -> None:
        row = self._root(user, root)
        with self.store.transaction():
            self.store.execute("DELETE FROM library_roots WHERE id = ? AND user = ?", (row["id"], user))
            self.assistant.audit.record(actor(user), "library.remove", f"root:{row['id']}", {"user": user})

    def roots(self, user: str) -> list[dict[str, Any]]:
        rows = self.store.query(
            "SELECT r.*, count(DISTINCT f.id) AS files, count(c.id) AS chunks FROM library_roots r "
            "LEFT JOIN library_files f ON f.root_id = r.id LEFT JOIN library_chunks c ON c.file_id = f.id "
            "WHERE r.user = ? GROUP BY r.id ORDER BY r.id",
            (user,),
        )
        return [dict(row) for row in rows]

    def _root(self, user: str, root: int | str) -> Any:
        by_number = str(root).isdigit()
        if by_number:
            row = self.store.one("SELECT * FROM library_roots WHERE id = ? AND user = ?", (int(str(root)), user))
        else:
            row = self.store.one("SELECT * FROM library_roots WHERE path = ? AND user = ?", (str(Path(str(root)).resolve()), user))
        if row is None:
            raise LibraryRefused(f"There is no library folder {root}. `library list` shows them")
        return row

    # Indexing

    def index(self, user: str, root: int | str | None = None) -> IndexReport:
        rows = [self._root(user, root)] if root is not None else self.store.query("SELECT * FROM library_roots WHERE user = ? ORDER BY id", (user,))
        report = IndexReport()
        for row in rows:
            report.merge(self._index_root(row, user))
        return report

    def _index_root(self, root: Any, user: str) -> IndexReport:
        started = time.perf_counter()
        report = IndexReport()
        path = root["path"]
        if not os.path.isdir(path):
            report.problems.append(f"{path} cannot be reached, so its index was left as it was. Is the drive connected?")
            return report
        known = {row["path"]: row for row in self.store.query("SELECT id, path, size, mtime_ns, sha256 FROM library_files WHERE root_id = ?", (root["id"],))}
        unreadable: list[str] = []
        seen: set[str] = set()
        batch: list[tuple[Any, ...]] = []
        for relative, status in walk(path, unreadable):
            old = known.get(relative)
            if old is not None and old["size"] == status.st_size and old["mtime_ns"] == status.st_mtime_ns:
                seen.add(relative)
                report.unchanged += 1
                continue
            try:
                entry = self._read(path, relative, status, old, report)
            except OSError:  # often just locked by whatever has it open, which is not deleted
                seen.add(relative)
                report.skipped += 1
                continue
            if entry is None:
                continue
            seen.add(relative)
            batch.append(entry)
            if len(batch) >= BATCH_FILES:
                self._write(root["id"], batch, report)
                batch = []
        self._write(root["id"], batch, report)
        report.problems += [f"{path}/{folder} could not be read, so what was indexed from it was kept" for folder in unreadable]
        self._remove_missing(known, seen, unreadable, report)
        self.store.execute("UPDATE library_roots SET indexed_at = ? WHERE id = ?", (self.assistant.clock(), root["id"]))
        self._embed_missing(root, report)
        report.seconds = time.perf_counter() - started
        self.assistant.audit.record(
            actor(user), "library.index", f"root:{root['id']}",
            {"user": user, "added": report.added, "updated": report.updated, "removed": report.removed},
        )
        return report

    def _read(self, root: str, relative: str, status: os.stat_result, old: Any, report: IndexReport) -> tuple[Any, ...] | None:
        full = os.path.join(root, relative.replace("/", os.sep))
        if status.st_size > MAX_FILE_BYTES or not inside(root, full):
            report.skipped += 1
            return None
        with open(full, "rb") as file:
            data = file.read()
        digest = hashlib.sha256(data).hexdigest()
        if old is not None and old["sha256"] == digest:
            return (relative, status.st_size, status.st_mtime_ns, digest, "", old, None)
        kind = kind_of(relative)
        chunks = self._chunks(full, kind, data, report)
        if chunks is None:
            report.skipped += 1
            return None
        return (relative, status.st_size, status.st_mtime_ns, digest, kind, old, chunks)

    def _chunks(self, full: str, kind: str, data: bytes, report: IndexReport) -> list[Chunk] | None:
        if kind == "pdf":
            try:
                return chunk_pdf(full)
            except ImportError:
                message = "PDF files were skipped. To index them: pip install pypdf"
                if message not in report.problems:
                    report.problems.append(message)
            except Exception as error:  # a damaged PDF should not stop a whole folder
                report.problems.append(f"{full} could not be read as a PDF: {error}")
            return None
        text = decode(data)
        if text is None:
            return None
        return chunk_markdown(text) if kind == "markdown" else chunk_blocks(text)

    def _write(self, root_id: int, batch: list[tuple[Any, ...]], report: IndexReport) -> None:
        if not batch:
            return
        now = self.assistant.clock()
        with self.store.transaction() as connection:
            for relative, size, mtime_ns, digest, kind, old, chunks in batch:
                if chunks is None:  # touched, but the same bytes
                    connection.execute("UPDATE library_files SET size = ?, mtime_ns = ? WHERE id = ?", (size, mtime_ns, old["id"]))
                    report.unchanged += 1
                    continue
                if old is not None:
                    connection.execute("DELETE FROM library_files WHERE id = ?", (old["id"],))
                file_id = connection.execute(
                    "INSERT INTO library_files (root_id, path, size, mtime_ns, sha256, kind, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (root_id, relative, size, mtime_ns, digest, kind, now),
                ).lastrowid
                connection.executemany(
                    "INSERT INTO library_chunks (file_id, ordinal, heading, line, end_line, page, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(file_id, number, chunk.heading, chunk.line, chunk.end_line, chunk.page, chunk.text) for number, chunk in enumerate(chunks)],
                )
                report.chunks += len(chunks)
                if old is None:
                    report.added += 1
                else:
                    report.updated += 1

    def _remove_missing(self, known: dict[str, Any], seen: set[str], unreadable: list[str], report: IndexReport) -> None:
        if "." in unreadable:  # a root that cannot be listed says nothing about what is in it
            return
        kept = tuple(f"{folder}/" for folder in unreadable)
        gone = [row["id"] for path, row in known.items() if path not in seen and not (kept and path.startswith(kept))]
        if not gone:
            return
        with self.store.transaction() as connection:
            connection.executemany("DELETE FROM library_files WHERE id = ?", [(file_id,) for file_id in gone])
        report.removed += len(gone)

    # Vectors, when a provider offers them

    def _embedder(self) -> Any:
        router = getattr(self.assistant, "router", None)
        return router if router is not None and hasattr(router, "embed") and getattr(router, "can_embed", lambda: False)() else None

    def _embed_missing(self, root: Any, report: IndexReport) -> None:
        router = self._embedder()
        if router is None:
            return
        reach = Reach(root["reach"])
        while True:
            rows = self.store.query(
                "SELECT c.id, c.text FROM library_chunks c JOIN library_files f ON f.id = c.file_id "
                "LEFT JOIN library_vectors v ON v.chunk_id = c.id WHERE f.root_id = ? AND v.chunk_id IS NULL LIMIT ?",
                (root["id"], EMBED_BATCH),
            )
            if not rows:
                return
            try:
                vectors, model = router.embed([row["text"] for row in rows], reach)
            except Exception as error:
                # The words are still searchable. The reason says what would fix it.
                report.problems.append(f"Keyword search only for {root['path']}: {error}")
                return
            with self.store.transaction() as connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO library_vectors (chunk_id, model, vector) VALUES (?, ?, ?)",
                    [(row["id"], model, np.asarray(vector, dtype=np.float32).tobytes()) for row, vector in zip(rows, vectors)],
                )
            report.embedded += len(rows)

    def _nearest(self, user: str, query: str, limit: int, reach: Reach) -> list[int]:
        router = self._embedder()
        if router is None:
            return []
        try:
            vectors, model = router.embed([query], reach)
        except Exception:
            return []  # the words alone can still answer this
        wanted = np.asarray(vectors[0], dtype=np.float32)
        rows = self.store.query(
            "SELECT v.chunk_id, v.vector FROM library_vectors v JOIN library_chunks c ON c.id = v.chunk_id "
            "JOIN library_files f ON f.id = c.file_id JOIN library_roots r ON r.id = f.root_id WHERE r.user = ? AND v.model = ?",
            (user, model),
        )
        usable = [row for row in rows if len(row["vector"]) == wanted.nbytes]
        if not usable:
            return []
        matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in usable])
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(wanted) or 1.0)
        similarity = matrix @ wanted / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-similarity, kind="stable")[:limit]
        return [usable[int(index)]["chunk_id"] for index in order]

    # Searching

    def search(self, user: str, query: str, limit: int = 8, reach: Reach = Reach.POOL) -> list[ContextItem]:
        """The best passages, each with a citation. ``reach`` limits who may see the query."""
        terms = search_terms(query)
        if not terms:
            return []
        rows = self.store.query(
            "SELECT c.id, bm25(library_fts) AS rank FROM library_fts JOIN library_chunks c ON c.id = library_fts.rowid "
            "JOIN library_files f ON f.id = c.file_id JOIN library_roots r ON r.id = f.root_id "
            "WHERE library_fts MATCH ? AND r.user = ? ORDER BY rank LIMIT ?",
            (terms, user, limit * 3),
        )
        keyword = [row["id"] for row in rows]
        nearest = self._nearest(user, query, limit * 3, reach)
        if nearest:
            scores = fuse([keyword, nearest])
        else:
            scores = {row["id"]: -row["rank"] for row in rows}
        best = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
        return self._items(user, best, scores)

    def _items(self, user: str, ids: list[int], scores: dict[int, float]) -> list[ContextItem]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.store.query(
            "SELECT c.*, f.path AS file, f.indexed_at, r.path AS root, r.reach FROM library_chunks c "
            "JOIN library_files f ON f.id = c.file_id JOIN library_roots r ON r.id = f.root_id "
            f"WHERE r.user = ? AND c.id IN ({marks})",
            [user, *ids],
        )
        found = {row["id"]: row for row in rows}
        return [self._item(found[chunk_id], scores[chunk_id]) for chunk_id in ids if chunk_id in found]

    @staticmethod
    def _item(row: Any, score: float) -> ContextItem:
        source = citation(row["root"], row["file"], row["line"], row["page"])
        origin = Provenance("library", source, row["file"], Trust.EXTERNAL, row["indexed_at"], Reach(row["reach"]))
        return ContextItem(row["text"], origin, score)

    def chunk_text(self, user: str, source_id: str) -> str | None:
        """What a citation currently says, for looking back at what a model saw."""
        for row in self.store.query("SELECT id, path FROM library_roots WHERE user = ?", (user,)):
            if not source_id.startswith(row["path"]):
                continue
            base, _, tail = source_id.rpartition("#page=") if "#page=" in source_id else source_id.rpartition(":")
            relative = os.path.relpath(base, row["path"]).replace(os.sep, "/")
            column = "c.page" if "#page=" in source_id else "c.line"
            match = self.store.one(
                f"SELECT c.text FROM library_chunks c JOIN library_files f ON f.id = c.file_id WHERE f.root_id = ? AND f.path = ? AND {column} = ?",
                (row["id"], relative, int(tail) if tail.isdigit() else -1),
            )
            if match:
                return match["text"]
        return None


def attach(assistant: Any) -> None:
    assistant.library = Library(assistant)


def add_commands(commands: Any) -> None:
    parser = commands.add_parser("library", help="folders you authorized, searchable with citations")
    actions = parser.add_subparsers(dest="library_command", required=True)
    add = actions.add_parser("add", help="authorize a folder")
    add.add_argument("path", help="the folder")
    add_reach_option(add, "pool", "text from this folder")
    add_user_option(add)
    add.set_defaults(run=plainly(run_add))
    remove = actions.add_parser("remove", help="stop using a folder and delete its index; the files are not touched")
    remove.add_argument("root", help="the folder, or its number from `library list`")
    add_user_option(remove)
    remove.set_defaults(run=plainly(run_remove))
    listing = actions.add_parser("list", help="show the authorized folders")
    add_user_option(listing)
    listing.set_defaults(run=plainly(run_list))
    index = actions.add_parser("index", help="bring the index up to date; only changed files are read")
    index.add_argument("root", nargs="?", default=None, help="one folder, or its number; default is all of them")
    add_user_option(index)
    index.set_defaults(run=plainly(run_index))
    search = actions.add_parser("search", help="search the library")
    search.add_argument("query", help="what to look for")
    search.add_argument("--limit", type=int, default=8, help="how many results to show")
    add_user_option(search)
    search.set_defaults(run=plainly(run_search))


def run_add(assistant: Any, args: argparse.Namespace) -> int:
    root_id = assistant.library.add(args.user, args.path, args.reach)
    emit(f"Added folder {root_id}. Run `library index` to read it.")
    return 0


def run_remove(assistant: Any, args: argparse.Namespace) -> int:
    assistant.library.remove(args.user, args.root)
    emit("Removed. The files themselves were not touched.")
    return 0


def run_list(assistant: Any, args: argparse.Namespace) -> int:
    roots = assistant.library.roots(args.user)
    for row in roots:
        emit(f"{row['id']:>4}  {Reach(row['reach']).name.lower():<8}  {row['files']:>7} files  {row['chunks']:>8} chunks  {row['path']}")
    if not roots:
        emit("The library is empty. Add a folder with `library add`.")
    return 0


def run_index(assistant: Any, args: argparse.Namespace) -> int:
    report = assistant.library.index(args.user, args.root)
    emit(
        f"{report.added} added, {report.updated} updated, {report.unchanged} unchanged, {report.removed} removed, "
        f"{report.skipped} skipped, {report.chunks} chunks, {report.embedded} embedded, in {report.seconds:.2f} s."
    )
    for problem in report.problems:
        emit(problem)
    return 0


def run_search(assistant: Any, args: argparse.Namespace) -> int:
    items = assistant.library.search(args.user, args.query, args.limit)
    for item in items:
        emit(item.provenance.source_id)
        emit("    " + " ".join(item.text.split())[:200])
    if not items:
        emit("Nothing in the library matches that.")
    return 0
