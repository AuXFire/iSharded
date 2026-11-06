"""The server: every part's HTTP surface, behind per-user keys.

This is not a part. It is built on the assistant, and holds everything people and
OpenAI clients reach it through. A key says who is calling, and nothing in a request
can say otherwise, so one person cannot ask for another's data. Every path needs a
key unless it is listed in PUBLIC_PATHS, which covers routes other parts add too.

Backups live here as well: one consistent copy of the one file everything is in.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import secrets
import sqlite3
import struct
import tempfile
import threading
import time
from typing import Any
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from shardmesh.pool import LOOPBACK
from shardmesh.serving import BodyLimit, NoDelayHTTPProtocol

from .broker import PEER_PREFIX
from .context import TurnNotFound
from .contracts import Provenance, Reach, Trust
from .conversation import TurnResult, decide_approval
from .memory import KINDS, MemoryNotFound, MemoryRefused
from .mind import LOCAL_USER, actor, emit, table

KEY_PREFIX = "sma-"
USER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PUBLIC_PATHS = frozenset({"/", "/assistant", "/assistant/", "/healthz"})
MAX_REQUEST_BYTES = 8 * 1024 * 1024
HTML_FILE = Path(__file__).with_name("assistant.html")
STREAM_POLL_SECONDS = 0.01

BACKUP_MAGIC = b"SMAB1\n"
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**15, 8, 1
SALT_BYTES = 16
NONCE_PREFIX_BYTES = 7  # plus a four byte counter and a one byte last-chunk flag
BACKUP_CHUNK = 1024 * 1024

USERS_SCHEMA = [
    """
    CREATE TABLE gateway_users (
        name TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        key_hash TEXT UNIQUE,
        added_at REAL NOT NULL,
        removed_at REAL
    );
    """,
]


def hash_key(key: str) -> str:
    # Plain SHA-256 is right here: the key is 256 random bits, so nothing can be guessed.
    return hashlib.sha256(key.encode()).hexdigest()


def write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(text)


class Users:
    """Who may connect. Only a hash of each key is kept."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.store.migrate("gateway", USERS_SCHEMA)

    def add(self, name: str, role: str | None = None, reclaim: bool = False) -> str:
        """Make a user and return their key, which cannot be recovered later.

        A removed name keeps its data, so giving it to someone new has to be deliberate.
        """
        if not USER_NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a usable name. Use letters, numbers, dots, dashes, or underscores")
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self.store.transaction():
            row = self.store.one("SELECT * FROM gateway_users WHERE name = ?", (name,))
            if row and row["removed_at"] is None:
                raise ValueError(f"{name} already exists")
            if row and not reclaim:
                raise ValueError(f"{name} was removed and their data is still here. Pass --reclaim to give it to this person")
            first = self.store.one("SELECT name FROM gateway_users WHERE removed_at IS NULL AND role = 'owner'") is None
            chosen = role or ("owner" if first else "member")
            self.store.execute(
                "INSERT INTO gateway_users (name, role, key_hash, added_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET role = excluded.role, key_hash = excluded.key_hash, added_at = excluded.added_at, removed_at = NULL",
                (name, chosen, hash_key(key), self.assistant.clock()),
            )
            self.assistant.audit.record(actor(LOCAL_USER), "user.add", f"user:{name}", {"role": chosen})
        return key

    def remove(self, name: str) -> None:
        with self.store.transaction():
            cursor = self.store.connection.execute(
                "UPDATE gateway_users SET key_hash = NULL, removed_at = ? WHERE name = ? AND removed_at IS NULL", (self.assistant.clock(), name)
            )
            if cursor.rowcount == 0:
                raise ValueError(f"There is no user {name}. `user list` shows them")
            self.assistant.audit.record(actor(LOCAL_USER), "user.remove", f"user:{name}")

    def entries(self) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT name, role, added_at FROM gateway_users WHERE removed_at IS NULL ORDER BY added_at, name")
        return [dict(row) for row in rows]

    def identify(self, key: str) -> tuple[str, str] | None:
        row = self.store.one("SELECT name, role FROM gateway_users WHERE key_hash = ? AND removed_at IS NULL", (hash_key(key),))
        return (row["name"], row["role"]) if row else None


class RequireKey:
    """Refuses anything without a good key, before a route can run.

    Plain ASGI, so it also covers a route another part forgot to protect, and so
    streams pass through untouched. A route reads ``request.state.user``. A part
    leaves a route open by listing it in its own ``PUBLIC_ROUTES``, as (method, path).
    """

    def __init__(self, app: Any, users: Users, public_routes: frozenset[tuple[str, str]] = frozenset()) -> None:
        self.app = app
        self.users = users
        self.public_routes = public_routes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            return  # no other kind of connection is offered, so none is accepted
        if scope["path"] in PUBLIC_PATHS and scope["method"] in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return
        if (scope["method"], scope["path"]) in self.public_routes:
            await self.app(scope, receive, send)
            return
        header = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        who = self.users.identify(header[7:].strip()) if header.startswith("Bearer ") else None
        if who is None:
            await self._refuse(send)
            return
        state = scope.setdefault("state", {})
        state["user"], state["role"] = who
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(send: Any) -> None:
        body = json.dumps({"detail": "Send your key as Authorization: Bearer <key>. `shardmesh assistant user add` makes one."}).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"www-authenticate", b"Bearer")]
        await send({"type": "http.response.start", "status": 401, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def public_memory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "text": row["text"],
        "subject": row["subject"],
        "source": row["source"],
        "source_id": row["source_id"],
        "author": row["author"],
        "trust": row["trust"],
        "reach": Reach(row["reach"]).name.lower(),
        "captured_at": row["captured_at"],
        "superseded_by": row["superseded_by"],
        "expires_at": row["expires_at"],
        "confirmed_at": row["confirmed_at"],
    }


def public_item(item: Any) -> dict[str, Any]:
    # No score: bm25 is computed across the whole index, so a score could hint
    # at what is in someone else's rows.
    origin = item.provenance
    return {
        "text": item.text,
        "source": origin.source,
        "source_id": origin.source_id,
        "author": origin.author,
        "trust": origin.trust.value,
        "reach": origin.reach.name.lower(),
        "captured_at": origin.captured_at,
    }


def public_result(result: TurnResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "text": result.text,
        "conversation": result.conversation_id,
        "turn": result.turn_id,
        "model": result.model,
        "provider": result.provider,
        "tainted": result.tainted,
        "reach": result.reach.name.lower(),
        "pending_approval": result.pending,
        "task": result.task_id,
        "tools": list(result.tools),
    }


def text_of(content: Any) -> str:
    """OpenAI clients send a string, or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def conversation_for(user: str, messages: list[Any], given: str | None) -> str:
    """A stable id for a client that resends its whole chat and names no conversation."""
    if given:
        return given
    first = next((text_of(message.content) for message in messages if message.role == "user"), "")
    return "chat-" + hashlib.sha256(f"{user}\0{first}".encode()).hexdigest()[:24]


def server_sent(payload: dict[str, Any] | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


def turn_in_background(assistant: Any, user: str, text: str, conversation_id: str) -> "queue.Queue[Any]":
    """Run a turn on its own thread, handing text over as it comes.

    The turn belongs to the assistant and not to the connection. If a client goes
    away the reply is still finished and stored, so nothing said is lost.
    """
    items: queue.Queue[Any] = queue.Queue()

    def work() -> None:
        try:
            for item in assistant.conversation.stream(user, text, conversation_id, learn="background"):
                items.put(item)
        except Exception as error:
            items.put(error)
        finally:
            items.put(None)
            assistant.store.close()

    threading.Thread(target=work, name="assistant-turn", daemon=True).start()
    return items


async def next_item(items: "queue.Queue[Any]") -> Any:
    while True:
        try:
            return items.get_nowait()
        except queue.Empty:
            await asyncio.sleep(STREAM_POLL_SECONDS)


class ClientMessage(BaseModel):
    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    messages: list[ClientMessage]
    model: str | None = None
    stream: bool = False
    conversation_id: str | None = Field(default=None, max_length=128)


class SayRequest(BaseModel):
    text: str
    conversation_id: str | None = Field(default=None, max_length=128)


class MemoryRequest(BaseModel):
    text: str
    kind: str = "user"
    subject: str | None = None
    reach: str = "pool"
    expires_at: float | None = None


class RootRequest(BaseModel):
    path: str
    reach: str = "pool"
    for_user: str | None = None


class DecisionRequest(BaseModel):
    approve: bool


class TaskRequest(BaseModel):
    plan: dict[str, Any]


@asynccontextmanager
async def running_tasks(app: FastAPI) -> AsyncIterator[None]:
    """Nothing runs a task unless its host says so, and this server is the host."""
    engine = getattr(app.state.assistant, "tasks", None)
    if engine is not None:
        engine.start()
    yield
    if engine is not None:
        await asyncio.to_thread(engine.stop)


def create_app(assistant: Any) -> FastAPI:
    users = Users(assistant)
    app = FastAPI(title="ShardMesh assistant", docs_url=None, redoc_url=None, openapi_url=None, lifespan=running_tasks)
    app.state.assistant = assistant
    app.state.users = users

    def me(request: Request) -> str:
        return request.state.user

    def owner_only(request: Request) -> None:
        if request.state.role != "owner":
            raise HTTPException(403, "Only the owner can do this")

    def persona() -> str:
        return re.sub(r"[^a-z0-9._-]+", "-", assistant.identity.name.lower()).strip("-") or "assistant"

    # OpenAI clients

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": persona(), "object": "model", "owned_by": "shardmesh-assistant"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request) -> Any:
        # Only the newest message is new. A client's history and system prompt are
        # not trusted: the assistant keeps its own record and its own identity.
        user = me(request)
        newest = next((message for message in reversed(body.messages) if message.role == "user"), None)
        text = text_of(newest.content).strip() if newest else ""
        if not text:
            raise HTTPException(400, "Send at least one user message")
        conversation_id = conversation_for(user, body.messages, body.conversation_id or request.headers.get("x-conversation-id"))
        identifier = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if body.stream:
            items = turn_in_background(assistant, user, text, conversation_id)
            return StreamingResponse(events(items, identifier, created), media_type="text/event-stream")
        result = await asyncio.to_thread(say_on_thread, user, text, conversation_id)
        return {
            "id": identifier,
            "object": "chat.completion",
            "created": created,
            "model": persona(),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            "assistant": public_result(result),
        }

    def say_on_thread(user: str, text: str, conversation_id: str | None) -> TurnResult:
        return assistant.conversation.say(user, text, conversation_id, learn="background")

    async def events(items: "queue.Queue[Any]", identifier: str, created: int) -> AsyncIterator[str]:
        envelope = {"id": identifier, "object": "chat.completion.chunk", "created": created, "model": persona()}
        sent = ""
        while True:
            item = await next_item(items)
            if item is None:
                break
            if isinstance(item, Exception):
                yield server_sent({"error": {"message": str(item)}})
            elif isinstance(item, TurnResult):
                rest = item.text[len(sent) :] if item.text.startswith(sent) else ""
                if rest:  # a notice that came after the model's words
                    yield server_sent({**envelope, "choices": [{"index": 0, "delta": {"content": rest}, "finish_reason": None}]})
                usage = {
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                    "total_tokens": item.prompt_tokens + item.completion_tokens,
                }
                yield server_sent({**envelope, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage, "assistant": public_result(item)})
            else:
                sent += item
                yield server_sent({**envelope, "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}]})
        yield server_sent("[DONE]")

    # The assistant's own API

    api = APIRouter(prefix="/assistant")

    @api.get("/whoami")
    def whoami(request: Request) -> dict[str, Any]:
        return {"user": me(request), "role": request.state.role, "assistant": assistant.identity.name}

    @api.get("/identity")
    def identity(request: Request) -> dict[str, Any]:
        return assistant.identity.describe(me(request))

    @api.post("/say")
    def say(body: SayRequest, request: Request) -> dict[str, Any]:
        return public_result(assistant.conversation.say(me(request), body.text, body.conversation_id, learn="background"))

    @api.get("/conversations")
    def conversations(request: Request) -> dict[str, Any]:
        return {"conversations": assistant.memory.conversations(me(request))}

    @api.get("/conversations/{conversation_id}")
    def conversation(conversation_id: str, request: Request) -> dict[str, Any]:
        rows = assistant.memory.recent(me(request), conversation_id, 200)
        keys = ("id", "role", "content", "model", "provider", "tool", "tainted", "created_at", "turn_id")
        return {"messages": [{key: row[key] for key in keys} for row in rows]}

    @api.get("/memory")
    def memories(request: Request, q: str = "", limit: int = 100, history: bool = False) -> dict[str, Any]:
        user = me(request)
        if q:
            return {"results": [public_item(item) for item in assistant.memory.search(user, q, min(limit, 100))]}
        return {"memories": [public_memory(row) for row in assistant.memory.entries(user, min(limit, 500), include_superseded=history)]}

    @api.post("/memory")
    def remember(body: MemoryRequest, request: Request) -> dict[str, Any]:
        user = me(request)
        if body.kind not in KINDS:
            raise HTTPException(400, f"Kind is one of: {', '.join(KINDS)}")
        try:
            reach = Reach[body.reach.upper()]
        except KeyError:
            raise HTTPException(400, "Reach is one of: machine, pool, peer, cloud") from None
        # Words from a key's holder are that person's. A request cannot claim more.
        origin = Provenance("user", "gateway", user, Trust.OWNER, assistant.clock(), reach)
        memory_id = assistant.memory.remember(user, body.kind, body.text, origin, subject=body.subject, expires_at=body.expires_at)
        return {"id": memory_id}

    @api.get("/memory/export")
    def export(request: Request) -> dict[str, Any]:
        return assistant.memory.export(me(request))

    @api.get("/memory/{memory_id}")
    def memory_history(memory_id: str, request: Request) -> dict[str, Any]:
        return {"history": [public_memory(row) for row in assistant.memory.history(me(request), memory_id)]}

    @api.delete("/memory/{memory_id}")
    def forget(memory_id: str, request: Request) -> dict[str, Any]:
        return {"forgotten": assistant.memory.forget(me(request), memory_id)}

    @api.post("/memory/{memory_id}/confirm")
    def confirm(memory_id: str, request: Request) -> dict[str, Any]:
        assistant.memory.confirm(me(request), memory_id)
        return {"confirmed": True}

    @api.get("/library")
    def library_roots(request: Request) -> dict[str, Any]:
        rows = assistant.library.roots(me(request))
        return {"roots": [{**row, "reach": Reach(row["reach"]).name.lower()} for row in rows]}

    @api.post("/library")
    def library_add(body: RootRequest, request: Request) -> dict[str, Any]:
        # Adding a folder reads this machine's disk as this process, which only the
        # owner may decide, or a member could read the owner's files through it.
        owner_only(request)
        return {"id": assistant.library.add(body.for_user or me(request), body.path, body.reach, by=me(request))}

    @api.delete("/library/{root}")
    def library_remove(root: str, request: Request) -> dict[str, Any]:
        assistant.library.remove(me(request), root)
        return {"removed": True}

    @api.post("/library/index")
    def library_index(request: Request) -> dict[str, Any]:
        report = assistant.library.index(me(request))
        return {key: getattr(report, key) for key in ("added", "updated", "unchanged", "removed", "skipped", "chunks", "embedded", "seconds", "problems")}

    @api.get("/library/search")
    def library_search(q: str, request: Request, limit: int = 8) -> dict[str, Any]:
        return {"results": [public_item(item) for item in assistant.library.search(me(request), q, min(limit, 50))]}

    def waiting_on(user: str, role: str) -> list[dict[str, Any]]:
        """What the broker is asking this person. A peer has no key, so its owner answers for it."""
        broker = getattr(assistant, "broker", None)
        if broker is None:
            return []
        rows = broker.approvals(user)
        if role == "owner":
            rows = rows + [row for row in broker.approvals() if row["user"].startswith(PEER_PREFIX)]
        return rows

    def approvals_of(user: str, role: str) -> list[dict[str, Any]]:
        held = assistant.conversation.pending(user)
        known = {row["approval_id"] for row in held}
        extra = [{**row, "approval_id": row["id"]} for row in waiting_on(user, role) if row["id"] not in known]
        return held + extra

    @api.get("/approvals")
    def approvals(request: Request) -> dict[str, Any]:
        return {"approvals": approvals_of(me(request), request.state.role)}

    @api.post("/approvals/{approval_id}")
    def decide(approval_id: str, body: DecisionRequest, request: Request) -> dict[str, Any]:
        user = me(request)
        if approval_id not in {row["approval_id"] for row in approvals_of(user, request.state.role)}:
            raise HTTPException(404, f"Nothing is waiting on {approval_id} for you")
        told = decide_approval(assistant, approval_id, body.approve, user)
        if not told and already_decided(approval_id) != ("approved" if body.approve else "denied"):
            raise HTTPException(409, f"{approval_id} is no longer waiting, so nothing changed")
        turn = assistant.conversation.turn_for_approval(user, approval_id)
        if turn is None:
            return {"decided": True}
        return {"decided": True, "turn": public_result(assistant.conversation.resume(user, turn, approved=body.approve, learn="background"))}

    def already_decided(approval_id: str) -> str | None:
        """A person may have answered from the command line while a turn still waits here."""
        decided = assistant.broker.approval(approval_id)
        return decided["status"] if decided else None

    def owned_task(user: str, task_id: str) -> dict[str, Any]:
        engine = getattr(assistant, "tasks", None)
        if engine is None:
            raise HTTPException(404, "Tasks are not installed")
        try:
            return engine.status(task_id, user)  # asked in the query, so a stranger's id finds nothing
        except ValueError:
            raise HTTPException(404, f"There is no task {task_id}") from None

    @api.get("/tasks")
    def tasks(request: Request, limit: int = 100) -> dict[str, Any]:
        engine = getattr(assistant, "tasks", None)
        if engine is None:
            return {"available": False, "tasks": []}
        return {"available": True, "tasks": engine.list_tasks(me(request), min(limit, 500))}

    @api.post("/tasks")
    def task_submit(body: TaskRequest, request: Request) -> dict[str, Any]:
        engine = getattr(assistant, "tasks", None)
        if engine is None:
            raise HTTPException(404, "Tasks are not installed")
        return {"id": engine.submit(me(request), body.plan)}

    @api.get("/tasks/{task_id}")
    def task_status(task_id: str, request: Request) -> dict[str, Any]:
        return owned_task(me(request), task_id)

    @api.post("/tasks/{task_id}/cancel")
    def task_cancel(task_id: str, request: Request) -> dict[str, Any]:
        owned_task(me(request), task_id)
        assistant.tasks.cancel(task_id, me(request))
        return {"cancelled": True}

    @api.get("/audit")
    def audit(request: Request, since: int = 0, limit: int = 200) -> dict[str, Any]:
        # The trail covers everyone, so only the owner reads all of it. It holds
        # ids and decisions, never what anyone said.
        size = min(limit, 1000)
        if request.state.role == "owner":
            rows = assistant.audit.read(since=since, limit=size)
        else:
            rows = assistant.audit.read(since=since, actor=actor(me(request)), limit=size)
        return {"entries": rows, "intact": assistant.audit.verify() is None}

    @api.get("/why/{turn}")
    def why(turn: str, request: Request) -> dict[str, Any]:
        return assistant.context.why(me(request), turn)

    @api.get("/turns")
    def turns(request: Request, limit: int = 20) -> dict[str, Any]:
        return {"turns": assistant.context.recent_turns(me(request), min(limit, 100))}

    @api.get("/models")
    def model_stats() -> dict[str, Any]:
        return {"stats": assistant.router.stats(), "routes": assistant.router.routes()}

    app.include_router(api)
    include_parts(app, assistant)

    @app.exception_handler(MemoryNotFound)
    async def not_found(_: Request, error: MemoryNotFound) -> Any:
        return plain_error(404, error)

    @app.exception_handler(TurnNotFound)
    async def turn_not_found(_: Request, error: TurnNotFound) -> Any:
        return plain_error(404, error)

    @app.exception_handler(MemoryRefused)
    async def refused(_: Request, error: MemoryRefused) -> Any:
        return plain_error(409, error)

    @app.exception_handler(ValueError)
    async def bad_value(_: Request, error: ValueError) -> Any:
        return plain_error(400, error)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    @app.get("/assistant", response_class=HTMLResponse)
    @app.get("/assistant/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        return page_response()

    public_routes = frozenset(route for part in assistant.parts for route in getattr(part, "PUBLIC_ROUTES", ()))
    app.add_middleware(RequireKey, users=users, public_routes=public_routes)
    app.add_middleware(BodyLimit, limit=MAX_REQUEST_BYTES)
    return app


def plain_error(status: int, error: Exception) -> JSONResponse:
    detail = error.args[0] if error.args else str(error)
    return JSONResponse({"detail": str(detail)}, status_code=status)


def include_parts(app: Any, assistant: Any) -> None:
    """Add every part's routes. A path already taken keeps its first owner."""
    taken = {(path, method) for route in app.routes for path in [getattr(route, "path", "")] for method in getattr(route, "methods", None) or ()}
    for part in assistant.parts:
        make = getattr(part, "routes", None)
        if make is None:
            continue
        router = make(assistant)
        name = part.__name__.rsplit(".", 1)[-1]
        for route in list(router.routes):
            clash = [method for method in getattr(route, "methods", None) or () if (route.path, method) in taken]
            if clash:
                router.routes.remove(route)
                assistant.audit.record("gateway", "gateway.route_shadowed", route.path, {"part": name})
        app.include_router(router)


def page_response() -> HTMLResponse:
    nonce = secrets.token_urlsafe(16)
    html = HTML_FILE.read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
    policy = (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; "
        "img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return HTMLResponse(
        html,
        headers={
            "Content-Security-Policy": policy,
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


def check_bind(host: str, tls: bool, trusted_network: bool) -> None:
    """Keys cross the network with every request, so they need a link nobody can read."""
    if host in LOOPBACK or tls or trusted_network:
        return
    raise ValueError(
        f"{host} is reachable from other machines, and this would send keys and conversations unencrypted. "
        "Pass --tls-cert and --tls-key, or --trusted-network if nobody else can read this network."
    )


# Backup and restore


def read_passphrase(path: str) -> bytes:
    try:
        text = Path(path).read_text(encoding="utf-8").rstrip("\r\n")
    except OSError:
        raise ValueError(f"The passphrase file {path} cannot be read") from None
    if not text:
        raise ValueError(f"The passphrase file {path} is empty")
    return text.encode()


def derive_key(passphrase: bytes, salt: bytes, n: int, r: int, p: int) -> bytes:
    try:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError:
        raise ValueError("Backups need the cryptography package. Run: pip install cryptography") from None
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase)


def chunk_nonce(prefix: bytes, index: int, last: bool) -> bytes:
    """Each chunk's position is in its nonce, so chunks cannot be reordered or cut off."""
    return prefix + struct.pack(">IB", index, int(last))


def encrypt_file(source: Path, target: Path, passphrase: bytes) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt, prefix = os.urandom(SALT_BYTES), os.urandom(NONCE_PREFIX_BYTES)
    header = BACKUP_MAGIC + struct.pack(">IBB", SCRYPT_N, SCRYPT_R, SCRYPT_P) + salt + prefix
    cipher = AESGCM(derive_key(passphrase, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P))
    temporary = target.with_name(target.name + ".part")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        writer.write(header)
        index, chunk = 0, reader.read(BACKUP_CHUNK)
        while True:
            following = reader.read(BACKUP_CHUNK)
            sealed = cipher.encrypt(chunk_nonce(prefix, index, not following), chunk, header)
            writer.write(struct.pack(">I", len(sealed)) + sealed)
            if not following:
                break
            index, chunk = index + 1, following
    os.replace(temporary, target)


def decrypt_file(source: Path, target: Path, passphrase: bytes) -> None:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    wrong = "This backup cannot be opened: the passphrase is wrong, or the file was damaged or changed."
    with source.open("rb") as reader, target.open("wb") as writer:
        magic = reader.read(len(BACKUP_MAGIC))
        if magic != BACKUP_MAGIC:
            raise ValueError(f"{source} is not an assistant backup")
        fixed = reader.read(6)
        salt = reader.read(SALT_BYTES)
        prefix = reader.read(NONCE_PREFIX_BYTES)
        if len(fixed) != 6 or len(salt) != SALT_BYTES or len(prefix) != NONCE_PREFIX_BYTES:
            raise ValueError(wrong)
        n, r, p = struct.unpack(">IBB", fixed)
        header = magic + fixed + salt + prefix
        try:
            cipher = AESGCM(derive_key(passphrase, salt, n, r, p))
        except (ValueError, MemoryError):  # a header changed to ask for impossible settings
            raise ValueError(wrong) from None
        index = 0
        size_bytes = reader.read(4)
        while True:
            if len(size_bytes) != 4:
                raise ValueError(wrong)
            sealed = reader.read(struct.unpack(">I", size_bytes)[0])
            size_bytes = reader.read(4)
            last = not size_bytes
            try:
                writer.write(cipher.decrypt(chunk_nonce(prefix, index, last), sealed, header))
            except InvalidTag:
                raise ValueError(wrong) from None
            if last:
                return
            index += 1


def backup(assistant: Any, target: Path, passphrase: bytes) -> dict[str, Any]:
    """One consistent copy, encrypted. The clear copy sits beside the database, as
    protected as the database is, and is gone when this returns."""
    target = Path(target)
    started = time.perf_counter()
    # Recorded first: if this cannot be written nothing has happened yet, a good
    # backup is never reported as failed, and the copy carries its own record.
    assistant.audit.record(actor(LOCAL_USER), "backup.started", target.name)
    with tempfile.TemporaryDirectory(dir=assistant.home, prefix="backup-") as folder:
        copy = Path(folder) / "copy.db"
        destination = sqlite3.connect(copy)
        try:
            assistant.store.connection.backup(destination)
        finally:
            destination.close()
        size = copy.stat().st_size
        encrypt_file(copy, target, passphrase)
    return {"bytes": size, "seconds": time.perf_counter() - started}


def restore(assistant: Any, source: Path, passphrase: bytes, force: bool = False) -> dict[str, Any]:
    """Replace this install's state with a backup, which is checked first."""
    has_data = assistant.store.one("SELECT count(*) AS total FROM audit")["total"] > 0
    if has_data and not force:
        raise ValueError(f"{assistant.store.path} already holds data. Pass --force to replace it, or restore into a new --home")
    with tempfile.TemporaryDirectory(dir=assistant.home, prefix="restore-") as folder:
        copy = Path(folder) / "copy.db"
        decrypt_file(Path(source), copy, passphrase)
        incoming = sqlite3.connect(copy)
        try:
            check = incoming.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise ValueError(f"This backup opened but its database is damaged: {check}")
            incoming.backup(assistant.store.connection)
        finally:
            incoming.close()
    broken = assistant.audit.verify()
    assistant.audit.record(actor(LOCAL_USER), "backup.restored", Path(source).name, {"chain_intact": broken is None})
    return {"chain_intact": broken is None, "broken_at": broken}


# Commands


def run_serve(assistant: Any, args: argparse.Namespace) -> int:
    import uvicorn

    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("TLS needs both --tls-cert and --tls-key")
    check_bind(args.host, bool(args.tls_cert), args.trusted_network)
    users = Users(assistant)
    if not users.entries():
        raise ValueError("Nobody could connect yet. Run `shardmesh assistant user add NAME` first")
    options = {"ssl_certfile": args.tls_cert, "ssl_keyfile": args.tls_key} if args.tls_cert else {}
    scheme = "https" if args.tls_cert else "http"
    emit(f"{assistant.identity.name} is at {scheme}://{args.host}:{args.port}/")
    federation = getattr(assistant, "federation", None)
    if federation is not None:
        federation.listening_at(args.host, f"{scheme}://{args.host}:{args.port}")
    uvicorn.run(create_app(assistant), host=args.host, port=args.port, http=NoDelayHTTPProtocol, access_log=False, log_level="warning", **options)
    return 0


def run_user_add(assistant: Any, args: argparse.Namespace) -> int:
    path = Path(args.key_file) if args.key_file else assistant.home / "keys" / f"{args.name}.key"
    if path.exists() and not args.reclaim:
        raise ValueError(f"{path} already exists. Choose another --key-file")
    key = Users(assistant).add(args.name, args.role, args.reclaim)
    path.unlink(missing_ok=True)
    write_private(path, key + "\n")
    # The key goes to a file and not the screen, where it would stay in scrollback and logs.
    emit(f"Added {args.name}. Their key is in {path} and cannot be shown again.")
    if args.name != LOCAL_USER:
        emit(f"Commands on this machine act for {LOCAL_USER} unless given --user {args.name}, so grants for them need it too.")
    return 0


def run_user_list(assistant: Any, args: argparse.Namespace) -> int:
    emit(table(Users(assistant).entries(), ["name", "role"]) or "Nobody yet. Add someone with `user add`.")
    return 0


def run_user_remove(assistant: Any, args: argparse.Namespace) -> int:
    Users(assistant).remove(args.name)
    emit(f"{args.name} can no longer connect. Their data stays here.")
    return 0


def run_backup(assistant: Any, args: argparse.Namespace) -> int:
    report = backup(assistant, Path(args.file), read_passphrase(args.passphrase_file))
    emit(f"Wrote {args.file}: {report['bytes']} bytes of state in {report['seconds']:.2f} s.")
    return 0


def run_restore(assistant: Any, args: argparse.Namespace) -> int:
    report = restore(assistant, Path(args.file), read_passphrase(args.passphrase_file), args.force)
    emit("Restored. Restart anything using this folder." if report["chain_intact"] else f"Restored, but the audit record was changed at entry {report['broken_at']}.")
    return 0 if report["chain_intact"] else 1
