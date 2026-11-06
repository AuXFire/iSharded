"""Stand-ins for models and the broker, shared by the assistant's tests. Holds no tests.

FakeModelServer is real HTTP on an ephemeral loopback port. It can refuse, fail
partway, or go silent while holding the connection open, because a lost link gives
no error at all and a killed process never shows that.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any

from shardmesh.assistant.contracts import (
    CallContext,
    Capability,
    Decision,
    Effect,
    ModelReply,
    ModelRequest,
    Provenance,
    Reach,
    ToolCall,
    ToolResult,
    Trust,
    Verdict,
)


@dataclass
class Reply:
    text: str = "Hello."
    tool_calls: list[dict[str, Any]] | None = None
    status: int = 200
    error_body: str = '{"detail": "nope"}'
    usage: dict[str, int] | None = field(default_factory=lambda: {"prompt_tokens": 7, "completion_tokens": 3})
    silent: bool = False  # hold the connection and say nothing at all
    silent_after: int | None = None  # the same, after this many stream pieces
    cut_after: int | None = None  # close the connection after this many pieces
    error_after: int | None = None  # an error event after this many pieces, as the mesh sends
    no_finish: bool = False  # end cleanly without saying it finished
    model: str | None = None


class FakeModelServer:
    def __init__(self, replies: list[Reply] | None = None, models: tuple[str, ...] = ("fake-model",), mesh_shape: bool = False) -> None:
        self.replies = list(replies or [])
        self.models = list(models)
        self.mesh_shape = mesh_shape
        self.requests: list[dict[str, Any]] = []
        self.bad_requests: list[str] = []
        self.model_list_silent = False
        self.release = threading.Event()
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:
                outer.handle_get(self)

            def do_POST(self) -> None:
                outer.handle_post(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "FakeModelServer":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()

    def next_reply(self) -> Reply:
        with self.lock:
            return self.replies.pop(0) if self.replies else Reply()

    def handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        if self.model_list_silent:
            self.release.wait(60)
            return
        if handler.path.endswith("/models"):
            self.send_json(handler, 200, {"object": "list", "data": [{"id": name, "object": "model"} for name in self.models]})
        else:
            self.send_json(handler, 404, {"detail": "not found"})

    @staticmethod
    def send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        body = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))))
        with self.lock:
            self.requests.append({"path": handler.path, "headers": dict(handler.headers), "body": body})
        if handler.path.endswith("/embeddings"):
            vectors = [{"index": index, "embedding": [float(len(text)), 1.0]} for index, text in enumerate(body["input"])]
            self.send_json(handler, 200, {"data": vectors})
            return
        if self.mesh_shape and not self.check_mesh_shape(handler, body):
            return
        reply = self.next_reply()
        if reply.silent:
            self.release.wait(60)
            return
        if reply.status != 200:
            data = reply.error_body.encode()
            handler.send_response(reply.status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
            return
        if body.get("stream"):
            self.stream(handler, reply)
        else:
            message: dict[str, Any] = {"role": "assistant", "content": reply.text}
            if reply.tool_calls:
                message["tool_calls"] = reply.tool_calls
            payload: dict[str, Any] = {"model": reply.model or body.get("model"), "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
            if reply.usage:
                payload["usage"] = reply.usage
            self.send_json(handler, 200, payload)

    def check_mesh_shape(self, handler: BaseHTTPRequestHandler, body: dict[str, Any]) -> bool:
        """Hold requests to the mesh's real schema, so a fake cannot drift from it."""
        from pydantic import ValidationError

        from shardmesh.protocol import ChatCompletionRequest

        try:
            ChatCompletionRequest.model_validate(body)
        except ValidationError as error:
            self.bad_requests.append(str(error))
            self.send_json(handler, 422, {"detail": str(error)})
            return False
        extra = set(body) - set(ChatCompletionRequest.model_fields)
        if extra:
            self.bad_requests.append(f"unknown fields: {sorted(extra)}")
        return True

    def stream(self, handler: BaseHTTPRequestHandler, reply: Reply) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True

        def send(payload: dict[str, Any] | str) -> None:
            data = payload if isinstance(payload, str) else json.dumps(payload)
            handler.wfile.write(f"data: {data}\n\n".encode())
            handler.wfile.flush()

        pieces = [reply.text[start : start + 5] for start in range(0, len(reply.text), 5)]
        for index, piece in enumerate(pieces):
            if index == reply.silent_after:
                self.release.wait(60)
                return
            if index == reply.cut_after:
                return
            if index == reply.error_after:
                send({"error": {"message": "a stage was lost", "code": 503}})
                send("[DONE]")
                return
            send({"model": reply.model or "fake-model", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        if reply.no_finish:
            return
        final: dict[str, Any] = {"model": reply.model or "fake-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        if reply.usage:
            final["usage"] = reply.usage
        send(final)
        send("[DONE]")


class FakeProvider:
    """An in-process provider, for what does not depend on HTTP."""

    kind = "fake"

    def __init__(
        self,
        name: str,
        reach: Reach = Reach.POOL,
        text: str = "ok",
        fail: bool = False,
        daily_tokens: int | None = None,
        model: str = "m",
        tokens: tuple[int, int] = (10, 5),
        seconds: float = 1.0,
    ) -> None:
        self.name = name
        self.reach = reach
        self.text = text
        self.fail = fail
        self.daily_tokens = daily_tokens
        self.model = model
        self.tokens = tokens
        self.seconds = seconds
        self.calls = 0
        self.model_asks = 0
        self.seen: list[ModelRequest] = []

    def current_model(self) -> str | None:
        self.model_asks += 1
        return self.model

    def health(self) -> dict[str, Any]:
        return {"up": not self.fail, "models": [self.model], "error": None, "seconds": 0.0}

    def complete(self, request: ModelRequest, model: str) -> ModelReply:
        self.calls += 1
        self.seen.append(request)
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        return ModelReply(self.text, (), model, self.name, self.tokens[0], self.tokens[1], self.seconds)

    def stream(self, request: ModelRequest, model: str) -> Iterator[Any]:
        reply = self.complete(request, model)
        for start in range(0, len(reply.text), 3):
            yield reply.text[start : start + 3]
        yield reply


class ScriptedRouter:
    """Implements contracts.Router. Each reply is text, a ModelReply, an exception, or a callable."""

    def __init__(self, *replies: Any, skip_workers: bool = False) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []
        self.skip_workers = skip_workers  # learning runs in the background, in no fixed order

    def complete(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        if self.skip_workers and request.role == "worker":
            return model_reply("[]")
        reply = self.replies.pop(0) if self.replies else "ok"
        if callable(reply) and not isinstance(reply, type):
            reply = reply(request)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, ModelReply):
            return reply
        return model_reply(str(reply))

    def by_role(self, role: str) -> list[ModelRequest]:
        return [request for request in self.requests if request.role == role]


def model_reply(text: str = "ok", *calls: ToolCall, model: str = "scripted", provider: str = "fake") -> ModelReply:
    return ModelReply(text, tuple(calls), model, provider, 5, 3, 0.1)


def capability(name: str = "notes.read", *effects: Effect, summary: str = "A tool.") -> Capability:
    return Capability(name=name, summary=summary, effects=frozenset(effects or {Effect.READ}))


class FakeBroker:
    """Implements contracts.Broker, with a person's approval decided by the test."""

    def __init__(self) -> None:
        self.tools: dict[str, tuple[Capability, Any]] = {}
        self.verdicts: dict[str, Verdict] = {}
        self.approved: set[str] = set()
        self.declined: set[str] = set()
        self.asked: dict[str, dict[str, Any]] = {}
        self.checks: list[tuple[ToolCall, CallContext]] = []
        self.invocations: list[tuple[ToolCall, CallContext]] = []

    def register(self, capability: Capability, handler: Any) -> None:
        self.tools[capability.name] = (capability, handler)

    def capabilities(self) -> list[Capability]:
        return [item[0] for item in self.tools.values()]

    def add(self, name: str, output: Any = "tool output", verdict: Verdict = Verdict.ALLOW, reach: Reach = Reach.POOL, effects: tuple[Effect, ...] = (Effect.READ,)) -> None:
        def handler(arguments: dict[str, Any], context: CallContext) -> ToolResult:
            if isinstance(output, BaseException):
                raise output
            text = output(arguments) if callable(output) else output
            return ToolResult(True, text, Provenance("tool", name, "tool", Trust.VERIFIED, 0.0, reach))

        self.register(capability(name, *effects), handler)
        self.verdicts[name] = verdict

    def check(self, call: ToolCall, context: CallContext) -> Decision:
        self.checks.append((call, context))
        verdict = self.verdicts.get(call.capability, Verdict.DENY)
        if verdict is Verdict.ASK:
            key = json.dumps([call.capability, call.arguments], sort_keys=True)
            if key in self.approved:
                return Decision(Verdict.ALLOW, "approved")
            if key in self.declined:
                return Decision(Verdict.DENY, "a person declined this")
            approval = f"approval-{len(self.asked) + 1}"
            self.asked[approval] = {"key": key, "user": context.user, "capability": call.capability}
            return Decision(Verdict.ASK, "a person must approve this", approval_id=approval)
        return Decision(verdict, "by policy" if verdict is Verdict.ALLOW else f"{call.capability} is not granted")

    def invoke(self, call: ToolCall, context: CallContext) -> ToolResult:
        self.invocations.append((call, context))
        return self.tools[call.capability][1](call.arguments, context)

    def decide(self, approval_id: str, approved: bool, by: str, user: str | None = None) -> bool:
        asked = self.asked.get(approval_id)
        if asked is None or asked.get("status", "pending") != "pending":
            return False
        asked["status"] = "approved" if approved else "denied"
        (self.approved if approved else self.declined).add(asked["key"])
        return True

    def approve(self, approval_id: str, by: str = "owner") -> None:
        self.decide(approval_id, True, by)

    def approval(self, approval_id: str) -> dict[str, Any] | None:
        asked = self.asked.get(approval_id)
        return {"id": approval_id, "status": asked.get("status", "pending"), **asked} if asked else None

    def approvals(self, user: str | None = None, status: str | None = "pending") -> list[dict[str, Any]]:
        rows = [self.approval(name) for name in self.asked]
        return [row for row in rows if user in (None, row["user"]) and status in (None, row["status"])]

    def grants(self, user: str | None = None, include_ended: bool = False) -> list[dict[str, Any]]:
        """Every tool counts as granted here. A test says what happens with `verdicts`."""
        return [{"capability": name} for name in self.tools]


def wait_until(condition: Any, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return bool(condition())
