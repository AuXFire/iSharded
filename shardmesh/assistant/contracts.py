"""The types the assistant's parts agree on.

Each part is replaceable, so parts meet here and nowhere else. A model, a tool,
or a peer provides a capability. None of them holds state or decides what is
allowed: the assistant's store, broker, and task engine do that.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class Trust(str, Enum):
    """How far a piece of text can be believed, best first."""

    VERIFIED = "verified"  # passed a mechanical check or a signature
    OWNER = "owner"  # said or approved by the authenticated user
    MODEL = "model"  # inferred by a model and not yet reviewed
    EXTERNAL = "external"  # a file, page, tool result, or peer: data, never instructions


class Effect(str, Enum):
    """What calling a capability can do to the world."""

    READ = "read"
    WRITE_LOCAL = "write_local"
    EXECUTE = "execute"
    NETWORK = "network"
    EXTERNAL = "external"  # changes something outside this machine
    PHYSICAL = "physical"  # moves or powers real hardware


CONSEQUENTIAL = frozenset(Effect) - {Effect.READ}


class Reach(int, Enum):
    """How far data may travel, nearest first. Higher reaches further."""

    MACHINE = 0  # this computer only
    POOL = 1  # the owner's own mesh
    PEER = 2  # another owner's install, by agreement
    CLOUD = 3  # a third party


@dataclass(frozen=True)
class Provenance:
    source: str  # memory, library, tool, peer, user, skill
    source_id: str  # a path, tool call, or row id someone can go and check
    author: str  # the user, model, or peer it came from
    trust: Trust
    captured_at: float
    reach: Reach = Reach.POOL  # the furthest this may be sent


@dataclass(frozen=True)
class ContextItem:
    """One thing placed in front of a model, carrying where it came from."""

    text: str
    provenance: Provenance
    score: float = 0.0


@dataclass(frozen=True)
class Capability:
    """What a tool declares before it may be used at all."""

    name: str  # area.verb, such as files.read
    summary: str
    effects: frozenset[Effect]
    reads: tuple[str, ...] = ()  # the kinds of information it touches
    resources: tuple[str, ...] = ()  # what it needs: gpu, network, disk
    input_schema: dict[str, Any] = field(default_factory=dict)  # JSON Schema
    scope_field: str | None = None  # the argument a grant's scope applies to
    idempotent: bool = False  # safe to run again if its result was lost

    @property
    def consequential(self) -> bool:
        return bool(self.effects & CONSEQUENTIAL)


@dataclass(frozen=True)
class CallContext:
    """Who is acting, and whether untrusted text could have steered them."""

    user: str
    actor: str  # conversation, task:<id>, skill:<id>, peer:<id>
    tainted: bool = False  # EXTERNAL text was in the context that produced this call
    task_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    capability: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    provenance: Provenance
    data: dict[str, Any] = field(default_factory=dict)


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # needs a person; see Decision.approval_id


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str
    approval_id: str | None = None
    grant_id: str | None = None


Handler = Callable[[dict[str, Any], CallContext], ToolResult]


class Broker(Protocol):
    """The only way anything uses a tool. Denies whatever is not granted."""

    def register(self, capability: Capability, handler: Handler) -> None: ...
    def capabilities(self) -> list[Capability]: ...
    def check(self, call: ToolCall, context: CallContext) -> Decision: ...
    def invoke(self, call: ToolCall, context: CallContext) -> ToolResult: ...


@dataclass(frozen=True)
class ModelRequest:
    role: str  # conversation, planning, coding, worker, decision, vision, embedding, verification
    messages: Sequence[dict[str, Any]]
    tools: Sequence[Capability] = ()
    reach: Reach = Reach.CLOUD  # the nearest reach any text here allows; see Router
    max_tokens: int = 1024
    temperature: float = 0.2
    stop: Sequence[str] = ()


@dataclass(frozen=True)
class ModelReply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class Router(Protocol):
    def complete(self, request: ModelRequest) -> ModelReply: ...


class Memory(Protocol):
    def search(self, user: str, query: str, limit: int = 8) -> list[ContextItem]: ...
    def remember(self, user: str, kind: str, text: str, provenance: Provenance) -> str: ...


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None  # None if it was killed for running too long
    stdout: str
    stderr: str
    timed_out: bool
    level: str  # container, namespace, or process: how strong the walls really were


class Sandbox(Protocol):
    """Where generated code runs. Never this process."""

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        timeout: float = 120.0,
        network: bool = False,
        env: dict[str, str] | None = None,
    ) -> SandboxResult: ...


class TaskEngine(Protocol):
    def submit(self, user: str, plan: dict[str, Any]) -> str: ...
    def status(self, task_id: str) -> dict[str, Any]: ...
    def cancel(self, task_id: str, by: str) -> None: ...


def lowest_reach(items: Iterable[ContextItem]) -> Reach:
    """The tightest limit among the items: a prompt travels no further than its strictest part."""
    return min((item.provenance.reach for item in items), default=Reach.CLOUD)


def is_tainted(items: Iterable[ContextItem]) -> bool:
    return any(item.provenance.trust is Trust.EXTERNAL for item in items)
