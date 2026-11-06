"""The only way anything uses a tool. Whatever is not granted is denied.

A grant is a standing permission: this user, this capability, this scope. An
approval is one person saying yes to one exact action, bound by a hash of its
arguments, and it works once. Models, skills, and peers can ask for things here
but can never decide anything here, so a poisoned page that steers a model gets
as far as a person's question and no further.

The decision and whatever it consumes commit together with the audit entry, and
the handler runs after that commit, so a slow tool never holds the write lock
and a use is never spent without a record of it.
"""
from __future__ import annotations

from dataclasses import replace
import fnmatch
import hashlib
import json
import os
import re
import secrets
import time
from typing import Any
from urllib.parse import urlsplit

from .contracts import (
    CallContext,
    Capability,
    Decision,
    Effect,
    Handler,
    Provenance,
    ToolCall,
    ToolResult,
    Trust,
    Verdict,
)
from .schema import errors
from .store import canonical

ANY_SCOPE = "*"
DEFAULT_USER = "owner"
PEER_PREFIX = "peer:"
SKILL_PREFIX = "skill:"
# Actors that act for a person but are not one. None of these may grant or decide.
NOT_A_PERSON = (PEER_PREFIX, SKILL_PREFIX, "task:", "model:", "plugin:", "conversation")
PATH_FIELDS = {"path", "file", "dir", "directory", "folder", "cwd", "workdir", "root", "repo", "repository"}
PATH_SUFFIXES = ("path", "dir", "file", "folder")
URL_SUFFIXES = ("url", "uri")
SECRET_NAME = re.compile(r"pass|secret|token|key|auth|cookie|credential", re.IGNORECASE)
HOST_REFUSED = re.compile(r"[\\\s\x00-\x1f]")

SCHEMA = [
    """
    CREATE TABLE grants (
        id TEXT PRIMARY KEY,
        user TEXT NOT NULL,
        capability TEXT NOT NULL,
        scope TEXT NOT NULL,
        expires_at REAL,
        uses_left INTEGER,
        granted_by TEXT NOT NULL,
        granted_at REAL NOT NULL,
        revoked_at REAL,
        revoked_by TEXT
    );
    CREATE INDEX grants_lookup ON grants (user, capability);
    CREATE TABLE approvals (
        id TEXT PRIMARY KEY,
        user TEXT NOT NULL,
        capability TEXT NOT NULL,
        args_hash TEXT NOT NULL,
        arguments TEXT NOT NULL,
        actor TEXT NOT NULL,
        task_id TEXT,
        status TEXT NOT NULL,
        reason TEXT NOT NULL,
        requested_at REAL NOT NULL,
        decided_by TEXT,
        decided_at REAL,
        used_at REAL
    );
    CREATE INDEX approvals_lookup ON approvals (user, capability, args_hash, status);
    CREATE INDEX approvals_status ON approvals (status, requested_at);
    """,
]


def argument_hash(arguments: Any) -> str:
    return hashlib.sha256(canonical(arguments).encode("utf-8", "surrogatepass")).hexdigest()


def is_person(name: str) -> bool:
    return bool(name) and not name.startswith(NOT_A_PERSON)


def redacted(arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments as a person may see them on a screen someone else can read."""
    return {key: "***" if SECRET_NAME.search(key) else value for key, value in arguments.items()}


def scope_kind(capability: Capability) -> str:
    """How a grant's scope is compared: by real path, by host, or as a glob."""
    field = capability.scope_field or ""
    properties = capability.input_schema.get("properties") if isinstance(capability.input_schema, dict) else None
    entry = properties.get(field) if isinstance(properties, dict) else None
    declared = entry.get("format") if isinstance(entry, dict) else None
    if declared in ("path", "uri", "url"):
        return "path" if declared == "path" else "url"
    name = field.lower()
    if name in PATH_FIELDS or name.endswith(PATH_SUFFIXES):
        return "path"
    return "url" if name.endswith(URL_SUFFIXES) else "glob"


def real_path(value: Any) -> str | None:
    """Where a path really leads, or None if it cannot be judged.

    A relative path means whatever the working directory happens to be, so a
    grant could not honestly be compared to it. It is refused, not guessed at.
    """
    if not isinstance(value, str) or not value or "\x00" in value or not os.path.isabs(value):
        return None
    try:
        return os.path.realpath(value)
    except (OSError, ValueError):
        return None


def path_within(candidate: str, root: str) -> bool:
    """A real path comparison: /work-private is not inside /work."""
    candidate, root = os.path.normcase(candidate), os.path.normcase(root)
    return candidate == root or candidate.startswith(root if root.endswith(os.sep) else root + os.sep)


def url_host(value: Any) -> tuple[str, int | None] | None:
    """The host and port a URL really goes to, or None if that is not clear."""
    if not isinstance(value, str) or HOST_REFUSED.search(value):
        return None
    try:
        parts = urlsplit(value if "://" in value else "//" + value)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    return (host.lower().rstrip("."), port) if host else None


def host_covered(scope: str, host: str, port: int | None) -> bool:
    wanted = url_host(scope)
    if wanted is None:
        return False
    return fnmatch.fnmatchcase(host, wanted[0]) and wanted[1] in (None, port)


def value_covered(kind: str, scope: str, value: Any) -> bool:
    if kind == "path":
        real = real_path(value)
        return real is not None and path_within(real, scope)
    if kind == "url":
        target = url_host(value)
        return target is not None and host_covered(scope, *target)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    return fnmatch.fnmatchcase(str(value), scope)


def scope_covers(capability: Capability, scope: str, arguments: dict[str, Any]) -> bool:
    field = capability.scope_field
    if scope == ANY_SCOPE:
        return True
    if field is None or field not in arguments:
        return False  # nothing to compare, so a narrow grant cannot be said to cover it
    value = arguments[field]
    values = value if isinstance(value, (list, tuple)) else [value]
    return bool(values) and all(value_covered(scope_kind(capability), scope, item) for item in values)


class CapabilityBroker:
    """Implements contracts.Broker."""

    def __init__(self, assistant: Any) -> None:
        self.assistant = assistant
        self.store = assistant.store
        self.audit = assistant.audit
        self.clock = getattr(assistant, "clock", time.time)
        self.registered: dict[str, tuple[Capability, Handler]] = {}
        self.store.migrate("broker", SCHEMA)

    # What exists.

    def register(self, capability: Capability, handler: Handler) -> None:
        if not capability.effects:
            raise ValueError(f"{capability.name} declares no effects. Every capability must say what it can do.")
        if capability.name in self.registered:
            raise ValueError(f"{capability.name} is already registered, and a tool cannot replace another one.")
        self.registered[capability.name] = (capability, handler)

    def unregister(self, name: str) -> None:
        self.registered.pop(name, None)

    def capabilities(self) -> list[Capability]:
        return [entry[0] for _, entry in sorted(self.registered.items())]

    def capability(self, name: str) -> Capability | None:
        entry = self.registered.get(name)
        return entry[0] if entry else None

    # Deciding.

    def check(self, call: ToolCall, context: CallContext) -> Decision:
        """What would happen. Spends nothing, though a question may be put to a person."""
        with self.store.transaction():
            decision, _ = self._decide(call, context, spend=False)
            self._record("check", call, context, decision)
        return decision

    def invoke(self, call: ToolCall, context: CallContext) -> ToolResult:
        with self.store.transaction():
            decision, arguments = self._decide(call, context, spend=True)
            self._record("invoke", call, context, decision)
        if decision.verdict is not Verdict.ALLOW:
            return self._refusal(call, decision)
        capability, handler = self.registered[call.capability]
        try:
            result = self._normal(handler(arguments, context), call, context)
        except Exception as error:  # a tool must never be able to take the assistant down
            self._outcome(call, context, False, type(error).__name__)
            return ToolResult(False, f"{call.capability} failed: {type(error).__name__}: {error}", self._provenance(call))
        self._outcome(call, context, result.ok, "")
        return result

    def covers(self, user: str, capability: str, arguments: dict[str, Any]) -> bool:
        """Whether a standing grant covers this, without asking or spending anything."""
        entry = self.capability(capability)
        return entry is not None and self._grant(entry, user, arguments) is not None

    def names_host(self, user: str, capability: str, url: str) -> bool:
        """Whether a live grant names this host outright, not through a wildcard."""
        target = url_host(url)
        if target is None:
            return False
        return any(
            "*" not in row["scope"] and host_covered(row["scope"], *target) for row in self._live_grants(user, capability)
        )

    def _decide(self, call: ToolCall, context: CallContext, spend: bool) -> tuple[Decision, dict[str, Any]]:
        capability = self.capability(call.capability)
        if capability is None:
            return Decision(Verdict.DENY, f"{call.capability} is not a registered capability."), {}
        refusal = self._malformed(call, capability, context)
        if refusal:
            return Decision(Verdict.DENY, refusal), {}
        arguments = self._resolved(capability, call.arguments)
        physical = Effect.PHYSICAL in capability.effects
        grant = None if physical else self._grant(capability, context.user, call.arguments)
        approval = self._approved(call, context, arguments)
        if context.actor.startswith(SKILL_PREFIX) and grant is None:
            return self._denied(capability, context, "A skill gets only what a grant already covers."), {}
        if physical:
            return (self._by_approval(approval, spend) if approval else self._ask(call, context, "physical", arguments)), arguments
        tainted = context.tainted or context.actor.startswith(PEER_PREFIX)
        if capability.consequential and tainted:
            if approval:
                return self._by_approval(approval, spend), arguments
            if grant is None:
                return self._denied(capability, context, "Nothing covers this, and it came from a tainted context."), {}
            return self._ask(call, context, "tainted", arguments), arguments
        if approval:
            return self._by_approval(approval, spend), arguments
        if grant is None:
            return self._denied(capability, context, "No grant covers this."), {}
        return self._by_grant(grant, spend), arguments

    def _malformed(self, call: ToolCall, capability: Capability, context: CallContext) -> str | None:
        if not context.user:
            return "A call must name the user it is for."
        if not isinstance(call.arguments, dict):
            return "Arguments must be a JSON object."
        try:
            canonical(call.arguments)
        except (TypeError, ValueError, RecursionError):
            return "Arguments must be plain JSON values, nested no deeper than JSON can carry."
        problems = errors(call.arguments, capability.input_schema, "arguments") if capability.input_schema else []
        if problems:
            return " ".join(problems[:5])
        field = capability.scope_field
        if field and field in call.arguments and scope_kind(capability) == "path":
            values = call.arguments[field]
            for value in values if isinstance(values, (list, tuple)) else [values]:
                if real_path(value) is None:
                    return f"{field} must be an absolute path, because a grant is compared to the real location."
        return None

    @staticmethod
    def _resolved(capability: Capability, arguments: dict[str, Any]) -> dict[str, Any]:
        """Hand the handler the path that was judged, so it cannot use a different one."""
        field = capability.scope_field
        if not field or field not in arguments or scope_kind(capability) != "path":
            return dict(arguments)
        value = arguments[field]
        if isinstance(value, (list, tuple)):
            return {**arguments, field: [real_path(item) for item in value]}
        return {**arguments, field: real_path(value)}

    def _denied(self, capability: Capability, context: CallContext, why: str) -> Decision:
        scope = "SCOPE " if capability.scope_field else ""
        return Decision(
            Verdict.DENY,
            f"{why} A person can allow {capability.name} with: "
            f"isharded assistant grant add {capability.name} {scope}--user {context.user}",
        )

    def _by_grant(self, grant: Any, spend: bool) -> Decision:
        if spend and grant["uses_left"] is not None:
            updated = self.store.connection.execute(
                "UPDATE grants SET uses_left = uses_left - 1 WHERE id = ? AND uses_left > 0", (grant["id"],)
            )
            if updated.rowcount != 1:
                return Decision(Verdict.DENY, "That grant has no uses left.", grant_id=grant["id"])
        return Decision(Verdict.ALLOW, "A grant covers this.", grant_id=grant["id"])

    def _by_approval(self, approval: Any, spend: bool) -> Decision:
        if spend:
            updated = self.store.connection.execute(
                "UPDATE approvals SET status = 'used', used_at = ? WHERE id = ? AND status = 'approved'",
                (self.clock(), approval["id"]),
            )
            if updated.rowcount != 1:
                return Decision(Verdict.DENY, "That approval was already used.", approval_id=approval["id"])
        return Decision(Verdict.ALLOW, "A person approved this exact action.", approval_id=approval["id"])

    def _ask(self, call: ToolCall, context: CallContext, why: str, arguments: dict[str, Any]) -> Decision:
        reason = {
            "physical": f"{call.capability} moves or powers real hardware, so a person approves every action.",
            "tainted": f"{call.capability} has consequences and untrusted text could have steered this call, "
            "so a standing grant does not apply. A person must approve this exact action.",
        }[why]
        approval_id = self.request_approval(context.user, call.capability, arguments, context.actor, context.task_id, reason)
        return Decision(Verdict.ASK, f"{reason} Run: isharded assistant approvals approve {approval_id}", approval_id=approval_id)

    def _record(self, stage: str, call: ToolCall, context: CallContext, decision: Decision) -> None:
        self.audit.record(
            context.actor,
            f"broker.{stage}",
            call.capability,
            {
                "user": context.user,
                "verdict": decision.verdict.value,
                "reason": decision.reason,
                "args_hash": self._safe_hash(call.arguments),
                "tainted": context.tainted or context.actor.startswith(PEER_PREFIX),
                "grant": decision.grant_id,
                "approval": decision.approval_id,
                "call": call.call_id,
            },
            context.task_id,
        )

    def _outcome(self, call: ToolCall, context: CallContext, ok: bool, error: str) -> None:
        detail = {"user": context.user, "ok": ok, "args_hash": self._safe_hash(call.arguments), "call": call.call_id, "error": error}
        self.audit.record(context.actor, "broker.result", call.capability, detail, context.task_id)

    @staticmethod
    def _safe_hash(arguments: Any) -> str:
        try:
            return argument_hash(arguments)
        except (TypeError, ValueError, RecursionError):
            return "unhashable"

    def _provenance(self, call: ToolCall) -> Provenance:
        return Provenance("tool", call.call_id or call.capability, call.capability, Trust.EXTERNAL, self.clock())

    def _refusal(self, call: ToolCall, decision: Decision) -> ToolResult:
        # These words are the broker's own, so they do not taint whoever reads them.
        provenance = Provenance("broker", call.capability, "broker", Trust.VERIFIED, self.clock())
        data = {"refused": True, "verdict": decision.verdict.value, "approval_id": decision.approval_id}
        return ToolResult(False, decision.reason, provenance, data)

    def _normal(self, result: Any, call: ToolCall, context: CallContext) -> ToolResult:
        """Nothing a tool returns is believed unless it proved it, by a check or signature."""
        if not isinstance(result, ToolResult):
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            return ToolResult(True, text, self._provenance(call))
        provenance = result.provenance
        if not isinstance(provenance, Provenance):
            return replace(result, provenance=self._provenance(call))
        return result if provenance.trust is Trust.VERIFIED else replace(result, provenance=replace(provenance, trust=Trust.EXTERNAL))

    # Grants.

    def _live_grants(self, user: str, capability: str) -> list[Any]:
        return self.store.query(
            "SELECT * FROM grants WHERE user = ? AND capability = ? AND revoked_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > ?) AND (uses_left IS NULL OR uses_left > 0) "
            "ORDER BY uses_left IS NOT NULL, granted_at, id",  # spend a limited grant only if nothing else covers it
            (user, capability, self.clock()),
        )

    def _grant(self, capability: Capability, user: str, arguments: dict[str, Any]) -> Any | None:
        if Effect.PHYSICAL in capability.effects:
            return None
        for row in self._live_grants(user, capability.name):
            if scope_covers(capability, row["scope"], arguments):
                return row
        return None

    def add_grant(
        self,
        user: str,
        capability: str,
        scope: str | None,
        by: str,
        expires_at: float | None = None,
        uses: int | None = None,
    ) -> str:
        declared = self.capability(capability)
        if declared is None:
            raise ValueError(f"{capability} is not a registered capability. See: isharded assistant grant list --capabilities")
        if Effect.PHYSICAL in declared.effects:
            raise ValueError(f"{capability} moves or powers real hardware, so it can have no standing grant. A person approves each action.")
        if not is_person(by):
            raise ValueError(f"{by or 'Nobody'} is not a person, and only a person can grant anything.")
        if uses is not None and uses < 1:
            raise ValueError("A grant needs at least one use. To end one, revoke it.")
        stored = self._stored_scope(declared, scope)
        grant_id = f"g-{secrets.token_hex(8)}"
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO grants (id, user, capability, scope, expires_at, uses_left, granted_by, granted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (grant_id, user, capability, stored, expires_at, uses, by, self.clock()),
            )
            detail = {"user": user, "scope": stored, "expires_at": expires_at, "uses": uses, "grant": grant_id}
            self.audit.record(by, "grant.add", capability, detail)
        return grant_id

    @staticmethod
    def _stored_scope(capability: Capability, scope: str | None) -> str:
        if capability.scope_field is None:
            if scope not in (None, ANY_SCOPE):
                raise ValueError(f"{capability.name} has nothing to scope, so a grant for it covers all of it. Leave the scope out.")
            return ANY_SCOPE
        if not scope:
            raise ValueError(f"{capability.name} needs a scope for {capability.scope_field}. To allow everything, say * on purpose.")
        if scope == ANY_SCOPE or scope_kind(capability) != "path":
            if scope != ANY_SCOPE and scope_kind(capability) == "url" and url_host(scope) is None:
                raise ValueError(f"{scope} is not a host name.")
            return scope
        real = real_path(scope)
        if real is None:
            raise ValueError(f"{scope} is not an absolute path.")
        return real  # settled now, so changing a link later cannot widen the grant

    def revoke_grant(self, grant_id: str, by: str) -> bool:
        if not is_person(by):
            raise ValueError(f"{by or 'Nobody'} is not a person, and only a person can revoke a grant.")
        with self.store.transaction():
            updated = self.store.connection.execute(
                "UPDATE grants SET revoked_at = ?, revoked_by = ? WHERE id = ? AND revoked_at IS NULL", (self.clock(), by, grant_id)
            )
            if updated.rowcount:
                self.audit.record(by, "grant.revoke", grant_id, {})
        return updated.rowcount == 1

    def revoke_capability(self, capability: str, by: str) -> int:
        """End every grant for a capability, so reinstalling a tool cannot inherit them."""
        rows = self.store.query("SELECT id FROM grants WHERE capability = ? AND revoked_at IS NULL", (capability,))
        return sum(self.revoke_grant(row["id"], by) for row in rows)

    def grants(self, user: str | None = None, include_ended: bool = False) -> list[dict[str, Any]]:
        now = self.clock()
        clauses, parameters = [], []
        if user is not None:
            clauses.append("user = ?")
            parameters.append(user)
        if not include_ended:
            clauses.append("revoked_at IS NULL AND (expires_at IS NULL OR expires_at > ?) AND (uses_left IS NULL OR uses_left > 0)")
            parameters.append(now)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return [dict(row) for row in self.store.query(f"SELECT * FROM grants {where} ORDER BY granted_at, id", parameters)]

    # Approvals.

    def _approved(self, call: ToolCall, context: CallContext, arguments: dict[str, Any]) -> Any | None:
        """Bound to who asked, so another actor cannot spend someone else's yes.

        And to where a path really leads, not how it was written, so moving a link
        after a person said yes makes it a different action, and shows them the real
        place when they are asked.
        """
        return self.store.one(
            "SELECT * FROM approvals WHERE user = ? AND capability = ? AND args_hash = ? AND actor = ? AND task_id IS ? "
            "AND status = 'approved' ORDER BY requested_at, id LIMIT 1",
            (context.user, call.capability, argument_hash(arguments), context.actor, context.task_id),
        )

    def request_approval(
        self,
        user: str,
        capability: str,
        arguments: dict[str, Any],
        actor: str,
        task_id: str | None = None,
        reason: str = "",
    ) -> str:
        """Put one exact action in front of a person. Asking twice is one question."""
        digest = argument_hash(arguments)
        with self.store.transaction():
            row = self.store.one(
                "SELECT id FROM approvals WHERE user = ? AND capability = ? AND args_hash = ? AND actor = ? AND task_id IS ? "
                "AND status = 'pending' LIMIT 1",
                (user, capability, digest, actor, task_id),
            )
            if row:
                return row["id"]
            approval_id = f"a-{secrets.token_hex(8)}"
            self.store.execute(
                "INSERT INTO approvals (id, user, capability, args_hash, arguments, actor, task_id, status, reason, requested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (approval_id, user, capability, digest, canonical(arguments), actor, task_id, reason, self.clock()),
            )
            self.audit.record(actor, "approval.request", capability, {"user": user, "args_hash": digest, "approval": approval_id}, task_id)
        return approval_id

    def approval(self, approval_id: str) -> dict[str, Any] | None:
        row = self.store.one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        return {**dict(row), "arguments": json.loads(row["arguments"])} if row else None

    def approvals(self, user: str | None = None, status: str | None = "pending") -> list[dict[str, Any]]:
        clauses, parameters = [], []
        if user is not None:
            clauses.append("user = ?")
            parameters.append(user)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.store.query(f"SELECT * FROM approvals {where} ORDER BY requested_at, id", parameters)
        return [{**dict(row), "arguments": json.loads(row["arguments"])} for row in rows]

    def withdraw(self, approval_id: str, why: str) -> bool:
        """The asker no longer needs this, so no person should be left deciding it."""
        with self.store.transaction():
            updated = self.store.connection.execute(
                "UPDATE approvals SET status = 'withdrawn', reason = ? WHERE id = ? AND status IN ('pending', 'approved')",
                (why, approval_id),
            )
            if updated.rowcount:
                self.audit.record("broker", "approval.withdraw", approval_id, {"why": why})
        return updated.rowcount == 1

    def decide(self, approval_id: str, approved: bool, by: str, user: str | None = None) -> bool:
        """A person's answer. Saying no also withdraws a yes that was not used yet."""
        if not is_person(by):
            raise ValueError(f"{by or 'Nobody'} is not a person, and only a person can decide an approval.")
        allowed_from = "('pending')" if approved else "('pending', 'approved')"
        with self.store.transaction():
            updated = self.store.connection.execute(
                f"UPDATE approvals SET status = ?, decided_by = ?, decided_at = ? WHERE id = ? AND status IN {allowed_from} "
                "AND (? IS NULL OR user = ?)",
                ("approved" if approved else "denied", by, self.clock(), approval_id, user, user),
            )
            if updated.rowcount:
                self.audit.record(by, "approval.approve" if approved else "approval.deny", approval_id, {})
        return updated.rowcount == 1


def attach(assistant: Any) -> None:
    assistant.broker = CapabilityBroker(assistant)


# The command line.


def add_commands(commands: Any) -> None:
    grant = commands.add_parser("grant", help="what the assistant may do without asking")
    actions = grant.add_subparsers(dest="grant_command", required=True)
    add = actions.add_parser("add", help="allow a capability within a scope")
    add.add_argument("capability", help="the capability, such as files.read")
    add.add_argument("scope", nargs="?", default=None, help="a folder, a host, or a glob; * allows everything, on purpose")
    add.add_argument("--user", default=DEFAULT_USER, help="who this is for")
    add.add_argument("--by", default=DEFAULT_USER, help="the person granting it, for the record")
    add.add_argument("--expires-in", type=float, default=None, help="seconds until it ends; default is never")
    add.add_argument("--uses", type=int, default=None, help="how many times it works; default is no limit")
    add.set_defaults(run=run_grant_add)
    listing = actions.add_parser("list", help="show grants")
    listing.add_argument("--user", default=None, help="only this user")
    listing.add_argument("--all", action="store_true", help="include ended grants")
    listing.add_argument("--capabilities", action="store_true", help="show the capabilities a grant can name instead")
    listing.set_defaults(run=run_grant_list)
    revoke = actions.add_parser("revoke", help="end a grant at once")
    revoke.add_argument("id", help="the grant, from grant list")
    revoke.add_argument("--by", default=DEFAULT_USER, help="the person revoking it, for the record")
    revoke.set_defaults(run=run_grant_revoke)

    approvals = commands.add_parser("approvals", help="actions waiting for a person")
    choices = approvals.add_subparsers(dest="approvals_command", required=True)
    pending = choices.add_parser("list", help="show what is waiting")
    pending.add_argument("--user", default=None, help="only this user")
    pending.add_argument("--all", action="store_true", help="include decided and used approvals")
    pending.set_defaults(run=run_approvals_list)
    for name, handler, words in (("approve", run_approvals_approve, "allow this one exact action, once"), ("deny", run_approvals_deny, "refuse it")):
        choice = choices.add_parser(name, help=words)
        choice.add_argument("id", help="the approval, from approvals list")
        choice.add_argument("--by", default=DEFAULT_USER, help="the person deciding, for the record")
        choice.set_defaults(run=handler)


def _when(moment: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(moment)) if moment else "never"


def run_grant_add(assistant: Any, args: Any) -> int:
    expires = assistant.clock() + args.expires_in if args.expires_in is not None else None
    try:
        grant_id = assistant.broker.add_grant(args.user, args.capability, args.scope, args.by, expires, args.uses)
    except ValueError as error:
        print(error)
        return 1
    print(f"Granted {args.capability} to {args.user} as {grant_id}.")
    return 0


def run_grant_list(assistant: Any, args: Any) -> int:
    if args.capabilities:
        for capability in assistant.broker.capabilities():
            effects = ",".join(sorted(effect.value for effect in capability.effects))
            print(f"{capability.name:<28} {effects:<28} {capability.summary}")
        return 0
    rows = assistant.broker.grants(args.user, args.all)
    for row in rows:
        uses = "no limit" if row["uses_left"] is None else f"{row['uses_left']} uses left"
        ended = f"  revoked by {row['revoked_by']}" if row["revoked_at"] else ""
        print(f"{row['id']}  {row['user']:<12} {row['capability']:<20} {row['scope']}  ends {_when(row['expires_at'])}, {uses}{ended}")
    if not rows:
        print("No grants. The assistant can do nothing that needs one.")
    return 0


def run_grant_revoke(assistant: Any, args: Any) -> int:
    try:
        ended = assistant.broker.revoke_grant(args.id, args.by)
    except ValueError as error:
        print(error)
        return 1
    print(f"Revoked {args.id}." if ended else f"{args.id} is not a live grant. See: isharded assistant grant list --all")
    return 0 if ended else 1


def run_approvals_list(assistant: Any, args: Any) -> int:
    rows = assistant.broker.approvals(args.user, None if args.all else "pending")
    for row in rows:
        task = f"  task {row['task_id']}" if row["task_id"] else ""
        print(f"{row['id']}  {row['status']:<9} {row['user']:<12} {row['capability']}  by {row['actor']}{task}")
        print(f"    {json.dumps(redacted(row['arguments']), ensure_ascii=False)}")
        print(f"    hash {row['args_hash']}")
        if row["reason"]:
            print(f"    {row['reason']}")
    if not rows:
        print("Nothing is waiting for a person.")
    return 0


def _decide_from_command(assistant: Any, args: Any, approved: bool) -> int:
    try:
        changed = assistant.broker.decide(args.id, approved, args.by)
    except ValueError as error:
        print(error)
        return 1
    print(f"{'Approved' if approved else 'Denied'} {args.id}." if changed else f"{args.id} is not waiting for a decision. See: isharded assistant approvals list --all")
    return 0 if changed else 1


def run_approvals_approve(assistant: Any, args: Any) -> int:
    return _decide_from_command(assistant, args, True)


def run_approvals_deny(assistant: Any, args: Any) -> int:
    return _decide_from_command(assistant, args, False)
