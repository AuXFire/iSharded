"""Who may do what, and a record of what they did.

Three kinds of caller reach a control plane. Nodes prove themselves with a
certificate, or with the pool token where there is no pool authority. People and
applications use API keys, which name them and carry a role. With none of these
the pool is open, which suits a trusted closed network.

A certificate is good for node routes and nothing else. With a pool authority the
token counts for nothing, so a revoked node that kept a copy of it stays out, and
operators use admin keys.

Keys are stored as hashes, so a key file gives nothing away. The list of keys and
revoked nodes is one versioned document. An admin changes it through any node, and
gossip carries it to the rest, so nothing has to be copied by hand. Revocations and
removed keys are kept for good and merged entry by entry, so neither a split nor a
forged document can bring one back. To return a machine, issue it a new name.

With a pool authority, every key and revocation is an entry signed with the
authority's private key, which stays off the nodes. Nodes check the signature and
ignore anything else, so a compromised node cannot add a key or revoke a peer.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable

ROLE_USER = "user"
ROLE_ADMIN = "admin"
ROLE_NODE = "node"
# A node proven by its certificate: node routes only, however it got here.
ROLE_PEER = "peer"
RANKS = {ROLE_PEER: 0, ROLE_USER: 1, ROLE_ADMIN: 2, ROLE_NODE: 3}
WORKER_KINDS = {"gpu", "cpu", "memory"}
AUDIT_ROTATE_BYTES = 64 * 1024 * 1024
MAX_JOIN_REQUESTS = 64


def node_of(worker_id: str) -> str:
    """The node a worker belongs to. Its workers are named for it."""
    node, _, kind = worker_id.rpartition("-")
    return node if node and kind in WORKER_KINDS else worker_id


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def canonical(entry: dict[str, Any]) -> bytes:
    """The exact bytes an entry's signature covers."""
    return json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()


class NeedsAuthority(ValueError):
    """Only the pool authority can make this change."""


class Authority:
    """Checks signatures against the pool authority's certificate."""

    def __init__(self, certificate_path: str) -> None:
        try:
            from cryptography import x509
        except ImportError as exc:
            raise SystemExit("A pool with an authority needs the cryptography package: pip install shardmesh[pool]") from exc
        self.key = x509.load_pem_x509_certificate(Path(certificate_path).read_bytes()).public_key()

    def verified(self, signed: Any) -> dict[str, Any] | None:
        """The entry, if the authority really signed it."""
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        try:
            entry, signature = signed["entry"], base64.b64decode(signed["signature"], validate=True)
            self.key.verify(signature, canonical(entry), ec.ECDSA(hashes.SHA256()))
        except (InvalidSignature, KeyError, TypeError, ValueError):
            return None
        return entry if isinstance(entry, dict) else None


@dataclass(slots=True)
class Caller:
    name: str
    role: str
    requests_per_minute: int = 0
    tokens: float = 0.0
    refilled_at: float | None = None


@dataclass
class AccessControl:
    pool_token: str | None = None
    keys: dict[str, Caller] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic
    version: int = 0
    revoked: set[str] = field(default_factory=set)
    # Hashes of removed keys, kept so a stale peer cannot hand one back.
    removed: set[str] = field(default_factory=set)
    # Nodes taken out of service on purpose, for maintenance.
    cordoned: set[str] = field(default_factory=set)
    path: Path | None = None
    # Nodes prove themselves with certificates, so nothing is open even with no token.
    identities: bool = False
    authority: Authority | None = None
    # Signed entries, by the hash of what they say, since one entry can carry many
    # valid signatures. Everything a pool with an authority shares.
    signed: dict[str, dict[str, Any]] = field(default_factory=dict)
    removed_names: set[str] = field(default_factory=set)
    # Requests to join, by node name, until the owner answers. Carried by gossip
    # unsigned: a request grants nothing, and the owner checks its fingerprint.
    join_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Certificates the owner issued, for new members to collect.
    certificates: dict[str, str] = field(default_factory=dict)
    denied: set[str] = field(default_factory=set)

    @classmethod
    def load(
        cls,
        pool_token: str | None,
        path: str | None,
        identities: bool = False,
        authority: Authority | None = None,
    ) -> "AccessControl":
        control = cls(pool_token=pool_token, path=Path(path) if path else None, identities=identities, authority=authority)
        if control.path and control.path.is_file():
            document = json.loads(control.path.read_text(encoding="utf-8"))
            # Keys in this file are trusted here, because an operator can write it,
            # and go no further. Signed entries are checked again, as peers' would be.
            control._apply(document)
            control._take_signed(document)
        return control

    def _apply(self, document: dict[str, Any]) -> None:
        keys: dict[str, Caller] = {}
        for item in document.get("keys", []):
            role = item.get("role", ROLE_USER)
            if role not in {ROLE_USER, ROLE_ADMIN}:
                raise ValueError(f"key {item.get('name')!r} has unknown role {role!r}")
            held = self.keys.get(item["sha256"])
            caller = Caller(item["name"], role, int(item.get("requests_per_minute", 0)))
            if held is not None:  # an allowance survives an unrelated change
                caller.tokens, caller.refilled_at = held.tokens, held.refilled_at
            keys[item["sha256"]] = caller
        self.keys = keys
        self.revoked = set(document.get("revoked_nodes", []))
        self.removed = set(document.get("removed_keys", []))
        self.cordoned = set(document.get("cordoned_nodes", []))
        self.version = int(document.get("version", 0))

    def document(self) -> dict[str, Any]:
        keys = [
            {"name": caller.name, "role": caller.role, "sha256": hashed, "requests_per_minute": caller.requests_per_minute}
            for hashed, caller in sorted(self.keys.items())
        ]
        return {
            "version": self.version,
            "keys": keys,
            "revoked_nodes": sorted(self.revoked),
            "removed_keys": sorted(self.removed),
            "cordoned_nodes": sorted(self.cordoned),
            "signed": [self.signed[held] for held in sorted(self.signed)],
            "join_requests": [self.join_requests[name] for name in sorted(self.join_requests)],
        }

    def _take_join_requests(self, document: dict[str, Any]) -> None:
        for held in document.get("join_requests", [])[:MAX_JOIN_REQUESTS]:
            if not isinstance(held, dict) or not isinstance(held.get("request"), str):
                continue
            try:
                from .pool import read_membership_request

                details = read_membership_request(held["request"])
            except (ImportError, ValueError):
                continue  # each node checks for itself, since gossip proves nothing
            self.add_join_request(details, held["request"])

    def _take_signed(self, document: dict[str, Any]) -> bool:
        """Apply new signed entries from a document. Anything unsigned is ignored."""
        if self.authority is None:
            return False
        changed = False
        for signed in document.get("signed", []):
            claimed = signed.get("entry") if isinstance(signed, dict) else None
            if not isinstance(claimed, dict):
                continue
            held = hashlib.sha256(canonical(claimed)).hexdigest()
            if held in self.signed:
                continue
            entry = self.authority.verified(signed)
            if entry is not None and self._apply_entry(entry):
                self.signed[held] = signed
                changed = True
        return changed

    def signed_revocations(self) -> list[dict[str, Any]]:
        return [signed for signed in self.signed.values() if signed["entry"].get("type") == "revoke"]

    def _apply_entry(self, entry: dict[str, Any]) -> bool:
        kind, name = entry.get("type"), str(entry.get("name", ""))
        if kind == "key" and entry.get("role") in {ROLE_USER, ROLE_ADMIN} and isinstance(entry.get("sha256"), str):
            if name not in self.removed_names:
                self.keys[entry["sha256"]] = Caller(name, entry["role"], int(entry.get("requests_per_minute", 0)))
            return True
        if kind == "remove_key" and name:
            self.removed_names.add(name)
            for hashed in [hashed for hashed, caller in self.keys.items() if caller.name == name]:
                del self.keys[hashed]
                self.removed.add(hashed)
            return True
        if kind == "revoke" and name:
            self.revoked.add(name)
            self.join_requests.pop(name, None)
            return True
        if kind == "member" and name and isinstance(entry.get("certificate"), str):
            self.certificates[name] = entry["certificate"]
            self.join_requests.pop(name, None)
            return True
        if kind == "deny" and name:
            # By the request, not the name: asking again with a new key is a new request.
            self.denied.add(str(entry.get("fingerprint", "")))
            held = self.join_requests.get(name)
            if held and held.get("fingerprint") in self.denied:
                del self.join_requests[name]
            return True
        return False

    def add_join_request(self, details: dict[str, Any], request: str) -> str:
        """Hold a request for the owner. Says what happened, for whoever asked."""
        name = details["name"]
        if name in self.revoked or details["fingerprint"] in self.denied:
            return "refused"
        if name in self.certificates:
            return "approved"
        if name not in self.join_requests and len(self.join_requests) >= MAX_JOIN_REQUESTS:
            return "busy"
        self.join_requests[name] = {**details, "request": request}
        return "waiting"

    def add_signed(self, signed: Any) -> bool:
        """Take one signed entry from an operator. False if the signature is wrong."""
        if self.authority is None or self.authority.verified(signed) is None:
            return False
        if self._take_signed({"signed": [signed]}):
            self._changed()
        return True

    def _merge_with_authority(self, document: dict[str, Any]) -> bool:
        mine = self.document()
        changed = self._take_signed(document)
        self._take_join_requests(document)
        if self._rank(document) > self._rank(mine):
            cordoned = set(document.get("cordoned_nodes", []))
            changed = changed or cordoned != self.cordoned
            self.cordoned = cordoned
            self.version = max(self.version, int(document.get("version", 0)))
        if changed:
            self._save()
        return changed

    def _rank(self, document: dict[str, Any]) -> tuple[int, str]:
        # Two sides of a split can reach one version. Any fixed rule settles it.
        return int(document.get("version", 0)), hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    def merge(self, document: dict[str, Any] | None) -> bool:
        """Combine a peer's document with this one. True if anything changed.

        Keys and revocations merge entry by entry, because either side of a split
        may hold the only copy of one. Cordons are routine, so the newer list wins.
        """
        if not document:
            return False
        if self.authority is not None:
            return self._merge_with_authority(document)
        mine = self.document()
        newest = document if self._rank(document) > self._rank(mine) else mine
        removed = self.removed | set(document.get("removed_keys", []))
        keys = {item["sha256"]: item for item in [*document.get("keys", []), *mine["keys"]]}
        merged = {
            "version": max(self.version, int(document.get("version", 0))),
            "keys": [item for hashed, item in sorted(keys.items()) if hashed not in removed],
            "revoked_nodes": sorted(self.revoked | set(document.get("revoked_nodes", []))),
            "removed_keys": sorted(removed),
            "cordoned_nodes": sorted(newest.get("cordoned_nodes", [])),
            "signed": mine["signed"],
            "join_requests": mine["join_requests"],
        }
        if merged == mine:
            return False
        self._apply(merged)
        self._save()
        return True

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".part")
        partial.write_text(json.dumps(self.document(), indent=2), encoding="utf-8")
        partial.replace(self.path)

    def _changed(self) -> None:
        self.version += 1
        self._save()

    def _refuse_with_authority(self, change: str) -> None:
        if self.authority is not None:
            raise NeedsAuthority(f"{change} needs the pool authority's signature; use the shardmesh command with --dir")

    def add_key(self, name: str, role: str, requests_per_minute: int = 0) -> str:
        self._refuse_with_authority("a new key")
        if role not in {ROLE_USER, ROLE_ADMIN}:
            raise ValueError(f"role must be {ROLE_USER} or {ROLE_ADMIN}")
        if any(caller.name == name for caller in self.keys.values()):
            raise ValueError(f"a key named {name!r} already exists")
        key = "smk-" + secrets.token_urlsafe(32)
        self.keys[digest(key)] = Caller(name, role, requests_per_minute)
        self._changed()
        return key

    def remove_key(self, name: str) -> bool:
        self._refuse_with_authority("removing a key")
        found = [hashed for hashed, caller in self.keys.items() if caller.name == name]
        for hashed in found:
            del self.keys[hashed]
            self.removed.add(hashed)
        if found:
            self._changed()
        return bool(found)

    def revoke_node(self, name: str) -> None:
        self._refuse_with_authority("revoking a node")
        self.revoked.add(name)
        self._changed()

    def cordon(self, name: str) -> None:
        self.cordoned.add(name)
        self._changed()

    def uncordon(self, name: str) -> None:
        self.cordoned.discard(name)
        self._changed()

    @property
    def open(self) -> bool:
        return not self.pool_token and not self.keys and not self.identities

    def _bearer(self, authorization: str | None) -> str | None:
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:]
        return None

    def identify(self, authorization: str | None, peer: str | None = None) -> Caller | None:
        presented = self._bearer(authorization)
        if peer is not None:
            # The TLS layer verified this name against the pool's authority.
            if peer in self.revoked:
                return None
            return (self.keys.get(digest(presented)) if presented else None) or Caller(peer, ROLE_PEER)
        if self.open:
            return Caller("anonymous", ROLE_NODE)
        if presented is None:
            return None
        if self.pool_token and not self.identities and hmac.compare_digest(presented, self.pool_token):
            return Caller("pool", ROLE_NODE)
        return self.keys.get(digest(presented))

    def permits(self, caller: Caller | None, role: str) -> bool:
        if caller is None:
            return False
        if caller.role == ROLE_PEER:
            return role == ROLE_NODE
        return RANKS[caller.role] >= RANKS[role]

    def spend(self, caller: Caller) -> float:
        """Take one request from a caller's allowance. Returns seconds to wait, or 0."""
        if not caller.requests_per_minute:
            return 0.0
        now = self.clock()
        rate = caller.requests_per_minute / 60.0
        if caller.refilled_at is not None:
            caller.tokens = min(float(caller.requests_per_minute), caller.tokens + (now - caller.refilled_at) * rate)
        else:
            caller.tokens = float(caller.requests_per_minute)
        caller.refilled_at = now
        if caller.tokens >= 1.0:
            caller.tokens -= 1.0
            return 0.0
        return (1.0 - caller.tokens) / rate


def create_key(path: str, name: str, role: str, requests_per_minute: int = 0) -> str:
    """Add a key to a key file and return it. It cannot be read back afterwards."""
    if role not in {ROLE_USER, ROLE_ADMIN}:
        raise ValueError(f"role must be {ROLE_USER} or {ROLE_ADMIN}")
    file = Path(path)
    data = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {"keys": []}
    if any(item["name"] == name for item in data["keys"]):
        raise ValueError(f"a key named {name!r} already exists")
    key = "smk-" + secrets.token_urlsafe(32)
    data["keys"].append({"name": name, "role": role, "sha256": digest(key), "requests_per_minute": requests_per_minute})
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return key


class AuditLog:
    """Who did what, as JSON lines. Never the content of a prompt or reply."""

    def __init__(self, path: str | None, node: str) -> None:
        self.path = Path(path) if path else None
        self.node = node
        self.lock = threading.Lock()

    def record(self, actor: str, method: str, path: str, status: int, client: str | None) -> None:
        if self.path is None:
            return
        entry: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "node": self.node,
            "actor": actor,
            "method": method,
            "path": path,
            "status": status,
            "client": client,
        }
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file() and self.path.stat().st_size > AUDIT_ROTATE_BYTES:
                self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
