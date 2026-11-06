"""Proving that a person asked, without carrying their key.

A person gives an API key to one node, and it stays there, because anyone can claim
to coordinate. When that node passes the request on, the next node must decide who
asked. A node's certificate cannot answer that: it proves which node is talking, and
every member holds one. Nothing built on a key's hash can answer it either, because
every member holds the list of hashes.

So each key also stands for an Ed25519 key pair, derived from it. The public half, the
verifier, sits beside the hash in the signed key entry. The node holding the key signs
a short statement of exactly what was asked, by which node, of which control plane,
and when. The node receiving it takes the name and role from the key entry and nothing
from the request, so a node can only claim what it can prove.

A front door reached through a forward has no key to sign with, and may still have to
pass the request to a lane. It sends the statement it was given. A lane accepts that
only from the node the statement was addressed to, and it cannot check the body, which
the front door rewrites. The most a dishonest coordinator gets from this is the action
that person asked for, on a lane, once for each lane.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from typing import Any

from .access import ROLE_USER, digest
from .tickets import CLOCK_SKEW_SECONDS, _decode, _encode

# Long enough for a large request to arrive, since a statement is signed as it is sent.
STATEMENT_SECONDS = 30
# A front door waits for a new lane's workers before passing on a load, and it has no
# key to sign a newer statement with. Above LANE_WORKERS_WAIT_SECONDS.
PASSED_ON_SECONDS = 240
# Per key, so that nobody who holds a key can use up anyone else's.
MAX_NONCES = 16384
DERIVATION = b"isharded forward statement 1"
VERIFIER_LENGTH = 43  # 32 bytes, unpadded


class Refused(ValueError):
    """A statement proves nothing. Says why, for the node that sent it."""


def _signing_key(key: str) -> Any:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    seed = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=DERIVATION).derive(key.encode())
    return Ed25519PrivateKey.from_private_bytes(seed)


def key_verifier(key: str) -> str:
    """What the access list holds so that a key can prove itself. It reveals nothing of the key."""
    from cryptography.hazmat.primitives import serialization

    public = _signing_key(key).public_key()
    return _encode(public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))


def body_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def make_statement(key: str, sender: str, target: str, method: str, path: str, body: bytes = b"", clock: Any = time.time) -> str:
    """This node's word, signed with the person's key, that they asked for exactly this."""
    claims = {
        "key": digest(key),
        "from": sender,
        "to": target,
        "method": method,
        "path": path,
        "body": body_hash(body),
        "at": clock(),
        "nonce": secrets.token_hex(16),
    }
    text = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode()
    return f"{_encode(text)}.{_encode(_signing_key(key).sign(text))}"


def _read(statement: str) -> tuple[dict[str, Any], bytes, bytes]:
    try:
        text, signature = statement.split(".")
        raw = _decode(text)
        claims = json.loads(raw)
        if not isinstance(claims, dict):
            raise TypeError
        claims["at"] = float(claims["at"])
        for field in ("key", "from", "to", "method", "path", "body", "nonce"):
            claims[field] = str(claims[field])
        return claims, raw, _decode(signature)
    except (KeyError, RecursionError, TypeError, ValueError) as exc:
        raise Refused("this statement cannot be read; if a node was just updated, update them all") from exc


class Statements:
    """Checks statements, and remembers them for as long as each could be used again."""

    def __init__(self, clock: Any = time.time) -> None:
        self.clock = clock
        self.seen: dict[str, dict[str, float]] = {}

    def _first_use(self, key: str, nonce: str, forget_at: float, now: float) -> bool:
        held = self.seen.setdefault(key, {})
        if nonce in held:
            return False
        if len(held) >= MAX_NONCES:
            held = self.seen[key] = {item: until for item, until in held.items() if until > now}
            if len(held) >= MAX_NONCES:
                return False  # refusing is safe, and forgetting one would let it work twice
        held[nonce] = forget_at
        return True

    def check(self, access: Any, statement: str, peer: str, here: str, method: str, path: str, body: bytes, in_lane: bool = False) -> tuple[str, str]:
        """Who a statement proves is asking, as (name, role). Refused if it proves nothing.

        ``peer`` is the verified node on the connection, and ``here`` is how a statement
        names this control plane.
        """
        claims, raw, signature = _read(statement)
        caller = access.keys.get(claims["key"])
        if caller is None:
            raise Refused("this key is not on the access list here")
        passed_on = claims["to"] != here
        if passed_on and not (in_lane and claims["to"] == peer):
            raise Refused(f"this statement is for {claims['to'][:64]}, not {here}")
        if not passed_on and claims["from"] != peer:
            raise Refused(f"this statement was made by another node, not {peer}")
        # A front door rewrites what it passes on, so only the first hop can compare it all.
        said = (claims["method"], claims["path"].partition("?")[0]) if passed_on else (claims["method"], claims["path"], claims["body"])
        asked = (method, path.partition("?")[0]) if passed_on else (method, path, body_hash(body))
        if said != asked:
            raise Refused("this statement is for another request")
        window = (PASSED_ON_SECONDS if passed_on else STATEMENT_SECONDS) + CLOCK_SKEW_SECONDS
        now = self.clock()
        if not math.isfinite(claims["at"]) or now - claims["at"] > window or claims["at"] - now > CLOCK_SKEW_SECONDS:
            raise Refused("this statement is too old, or the clocks on these nodes differ by over a minute")
        if not caller.verifier:
            # Made before keys had verifiers. Every member could say this, so it proves
            # nothing, and it buys nothing a member could not reach anyway.
            return caller.name, ROLE_USER
        self._verify(caller.verifier, signature, raw)
        if not self._first_use(claims["key"], claims["nonce"], claims["at"] + window, now):
            raise Refused("this statement has already been used")
        return caller.name, caller.role

    @staticmethod
    def _verify(verifier: str, signature: bytes, raw: bytes) -> None:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(_decode(verifier)).verify(signature, raw)
        except (InvalidSignature, ValueError) as exc:
            raise Refused("this statement was not signed with this key") from exc
