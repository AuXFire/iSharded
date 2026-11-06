"""A network nobody owns.

There is no authority here. Each node makes its own key, and its name is derived
from that key, so a name cannot be taken or handed out. Joining is open, or takes
vouches from nodes this node already accepts. Every node works out who it accepts
for itself, so two nodes may disagree, and neither can overrule the other.

This module stands where a pool authority would, with no power over anyone. A
node's operator is an admin of that node and nothing else. To the rest of the
network they are a member, who can propose a model and take their own node out.
Blocks are local. Each node decides what it hosts. Nobody can order a node to do
anything, which also means nobody can remove a bad node for everyone.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .access import canonical
from .tickets import TicketChecker

NAME_PREFIX = "n-"
POLICY_FILE = "policy.json"
MEMBERS_FILE = "members.json"
MEMBERS_CHECK_SECONDS = 1.0
IDENTITY_YEARS = 20
JOINING_OPEN = "open"
JOINING_VOUCHED = "vouched"


def name_of(public_key: Any) -> str:
    """A node's name is its key's fingerprint, so only that key can claim it."""
    from cryptography.hazmat.primitives import serialization

    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return NAME_PREFIX + hashlib.sha256(der).hexdigest()[:20]


def certificate_file(folder: Path | str) -> Path:
    return Path(folder) / "node.pem"


def key_file(folder: Path | str) -> Path:
    return Path(folder) / "node.key"


def make_identity(folder: Path | str) -> str:
    """Make this node's key, if it has none. Returns its name."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    from .pool import _write_private

    root = Path(folder)
    if certificate_file(root).is_file():
        return identity_name(root)
    root.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = name_of(key.public_key())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * IDENTITY_YEARS))
        .sign(key, hashes.SHA256())
    )
    _write_private(
        key_file(root),
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )
    certificate_file(root).write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return name


def identity_name(folder: Path | str) -> str:
    from cryptography import x509

    return name_of(x509.load_pem_x509_certificate(certificate_file(folder).read_bytes()).public_key())


def write_policy(folder: Path | str, joining: str = JOINING_OPEN, quorum: int = 1, trust: list[str] | None = None) -> None:
    if joining not in {JOINING_OPEN, JOINING_VOUCHED}:
        raise ValueError(f"joining must be {JOINING_OPEN} or {JOINING_VOUCHED}")
    Path(folder, POLICY_FILE).write_text(json.dumps({"joining": joining, "quorum": max(int(quorum), 1), "trust": trust or []}), encoding="utf-8")


def read_policy(folder: Path | str) -> dict[str, Any]:
    try:
        policy = json.loads(Path(folder, POLICY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        policy = {}
    return {"joining": policy.get("joining", JOINING_OPEN), "quorum": max(int(policy.get("quorum", 1)), 1), "trust": list(policy.get("trust", []))}


def sign_vouch(folder: Path | str, subject: str) -> dict[str, Any]:
    """This node's word that it accepts another. Nodes that accept this one may count it."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = serialization.load_pem_private_key(key_file(folder).read_bytes(), password=None)
    entry = {"type": "vouch", "name": subject, "by": identity_name(folder), "at": int(time.time())}
    return {
        "entry": entry,
        "signature": base64.b64encode(key.sign(canonical(entry), ec.ECDSA(hashes.SHA256()))).decode(),
        "certificate": certificate_file(folder).read_text(encoding="utf-8"),
    }


def _self_made(certificate: Any) -> None:
    """Raises unless this certificate signed itself and carries the name its key gives."""
    from cryptography.x509.oid import NameOID

    certificate.verify_directly_issued_by(certificate)
    names = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if not names or names[0].value != name_of(certificate.public_key()):
        raise ValueError("a node's name must come from its key")


class Commons:
    """Stands where a pool authority would. It checks signatures and decides nothing for others."""

    ownerless = True

    def __init__(self, folder: Path | str, clock: Any = time.monotonic) -> None:
        self.folder = Path(folder)
        self.me = identity_name(folder)
        self.policy = read_policy(folder)
        self.clock = clock
        self.path = self.folder / MEMBERS_FILE
        self.checked_at = float("-inf")
        self.accepted_names: set[str] | None = None
        self.blocked: set[str] = set()

    def verified(self, signed: Any) -> dict[str, Any] | None:
        """A vouch really made by the node it names. Nothing else can be signed here."""
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        try:
            entry = signed["entry"]
            certificate = x509.load_pem_x509_certificate(str(signed["certificate"]).encode())
            _self_made(certificate)
            key = certificate.public_key()
            if entry.get("type") != "vouch" or entry.get("by") != name_of(key):
                return None
            key.verify(base64.b64decode(signed["signature"], validate=True), canonical(entry), ec.ECDSA(hashes.SHA256()))
        except (AttributeError, InvalidSignature, KeyError, TypeError, ValueError):
            return None
        return entry

    def members(self, signed: dict[str, dict[str, Any]]) -> set[str] | None:
        """Who this node accepts, or None for anyone. It starts from itself and whom it trusts."""
        if self.policy["joining"] == JOINING_OPEN:
            return None
        accepted = {self.me, *self.policy["trust"]}
        vouches: dict[str, set[str]] = {}
        for held in signed.values():
            entry = held["entry"]
            vouches.setdefault(str(entry.get("name")), set()).add(str(entry.get("by")))
        grew = True
        while grew:
            grew = False
            for subject, vouchers in vouches.items():
                if subject not in accepted and len(vouchers & accepted) >= self.policy["quorum"]:
                    accepted.add(subject)
                    grew = True
        return accepted

    def saved(self, access: Any) -> None:
        """Pass the verdict to this node's workers, which check tickets on their own."""
        if os.getenv("ISHARDED_LANE"):
            return  # the node's own view lives with its main control plane
        members = self.members(access.signed)
        document = {"accepted": sorted(members) if members is not None else None, "blocked": sorted(access.revoked)}
        partial = self.path.with_suffix(".part")
        partial.write_text(json.dumps(document), encoding="utf-8")
        partial.replace(self.path)
        self.checked_at = float("-inf")

    def _refresh(self) -> None:
        now = self.clock()
        if now - self.checked_at < MEMBERS_CHECK_SECONDS:
            return
        self.checked_at = now
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            document = {}
        except (OSError, ValueError):
            return  # being replaced right now, and a bad read must not forget everyone
        accepted = document.get("accepted")
        trusted = {self.me, *self.policy["trust"]}
        self.accepted_names = None if self.policy["joining"] == JOINING_OPEN else {*(accepted or ()), *trusted}
        self.blocked = set(document.get("blocked", []))

    def accepts(self, name: str) -> bool:
        self._refresh()
        return name not in self.blocked and (self.accepted_names is None or name in self.accepted_names)


class CommonsChecker(TicketChecker):
    """Checks tickets from nodes that made their own identity."""

    def __init__(self, folder: Path | str, audiences: set[str], clock: Any = time.time) -> None:
        self.commons = Commons(folder)
        self.audiences = audiences
        self.clock = clock
        self.known: dict[str, tuple[float, str]] = {}

    def check_issuer(self, certificate: Any) -> None:
        _self_made(certificate)

    def name(self, ticket: str) -> str | None:
        name = super().name(ticket)
        return name if name and self.commons.accepts(name) else None


def ownerless_folder() -> str | None:
    return os.getenv("ISHARDED_OWNERLESS") or None
