"""Node identity that works where TLS client certificates cannot.

Antivirus products often intercept TLS and replace the certificates in it, which
ends mutual TLS between nodes. Failing closed is right, but a pool should not need
every machine's antivirus reconfigured. So a node can prove its name with a ticket
instead: a few signed lines in an ordinary header, made with the same certificate
and key, and checked against the same authority.

A ticket names the node it is meant for, so a peer that receives one cannot use it
elsewhere, and it expires in minutes. Tickets prove identity and encrypt nothing,
so the link still needs Tailscale, or a network nobody else can read.
"""

from __future__ import annotations

import base64
import datetime
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .tls import PEER, RETIRED_KEYS

PREFIX = "smn-"
# On a request: who it is for, when that is not who the URL names.
AUDIENCE = "isharded_audience"
TICKET_SECONDS = 120
# Clocks differ between machines, and a ticket from slightly ahead is still good.
CLOCK_SKEW_SECONDS = 60
MAX_REMEMBERED = 4096


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TicketMaker:
    """Makes this node's tickets, one per destination, and reuses each for a while."""

    def __init__(self, certificate_path: str, key_path: str, clock: Any = time.time) -> None:
        from cryptography.hazmat.primitives import serialization

        self.certificate = Path(certificate_path).read_bytes()
        self.key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
        self.clock = clock
        self.held: dict[str, tuple[float, str]] = {}
        self.lock = threading.Lock()

    def ticket(self, audience: str) -> str:
        now = self.clock()
        with self.lock:
            held = self.held.get(audience)
            if held and held[0] > now + TICKET_SECONDS / 2:
                return held[1]
            expires = now + TICKET_SECONDS
            ticket = self._signed(audience, expires)
            self.held[audience] = (expires, ticket)
            return ticket

    def fresh(self, audience: str) -> str:
        """A ticket for one use, so nothing here grows with each challenge."""
        return self._signed(audience, self.clock() + TICKET_SECONDS)

    def _signed(self, audience: str, expires: float) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        body = json.dumps(
            {"aud": audience, "exp": expires, "cert": self.certificate.decode()}, sort_keys=True, separators=(",", ":")
        ).encode()
        signature = self.key.sign(body, ec.ECDSA(hashes.SHA256()))
        return f"{PREFIX}{_encode(body)}.{_encode(signature)}"


class TicketChecker:
    """Checks tickets against the pool authority. Each one is only checked once."""

    def __init__(self, authority_path: str, audiences: set[str], clock: Any = time.time) -> None:
        from cryptography import x509

        self.authority = x509.load_pem_x509_certificate(Path(authority_path).read_bytes())
        self.audiences = audiences
        self.clock = clock
        self.known: dict[str, tuple[float, str, str]] = {}

    def name(self, ticket: str) -> str | None:
        """The verified node name in a ticket, if it is good for this node right now."""
        now = self.clock()
        known = self.known.get(ticket)
        if known:
            # A key retired after its ticket was checked counts for nothing from then on.
            return known[1] if known[0] > now and known[2] not in RETIRED_KEYS else None
        verified = self._verify(ticket, now)
        if verified is None or verified[2] in RETIRED_KEYS:
            return None
        if len(self.known) >= MAX_REMEMBERED:
            self.known = {key: value for key, value in self.known.items() if value[0] > now}
        self.known[ticket] = verified
        return verified[1]

    def check_issuer(self, certificate: Any) -> None:
        certificate.verify_directly_issued_by(self.authority)

    def _verify(self, ticket: str, now: float) -> tuple[float, str, str] | None:
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        from .pool import key_fingerprint

        try:
            body_text, signature_text = ticket[len(PREFIX) :].split(".")
            body = _decode(body_text)
            claims = json.loads(body)
            expires = float(claims["exp"])
            if not math.isfinite(expires) or claims["aud"] not in self.audiences:
                return None
            if expires <= now or expires > now + TICKET_SECONDS + CLOCK_SKEW_SECONDS:
                return None
            certificate = x509.load_pem_x509_certificate(str(claims["cert"]).encode())
            self.check_issuer(certificate)
            moment = datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
            if not certificate.not_valid_before_utc <= moment <= certificate.not_valid_after_utc:
                return None
            certificate.public_key().verify(_decode(signature_text), body, ec.ECDSA(hashes.SHA256()))
            names = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            return expires, str(names[0].value), key_fingerprint(certificate.public_key())
        except (InvalidSignature, KeyError, IndexError, TypeError, ValueError):
            return None


class NodeTickets(httpx.Auth):
    """Adds a ticket for whoever a request goes to, unless it already speaks for someone."""

    def __init__(self, maker: TicketMaker) -> None:
        self.maker = maker

    def auth_flow(self, request: httpx.Request) -> Any:
        if "authorization" not in request.headers:
            audience = request.extensions.get(AUDIENCE) or request.url.netloc.decode()
            request.headers["Authorization"] = f"Bearer {self.maker.ticket(audience)}"
        yield request


def reply_audience(url: str) -> str:
    """A gossip reply's proof names who asked, so it is no use anywhere else."""
    return f"reply:{url}"


def client_auth() -> NodeTickets | None:
    """Tickets for this node's outgoing requests, if it has an identity for them."""
    certificate, key = os.getenv("ISHARDED_IDENTITY_CERT"), os.getenv("ISHARDED_IDENTITY_KEY")
    return NodeTickets(TicketMaker(certificate, key)) if certificate and key else None


def own_audiences() -> set[str]:
    """The addresses a ticket may name to be good here."""
    audiences = set()
    for variable in ("ISHARDED_NODE_URL", "ISHARDED_ADVERTISE_URL"):
        parts = urlsplit(os.getenv(variable) or "")
        if parts.netloc:
            audiences.add(parts.netloc)
            if parts.port:
                audiences |= {f"127.0.0.1:{parts.port}", f"localhost:{parts.port}"}
    return audiences


def make_checker(audiences: set[str]) -> TicketChecker | None:
    """The checker for this network, whether it has an authority or nobody owns it."""
    folder = os.getenv("ISHARDED_OWNERLESS")
    if folder:
        from .commons import CommonsChecker

        return CommonsChecker(folder, audiences)
    authority = os.getenv("ISHARDED_CA_FILE")
    return TicketChecker(authority, audiences) if authority else None


class TicketIdentity:
    """Treats a good ticket as the TLS layer treats a verified certificate."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.checker: TicketChecker | None = None
        self.ready = False

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            if not self.ready:
                self.ready = True
                self.checker = make_checker(own_audiences())
            if self.checker is not None:
                presented = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
                if presented.startswith("Bearer " + PREFIX):
                    name = self.checker.name(presented[7:])
                    if name is not None:
                        PEER.set(name)
        await self.app(scope, receive, send)
