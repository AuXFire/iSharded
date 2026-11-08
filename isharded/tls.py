"""Encryption and identity between nodes.

Tailscale encrypts its own links, and a closed network may not need to. A set of
remote machines over the public internet does. With a pool authority, nodes also
prove who they are to each other, so one node can be revoked by name.

Client certificates are optional at the TLS layer, because people connect with
browsers and API keys. The node protocol checks for one, or for the pool token,
on every request.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
import os
import socket
import ssl
from typing import Any

# The verified node name on the connection serving this request, if any.
PEER: ContextVar[str | None] = ContextVar("isharded_peer", default=None)
# Who a request is proven to come from, when a node passed it on: (name, role).
BEHALF: ContextVar[tuple[str, str] | None] = ContextVar("isharded_behalf", default=None)
# The key a person gave this node, for as long as their request lasts here.
KEY_HELD: ContextVar[str | None] = ContextVar("isharded_key_held", default=None)
# The statement that proved BEHALF, when the person's key stayed on another node.
STATEMENT: ContextVar[str | None] = ContextVar("isharded_statement", default=None)
# The connection serving this request, for marking it as one a node holds.
CONNECTION: ContextVar[Any] = ContextVar("isharded_connection", default=None)
# Fingerprints of node keys the owner retired, in this process. A certificate for one still
# checks out against the authority, so every place that takes a node's name looks here too.
RETIRED_KEYS: set[str] = set()
FORWARD_MARK = "x-isharded-forwarded"
BEHALF_HEADER = "x-isharded-on-behalf"


OWN_ADDRESS = "ISHARDED_OWN_ADDRESS"
# Measured from a home connection to five data centers: about 1 in 10 new connections
# failed, against none between the data centers. A failed connect never reached the
# other side, so trying again is always safe.
CONNECT_RETRIES = 3
# A new connection costs two or three round trips, one hop after another down a route,
# so a reply after a pause should find the old ones still open.
NODE_KEEP_ALIVE_SECONDS = 3600.0
# An idle connection gets a probe this often. That holds its place in a home router, which
# can forget one in minutes, and finds a peer that went without a word before a reply does.
PROBE_IDLE_SECONDS = 30
PROBE_INTERVAL_SECONDS = 10
PROBE_COUNT = 3


@lru_cache(maxsize=8)
def _own_addresses(setting: str) -> frozenset[str]:
    return frozenset(item.strip() for item in setting.split(",") if item.strip())


def stay_on_this_machine(request: Any) -> None:
    """Send a request for this node's own public address over loopback.

    Most NAT, cloud networks and home routers included, will not loop a machine's
    public address back to it, so a worker could never reach the control plane
    beside it. Certificates cover 127.0.0.1, and the Host header stays as it was.

    The setting is a host, or for a node behind a relay it is a list of host:port,
    as other nodes can share that host.
    """
    setting = os.getenv(OWN_ADDRESS)
    if not setting:
        return
    own = _own_addresses(setting)
    if request.url.host in own or f"{request.url.host}:{request.url.port}" in own:
        request.url = request.url.copy_with(host="127.0.0.1")


async def stay_on_this_machine_async(request: Any) -> None:
    stay_on_this_machine(request)


def node_proven() -> None:
    """Say that this request's connection belongs to a node, so it can stay open for long."""
    connection = CONNECTION.get()
    if connection is not None:
        connection.proven = True


def probe_options() -> list[tuple[int, int, int]]:
    """Socket options that have the system probe a connection while it sits idle."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = "TCP_KEEPIDLE" if hasattr(socket, "TCP_KEEPIDLE") else "TCP_KEEPALIVE"  # a Mac has only the second
    timing = {idle: PROBE_IDLE_SECONDS, "TCP_KEEPINTVL": PROBE_INTERVAL_SECONDS, "TCP_KEEPCNT": PROBE_COUNT}
    return options + [(socket.IPPROTO_TCP, getattr(socket, name), value) for name, value in timing.items() if hasattr(socket, name)]


def node_transport() -> Any:
    """How one node reaches another: it tries a failed connection again, and keeps a good one."""
    import httpx

    limits = httpx.Limits(max_connections=100, max_keepalive_connections=100, keepalive_expiry=NODE_KEEP_ALIVE_SECONDS)
    return httpx.AsyncHTTPTransport(retries=CONNECT_RETRIES, verify=client_verify(), limits=limits, socket_options=probe_options())


def client_verify() -> Any:
    """What internal clients trust, and who they say they are."""
    authority = os.getenv("ISHARDED_CA_FILE")
    if not authority:
        return True
    context = ssl.create_default_context(cafile=authority)
    certificate, key = os.getenv("ISHARDED_TLS_CERT"), os.getenv("ISHARDED_TLS_KEY")
    if certificate and key:
        context.load_cert_chain(certificate, key)
    return context


# Uvicorn's default, "TLSv1", leaves TLS 1.2 with old CBC ciphers and anonymous ones, and
# nothing a current Python client offers. These have forward secrecy and authenticated
# encryption. TLS 1.3 has its own list, so this leaves it alone.
SERVER_CIPHERS = "ECDHE+AESGCM:ECDHE+CHACHA20"


def server_options() -> dict[str, Any]:
    certificate = os.getenv("ISHARDED_TLS_CERT")
    key = os.getenv("ISHARDED_TLS_KEY")
    if bool(certificate) != bool(key):
        raise ValueError("TLS needs both a certificate and a key")
    if not certificate:
        return {}
    options: dict[str, Any] = {"ssl_certfile": certificate, "ssl_keyfile": key, "ssl_ciphers": SERVER_CIPHERS}
    authority = os.getenv("ISHARDED_CA_FILE")
    if authority:
        options["ssl_ca_certs"] = authority
        options["ssl_cert_reqs"] = ssl.CERT_OPTIONAL
    return options


def intercepted(error: BaseException) -> bool:
    """Whether a failed connection looks like someone replacing certificates."""
    return "CERTIFICATE_VERIFY_FAILED" in str(error)


INTERCEPTED = (
    "a node's certificate did not come from the pool authority. If antivirus scans HTTPS on this "
    "machine, it is probably replacing certificates: exclude this pool's ports from scanning, or use "
    "tickets (--identity-cert, --identity-key) on a private link. See docs/SECURITY.md"
)


def peer_name(certificate: dict[str, Any] | None) -> str | None:
    """The node name in a certificate the TLS layer has already verified."""
    for field in (certificate or {}).get("subject", ()):
        for key, value in field:
            if key == "commonName":
                return str(value)
    return None


def retire_keys(fingerprints: Any) -> None:
    """Add to this process's retired keys. Only ever more: a retirement is for good."""
    RETIRED_KEYS.update(str(item) for item in fingerprints)


def retired_certificate(certificate: bytes | None) -> bool:
    """Whether a verified certificate, in DER, is for a key the owner retired."""
    if not RETIRED_KEYS or not certificate:
        return False
    from .pool import certificate_fingerprint

    return certificate_fingerprint(certificate) in RETIRED_KEYS
