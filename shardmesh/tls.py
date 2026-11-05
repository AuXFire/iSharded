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
import os
import ssl
from typing import Any

# The verified node name on the connection serving this request, if any.
PEER: ContextVar[str | None] = ContextVar("shardmesh_peer", default=None)


def client_verify() -> Any:
    """What internal clients trust, and who they say they are."""
    authority = os.getenv("SHARDMESH_CA_FILE")
    if not authority:
        return True
    context = ssl.create_default_context(cafile=authority)
    certificate, key = os.getenv("SHARDMESH_TLS_CERT"), os.getenv("SHARDMESH_TLS_KEY")
    if certificate and key:
        context.load_cert_chain(certificate, key)
    return context


def server_options() -> dict[str, Any]:
    certificate = os.getenv("SHARDMESH_TLS_CERT")
    key = os.getenv("SHARDMESH_TLS_KEY")
    if bool(certificate) != bool(key):
        raise ValueError("TLS needs both a certificate and a key")
    if not certificate:
        return {}
    options: dict[str, Any] = {"ssl_certfile": certificate, "ssl_keyfile": key}
    authority = os.getenv("SHARDMESH_CA_FILE")
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
